"""
Image YOLOv3 detection layer.
Source: Image_Yolo_Yolov3/models/yolo_layer.py - algorithm unchanged.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# [Unified] no-object ignore rule (2026-10-01).
#   "pred"   : original Darknet YOLOv3 (pjreddie/darknet src/yolo_layer.c, forward_yolo_layer):
#              a prediction whose decoded box overlaps ANY ground truth of the image by more than
#              ignore_thresh (.7 in yolov3.cfg) gets no no-object penalty - at every cell and scale.
#   "anchor" : the Legacy port: only anchors of the ground-truth cell whose SHAPE overlaps the
#              ground truth are ignored, so well-placed predictions of neighbouring cells and of
#              the other two scales are pushed to objectness 0 (hurts multi-scale models).
# The environment variable is only an ablation switch for the convergence study.
IGNORE_MODE = os.environ.get("UNIFIED_YOLO_IGNORE", "pred")

# [Unified] class-loss normalisation (2026-10-02).
#   "pos"  : sum over the classes, mean over the positives - as Darknet (delta for every class of every
#            responsible prediction) and as the anchor-free detectors of this framework (sum / num_fg)
#   "elem" : the Legacy port: mean over positives x classes, i.e. the class signal divided by the number
#            of classes (36 on FashionLetters36, 20 on VOC) - classification learned far too slowly
#            (train cls acc ~45 % with near-zero loss for v3 / v4 / v5 in the 30-epoch validation)
CLS_NORM = os.environ.get("UNIFIED_YOLO_CLS_NORM", "pos")


def _iou_cxcywh(a, b):
    """a [N, 4], b [M, 4] in (cx, cy, w, h) -> IoU [N, M]."""
    a1, a2 = a[:, None, :2] - a[:, None, 2:] / 2, a[:, None, :2] + a[:, None, 2:] / 2
    b1, b2 = b[None, :, :2] - b[None, :, 2:] / 2, b[None, :, :2] + b[None, :, 2:] / 2
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(min=0).prod(-1)
    union = a[:, None, 2:].prod(-1) + b[None, :, 2:].prod(-1) - inter
    return inter / union.clamp(min=1e-9)


def bbox_wh_iou(anchor_wh, target_wh):
    """
    IoU between one anchor (w,h) and N target boxes (w,h).
    Boxes are assumed to share the same center.
    """

    anchor_wh = anchor_wh.view(1, 2)

    inter = torch.min(
        anchor_wh,
        target_wh,
    ).prod(dim=1)

    anchor_area = anchor_wh.prod(dim=1)
    target_area = target_wh.prod(dim=1)

    return inter / (
        anchor_area + target_area - inter + 1e-16
    )


def bbox_iou_xywh(box1, box2):
    """
    Element-wise IoU for boxes in center xywh format.
    box1 and box2 must have shape [N,4].
    """

    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter_w = (
        torch.min(b1_x2, b2_x2)
        - torch.max(b1_x1, b2_x1)
    ).clamp(min=0)

    inter_h = (
        torch.min(b1_y2, b2_y2)
        - torch.max(b1_y1, b2_y1)
    ).clamp(min=0)

    inter_area = inter_w * inter_h

    area1 = (
        (b1_x2 - b1_x1).clamp(min=0)
        * (b1_y2 - b1_y1).clamp(min=0)
    )

    area2 = (
        (b2_x2 - b2_x1).clamp(min=0)
        * (b2_y2 - b2_y1).clamp(min=0)
    )

    return inter_area / (
        area1 + area2 - inter_area + 1e-16
    )


class YoloLayer(nn.Module):
    """
    Classical 2D YOLOv3 detection layer.

    Prediction:
        x, y, w, h, objectness, classes...

    Training target:
        batch_index, class_id, x, y, w, h

    Target xywh values are normalized to [0,1].
    """

    def __init__(
        self,
        anchors,
        num_classes,
        img_size=416,
        all_anchors=None,
        anchor_mask=None,
        ignore_thres=0.7,
    ):
        super(YoloLayer, self).__init__()

        self.num_classes = num_classes

        # Local anchors used by this detection scale.
        self.anchors = anchors
        self.num_anchors = len(anchors)

        # Global YOLOv3 anchor table and the global anchor
        # indices assigned to this scale.
        self.all_anchors = (
            all_anchors
            if all_anchors is not None
            else anchors
        )

        self.anchor_mask = (
            anchor_mask
            if anchor_mask is not None
            else list(range(len(anchors)))
        )

        self.ignore_thres = float(ignore_thres)

        self.obj_scale = 1.0
        self.noobj_scale = 40.0

        self.grid_size = 0
        self.img_size = img_size
        self.metrics = {}

        # x, y, w, h, objectness
        self.num_b_b_attr = 5

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size

        g = grid_size
        self.stride = self.img_size / g

        self.grid_x = (
            torch.arange(
                g,
                device=self.device,
                dtype=torch.float32,
            )
            .repeat(g, 1)
            .view(1, 1, g, g)
        )

        self.grid_y = (
            torch.arange(
                g,
                device=self.device,
                dtype=torch.float32,
            )
            .repeat(g, 1)
            .t()
            .view(1, 1, g, g)
        )

        self.scaled_anchors = torch.tensor(
            [
                (
                    anchor_w / self.stride,
                    anchor_h / self.stride,
                )
                for anchor_w, anchor_h in self.anchors
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.scaled_all_anchors = torch.tensor(
            [
                (
                    anchor_w / self.stride,
                    anchor_h / self.stride,
                )
                for anchor_w, anchor_h in self.all_anchors
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.anchor_w = (
            self.scaled_anchors[:, 0]
            .view(1, self.num_anchors, 1, 1)
        )

        self.anchor_h = (
            self.scaled_anchors[:, 1]
            .view(1, self.num_anchors, 1, 1)
        )

    def build_targets(
        self,
        pred_boxes,
        pred_cls,
        target,
    ):
        nB, nA, nG, _, nC = pred_cls.shape

        obj_mask = torch.zeros(
            nB, nA, nG, nG,
            dtype=torch.bool,
            device=self.device,
        )

        noobj_mask = torch.ones(
            nB, nA, nG, nG,
            dtype=torch.bool,
            device=self.device,
        )

        class_mask = torch.zeros(
            nB, nA, nG, nG,
            device=self.device,
        )

        iou_scores = torch.zeros(
            nB, nA, nG, nG,
            device=self.device,
        )

        tx = torch.zeros(
            nB, nA, nG, nG,
            device=self.device,
        )

        ty = torch.zeros_like(tx)
        tw = torch.zeros_like(tx)
        th = torch.zeros_like(tx)

        tcls = torch.zeros(
            nB, nA, nG, nG, nC,
            device=self.device,
        )

        tconf = torch.zeros(
            nB, nA, nG, nG,
            device=self.device,
        )

        if target is None or target.numel() == 0:
            return (
                iou_scores,
                class_mask,
                obj_mask,
                noobj_mask,
                tx,
                ty,
                tw,
                th,
                tcls,
                tconf,
            )

        target = target.to(
            device=self.device,
            dtype=torch.float32,
        )

        # ---------------------------------------------------------
        # Normalized target xywh -> grid-space xywh
        # ---------------------------------------------------------

        gxy = target[:, 2:4] * nG
        gwh = target[:, 4:6] * nG

        b = target[:, 0].long()
        target_labels = target[:, 1].long()

        # Protect grid indexing when x/y == 1.0.
        gi = gxy[:, 0].long().clamp(
            min=0,
            max=nG - 1,
        )

        gj = gxy[:, 1].long().clamp(
            min=0,
            max=nG - 1,
        )

        # ---------------------------------------------------------
        # Global 9-anchor matching
        # ---------------------------------------------------------

        # Compare every GT against all 9 classical YOLOv3
        # anchors. A GT becomes positive only in the scale
        # containing its globally best anchor.
        global_anchor_ious = torch.stack(
            [
                bbox_wh_iou(
                    anchor,
                    gwh,
                )
                for anchor in self.scaled_all_anchors
            ],
            dim=0,
        )

        _, global_best_n = global_anchor_ious.max(
            dim=0
        )

        # Local anchor IoUs are still used for the YOLO
        # no-object ignore rule on this detection scale.
        local_anchor_ious = global_anchor_ious[
            self.anchor_mask
        ]

        if IGNORE_MODE == "pred":
            # [Unified] Darknet rule: ignore predictions that already overlap any ground truth
            with torch.no_grad():
                pb = pred_boxes.detach().reshape(nB, -1, 4)
                gt = target[:, 2:6] * nG
                for bi in torch.unique(b):
                    sel = b == bi
                    best = _iou_cxcywh(pb[bi], gt[sel]).max(1).values
                    noobj_mask[bi] &= ~(best > self.ignore_thres).view(nA, nG, nG)
        else:
            for target_index in range(target.size(0)):
                noobj_mask[
                    b[target_index],
                    local_anchor_ious[:, target_index]
                    > self.ignore_thres,
                    gj[target_index],
                    gi[target_index],
                ] = False

        # Select only GTs whose global best anchor belongs
        # to this YOLO head.
        positive_target_indices = []
        local_best_indices = []

        global_to_local = {
            global_index: local_index
            for local_index, global_index
            in enumerate(self.anchor_mask)
        }

        for target_index, global_anchor_index in enumerate(
            global_best_n.tolist()
        ):
            if global_anchor_index in global_to_local:
                positive_target_indices.append(
                    target_index
                )
                local_best_indices.append(
                    global_to_local[
                        global_anchor_index
                    ]
                )

        if len(positive_target_indices) == 0:
            return (
                iou_scores,
                class_mask,
                obj_mask,
                noobj_mask,
                tx,
                ty,
                tw,
                th,
                tcls,
                tconf,
            )

        positive_target_indices = torch.tensor(
            positive_target_indices,
            device=self.device,
            dtype=torch.long,
        )

        best_n = torch.tensor(
            local_best_indices,
            device=self.device,
            dtype=torch.long,
        )

        # From this point onward only this scale's positive
        # GT subset participates in regression/class targets.
        b = b[positive_target_indices]
        target_labels = target_labels[
            positive_target_indices
        ]

        gxy = gxy[positive_target_indices]
        gwh = gwh[positive_target_indices]

        gi = gi[positive_target_indices]
        gj = gj[positive_target_indices]

        obj_mask[
            b,
            best_n,
            gj,
            gi,
        ] = True

        noobj_mask[
            b,
            best_n,
            gj,
            gi,
        ] = False

        # ---------------------------------------------------------
        # Regression targets
        # ---------------------------------------------------------

        gx = gxy[:, 0]
        gy = gxy[:, 1]
        gw = gwh[:, 0]
        gh = gwh[:, 1]

        tx[
            b,
            best_n,
            gj,
            gi,
        ] = gx - gx.floor()

        ty[
            b,
            best_n,
            gj,
            gi,
        ] = gy - gy.floor()

        selected_anchors = self.scaled_anchors[
            best_n
        ]

        tw[
            b,
            best_n,
            gj,
            gi,
        ] = torch.log(
            gw / (
                selected_anchors[:, 0]
                + 1e-16
            )
            + 1e-16
        )

        th[
            b,
            best_n,
            gj,
            gi,
        ] = torch.log(
            gh / (
                selected_anchors[:, 1]
                + 1e-16
            )
            + 1e-16
        )

        # ---------------------------------------------------------
        # Class / confidence targets
        # ---------------------------------------------------------

        tcls[
            b,
            best_n,
            gj,
            gi,
            target_labels,
        ] = 1.0

        tconf[
            b,
            best_n,
            gj,
            gi,
        ] = 1.0

        class_mask[
            b,
            best_n,
            gj,
            gi,
        ] = (
            pred_cls[
                b,
                best_n,
                gj,
                gi,
            ].argmax(dim=-1)
            == target_labels
        ).float()

        target_boxes_grid = torch.cat(
            (
                gxy,
                gwh,
            ),
            dim=1,
        )

        predicted_positive_boxes = pred_boxes[
            b,
            best_n,
            gj,
            gi,
        ]

        iou_scores[
            b,
            best_n,
            gj,
            gi,
        ] = bbox_iou_xywh(
            predicted_positive_boxes,
            target_boxes_grid,
        )

        return (
            iou_scores,
            class_mask,
            obj_mask,
            noobj_mask,
            tx,
            ty,
            tw,
            th,
            tcls,
            tconf,
        )

    @staticmethod
    def safe_mse(pred, target):
        if pred.numel() == 0:
            return pred.sum() * 0.0

        return F.mse_loss(
            pred,
            target,
            reduction="mean",
        )

    @staticmethod
    def safe_bce(pred, target):
        if pred.numel() == 0:
            return pred.sum() * 0.0

        return F.binary_cross_entropy(
            pred,
            target,
            reduction="mean",
        )

    def forward(
        self,
        x,
        targets=None,
        img_size=608,
        use_giou_loss=False,
    ):
        self.img_size = img_size
        self.device = x.device

        num_samples = x.size(0)
        grid_size = x.size(2)

        expected_channels = (
            self.num_anchors
            * (
                self.num_classes
                + self.num_b_b_attr
            )
        )

        if x.size(1) != expected_channels:
            raise RuntimeError(
                "YOLO head channel mismatch: "
                f"received={x.size(1)}, "
                f"expected={expected_channels}"
            )

        prediction = (
            x.view(
                num_samples,
                self.num_anchors,
                self.num_classes
                + self.num_b_b_attr,
                grid_size,
                grid_size,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        pred_x = torch.sigmoid(
            prediction[..., 0]
        )

        pred_y = torch.sigmoid(
            prediction[..., 1]
        )

        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]

        pred_conf = torch.sigmoid(
            prediction[..., 4]
        )

        pred_cls = torch.sigmoid(
            prediction[..., 5:]
        )

        if grid_size != self.grid_size:
            self.compute_grid_offsets(
                grid_size
            )

        pred_boxes = torch.empty(
            (
                num_samples,
                self.num_anchors,
                grid_size,
                grid_size,
                4,
            ),
            device=self.device,
            dtype=prediction.dtype,
        )

        pred_boxes[..., 0] = (
            pred_x + self.grid_x
        )

        pred_boxes[..., 1] = (
            pred_y + self.grid_y
        )

        pred_boxes[..., 2] = (
            torch.exp(pred_w).clamp(max=1e3)
            * self.anchor_w
        )

        pred_boxes[..., 3] = (
            torch.exp(pred_h).clamp(max=1e3)
            * self.anchor_h
        )

        output = torch.cat(
            (
                pred_boxes.view(
                    num_samples,
                    -1,
                    4,
                )
                * self.stride,

                pred_conf.view(
                    num_samples,
                    -1,
                    1,
                ),

                pred_cls.view(
                    num_samples,
                    -1,
                    self.num_classes,
                ),
            ),
            dim=-1,
        )

        if targets is None:
            return output, 0

        (
            iou_scores,
            class_mask,
            obj_mask,
            noobj_mask,
            tx,
            ty,
            tw,
            th,
            tcls,
            tconf,
        ) = self.build_targets(
            pred_boxes,
            pred_cls,
            targets,
        )

        # ---------------------------------------------------------
        # Classical YOLOv3-style losses
        # ---------------------------------------------------------

        loss_x = self.safe_mse(
            pred_x[obj_mask],
            tx[obj_mask],
        )

        loss_y = self.safe_mse(
            pred_y[obj_mask],
            ty[obj_mask],
        )

        loss_w = self.safe_mse(
            pred_w[obj_mask],
            tw[obj_mask],
        )

        loss_h = self.safe_mse(
            pred_h[obj_mask],
            th[obj_mask],
        )

        loss_conf_obj = self.safe_bce(
            pred_conf[obj_mask],
            tconf[obj_mask],
        )

        loss_conf_noobj = self.safe_bce(
            pred_conf[noobj_mask],
            tconf[noobj_mask],
        )

        loss_obj = (
            self.obj_scale
            * loss_conf_obj
            + self.noobj_scale
            * loss_conf_noobj
        )

        loss_cls = self.safe_bce(
            pred_cls[obj_mask],
            tcls[obj_mask],
        )
        if CLS_NORM == "pos":
            loss_cls = loss_cls * pred_cls.shape[-1]      # [Unified] sum over classes / num positives

        total_loss = (
            loss_x
            + loss_y
            + loss_w
            + loss_h
            + loss_obj
            + loss_cls
        )

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        with torch.no_grad():

            if obj_mask.any():

                cls_acc = (
                    100.0
                    * class_mask[obj_mask].mean()
                )

                conf_obj = (
                    pred_conf[obj_mask].mean()
                )

                iou50 = (
                    iou_scores[obj_mask] > 0.5
                ).float()

                iou75 = (
                    iou_scores[obj_mask] > 0.75
                ).float()

                recall50 = iou50.mean()
                recall75 = iou75.mean()

            else:

                zero = total_loss.detach() * 0.0

                cls_acc = zero
                conf_obj = zero
                recall50 = zero
                recall75 = zero

            if noobj_mask.any():
                conf_noobj = (
                    pred_conf[
                        noobj_mask
                    ].mean()
                )
            else:
                conf_noobj = (
                    total_loss.detach() * 0.0
                )

        self.metrics = {
            "loss": float(
                total_loss.detach().cpu()
            ),
            "loss_x": float(
                loss_x.detach().cpu()
            ),
            "loss_y": float(
                loss_y.detach().cpu()
            ),
            "loss_w": float(
                loss_w.detach().cpu()
            ),
            "loss_h": float(
                loss_h.detach().cpu()
            ),
            "loss_obj": float(
                loss_obj.detach().cpu()
            ),
            "loss_cls": float(
                loss_cls.detach().cpu()
            ),
            "cls_acc": float(
                cls_acc.detach().cpu()
            ),
            "recall50": float(
                recall50.detach().cpu()
            ),
            "recall75": float(
                recall75.detach().cpu()
            ),
            "conf_obj": float(
                conf_obj.detach().cpu()
            ),
            "conf_noobj": float(
                conf_noobj.detach().cpu()
            ),
            "grid_size": grid_size,
        }

        return output, total_loss
