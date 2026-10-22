"""
YOLOv5-seg builder (seg task) - the framework's YOLOv5-n (detectors/yolov5, unchanged) + the YOLOv5 Segment head.

Reference: Ultralytics YOLOv5 v7.0 (2022-11), models/segment/yolov5n-seg.yaml (head `Segment, [nc, anchors, 32,
256]`), models/yolo.py Segment, models/common.py Proto, utils/segment/loss.py ComputeLoss.single_mask_loss,
utils/general.py crop_mask. AGPL-3.0. Reimplemented here.

    Proto      on P3: Conv 3x3 (c_) -> Upsample x2 nearest -> Conv 3x3 (c_) -> Conv 1x1 (32)
               -> 32 prototypes at stride 4; c_ = 256 x width 0.25 = 64 (parse_model scales Segment's npr)
               (YOLOv8/11 replace the nearest upsample with a ConvTranspose - this is the v5 block)
    coef       Segment's 1x1 conv emits na x (5 + nc + 32) per cell; here the detection 1x1 conv is the
               framework's YOLOv5 head unchanged and a second 1x1 conv emits the na x 32 coefficients - the same
               linear map split in two (no activation on the coefficients, as Segment)
    mask       sigmoid(coefficients . prototypes), cropped with the box
    mask loss  single_mask_loss: per positive, BCE(coef . proto, gt mask at proto resolution) averaged inside the
               gt box (crop_mask), divided by the normalised gt box area, mean over the positives of an image;
               summed over images and scales, x box gain / batch size

[Unified]
  - detection part = the framework's YOLOv5 (detectors/yolov5: v5 backbone + PaFPN, v3 YOLO layers, global
    9-anchor matching, one positive per object) exactly as on the detection stage; the mask loss uses those
    positives (YOLOv5's own matcher with neighbour cells is not used there, so not here either)
  - mask gain: Ultralytics sets lseg x hyp box, i.e. the mask term carries the box-loss gain. The v3 layer's box
    term (x + y + w + h MSE) has gain 1, so the mask gain is 1. Ultralytics' final x batch size cancels with the
    / batch size above and is not applied to the other terms in this framework, so it is left out.
  - ground-truth masks: dataset instance masks downsampled to proto resolution, nearest (as YOLO11-/YOLOv8-seg)
  - no pretrained weights, common init and recipe, common post-processing (as every model of the framework)

Raw contract: v3 YOLO layer rows [B, M, 5 + C] (anchor-major per scale, then y, x); aux = coefficients [B, M, 32]
in the same row order + prototypes. decode_masks returns kind "full" at proto resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.yolo11.modules import Conv
from detectors.yolo11_seg.build import crop_mask
from detectors.yolov5.build import VARIANT, YOLOv5
from detectors.yolov5.yolov5_config import yolov5_cfg

NM, NPR = 32, 256


class Proto(nn.Module):
    """YOLOv5 v7.0 models/common.py Proto (nearest upsample)."""

    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


def record_positives(layer, target):
    """The (b, anchor, gj, gi, target row) of every positive of a v3 image YOLO layer - the same selection as
    YoloLayer.build_targets (global best of the 9 anchors, kept if it belongs to this scale; cell of the centre)."""
    nG = layer.grid_size
    gxy, gwh = target[:, 2:4] * nG, target[:, 4:6] * nG
    ious = torch.stack([torch.min(a.view(1, 2), gwh).prod(1)
                        / (a.prod() + gwh.prod(1) - torch.min(a.view(1, 2), gwh).prod(1) + 1e-16)
                        for a in layer.scaled_all_anchors], 0)
    best = ious.max(0).indices
    local = {g: l for l, g in enumerate(layer.anchor_mask)}
    rows = [i for i, g in enumerate(best.tolist()) if g in local]
    if not rows:
        return None
    rows = torch.tensor(rows, device=target.device)
    a = torch.tensor([local[g] for g in best[rows].tolist()], device=target.device)
    gi = gxy[rows, 0].long().clamp(0, nG - 1)
    gj = gxy[rows, 1].long().clamp(0, nG - 1)
    return target[rows, 0].long(), a, gj, gi, rows


class YOLOv5Seg(YOLOv5):
    def __init__(self, num_classes):
        super().__init__("image", num_classes)
        width = yolov5_cfg[VARIANT]["width"]                             # yolov5_n: 0.25
        npr = max(int(-(-NPR * width // 8) * 8), 8)                      # make_divisible(256 x 0.25, 8) = 64
        self.na = [layer.num_anchors for layer in self.yolo_modules]
        self.proto = Proto(self.fpn.out_dim[0], npr, NM)
        self.coef_heads = nn.ModuleList(nn.Conv2d(d, na * NM, 1) for d, na in zip(self.fpn.out_dim, self.na))
        self.mask_gain = 1.0

    def forward(self, x, targets=None):
        img_dim, B = x.shape[2], x.shape[0]
        feats = self.fpn(self.backbone(x))
        boxes = targets["boxes"] if targets is not None else None
        outputs, coefs, loss = [], [], 0
        for feat, head, ch, layer in zip(feats, self.heads, self.coef_heads, self.yolo_modules):
            out, layer_loss = layer(head(feat), boxes, img_dim)
            g = feat.shape[2]
            coefs.append(ch(feat).view(B, layer.num_anchors, NM, g, g).permute(0, 1, 3, 4, 2).reshape(B, -1, NM))
            outputs.append(out)
            loss += layer_loss
        raw, proto = torch.cat(outputs, 1), self.proto(feats[0])
        if targets is None:
            return raw if self.training else (raw, {"coef": torch.cat(coefs, 1), "proto": proto})
        loss_mask = self.mask_loss(coefs, proto, targets, img_dim)
        for layer in self.yolo_modules:
            layer.metrics["loss_mask"] = float(loss_mask.detach())
        return loss + self.mask_gain * loss_mask, raw

    def mask_loss(self, coefs, proto, targets, img_size):
        """YOLOv5 v7.0 ComputeLoss mask term (overlap=False path)."""
        boxes, masks = targets["boxes"], targets["masks"]
        B, nm, mh, mw = proto.shape
        total = proto.sum() * 0.0
        if boxes.numel() == 0:
            return total
        gt_all = None
        for coef, layer in zip(coefs, self.yolo_modules):
            pos = record_positives(layer, boxes)
            if pos is None:
                continue
            b, a, gj, gi, rows = pos
            if gt_all is None:
                gt_all = F.interpolate(masks[None].float(), (mh, mw), mode="nearest")[0]      # [N, mh, mw]
            g = layer.grid_size
            pmask = coef.view(B, layer.num_anchors, g, g, nm)[b, a, gj, gi]                  # [n, 32]
            xywhn = boxes[rows, 2:6]
            mxyxy = torch.cat([xywhn[:, :2] - xywhn[:, 2:] / 2, xywhn[:, :2] + xywhn[:, 2:] / 2], 1) \
                * torch.tensor([mw, mh, mw, mh], device=boxes.device, dtype=boxes.dtype)
            marea = xywhn[:, 2:].prod(1).clamp(min=1e-9)
            for bi in b.unique():
                j = b == bi
                pred = (pmask[j] @ proto[bi].view(nm, -1)).view(-1, mh, mw)
                loss = F.binary_cross_entropy_with_logits(pred, gt_all[rows[j]], reduction="none")
                total = total + (crop_mask(loss, mxyxy[j]).mean(dim=(1, 2)) / marea[j]).mean()
        return total / B

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        proto = aux["proto"][b]
        nm, mh, mw = proto.shape
        return {"kind": "full", "prob": (aux["coef"][b, rows] @ proto.view(nm, -1)).sigmoid().view(-1, mh, mw)}


def build_model(task, num_classes):
    return YOLOv5Seg(num_classes)
