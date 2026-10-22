"""
YOLOv5 builders for the three tasks.

Architecture (algorithm-specific, this folder):
    CSPDarknet (C3 blocks, SiLU, SPPF) + PaFPN neck, three scales (8, 16, 32)
    -> yolov5_backbone.py / yolov5_pafpn.py / yolov5_basic.py / yolov5_neck.py
       copied unchanged from the Legacy Image_Yolo_Yolov5 project
    -> head: one 1x1 convolution per scale (original YOLOv5 "Detect")
    -> size: VARIANT (legacy projects used "yolov5_n")

Detection layers: reused unchanged from detectors/yolov3, exactly as v2 / v4.
Anchors: taken from the v3 cfg files (Image: COCO anchors, KITTI: v3 anchors).

Not used from the Legacy v5 projects: the decoupled head, the YOLOv5
anchor-ratio matcher and its loss (replaced by the shared YOLO layers), so v5
differs from v3 in backbone / neck only.
"""

from pathlib import Path

import torch
import torch.nn as nn

from common.darknet import parse_model_config
from detectors.yolov3.build import complex_factory, image_factory, yolo3d_factory
from detectors.yolov5.yolov5_backbone import build_backbone
from detectors.yolov5.yolov5_config import yolov5_cfg
from detectors.yolov5.yolov5_pafpn import build_fpn

VARIANT = "yolov5_n"
V3_CFG = Path(__file__).resolve().parents[1] / "yolov3" / "cfg"

SPECS = {
    #          anchors source (v3 cfg)   factory          box attributes
    "image":   ("image_yolov3.cfg",   image_factory,   5),
    "complex": ("complex_yolov3.cfg", complex_factory, 7),
    "yolo3d":  ("yolo3d_yolov3.cfg",  yolo3d_factory,  8),
}


class YOLOv5(nn.Module):
    """Same contract as common.darknet.Darknet: forward(x[, targets])."""

    def __init__(self, task, num_classes, variant=VARIANT):
        super().__init__()
        cfg_name, factory, num_box_attr = SPECS[task]
        cfg = yolov5_cfg[variant]
        self.backbone, feat_dims = build_backbone(cfg)
        self.fpn = build_fpn(cfg, feat_dims[-3:])

        blocks = parse_model_config(V3_CFG / cfg_name)
        hyper = blocks[0]
        yolo_blocks = sorted((b for b in blocks if b["type"] == "yolo"),
                             key=lambda b: min(int(v) for v in b["mask"].split(",")))  # P3, P4, P5
        heads, layers = [], []
        for dim, block in zip(self.fpn.out_dim, yolo_blocks):
            block = dict(block, classes=str(num_classes))
            n_anchor = len(block["mask"].split(","))
            heads.append(nn.Conv2d(dim, n_anchor * (num_box_attr + num_classes), 1))
            layers.append(factory(block, hyper))
        self.heads = nn.ModuleList(heads)
        self.yolo_modules = nn.ModuleList(layers)
        self.yolo_layers = list(layers)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        feats = self.fpn(self.backbone(x))
        outputs, loss = [], 0
        for feat, head, layer in zip(feats, self.heads, self.yolo_modules):
            out, layer_loss = layer(head(feat), targets, img_dim)
            outputs.append(out)
            loss += layer_loss
        outputs = torch.cat(outputs, 1)
        return outputs if targets is None else (loss, outputs)

    def load_darknet_weights(self, path):
        raise NotImplementedError("YOLOv5 has no Darknet weights; train from scratch.")
