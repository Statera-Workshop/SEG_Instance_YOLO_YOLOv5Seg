"""
Rotated (BEV) IoU helpers used inside the Complex-YOLO loss (anchor matching,
IoU metrics). Copied from Complex_Yolo_Yolov3/utils/utils.py (the CPU-safe
version); the function bodies are unchanged.

Box format everywhere: [x, y, w, l, im, re]  (im = sin(yaw), re = cos(yaw)).
"""

import numpy as np
import torch
from shapely.geometry import Polygon

from common.kitti.kitti_bev_utils import get_corners


def to_cpu(tensor):
    return tensor.detach().cpu()


def convert_format(boxes_array):
    return np.array([Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array])


def compute_iou(box, boxes):
    iou = [box.intersection(b).area / (box.union(b).area + 1e-12) for b in boxes]
    return np.array(iou, dtype=np.float32)


def rotated_bbox_iou_polygon(box1, box2):
    box1 = to_cpu(box1).numpy()
    box2 = to_cpu(box2).numpy()

    x, y, w, l, im, re = box1
    angle = np.arctan2(im, re)
    bbox1 = convert_format(np.array(get_corners(x, y, w, l, angle)).reshape(-1, 4, 2))

    bbox2 = []
    for i in range(box2.shape[0]):
        x, y, w, l, im, re = box2[i, :]
        bbox2.append(get_corners(x, y, w, l, np.arctan2(im, re)))
    bbox2 = convert_format(np.array(bbox2))
    return compute_iou(bbox1[0], bbox2)


def rotated_box_wh_iou_polygon(anchor, wh, imre):
    w1, h1, im1, re1 = anchor[0], anchor[1], anchor[2], anchor[3]
    wh = wh.t()
    imre = imre.t()
    w2, h2, im2, re2 = wh[0], wh[1], imre[0], imre[1]

    anchor_box = torch.stack([
        anchor.new_tensor(100.0), anchor.new_tensor(100.0), w1, h1, im1, re1,
    ]).view(-1, 6)
    target_boxes = torch.full((w2.shape[0], 6), 100.0, device=anchor.device, dtype=anchor.dtype)
    target_boxes[:, 2] = w2
    target_boxes[:, 3] = h2
    target_boxes[:, 4] = im2
    target_boxes[:, 5] = re2
    return torch.from_numpy(rotated_bbox_iou_polygon(anchor_box[0], target_boxes))


def rotated_box_11_iou_polygon(box1, box2, nG):
    box1_new = torch.zeros((box1.shape[0], 6), device=box1.device, dtype=box1.dtype)
    box2_new = torch.zeros((box2.shape[0], 6), device=box2.device, dtype=box2.dtype)
    box1_new[:, :4] = box1[:, :4]
    box1_new[:, 4:] = box1[:, 4:]
    box2_new[:, :4] = box2[:, :4] * nG
    box2_new[:, 4:] = box2[:, 4:]

    ious = []
    for i in range(box1_new.shape[0]):
        iou = rotated_bbox_iou_polygon(box1_new[i], box2_new[i].view(-1, 6)).squeeze()
        ious.append(iou)
    return torch.from_numpy(np.array(ious))
