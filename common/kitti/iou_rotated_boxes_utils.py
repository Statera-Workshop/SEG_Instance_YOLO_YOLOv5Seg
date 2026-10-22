"""
# Description: This script for iou calculation of rotated boxes (on GPU)

"""

from __future__ import division

import torch
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull


class Line:
    """2D line segment helper used by rotated-rectangle intersection."""

    def __init__(self, p, q):
        self.p = p
        self.q = q

        self.dx = q[0] - p[0]
        self.dy = q[1] - p[1]

    def cal_values(self, points):
        """Evaluate the signed line equation for points."""
        return (
            self.dx * (points[:, 1] - self.p[1])
            - self.dy * (points[:, 0] - self.p[0])
        )

    def find_intersection(self, other):
        """Return intersection point of two infinite 2D lines."""
        x1, y1 = self.p
        x2, y2 = self.q
        x3, y3 = other.p
        x4, y4 = other.q

        denominator = (
            (x1 - x2) * (y3 - y4)
            - (y1 - y2) * (x3 - x4)
        )

        if torch.abs(denominator) < 1e-12:
            return self.p

        t = (
            (x1 - x3) * (y3 - y4)
            - (y1 - y3) * (x3 - x4)
        ) / denominator

        return torch.stack([
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1),
        ])


def intersection_area(rect1, rect2):
    """Calculate the inter

    Args:
        rect1: vertices of the rectangles (4, 2)
        rect2: vertices of the rectangles (4, 2)

    Returns:

    """

    # Use the vertices of the first rectangle as, starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    roll_rect2 = torch.roll(rect2, -1, dims=0)
    for p, q in zip(rect2, roll_rect2):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".
        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = line.cal_values(intersection)
        roll_intersection = torch.roll(intersection, -1, dims=0)
        roll_line_values = torch.roll(line_values, -1, dims=0)
        for s, t, s_value, t_value in zip(intersection, roll_intersection, line_values, roll_line_values):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.find_intersection(Line(s, t))
                new_intersection.append(intersection_point)

        if len(new_intersection) > 0:
            intersection = torch.stack(new_intersection)
        else:
            break

    # Calculate area
    if len(intersection) <= 2:
        return 0.

    return PolyArea2D(intersection)

def PolyArea2D(pts):
    roll_pts = torch.roll(pts, -1, dims=0)
    area = (pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]).sum().abs() * 0.5
    return area

def cvt_box_2_polygon(box):
    """
    :param array: an array of shape [num_conners, 2]
    :return: a shapely.geometry.Polygon object
    """
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)


def get_corners_3d(x, y, z, h, w, l, yaw):
    """bev image coordinates format - vectorization
    :param x, y, z, h, w, l, yaw: [num_boxes,]
    :return: num_boxes x (x, y, z) of 8 conners
    """
    device = x.device
    box_conners = torch.zeros((x.size(0), 8, 3), device=device, dtype=torch.float)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # front left
    box_conners[:, 0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 0, 2] = z - h / 2

    # rear left
    box_conners[:, 1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 1, 2] = z - h / 2

    # rear right
    box_conners[:, 2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 2, 2] = z - h / 2

    # front right
    box_conners[:, 3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 3, 2] = z - h / 2

    box_conners[:, 4, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 4, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 4, 2] = z + h / 2

    # rear left
    box_conners[:, 5, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 5, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 5, 2] = z + h / 2

    # rear right
    box_conners[:, 6, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 6, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 6, 2] = z + h / 2

    # front right
    box_conners[:, 7, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 7, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 7, 2] = z + h / 2

    return box_conners


def get_polygons_areas_fix_xyz(boxes, fix_xyz=100.):
    """
    Args:
        box: (num_boxes, 4) --> w, l, im, re
    """
    device = boxes.device
    n_boxes = boxes.size(0)
    x = torch.full(size=(n_boxes,), fill_value=fix_xyz, device=device, dtype=torch.float)
    y = torch.full(size=(n_boxes,), fill_value=fix_xyz, device=device, dtype=torch.float)
    z = torch.full(size=(n_boxes,), fill_value=fix_xyz, device=device, dtype=torch.float)
    h, w, l, im, re = boxes.t()
    yaw = torch.atan2(im, re)
    boxes_conners_3d = get_corners_3d(x, y, z, h, w, l, yaw)
    boxes_polygons = [cvt_box_2_polygon(box_) for box_ in
                      boxes_conners_3d[:, :4, :2]]  # Take (x,y) of the 4 first conners
    boxes_volumes = h * w * l
    low_h = boxes_conners_3d[:, 0, 2]
    high_h = boxes_conners_3d[:, -1, 2]

    return boxes_polygons, boxes_volumes, low_h, high_h


def iou_rotated_boxes_targets_vs_anchors(a_polygons, a_volumes, a_low_hs, a_high_hs, tg_polygons, tg_volumes, tg_low_hs,
                                         tg_high_hs):
    device = a_volumes.device
    num_anchors = len(a_volumes)
    num_targets_boxes = len(tg_volumes)

    ious = torch.zeros(size=(num_anchors, num_targets_boxes), device=device, dtype=torch.float)

    for a_idx in range(num_anchors):
        for tg_idx in range(num_targets_boxes):
            inter_area = a_polygons[a_idx].intersection(tg_polygons[tg_idx]).area
            low_inter_h = max(a_low_hs[a_idx], tg_low_hs[tg_idx])
            high_inter_h = min(a_high_hs[a_idx], tg_high_hs[tg_idx])
            inter_volume = (high_inter_h - low_inter_h) * inter_area
            iou = inter_volume / (a_volumes[a_idx] + tg_volumes[tg_idx] - inter_volume + 1e-16)
            ious[a_idx, tg_idx] = iou

    return ious

def iou_pred_vs_target_boxes(pred_boxes, target_boxes, GIoU=False, DIoU=False, CIoU=False):
    assert pred_boxes.size() == target_boxes.size(), "Unmatch size of pred_boxes and target_boxes"
    device = pred_boxes.device
    n_boxes = pred_boxes.size(0)

    t_x, t_y, t_z, t_h, t_w, t_l, t_im, t_re = target_boxes.t()
    t_yaw = torch.atan2(t_im, t_re)
    t_conners = get_corners_3d(t_x, t_y, t_z, t_h, t_w, t_l, t_yaw)
    t_volumes = t_h * t_w * t_l
    t_low_hs = t_conners[:, 0, 2]
    t_high_hs = t_conners[:, -1, 2]

    p_x, p_y, p_z, p_h, p_w, p_l, p_im, p_re = pred_boxes.t()
    p_yaw = torch.atan2(p_im, p_re)
    p_conners = get_corners_3d(p_x, p_y, p_z, p_h, p_w, p_l, p_yaw)
    p_volumes = p_h * p_w * p_l
    p_low_hs = p_conners[:, 0, 2]
    p_high_hs = p_conners[:, -1, 2]

    ious = []
    giou_loss = torch.tensor([0.], device=device, dtype=torch.float)
    # Thinking to apply vectorization this step
    for box_idx in range(n_boxes):
        p_cons, t_cons = p_conners[box_idx], t_conners[box_idx]
        if not GIoU:
            p_poly, t_poly = cvt_box_2_polygon(p_cons[:4, :2]), cvt_box_2_polygon(t_cons[:4, :2])  # (x, y) of 4 first
            inter_area = p_poly.intersection(t_poly).area
        else:
            inter_area = intersection_area(p_cons[:4, :2], t_cons[:4, :2])

        low_inter_h = max(p_low_hs[box_idx], t_low_hs[box_idx])
        high_inter_h = min(p_high_hs[box_idx], t_high_hs[box_idx])
        inter_h = max(0., high_inter_h - low_inter_h)
        inter_volume = inter_h * inter_area
        union_volume = p_volumes[box_idx] + t_volumes[box_idx] - inter_volume
        iou = inter_volume / (union_volume + 1e-16)

        if GIoU:
            convex_conners = torch.cat((p_cons[:4, :2], t_cons[:4, :2]), dim=0)
            hull = ConvexHull(convex_conners.clone().detach().cpu().numpy())  # done on cpu, just need indices output
            convex_conners = convex_conners[hull.vertices]
            convex_area = PolyArea2D(convex_conners)
            low_convex_h = min(p_low_hs[box_idx], t_low_hs[box_idx])
            high_convex_h = max(p_high_hs[box_idx], t_high_hs[box_idx])
            convex_h = max(0., high_convex_h - low_convex_h)
            convex_volume = convex_h * convex_area
            giou_loss += 1. - (iou - (convex_volume - union_volume) / (convex_volume + 1e-16))
        else:
            giou_loss += 1. - iou

        if DIoU or CIoU:
            raise NotImplementedError

        ious.append(iou)

    return torch.tensor(ious, device=device, dtype=torch.float), giou_loss
