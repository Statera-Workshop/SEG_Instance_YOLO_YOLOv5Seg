"""
KITTI LiDAR -> Bird's-Eye-View utilities shared by Complex-YOLO and YOLO3D.

Origin
    Complex_Yolo_Yolov3/data_process/kitti_bev_utils.py   (BEV map, drawing)
    Yolo_3d_Yolov3/data_process/kitti_bev_utils.py        (3D target builder)

The BEV map and the geometry are identical in both legacy projects, so they
live here once. Only the target encoders differ (task-specific):

    build_complex_target  -> [cls, x_col, y_row, w, l, im, re]
    build_yolo3d_target   -> [cls, x_col, y_row, z, h, w, l, im, re]

All values are normalized to [0, 1] exactly as in the legacy code.
"""


import numpy as np


def get_corners(x, y, w, l, yaw):
    """BEV image-coordinate corners (legacy definition)."""
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)  # front left
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)  # rear left
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)  # rear right
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)  # front right
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)
    return bev_corners
