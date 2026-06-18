import cv2
import numpy as np

def calculate_homography(image_points, gps_points):
    """
    Computes a homography matrix that maps 2D camera pixels to local flat metric meters.
    1. Projects GPS points to a flat metric space centered at the first GPS coordinate.
    2. Calculates the 3x3 matrix using cv2.findHomography.
    """
    from .geospatial import LocalCartesianProjection

    if len(image_points) < 4 or len(gps_points) < 4:
        raise ValueError("At least 4 Ground Control Points (GCPs) are required.")

    # 1. Project GPS coordinates to a local flat metric cartesian grid (meters)
    lat_origin = float(gps_points[0][0])
    lon_origin = float(gps_points[0][1])
    projection = LocalCartesianProjection(lat_origin, lon_origin)

    metric_points = []
    for lat, lon in gps_points:
        mx, my = projection.gps_to_metric(lat, lon)
        metric_points.append([mx, my])

    # 2. Use cv2.findHomography to map pixel space to metric space
    src_pts = np.array(image_points, dtype=np.float32)
    dst_pts = np.array(metric_points, dtype=np.float32)

    # Use RANSAC if more than 4 points are supplied, otherwise standard DLT
    method = cv2.RANSAC if len(image_points) > 4 else 0
    H, status = cv2.findHomography(src_pts, dst_pts, method)

    if H is None:
        raise ValueError("Homography matrix calculation failed due to collinear or degenerate point patterns.")

    return H.tolist(), (lat_origin, lon_origin)


def pixel_to_metric(x, y, homography_matrix):
    """
    Transforms pixel coordinate (x, y) into local metric coordinate (mx, my)
    using the computed homography matrix.
    """
    H = np.array(homography_matrix, dtype=np.float64)
    # cv2.perspectiveTransform expects shape (N, 1, 2)
    pts = np.array([[[float(x), float(y)]]], dtype=np.float64)
    transformed_pts = cv2.perspectiveTransform(pts, H)
    mx, my = transformed_pts[0][0]
    return mx, my
