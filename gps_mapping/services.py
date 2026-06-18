import datetime
from .models import CameraGCPConfiguration
from .homography import calculate_homography, pixel_to_metric
from .geospatial import LocalCartesianProjection, calculate_polygon_area

def save_gcp_config(camera_id, camera_name, gcps):
    """
    Validates and saves a Ground Control Point configuration for a camera.
    
    gcps is a dictionary containing coordinate definitions for the 4 corners:
    {
        "top_left": {"x": float, "y": float, "lat": float, "lon": float},
        "top_right": {"x": float, "y": float, "lat": float, "lon": float},
        "bottom_right": {"x": float, "y": float, "lat": float, "lon": float},
        "bottom_left": {"x": float, "y": float, "lat": float, "lon": float}
    }
    """
    # 1. Arrange points in consistent order: Top Left, Top Right, Bottom Right, Bottom Left
    image_points = [
        [gcps["top_left"]["x"], gcps["top_left"]["y"]],
        [gcps["top_right"]["x"], gcps["top_right"]["y"]],
        [gcps["bottom_right"]["x"], gcps["bottom_right"]["y"]],
        [gcps["bottom_left"]["x"], gcps["bottom_left"]["y"]]
    ]
    
    gps_points = [
        [gcps["top_left"]["lat"], gcps["top_left"]["lon"]],
        [gcps["top_right"]["lat"], gcps["top_right"]["lon"]],
        [gcps["bottom_right"]["lat"], gcps["bottom_right"]["lon"]],
        [gcps["bottom_left"]["lat"], gcps["bottom_left"]["lon"]]
    ]
    
    # 2. Compute homography matrix
    homography_matrix, _ = calculate_homography(image_points, gps_points)
    
    # 3. Compute geospatial metrics (area, perimeter)
    area_stats = calculate_polygon_area(gps_points)
    
    # 4. Save to Database
    config, created = CameraGCPConfiguration.objects.update_or_create(
        camera_id=str(camera_id),
        defaults={
            "camera_name": camera_name,
            
            "top_left_pixel_x": gcps["top_left"]["x"],
            "top_left_pixel_y": gcps["top_left"]["y"],
            "top_left_lat": gcps["top_left"]["lat"],
            "top_left_lon": gcps["top_left"]["lon"],
            
            "top_right_pixel_x": gcps["top_right"]["x"],
            "top_right_pixel_y": gcps["top_right"]["y"],
            "top_right_lat": gcps["top_right"]["lat"],
            "top_right_lon": gcps["top_right"]["lon"],
            
            "bottom_right_pixel_x": gcps["bottom_right"]["x"],
            "bottom_right_pixel_y": gcps["bottom_right"]["y"],
            "bottom_right_lat": gcps["bottom_right"]["lat"],
            "bottom_right_lon": gcps["bottom_right"]["lon"],
            
            "bottom_left_pixel_x": gcps["bottom_left"]["x"],
            "bottom_left_pixel_y": gcps["bottom_left"]["y"],
            "bottom_left_lat": gcps["bottom_left"]["lat"],
            "bottom_left_lon": gcps["bottom_left"]["lon"],
            
            "area_m2": area_stats["area_m2"],
            "area_hectare": area_stats["area_hectare"],
            "perimeter_m": area_stats["perimeter_m"],
            "homography_matrix": homography_matrix
        }
    )
    
    return config


def pixel_to_gps(camera_id, x, y):
    """
    Transforms pixel coordinate (x, y) on camera_id feed to Estimated GPS latitude/longitude.
    """
    try:
        config = CameraGCPConfiguration.objects.get(camera_id=str(camera_id))
    except CameraGCPConfiguration.DoesNotExist:
        raise ValueError(f"No GCP configuration exists for Camera ID: {camera_id}")
        
    if not config.homography_matrix:
        raise ValueError(f"GCP configuration for Camera ID: {camera_id} is incomplete.")

    # 1. Transform pixels to local flat metric coordinate (mx, my) using homography
    mx, my = pixel_to_metric(x, y, config.homography_matrix)

    # 2. Convert local metric coordinate back to GPS coordinates (lat, lon)
    # Origin is the top_left coordinate since it was index 0 in calculate_homography
    projection = LocalCartesianProjection(config.top_left_lat, config.top_left_lon)
    lat, lon = projection.metric_to_gps(mx, my)

    return {
        "latitude": round(lat, 8),
        "longitude": round(lon, 8)
    }


def convert_bbox_to_gps(camera_id, bbox, object_id=None):
    """
    Helper for AI integrations. Takes a 2D bounding box [x1, y1, x2, y2],
    extracts the bottom-center point, and projects it to geographic GPS coordinates.
    
    Returns:
        {
            "object_id": id,
            "latitude": float,
            "longitude": float,
            "timestamp": ISO string
        }
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = y2  # Bottom center is y2
    
    gps_coords = pixel_to_gps(camera_id, cx, cy)
    
    return {
        "object_id": object_id,
        "latitude": gps_coords["latitude"],
        "longitude": gps_coords["longitude"],
        "timestamp": datetime.datetime.now().isoformat()
    }
