import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import CameraGCPConfiguration
from .services import save_gcp_config, pixel_to_gps
from .geospatial import calculate_polygon_area

@csrf_exempt
@require_http_methods(["POST"])
def save_gcp(request):
    """
    POST /api/gcp/save/
    Saves or clears a camera's GCP configuration.
    """
    try:
        data = json.loads(request.body)
        camera_id = data.get("camera_id")
        action = data.get("action")
        
        if not camera_id:
            return JsonResponse({"status": "error", "message": "Missing camera_id"}, status=400)
            
        if action == "clear":
            CameraGCPConfiguration.objects.filter(camera_id=str(camera_id)).delete()
            return JsonResponse({"status": "success", "message": "GCP configuration cleared"})
            
        camera_name = data.get("camera_name", f"Camera {camera_id}")
        gcps = data.get("gcps")
        
        if not gcps or not all(k in gcps for k in ["top_left", "top_right", "bottom_right", "bottom_left"]):
            return JsonResponse({"status": "error", "message": "Incomplete GCP coordinates supplied"}, status=400)
            
        config = save_gcp_config(camera_id, camera_name, gcps)
        
        return JsonResponse({
            "status": "success",
            "camera_id": config.camera_id,
            "area_m2": config.area_m2,
            "area_hectare": config.area_hectare,
            "perimeter_m": config.perimeter_m
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@require_http_methods(["GET"])
def get_gcp(request, camera_id):
    """
    GET /api/gcp/<camera_id>/
    Retrieves the camera GCP coordinates and calculated geometry.
    """
    try:
        config = CameraGCPConfiguration.objects.get(camera_id=str(camera_id))
        return JsonResponse({
            "configured": True,
            "camera_id": config.camera_id,
            "camera_name": config.camera_name,
            "gcps": {
                "top_left": {"x": config.top_left_pixel_x, "y": config.top_left_pixel_y, "lat": config.top_left_lat, "lon": config.top_left_lon},
                "top_right": {"x": config.top_right_pixel_x, "y": config.top_right_pixel_y, "lat": config.top_right_lat, "lon": config.top_right_lon},
                "bottom_right": {"x": config.bottom_right_pixel_x, "y": config.bottom_right_pixel_y, "lat": config.bottom_right_lat, "lon": config.bottom_right_lon},
                "bottom_left": {"x": config.bottom_left_pixel_x, "y": config.bottom_left_pixel_y, "lat": config.bottom_left_lat, "lon": config.bottom_left_lon}
            },
            "area_m2": config.area_m2,
            "area_hectare": config.area_hectare,
            "perimeter_m": config.perimeter_m
        })
    except CameraGCPConfiguration.DoesNotExist:
        return JsonResponse({
            "configured": False,
            "camera_id": str(camera_id)
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def calculate_area_api(request):
    """
    POST /api/gcp/calculate-area/
    Directly runs Geodesic area calculation on raw GPS points.
    """
    try:
        data = json.loads(request.body)
        points = data.get("points")
        if not points or len(points) < 3:
            return JsonResponse({"status": "error", "message": "At least 3 coordinates are required"}, status=400)
            
        area_stats = calculate_polygon_area(points)
        return JsonResponse(area_stats)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def pixel_to_gps_api(request):
    """
    POST /api/gcp/pixel-to-gps/
    Transforms pixel coordinates on feed to geographic latitude/longitude.
    """
    try:
        data = json.loads(request.body)
        camera_id = data.get("camera_id")
        x = data.get("x")
        y = data.get("y")
        
        if camera_id is None or x is None or y is None:
            return JsonResponse({"status": "error", "message": "Missing camera_id, x, or y"}, status=400)
            
        coords = pixel_to_gps(camera_id, x, y)
        return JsonResponse(coords)
    except ValueError as ve:
        return JsonResponse({"status": "error", "message": str(ve)}, status=400)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@require_http_methods(["GET"])
def list_cameras(request):
    """
    GET /api/gcp/cameras/list/
    Lists all available cameras with their GCP configuration status.
    """
    try:
        # CCTV app primarily streams Camera ID "0"
        try:
            config = CameraGCPConfiguration.objects.get(camera_id="0")
            configured = True
        except CameraGCPConfiguration.DoesNotExist:
            configured = False
            
        return JsonResponse({
            "cameras": [
                {
                    "camera_id": "0",
                    "camera_name": "Camera 1",
                    "configured": configured,
                    "preview_url": "/video_feed/"
                }
            ]
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
