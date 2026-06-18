import pyproj
from pyproj import Geod

class LocalCartesianProjection:
    """
    Projects geographic (lat, lon) coordinates to flat metric meters cartesian grid (x, y)
    using Transverse Mercator centered around a local origin to maintain high accuracy.
    """
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = float(lat_origin)
        self.lon_origin = float(lon_origin)
        self.proj = pyproj.Proj(
            proj='tmerc',
            lat_0=self.lat_origin,
            lon_0=self.lon_origin,
            datum='WGS84',
            units='m'
        )

    def gps_to_metric(self, lat, lon):
        x, y = self.proj(float(lon), float(lat))
        return x, y

    def metric_to_gps(self, x, y):
        lon, lat = self.proj(float(x), float(y), inverse=True)
        return lat, lon


def calculate_polygon_area(gps_points):
    """
    gps_points is a list/tuple of [lat, lon] coordinates.
    Calculates the geodesic area and perimeter on the WGS84 ellipsoid.
    """
    if len(gps_points) < 3:
        return {
            "area_m2": 0.0,
            "area_hectare": 0.0,
            "perimeter_m": 0.0
        }

    geod = Geod(ellps="WGS84")
    lons = [float(p[1]) for p in gps_points]
    lats = [float(p[0]) for p in gps_points]

    area_m2, perimeter_m = geod.polygon_area_perimeter(lons, lats)
    area_m2 = abs(area_m2)
    perimeter_m = abs(perimeter_m)
    area_hectare = area_m2 / 10000.0

    return {
        "area_m2": round(area_m2, 2),
        "area_hectare": round(area_hectare, 6),
        "perimeter_m": round(perimeter_m, 2)
    }
