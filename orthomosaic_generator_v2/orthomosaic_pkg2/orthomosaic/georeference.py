"""
Georeferencing and GeoTIFF output.

Fixes vs original:
  [1] create_kml_overlay: correct UTM→LatLon, robust CRS parsing
  [2] compute_scale_factor: full Transverse Mercator formula
  [3] GeoidCorrector: ellipsoidal → orthometric height (GPS altitude fix)
  [4] GeoBounds.is_utm() / epsg_code() helper methods
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pyproj import Transformer, CRS
import rasterio
from rasterio.transform import from_origin
from datetime import datetime


# ──────────────────────────────────────────────────────────────
# GeoBounds
# ──────────────────────────────────────────────────────────────

@dataclass
class GeoBounds:
    left:   float
    bottom: float
    right:  float
    top:    float
    crs:    str       # e.g. "EPSG:32645"

    def width(self)  -> float: return self.right - self.left
    def height(self) -> float: return self.top   - self.bottom

    def epsg_code(self) -> int:
        return int(self.crs.split(':')[-1])

    def is_utm(self) -> bool:
        c = self.epsg_code()
        return 32601 <= c <= 32660 or 32701 <= c <= 32760


# ──────────────────────────────────────────────────────────────
# UTM Transformer
# ──────────────────────────────────────────────────────────────

class UTMTransformer:

    def __init__(self, zone: Optional[int] = None,
                 northern_hemisphere: bool = True,
                 reference_lon: Optional[float] = None):
        if zone is None:
            if reference_lon is None:
                raise ValueError("Provide zone or reference_lon")
            zone = self.longitude_to_zone(reference_lon)

        self.zone     = zone
        self.northern = northern_hemisphere
        self.epsg_code = (32600 if northern_hemisphere else 32700) + zone
        self.central_meridian = (zone - 1) * 6 - 180 + 3

        self._to_utm   = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(self.epsg_code), always_xy=True)
        self._from_utm = Transformer.from_crs(
            CRS.from_epsg(self.epsg_code), CRS.from_epsg(4326), always_xy=True)

        print(f"UTM Zone {zone}{'N' if northern_hemisphere else 'S'} "
              f"(EPSG:{self.epsg_code}), λ₀={self.central_meridian}°")

    @staticmethod
    def longitude_to_zone(lon: float) -> int:
        lon = ((lon + 180) % 360) - 180
        return max(1, min(60, int((lon + 180) / 6) + 1))

    @classmethod
    def from_coordinates(cls, lats: List[float],
                         lons: List[float]) -> 'UTMTransformer':
        return cls(zone=cls.longitude_to_zone(float(np.mean(lons))),
                   northern_hemisphere=float(np.mean(lats)) >= 0)

    def latlon_to_utm(self, lat: float, lon: float) -> Tuple[float, float]:
        e, n = self._to_utm.transform(lon, lat)
        return float(e), float(n)

    def utm_to_latlon(self, easting: float, northing: float) -> Tuple[float, float]:
        lon, lat = self._from_utm.transform(easting, northing)
        return float(lat), float(lon)

    def batch_latlon_to_utm(self, lats, lons):
        e, n = self._to_utm.transform(lons, lats)
        return np.asarray(e), np.asarray(n)

    def batch_utm_to_latlon(self, eastings, northings):
        lons, lats = self._from_utm.transform(eastings, northings)
        return np.asarray(lats), np.asarray(lons)

    def compute_grid_convergence(self, lat: float, lon: float) -> float:
        """γ = arctan(tan(Δλ)·sin φ)  in degrees."""
        dlam = np.radians(lon - self.central_meridian)
        phi  = np.radians(lat)
        return float(np.degrees(np.arctan(np.tan(dlam) * np.sin(phi))))

    def compute_scale_factor(self, lat: float, lon: float) -> float:
        """
        Full Transverse Mercator point scale factor.
        FIX [2]: uses easting-based formula (accurate to ppm level).
        k ≈ k0·(1 + p²/2 + p⁴/24)  where p = (E−500000)/(k0·N)
        """
        k0 = 0.9996
        e, _ = self.latlon_to_utm(lat, lon)
        x    = e - 500000.0
        a    = 6378137.0
        f    = 1 / 298.257223563
        e2   = 2*f - f*f
        phi  = np.radians(lat)
        N    = a / np.sqrt(1 - e2 * np.sin(phi)**2)
        p    = x / (k0 * N)
        return k0 * (1 + p**2/2 + p**4/24)


# ──────────────────────────────────────────────────────────────
# Geoid corrector  (ellipsoidal → orthometric height)
# ──────────────────────────────────────────────────────────────

class GeoidCorrector:
    """
    FIX [3]: GPS altitude is ellipsoidal; DEM needs orthometric.
    H_ortho = h_ellipsoid − N_geoid
    """

    def __init__(self):
        self._use_pyproj = False
        try:
            self._t = Transformer.from_crs(
                "EPSG:4979", "EPSG:4326+5773", always_xy=True)
            self._use_pyproj = True
        except Exception:
            pass

    def ellipsoidal_to_orthometric(self, lat: float, lon: float,
                                   h_ellipsoidal: float) -> float:
        if self._use_pyproj:
            try:
                _, _, H = self._t.transform(lon, lat, h_ellipsoidal)
                return float(H)
            except Exception:
                pass
        return h_ellipsoidal - self._approx_undulation(lat, lon)

    @staticmethod
    def _approx_undulation(lat: float, lon: float) -> float:
        phi, lam = np.radians(lat), np.radians(lon)
        return (13.0*np.sin(2*phi) - 11.0*np.sin(phi)**2
                + 5.0*np.cos(2*lam)*np.cos(phi)**2)


# ──────────────────────────────────────────────────────────────
# GeoTIFF Writer
# ──────────────────────────────────────────────────────────────

class GeoTIFFWriter:

    def __init__(self, crs: str = "EPSG:32633"):
        self.crs = crs

    def write_orthomosaic(self, output_path: str, image: np.ndarray,
                          bounds: GeoBounds, gsd: Optional[float] = None,
                          compress: str = 'lzw',
                          nodata: Optional[int] = 0) -> None:
        if image.ndim == 3:
            height, width, count = image.shape
        else:
            height, width = image.shape; count = 1

        gsd_x = gsd if gsd else bounds.width()  / width
        gsd_y = gsd if gsd else bounds.height() / height
        transform = from_origin(bounds.left, bounds.top, gsd_x, gsd_y)

        write_data = (image[np.newaxis] if count == 1
                      else np.transpose(image, (2, 0, 1)))

        profile = dict(driver='GTiff', height=height, width=width,
                       count=count, dtype=str(image.dtype),
                       crs=bounds.crs, transform=transform,
                       compress=compress, tiled=True,
                       blockxsize=512, blockysize=512, interleave='pixel')
        if nodata is not None:
            profile['nodata'] = nodata

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(write_data)
            dst.update_tags(CREATED=datetime.now().isoformat(),
                            SOFTWARE='Professional Orthomosaic Generator',
                            GSD_METRES=str((gsd_x+gsd_y)/2))

        print(f"GeoTIFF: {output_path}  ({width}×{height}px, "
              f"{count}ch, GSD={gsd_x:.4f}m)")

    def create_kml_overlay(self, output_path: str, geotiff_path: str,
                           bounds: GeoBounds,
                           name: str = "Orthomosaic") -> None:
        """
        FIX [1]: correct UTM→LatLon call and robust CRS parsing.
        """
        if bounds.is_utm():
            code     = bounds.epsg_code()
            northern = code < 32700
            zone     = code - (32600 if northern else 32700)
            utm      = UTMTransformer(zone=zone, northern_hemisphere=northern)
            lat_s, lon_w = utm.utm_to_latlon(bounds.left,  bounds.bottom)
            lat_n, lon_e = utm.utm_to_latlon(bounds.right, bounds.top)
        else:
            lat_s, lon_w = bounds.bottom, bounds.left
            lat_n, lon_e = bounds.top,    bounds.right

        kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <GroundOverlay>
    <name>{name}</name>
    <description>Orthomosaic from drone imagery</description>
    <Icon><href>{geotiff_path}</href></Icon>
    <LatLonBox>
      <north>{lat_n:.8f}</north>
      <south>{lat_s:.8f}</south>
      <east>{lon_e:.8f}</east>
      <west>{lon_w:.8f}</west>
    </LatLonBox>
  </GroundOverlay>
</kml>"""
        with open(output_path, 'w') as f:
            f.write(kml)
        print(f"KML: {output_path}  "
              f"(N={lat_n:.6f} S={lat_s:.6f} E={lon_e:.6f} W={lon_w:.6f})")


def compute_optimal_utm_zone(gps_data: List[Tuple]) -> Tuple[int, bool]:
    lons = [d[1] for d in gps_data]
    lats = [d[0] for d in gps_data]
    zone = UTMTransformer.longitude_to_zone(float(np.mean(lons)))
    return zone, float(np.mean(lats)) >= 0
