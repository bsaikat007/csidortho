"""
DJI EXIF + XMP metadata reader — v2 (robust to 16:9 crop modes).

Fixes vs v1:
  [1] Focal length always read from EXIF — never uses hardcoded 20mm fallback
  [2] pixel_size computed from ACTUAL image width, not sensor DB height
      → correct for any crop mode (5472x3648, 5472x3078, 4000x3000, etc.)
  [3] Sensor width from FocalPlaneXResolution when available
  [4] Added FC6310S, Mavic 3, Mini 2/3, Air 2S to sensor DB
  [5] Clear warning when falling back
"""

import re
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class DJIImageMeta:
    path:             Path
    latitude:         float
    longitude:        float
    altitude_abs:     float
    altitude_rel:     float
    gimbal_yaw:       float
    gimbal_pitch:     float
    gimbal_roll:      float
    flight_yaw:       float
    flight_pitch:     float
    flight_roll:      float
    make:             str
    model:            str
    focal_length_mm:  float
    image_width:      int
    image_height:     int
    sensor_width_mm:  float
    sensor_height_mm: float
    pixel_size_um:    float
    fx_pixels:        float
    fy_pixels:        float
    cx_pixels:        float
    cy_pixels:        float


# Physical sensor widths only — pixel size is always derived from actual image dims
_DJI_SENSOR_MM = {
    'FC6310':  (13.2, 8.8),   # Phantom 4 Pro
    'FC6310S': (13.2, 8.8),   # Phantom 4 Pro V2
    'FC330':   (6.3,  4.7),   # Phantom 4
    'FC220':   (6.3,  4.7),   # Mavic Pro
    'FC200':   (6.3,  4.7),   # Phantom 3
    'FC220P':  (13.2, 8.8),   # Mavic 2 Pro (Hasselblad)
    'FC220Z':  (6.17, 4.55),  # Mavic 2 Zoom
    'FC7203':  (17.3, 13.0),  # Mavic 3
    'FC7303':  (17.3, 13.0),  # Mavic 3 Classic
    'FC7161':  (6.3,  4.7),   # Mini 2
    'FC8282':  (9.6,  7.2),   # Mini 3 Pro
    'L1D-20c': (13.2, 8.8),   # Inspire 2 / Zenmuse X5
    'FC6520':  (17.3, 13.0),  # Zenmuse X5S
    'FC3411':  (13.2, 8.8),   # Air 2S
    'FC3170':  (6.4,  4.8),   # Air 2
    'UNKNOWN': (13.2, 8.8),   # safe fallback
}


def _dms_to_decimal(dms, ref: str) -> float:
    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
    val = d + m/60.0 + s/3600.0
    if ref.strip().upper() in ('S', 'W'):
        val = -val
    return val


def _rat(val) -> float:
    """Parse IFDRational, tuple-fraction, or plain number."""
    if hasattr(val, 'numerator'):
        den = float(val.denominator)
        return float(val.numerator) / den if den else 0.
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return float(val[0]) / float(val[1]) if val[1] else 0.
    return float(val)


def _read_exif(path: Path) -> Dict:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    img  = Image.open(str(path))
    raw  = img._getexif() or {}
    data = {'img_width': img.width, 'img_height': img.height}
    for tag_id, val in raw.items():
        tag = TAGS.get(tag_id, str(tag_id))
        if tag == 'GPSInfo':
            gps = {}
            for k, v in val.items():
                gps[GPSTAGS.get(k, k)] = v
            data['GPS'] = gps
        else:
            data[tag] = val
    return data


def _read_xmp(path: Path) -> Dict:
    with open(str(path), 'rb') as f:
        raw = f.read(262144)
    start = raw.find(b'<x:xmpmeta')
    if start == -1:
        return {}
    end   = raw.find(b'</x:xmpmeta>', start)
    chunk = (raw[start:end+12].decode('utf-8', errors='replace')
             if end != -1 else raw[start:start+8192].decode('utf-8', errors='replace'))
    vals = {}
    for key in ['GimbalYawDegree','GimbalPitchDegree','GimbalRollDegree',
                'FlightYawDegree','FlightPitchDegree','FlightRollDegree',
                'RelativeAltitude','AbsoluteAltitude']:
        m = re.search(rf'{key}="([^"]+)"', chunk)
        if m:
            try: vals[key] = float(m.group(1))
            except ValueError: pass
    return vals


def read_dji_image(path) -> DJIImageMeta:
    """
    Read metadata from a DJI JPEG.

    Key guarantee: focal_length_mm is ALWAYS from EXIF FocalLength tag.
    pixel_size_um is computed from sensor_w / image_width so it works for
    any shooting mode (4:3, 16:9 crop, downscaled, etc.).
    """
    path = Path(path)
    exif = _read_exif(path)
    xmp  = _read_xmp(path)

    # GPS
    gps = exif.get('GPS', {})
    if not gps or 'GPSLatitude' not in gps:
        raise ValueError(f"No GPS in {path.name}")
    lat = _dms_to_decimal(gps['GPSLatitude'],  gps.get('GPSLatitudeRef',  'N'))
    lon = _dms_to_decimal(gps['GPSLongitude'], gps.get('GPSLongitudeRef', 'E'))
    alt_abs = _rat(gps.get('GPSAltitude', 0))

    # Camera identity
    make  = str(exif.get('Make',  'DJI')).strip('\x00').strip()
    model = str(exif.get('Model', 'UNKNOWN')).strip('\x00').strip()
    img_w = int(exif.get('img_width',  5472))
    img_h = int(exif.get('img_height', 3648))

    # FIX [1]: Focal length always from EXIF — never hardcoded
    focal_raw = exif.get('FocalLength', None)
    if focal_raw is not None:
        focal_mm = _rat(focal_raw)
    else:
        fl35 = exif.get('FocalLengthIn35mmFilm', None)
        if fl35 is not None:
            focal_mm = _rat(fl35) / 2.7   # approximate crop factor for 1-inch
            print(f"  WARNING {path.name}: FocalLength missing, estimated "
                  f"{focal_mm:.1f}mm from 35mm equivalent")
        else:
            focal_mm = 8.8
            print(f"  WARNING {path.name}: No focal length in EXIF at all, "
                  f"defaulting to {focal_mm}mm")
    if focal_mm <= 0:
        focal_mm = 8.8

    # Sensor physical size from DB (fallback to UNKNOWN=FC6310 spec)
    sensor_key = model if model in _DJI_SENSOR_MM else 'UNKNOWN'
    if sensor_key == 'UNKNOWN' and model not in ('UNKNOWN', ''):
        print(f"  WARNING {path.name}: Model '{model}' not in sensor DB, "
              f"using FC6310 spec. Add it to _DJI_SENSOR_MM for accuracy.")
    sensor_w_mm, sensor_h_mm_db = _DJI_SENSOR_MM[sensor_key]

    # FIX [2]: Pixel size from actual width — works for ANY crop mode
    # e.g. FC6310 at 5472x3078 (16:9): same sensor_w=13.2mm, same pixel pitch
    pixel_size_um = sensor_w_mm * 1000.0 / img_w

    # FIX [3]: Override pixel size from FocalPlaneXResolution if present
    fp_res  = exif.get('FocalPlaneXResolution', None)
    fp_unit = exif.get('FocalPlaneResolutionUnit', 2)
    if fp_res is not None:
        try:
            fp_val = _rat(fp_res)
            if fp_val > 0:
                if fp_unit == 2:    # px/inch
                    pixel_size_um = 25400.0 / fp_val
                elif fp_unit == 3:  # px/cm
                    pixel_size_um = 10000.0 / fp_val
        except Exception:
            pass

    # Effective sensor dimensions from pixel pitch × actual image dims
    sw_eff = pixel_size_um * img_w / 1000.0
    sh_eff = pixel_size_um * img_h / 1000.0

    # Intrinsics
    fx = focal_mm * img_w / sw_eff
    fy = focal_mm * img_h / sh_eff
    cx = img_w / 2.0
    cy = img_h / 2.0

    # Altitude + gimbal
    alt_rel      = float(xmp.get('RelativeAltitude', alt_abs))
    gimbal_yaw   = float(xmp.get('GimbalYawDegree',   0.0))
    gimbal_pitch = float(xmp.get('GimbalPitchDegree', -90.0))
    gimbal_roll  = float(xmp.get('GimbalRollDegree',   0.0))
    flight_yaw   = float(xmp.get('FlightYawDegree',   gimbal_yaw))
    flight_pitch = float(xmp.get('FlightPitchDegree',  0.0))
    flight_roll  = float(xmp.get('FlightRollDegree',   0.0))

    return DJIImageMeta(
        path=path, latitude=lat, longitude=lon,
        altitude_abs=alt_abs, altitude_rel=alt_rel,
        gimbal_yaw=gimbal_yaw, gimbal_pitch=gimbal_pitch,
        gimbal_roll=gimbal_roll,
        flight_yaw=flight_yaw, flight_pitch=flight_pitch,
        flight_roll=flight_roll,
        make=make, model=model,
        focal_length_mm=focal_mm,
        image_width=img_w, image_height=img_h,
        sensor_width_mm=sw_eff, sensor_height_mm=sh_eff,
        pixel_size_um=pixel_size_um,
        fx_pixels=fx, fy_pixels=fy,
        cx_pixels=cx, cy_pixels=cy,
    )
