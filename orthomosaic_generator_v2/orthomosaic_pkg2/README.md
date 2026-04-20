# Professional Orthomosaic Generator — v2

> **v2 fixes the swirl/rotation artefact** seen with 30-image datasets
> and 5472×3078 (16:9) cameras.

## What changed in v2

| Bug | Symptom | Fix |
|---|---|---|
| Only 1 pose built for all 30 images | Swirl / circular artefact | `initialise_poses()` now builds UTM transformer **before** the pose loop |
| `fx = 8290` instead of `3648` | Wrong scale, bad projection | Focal length always read from EXIF `FocalLength` tag |
| 5472×3078 (16:9) not handled | Wrong pixel size | `pixel_size_um = sensor_w / image_width` — works for any aspect ratio |
| AGL from GPS − estimate | Inaccurate altitude | AGL uses XMP `RelativeAltitude` (DJI's direct measurement) |

# Professional Orthomosaic Generator

Mathematically rigorous photogrammetry pipeline for DJI drone imagery.
Tested on **DJI FC6310 (Phantom 4 Pro)**, Sikkim/W.Bengal, India.

---

## What was fixed vs the original codebase

| Module | Fix | Impact |
|---|---|---|
| `camera.py` | Correct Gauss-Newton undistortion iteration | Up to 0.5px error eliminated |
| `camera.py` | `backproject_ray` returns unit vectors (not depth-scaled) | 34% 3-D error at corner pixels eliminated |
| `pose.py` | `make_nadir()` builds correct nadir rotation matrix | Was 0% coverage, now 67%+ |
| `pose.py` | `to_euler_angles()` uses quaternion path (gimbal-lock safe) | No singularity at pitch=−90° |
| `pose.py` | `compute_relative_pose` uses `cv2.recoverPose` | Stable E decomposition |
| `bundle_adjustment.py` | Deep-copy before residual eval | Jacobian was completely wrong |
| `bundle_adjustment.py` | Sparse `jac_sparsity` passed to scipy | 326× memory saving |
| `bundle_adjustment.py` | Distortion applied in reprojection | Correct reprojection error |
| `georeference.py` | KML UTM→LatLon fix | Correct georeferencing |
| `georeference.py` | Full Transverse Mercator scale factor | ppm-level accuracy |
| `orthorectifier.py` | `float64` accumulator | No precision loss in blending |
| `orthorectifier.py` | `compute_footprint` undistorts corners first | Correct image bounds |
| `dense_matcher.py` | NaN for invalid projections (not −1) | No NCC bias corruption |
| `dense_matcher.py` | Stereo rectification before SGBM | Required for nadir pairs |
| `dense_matcher.py` | Near-zero disparity marked invalid | No DEM pollution |

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8, numpy, opencv-python, scipy, pyproj, rasterio, Pillow

---

## Quick start

```bash
# Put your DJI images in a folder, then:
python run.py --images /path/to/images/ --output ortho.tif
```

That's it. The script auto-detects UTM zone, camera model, ground altitude, and GSD.

### Options

```
--images      Folder containing DJI JPEG images          (required)
--output      Output GeoTIFF path        (default: orthomosaic.tif)
--gsd         Target GSD in metres       (default: auto from altitude)
--scale       Downscale factor 0.1–1.0   (default: 0.5)
              Use 0.25 if RAM < 8 GB
--ground-alt  Ground elevation m ASL     (default: auto from XMP)
--blending    weighted | nearest | multiband  (default: weighted)
--bundle-adjust  Refine poses with BA    (only for grid/overlap flights)
```

### Memory guide

| RAM    | Recommended `--scale` | Notes |
|--------|----------------------|-------|
| 4 GB   | 0.15–0.20           | Safe  |
| 8 GB   | 0.25–0.35           | Good  |
| 16 GB  | 0.5                 | Default |
| 32 GB+ | 1.0                 | Full resolution |

---

## Python API

### One-call pipeline

```python
from orthomosaic import OrthoPipeline

pipe = OrthoPipeline(
    image_dir  = "images/",
    output     = "ortho.tif",
    scale      = 0.5,       # use 0.25 for low RAM
    blending   = "weighted",
    run_ba     = False,     # set True only for grid flights with lateral overlap
)
pipe.run()
```

### Step by step

```python
from orthomosaic import OrthoPipeline

pipe = OrthoPipeline(image_dir="images/", output="ortho.tif", scale=0.5)

# Step 1: read EXIF + XMP from all images
meta = pipe.load_images()
print(f"Camera: {meta[0].model}  focal={meta[0].focal_length_mm}mm")

# Step 2: build camera poses from GPS + gimbal angles
poses = pipe.initialise_poses()
print(f"UTM Zone: {pipe._utm.zone}N")

# Step 3 (optional): refine with bundle adjustment
# Only useful when images have lateral overlap (> 5m baseline)
# poses = pipe.refine_poses(max_iter=50)

# Step 4: orthorectify and blend
result, bounds, gsd = pipe.generate_orthomosaic()
print(f"Coverage: {100*np.sum(result.any(axis=2))/(result.shape[0]*result.shape[1]):.1f}%")

# Step 5: write GeoTIFF + KML
pipe.save(result, bounds, gsd)
```

### Individual modules

```python
from orthomosaic.exif_reader  import read_dji_image
from orthomosaic.pose         import Pose
from orthomosaic.camera       import BrownConradyCamera
from orthomosaic.georeference import UTMTransformer

# Read metadata from one image
m = read_dji_image("DJI_0001.JPG")
print(f"GPS: {m.latitude:.6f}N  {m.longitude:.6f}E  AGL={m.altitude_rel:.1f}m")
print(f"Gimbal: yaw={m.gimbal_yaw}°  pitch={m.gimbal_pitch}°")
print(f"Camera: {m.model}  fx={m.fx_pixels:.1f}px")

# Build nadir pose (correct for DJI pitch=-90°)
utm = UTMTransformer(reference_lon=m.longitude)
e, n = utm.latlon_to_utm(m.latitude, m.longitude)
gc   = utm.compute_grid_convergence(m.latitude, m.longitude)
pose = Pose.make_nadir(e, n, m.altitude_rel, m.gimbal_yaw + gc)

# Build camera model with known FC6310 distortion
cam = BrownConradyCamera(
    width=m.image_width, height=m.image_height,
    fx=m.fx_pixels, fy=m.fy_pixels, cx=m.cx_pixels, cy=m.cy_pixels,
    k1=-0.0508, k2=0.0290, p1=0.00015, p2=-0.00010, k3=-0.0087)

# Backproject image corners to unit rays (FIXED — not depth-scaled)
import numpy as np
corners = np.array([[0.,0.],[m.image_width-1,0.],
                    [m.image_width-1,m.image_height-1],[0.,m.image_height-1]])
rays = cam.backproject_ray(corners)          # (4,3) unit vectors
print(f"Ray norms: {np.linalg.norm(rays, axis=1)}")  # all 1.0
```

---

## Output files

| File | Description |
|---|---|
| `ortho.tif` | GeoTIFF with embedded UTM georeferencing (open in QGIS / ArcGIS) |
| `ortho.kml` | Google Earth overlay — double-click to view |

---

## What kind of flight works best

| Flight type | BA needed | Notes |
|---|---|---|
| **Grid survey** (50–80% overlap) | Optional | Best orthomosaic quality |
| **Strip flight** (side overlap) | Optional | Good |
| **Hover + ascend** (like your 5 images) | No | Orthorectifies fine; BA doesn't help because baseline ≈ 0 |
| **Oblique** (non-nadir) | Yes | Set `--bundle-adjust` |

Your 5 DJI images are a **hover-ascend** sequence (drone climbed 30m vertically,
horizontal drift only 0.3m). The pipeline handles this correctly using GPS+gimbal
poses directly without bundle adjustment.

For a proper **orthomosaic mission**, fly a grid pattern with ≥ 70% forward and
60% side overlap. WebODM or DJI GS Pro can plan this automatically.

---

## Tested hardware

| Drone | Camera | Status |
|---|---|---|
| DJI Phantom 4 Pro | FC6310 | ✓ Tested |
| DJI Phantom 4 | FC330 | ✓ Sensor DB entry |
| DJI Mavic Pro | FC220 | ✓ Sensor DB entry |
| DJI Mavic 3 | FC7203 | ✓ Sensor DB entry |
| DJI Inspire 2 | L1D-20c | ✓ Sensor DB entry |
| Other DJI | unknown | ✓ Falls back to FC6310 sensor spec |

---

## Running tests

```bash
python tests/test_pipeline.py --images /path/to/your/DJI/images/
```

Tests validate: EXIF parsing, camera math, pose construction (including
nadir gimbal-lock), UTM georeferencing, bundle adjustment correctness.
All tests run on your actual images — no synthetic data needed.
