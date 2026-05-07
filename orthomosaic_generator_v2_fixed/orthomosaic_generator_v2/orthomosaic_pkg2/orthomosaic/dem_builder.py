"""
dem_builder.py — True DSM generation from overlapping drone nadir imagery.

Two modes
---------
'sparse'  SIFT triangulation only    ~500–5 000 pts/pair,  < 1 s/pair
'dense'   SGBM semi-global matching  ~50 000 pts/pair,     3–10 s/pair
'both'    sparse + dense combined

Algorithm
---------
1. Pair selection  – footprint bounding-box IoU > min_overlap
2. Per pair        – SIFT match+triangulate (sparse) and/or SGBM (dense)
3. Point fusion    – median-Z binning onto UTM grid
4. Post-process    – scipy hole-fill, Gaussian smooth

Output: (H, W) float32 DSM, north-at-top (row-0 = y_max), same bounds
        as the orthorectifier canvas.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from .pose import Pose, triangulate_points


# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DEMParams:
    dem_gsd:             float = 1.0    # output DSM resolution [m/px]
    stereo_scale:        float = 0.25   # image scale for stereo matching
    min_overlap:         float = 0.35   # min footprint IoU to form a pair
    max_pairs_per_image: int   = 5      # neighbours per image
    mode:                str   = 'sparse'  # 'sparse' | 'dense' | 'both'
    terrain_range:       float = 80.0   # max terrain height variation [m]
    smooth_sigma:        float = 2.0    # Gaussian smooth [DEM pixels]
    min_points_pair:     int   = 8      # reject pair if fewer inlier matches


# ──────────────────────────────────────────────────────────────────────────────
# Footprint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _footprint_bbox(pose: Pose, cam, ground_z: float = 0.0
                    ) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box (xmin, ymin, xmax, ymax) of image footprint."""
    corners = np.array([[0., 0.],
                         [cam.width - 1., 0.],
                         [cam.width - 1., cam.height - 1.],
                         [0., cam.height - 1.]])
    rays_c = cam.backproject_ray(corners)
    rays_w = (pose.R.T @ rays_c.T).T
    C = pose.C
    pts = []
    for ray in rays_w:
        if abs(ray[2]) > 1e-6:
            t = (ground_z - C[2]) / ray[2]
            if t > 0:
                pts.append(C + t * ray)
    if len(pts) < 2:
        return C[0], C[1], C[0], C[1]
    gp = np.array(pts)
    return float(gp[:, 0].min()), float(gp[:, 1].min()), \
           float(gp[:, 0].max()), float(gp[:, 1].max())


def _iou(b1: tuple, b2: tuple) -> float:
    """Intersection-over-Union of two (xmin,ymin,xmax,ymax) bboxes."""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Pair selection
# ──────────────────────────────────────────────────────────────────────────────

def select_pairs(poses:    List[Pose],
                 cameras:  List,
                 ground_z: float,
                 params:   DEMParams) -> List[Tuple[int, int]]:
    """Return (i, j) pairs ordered by descending overlap IoU."""
    N = len(poses)
    bboxes = [_footprint_bbox(p, c, ground_z) for p, c in zip(poses, cameras)]

    candidates: List[Tuple[float, int, int]] = []
    for i in range(N):
        row = [(   _iou(bboxes[i], bboxes[j]), i, j)
               for j in range(N) if j != i and _iou(bboxes[i], bboxes[j]) >= params.min_overlap]
        row.sort(reverse=True)
        for cnt, (iou, ii, jj) in enumerate(row):
            if cnt >= params.max_pairs_per_image:
                break
            pair = (min(ii, jj), max(ii, jj))
            candidates.append((iou, pair[0], pair[1]))

    # Deduplicate while preserving highest-IoU entry
    seen: set = set()
    pairs: List[Tuple[int, int]] = []
    for _, i, j in sorted(candidates, reverse=True):
        if (i, j) not in seen:
            seen.add((i, j))
            pairs.append((i, j))

    print(f"  DEM: {N} images → {len(pairs)} stereo pairs selected")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Sparse cloud: SIFT triangulation
# ──────────────────────────────────────────────────────────────────────────────

def _sparse_cloud(img1:    np.ndarray,
                  img2:    np.ndarray,
                  pose1:   Pose,
                  pose2:   Pose,
                  K:       np.ndarray,
                  z_min:   float,
                  z_max:   float,
                  min_pts: int = 8) -> Optional[np.ndarray]:
    """
    SIFT match → RANSAC → cv2.triangulatePoints → filtered (N,3) world pts.

    K is the scaled intrinsic matrix matching img1/img2 resolution.
    z_min/z_max are absolute Z bounds (terrain range in AGL frame).
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.018,
                            edgeThreshold=12)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < min_pts or len(kp2) < min_pts:
        return None

    # FLANN k-NN
    index_params  = {'algorithm': 1, 'trees': 5}
    search_params = {'checks': 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        raw = flann.knnMatch(d1, d2, k=2)
    except cv2.error:
        return None

    good = [m for m, n in raw
            if len([m, n]) == 2 and m.distance < 0.72 * n.distance]
    if len(good) < min_pts:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Fundamental-matrix RANSAC to reject outlier matches
    if len(pts1) >= 8:
        _, mask_f = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.5,
            confidence=0.999)
        if mask_f is not None:
            m = mask_f.ravel().astype(bool)
            pts1, pts2 = pts1[m], pts2[m]

    if len(pts1) < min_pts:
        return None

    # Triangulate (pose.py helper uses cv2.triangulatePoints internally)
    pts3d = triangulate_points(pts1, pts2, pose1, pose2, K)

    # Filter: finite, within terrain Z bounds, in front of both cams
    ok = (np.isfinite(pts3d).all(axis=1) &
          (pts3d[:, 2] > z_min) & (pts3d[:, 2] < z_max))
    pts3d = pts3d[ok]
    if len(pts3d) == 0:
        return None

    for pose in (pose1, pose2):
        Xc = pose.transform_to_camera(pts3d)
        ok2 = Xc[:, 2] > 0.1
        pts3d = pts3d[ok2]

    return pts3d.astype(np.float32) if len(pts3d) >= 3 else None


# ──────────────────────────────────────────────────────────────────────────────
# Dense cloud: SGBM stereo
# ──────────────────────────────────────────────────────────────────────────────

def _dense_cloud(img1:    np.ndarray,
                 img2:    np.ndarray,
                 pose1:   Pose,
                 pose2:   Pose,
                 K1:      np.ndarray,
                 dist1:   np.ndarray,
                 K2:      np.ndarray,
                 dist2:   np.ndarray,
                 z_min:   float,
                 z_max:   float) -> Optional[np.ndarray]:
    """
    Stereo rectify → SGBM → reproject to UTM world points.

    K1/K2 and dist1/dist2 must match img1/img2 resolution.
    dist coefficients are in normalised-coordinate space (scale-invariant).
    """
    h, w = img1.shape[:2]
    img_sz = (w, h)

    # Relative pose: camera-1 → camera-2
    R_rel = (pose2.R @ pose1.R.T).astype(np.float64)
    t_rel = (pose2.R @ (pose1.C - pose2.C)).reshape(3, 1).astype(np.float64)

    # Stereo rectification
    R1_r, R2_r, P1_r, P2_r, Q, _, _ = cv2.stereoRectify(
        K1.astype(np.float64), dist1.astype(np.float64),
        K2.astype(np.float64), dist2.astype(np.float64),
        img_sz, R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        newImageSize=img_sz)

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, dist1, R1_r, P1_r, img_sz, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, dist2, R2_r, P2_r, img_sz, cv2.CV_32FC1)

    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    g1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY) if rect1.ndim == 3 else rect1
    g2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY) if rect2.ndim == 3 else rect2

    # Expected disparity from known baseline and camera height
    baseline  = float(np.linalg.norm(pose1.C[:2] - pose2.C[:2]))  # horizontal baseline
    fx_r      = float(P1_r[0, 0])
    cam_z     = float(pose1.C[2])
    if baseline < 0.5 or fx_r < 1.0:
        return None  # degenerate pair (same position or zero focal)

    # Disparity at scene height extremes (ground ≈0, terrain varies)
    d_ground = fx_r * baseline / max(cam_z - z_min, 1.)
    d_top    = fx_r * baseline / max(cam_z - z_max, 1.)
    min_disp = max(0, int(min(d_ground, d_top)) - 32)
    max_disp = int(max(d_ground, d_top)) + 32
    num_disp = max(16, min(512, ((max_disp - min_disp) // 16 + 1) * 16))
    min_disp = max(0, min_disp)

    win = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity    = min_disp,
        numDisparities  = num_disp,
        blockSize       = win,
        P1              = 8  * 3 * win * win,
        P2              = 32 * 3 * win * win,
        disp12MaxDiff   = 2,
        uniquenessRatio = 8,
        speckleWindowSize = 120,
        speckleRange    = 2,
        preFilterCap    = 63,
        mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disp_raw = sgbm.compute(g1, g2)  # fixed-point × 16

    # WLS disparity filter (needs opencv-contrib; graceful fallback)
    try:
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        disp_r_raw    = right_matcher.compute(g2, g1)
        wls           = cv2.ximgproc.createDisparityWLSFilter(sgbm)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        disp_raw = wls.filter(disp_raw, g1, disparity_map_right=disp_r_raw)
    except AttributeError:
        pass  # opencv-contrib not installed

    disp = disp_raw.astype(np.float32) / 16.0
    valid = (disp > min_disp + 0.5) & np.isfinite(disp)

    # Reproject to 3-D in rectified left-camera frame
    pts_rect = cv2.reprojectImageTo3D(disp, Q)  # (H, W, 3)

    # Transform: rectified cam1 → original cam1 → world
    # X_world = R1.T @ R1_r.T @ X_rect + C1
    R_to_world = pose1.R.T @ R1_r.T
    flat       = pts_rect.reshape(-1, 3)
    valid_flat = valid.ravel() & np.isfinite(flat).all(axis=1)

    pts_w = (R_to_world @ flat[valid_flat].T).T + pose1.C

    # Height filter
    ok = (pts_w[:, 2] > z_min) & (pts_w[:, 2] < z_max)
    pts_w = pts_w[ok]

    return pts_w.astype(np.float32) if len(pts_w) >= 10 else None


# ──────────────────────────────────────────────────────────────────────────────
# Point-cloud → DSM raster
# ──────────────────────────────────────────────────────────────────────────────

def _points_to_dsm(pts:          np.ndarray,
                   bounds:       Tuple[float, float, float, float],
                   dem_gsd:      float,
                   smooth_sigma: float) -> np.ndarray:
    """
    Bin (N,3) UTM points into a north-at-top float32 raster.

    Per-cell statistic: median Z (robust to outliers).
    Gaps filled with scipy griddata (linear, then nearest for boundary).
    """
    xmin, ymin, xmax, ymax = bounds
    W = max(2, int(round((xmax - xmin) / dem_gsd)))
    H = max(2, int(round((ymax - ymin) / dem_gsd)))

    # Pixel indices — row 0 = north (y_max)
    col = np.clip(((pts[:, 0] - xmin) / dem_gsd).astype(np.int32), 0, W - 1)
    row = np.clip(((ymax - pts[:, 1]) / dem_gsd).astype(np.int32), 0, H - 1)

    cell_idx = row * W + col
    order    = np.argsort(cell_idx)
    ci_s     = cell_idx[order]
    z_s      = pts[order, 2]

    # Median per cell via group boundaries
    boundaries = np.flatnonzero(np.diff(ci_s)) + 1
    boundaries = np.concatenate([[0], boundaries, [len(ci_s)]])

    dsm_flat = np.full(H * W, np.nan, dtype=np.float32)
    for k in range(len(boundaries) - 1):
        a, b = boundaries[k], boundaries[k + 1]
        dsm_flat[ci_s[a]] = float(np.median(z_s[a:b]))

    dsm = dsm_flat.reshape(H, W)

    # Outlier clamp — 1st / 99th percentile of all valid values
    valid_vals = dsm[np.isfinite(dsm)]
    if len(valid_vals) >= 4:
        lo, hi = np.percentile(valid_vals, [1, 99])
        dsm = np.clip(dsm, lo, hi)

    # Hole fill
    valid_mask = np.isfinite(dsm)
    if valid_mask.sum() >= 4:
        yy, xx = np.mgrid[0:H, 0:W]
        vpts = np.column_stack([xx[valid_mask], yy[valid_mask]])
        vz   = dsm[valid_mask]

        filled = griddata(vpts, vz, (xx, yy), method='linear')
        border = ~np.isfinite(filled)
        if border.any():
            filled[border] = griddata(vpts, vz,
                                       (xx[border], yy[border]),
                                       method='nearest')
        dsm = filled.astype(np.float32)

    # Smooth to reduce SIFT noise
    if smooth_sigma > 0:
        dsm = gaussian_filter(dsm, sigma=smooth_sigma).astype(np.float32)

    return dsm


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class DEMBuilder:
    """
    Build a DSM raster from overlapping drone nadir images.

    Usage::
        from orthomosaic.dem_builder import DEMBuilder, DEMParams

        builder = DEMBuilder(DEMParams(mode='sparse', dem_gsd=1.0))
        dsm, gsd = builder.build(
            image_paths, poses, cameras, bounds,
            pipeline_scale=0.5,          # scale that cameras were built at
            dist_coeffs_list=dist_list,  # list of [k1,k2,p1,p2,k3] per image
            ground_z=0.0)                # reference ground elevation
    """

    def __init__(self, params: Optional[DEMParams] = None):
        self.params = params or DEMParams()

    # ── main entry point ──────────────────────────────────────────────────────

    def build(self,
              image_paths:      List,
              poses:            List[Pose],
              cameras:          List,
              bounds:           Tuple[float, float, float, float],
              pipeline_scale:   float = 0.5,
              dist_coeffs_list: Optional[List] = None,
              ground_z:         float = 0.0
              ) -> Tuple[np.ndarray, float]:
        """
        Returns
        -------
        dsm     : (H, W) float32 elevation raster, north-at-top
        dem_gsd : resolution in metres/pixel
        """
        p  = self.params
        N  = len(image_paths)
        if N < 2:
            raise ValueError("Need at least 2 images")

        print(f"\n── DEM Builder ──────────────────────────────")
        print(f"  mode={p.mode}  dem_gsd={p.dem_gsd}m  "
              f"stereo_scale={p.stereo_scale}  images={N}")

        # Scale factor from pipeline cameras to DEM matching resolution
        adj = p.stereo_scale / pipeline_scale  # e.g. 0.25/0.5 = 0.5

        # Pair selection uses pipeline-scale cameras (footprints are scale-invariant)
        pairs = select_pairs(poses, cameras, ground_z, p)
        if not pairs:
            raise RuntimeError(
                "No overlapping pairs — check image overlap or reduce min_overlap")

        # Z bounds for point filtering (terrain in AGL frame, ground ≈ 0)
        agls = [pose.C[2] for pose in poses]
        median_agl = float(np.median(agls))
        z_min = ground_z - p.terrain_range
        z_max = ground_z + p.terrain_range

        # Pre-load images at DEM scale
        print(f"  Loading {N} images at ×{p.stereo_scale:.2f} …")
        imgs: List[Optional[np.ndarray]] = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                imgs.append(None)
                continue
            nh = int(img.shape[0] * p.stereo_scale)
            nw = int(img.shape[1] * p.stereo_scale)
            imgs.append(cv2.resize(img, (nw, nh)))
        gc.collect()

        # Scaled K matrices (same for all — single-camera drone)
        Ks = []
        for cam in cameras:
            K = cam.K().copy()
            K[:2] *= adj   # scale fx, fy, cx, cy
            Ks.append(K)

        dist_list = dist_coeffs_list or [np.zeros(5, dtype=np.float64)] * N

        # Process pairs
        all_pts: List[np.ndarray] = []
        for idx, (i, j) in enumerate(pairs):
            img1, img2 = imgs[i], imgs[j]
            if img1 is None or img2 is None:
                continue

            baseline = float(np.linalg.norm(poses[i].C[:2] - poses[j].C[:2]))
            print(f"  Pair {idx+1:3d}/{len(pairs)}: "
                  f"#{i+1}↔#{j+1}  B={baseline:.1f}m", end="")

            if p.mode in ('sparse', 'both'):
                sp = _sparse_cloud(img1, img2, poses[i], poses[j],
                                   Ks[i], z_min, z_max, p.min_points_pair)
                if sp is not None:
                    all_pts.append(sp)
                    print(f"  sparse:{len(sp)}", end="")

            if p.mode in ('dense', 'both'):
                d1 = np.asarray(dist_list[i], dtype=np.float64).ravel()
                d2 = np.asarray(dist_list[j], dtype=np.float64).ravel()
                if len(d1) < 4: d1 = np.zeros(5)
                if len(d2) < 4: d2 = np.zeros(5)
                dp = _dense_cloud(img1, img2, poses[i], poses[j],
                                  Ks[i], d1, Ks[j], d2, z_min, z_max)
                if dp is not None:
                    all_pts.append(dp)
                    print(f"  dense:{len(dp)}", end="")

            print()

        del imgs; gc.collect()

        if not all_pts:
            raise RuntimeError(
                "No 3-D points generated — verify camera calibration and image overlap")

        combined = np.vstack(all_pts).astype(np.float32)
        print(f"  Total 3-D points: {len(combined):,}")

        # Remove global outliers (2nd/98th percentile in Z)
        z2, z98 = np.percentile(combined[:, 2], [2, 98])
        combined = combined[(combined[:, 2] > z2) & (combined[:, 2] < z98)]
        print(f"  After outlier clip: {len(combined):,}  "
              f"Z=[{combined[:,2].min():.1f}, {combined[:,2].max():.1f}]m")

        print(f"  Gridding at {p.dem_gsd}m/px …")
        dsm = _points_to_dsm(combined, bounds, p.dem_gsd, p.smooth_sigma)

        z_ok = dsm[np.isfinite(dsm)]
        print(f"  DSM: {dsm.shape[1]}×{dsm.shape[0]}px  "
              f"Z=[{z_ok.min():.1f}, {z_ok.max():.1f}]m  "
              f"relief={z_ok.max()-z_ok.min():.1f}m")
        print(f"── DEM Builder done ─────────────────────────")
        return dsm, p.dem_gsd

    # ── GeoTIFF I/O ───────────────────────────────────────────────────────────

    @staticmethod
    def save_geotiff(dsm:    np.ndarray,
                     path:   str,
                     bounds: Tuple[float, float, float, float],
                     gsd:    float,
                     crs:    str = "EPSG:32633") -> None:
        """Write the DSM as a single-band float32 GeoTIFF."""
        import rasterio
        from rasterio.transform import from_origin

        xmin, _, _, ymax = bounds
        transform = from_origin(xmin, ymax, gsd, gsd)
        profile = dict(driver='GTiff', height=dsm.shape[0], width=dsm.shape[1],
                       count=1, dtype='float32', crs=crs,
                       transform=transform, compress='lzw',
                       tiled=True, blockxsize=256, blockysize=256,
                       nodata=np.nan)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(dsm[np.newaxis])
        print(f"  DSM saved: {path}")

    @staticmethod
    def load_geotiff(path: str
                     ) -> Tuple[np.ndarray, Tuple[float,float,float,float], float]:
        """
        Load a GeoTIFF DSM.

        Returns  (dsm, bounds, gsd)
        bounds   = (xmin, ymin, xmax, ymax)
        """
        import rasterio
        with rasterio.open(path) as src:
            dsm = src.read(1).astype(np.float32)
            t   = src.transform
            gsd = float((abs(t.a) + abs(t.e)) / 2)
            xmin = float(t.c)
            ymax = float(t.f)
            xmax = xmin + src.width  * abs(t.a)
            ymin = ymax - src.height * abs(t.e)
        return dsm, (xmin, ymin, xmax, ymax), gsd
