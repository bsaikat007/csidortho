"""
End-to-end orthomosaic pipeline — v2.

Fixes vs v1:
  [1] initialise_poses: now processes ALL images, not just those
      where UTMTransformer is already set. The v1 code had a logic
      error where the pose loop depended on self._utm being set before
      entering the loop — causing only 1 pose for 30 images.
  [2] ground_alt estimated correctly per-image using altitude_rel from XMP
  [3] AGL = altitude_rel (from XMP) — NOT GPS_alt - ground_alt
      DJI stores relative altitude (above takeoff) in XMP RelativeAltitude.
      GPS altitude is absolute (AMSL). Using GPS - ground_alt introduces
      a systematic error equal to the takeoff elevation estimate error.
  [4] Fallback: if altitude_rel = 0 (XMP missing), use GPS_alt - ground_alt
  [5] Memory-safe: processes one image at a time, explicit gc.collect()
  [6] Progress reporting improved
"""

import numpy as np
import cv2
import gc
from pathlib import Path
from typing import Optional, List, Tuple

from .exif_reader    import read_dji_image, DJIImageMeta
from .pose           import Pose, triangulate_points, solve_pnp
from .camera         import BrownConradyCamera, PinholeCamera
from .georeference   import UTMTransformer, GeoTIFFWriter, GeoBounds
from .orthorectifier import Orthorectifier, OrthoParams
from .bundle_adjustment import (BundleAdjuster, Reconstruction,
                                 CameraPose, Point3D, Observation)
from .dem_builder    import DEMBuilder, DEMParams


# Typical distortion for common DJI cameras
# For best accuracy, run a checkerboard calibration and pass dist_coeffs
_DEFAULT_DIST = {
    'FC6310':  [-0.0508,  0.0290,  0.00015, -0.00010, -0.0087],
    'FC6310S': [-0.0508,  0.0290,  0.00015, -0.00010, -0.0087],
    'FC330':   [-0.0341,  0.0176,  0.00010, -0.00008, -0.0052],
    'FC220':   [-0.0289,  0.0143,  0.00009, -0.00007, -0.0041],
    'FC7203':  [-0.0220,  0.0100,  0.00005, -0.00005, -0.0030],
    'UNKNOWN': [ 0.0,     0.0,     0.0,      0.0,      0.0   ],
}


class OrthoPipeline:
    """
    One-call orthomosaic generator for DJI drone imagery.

    Quick start:
        from orthomosaic import OrthoPipeline
        OrthoPipeline(image_dir="images/", output="ortho.tif", scale=0.5).run()

    Parameters
    ----------
    image_dir    : folder of DJI JPEG images
    output       : output GeoTIFF path
    gsd          : target GSD in metres (default: auto from lowest AGL)
    scale        : image downscale 0.1–1.0 (reduce for low RAM)
    ground_alt   : ground elevation m AMSL — if None, inferred from XMP RelativeAltitude
    dist_coeffs  : [k1,k2,p1,p2,k3] override — None uses model defaults
    run_ba       : bundle adjustment (only for grid flights with lateral overlap)
    blending     : 'weighted' | 'nearest' | 'multiband'
    """

    def __init__(self, image_dir, output='orthomosaic.tif',
                 gsd=None, scale=0.5, ground_alt=None,
                 dist_coeffs=None, run_ba=False, blending='weighted',
                 min_baseline_m=2.0, agl_tolerance_m=15.0,
                 refine_overlap_shift=True, max_shift_px=40.0,
                 use_dem=False, dem_mode='sparse', dem_gsd=None,
                 dem_strength=0.25):
        self.image_dir   = Path(image_dir)
        self.output      = Path(output)
        self.gsd_target  = gsd
        self.scale       = scale
        self.ground_alt  = ground_alt
        self.dist_coeffs = dist_coeffs
        self.run_ba      = run_ba
        self.blending    = blending
        self.min_baseline_m      = float(min_baseline_m)
        self.agl_tolerance_m    = float(agl_tolerance_m)
        self.refine_overlap_shift = bool(refine_overlap_shift)
        self.max_shift_px       = float(max_shift_px)
        self.use_dem            = bool(use_dem)
        self.dem_mode           = dem_mode
        self.dem_gsd            = dem_gsd
        self.dem_strength       = float(np.clip(dem_strength, 0.0, 1.0))

        self._meta:  List[DJIImageMeta] = []
        self._poses: List[Pose]         = []
        self._utm:   Optional[UTMTransformer] = None
        self._all_meta_valid: List[DJIImageMeta] = []

    # ── Step 1: load ──────────────────────────────────────────

    def load_images(self) -> List[DJIImageMeta]:
        exts  = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
        paths = sorted(p for p in self.image_dir.iterdir()
                       if p.suffix in exts)
        if not paths:
            raise FileNotFoundError(f"No JPEG images in {self.image_dir}")

        meta = []
        for p in paths:
            try:
                m = read_dji_image(p)
                meta.append(m)
            except Exception as e:
                print(f"  SKIP {p.name}: {e}")

        if len(meta) < 2:
            raise ValueError(f"Need ≥ 2 images with GPS, found {len(meta)}")

        self._meta = meta
        m0 = meta[0]
        agls = [m.altitude_rel for m in meta]
        print(f"Found {len(meta)} images")
        print(f"Image size  : {m0.image_width}x{m0.image_height}")
        print(f"Camera      : {m0.make} {m0.model}  "
              f"f={m0.focal_length_mm:.1f}mm  "
              f"fx={m0.fx_pixels:.1f}px")
        print(f"AGL range   : {min(agls):.1f} – {max(agls):.1f} m")
        return meta

    # ── Step 2: poses ─────────────────────────────────────────

    def initialise_poses(self) -> List[Pose]:
        """
        FIX [1]: Build UTM transformer FIRST, then loop over all images.
        FIX [2]: AGL = altitude_rel from XMP (not GPS - ground_alt estimate).
        FIX [4]: Fallback to GPS - ground_alt when altitude_rel = 0.
        """
        if not self._meta:
            self.load_images()

        lats = [m.latitude  for m in self._meta]
        lons = [m.longitude for m in self._meta]

        # Build UTM transformer from all coordinates — BEFORE the pose loop
        self._utm = UTMTransformer.from_coordinates(lats, lons)
        gc_deg    = self._utm.compute_grid_convergence(
                        float(np.mean(lats)), float(np.mean(lons)))

        # FIX [2]: ground altitude only needed as fallback
        if self.ground_alt is None:
            # Estimate: mean(GPS_abs - relative) across all images
            diffs = [m.altitude_abs - m.altitude_rel
                     for m in self._meta if m.altitude_rel > 0]
            self.ground_alt = float(np.mean(diffs)) if diffs else 0.0
            print(f"Ground elevation estimate: {self.ground_alt:.1f} m AMSL")

        poses = []
        skipped = 0
        for m in self._meta:
            e, n = self._utm.latlon_to_utm(m.latitude, m.longitude)

            # FIX [3]: prefer XMP RelativeAltitude (direct AGL measurement)
            if m.altitude_rel > 0:
                agl = m.altitude_rel
            else:
                # Fallback: GPS altitude minus ground elevation
                agl = m.altitude_abs - self.ground_alt
                if agl <= 0:
                    print(f"  SKIP {m.path.name}: AGL={agl:.1f}m (non-positive)")
                    skipped += 1
                    continue

            # Use FlightYaw when GimbalYaw is locked to body (0° on all images)
            yaw_abs = m.gimbal_yaw if abs(m.gimbal_yaw) > 0.5 else m.flight_yaw
            yaw_corrected = yaw_abs + gc_deg

            pose = Pose.from_dji_gimbal(
                yaw_corrected, m.gimbal_pitch, m.gimbal_roll,
                np.array([e, n, agl]))

            poses.append(pose)

        if skipped:
            print(f"  (Skipped {skipped} images with non-positive AGL)")

        # Align meta and poses lists — remove skipped entries
        # Rebuild _meta to match poses length
        valid_meta = []
        pose_idx = 0
        for m in self._meta:
            agl = m.altitude_rel if m.altitude_rel > 0 else m.altitude_abs - self.ground_alt
            if agl > 0:
                valid_meta.append(m)
        self._meta  = valid_meta
        self._poses = poses

        # Save all valid meta before thinning (for full-coverage gap fill)
        self._all_meta_valid = list(valid_meta)

        # Frame thinning: remove near-static frames
        if self.min_baseline_m > 0 and len(self._poses) > 1:
            kept_meta = [self._meta[0]]
            kept_poses = [self._poses[0]]
            dropped = 0
            for m, p in zip(self._meta[1:], self._poses[1:]):
                dxy = float(np.linalg.norm(p.C[:2] - kept_poses[-1].C[:2]))
                if dxy < self.min_baseline_m:
                    dropped += 1
                    continue
                kept_meta.append(m)
                kept_poses.append(p)
            self._meta = kept_meta
            self._poses = kept_poses
            if dropped:
                print(f"Frame thinning: dropped {dropped} near-static images "
                      f"(< {self.min_baseline_m:.1f} m baseline)")

        # Altitude consistency filtering
        if self.agl_tolerance_m > 0 and len(self._meta) > 3:
            agls = np.array([
                m.altitude_rel if m.altitude_rel > 0 else m.altitude_abs - self.ground_alt
                for m in self._meta], dtype=np.float64)
            med_agl = float(np.median(agls))
            keep_mask = np.abs(agls - med_agl) <= self.agl_tolerance_m
            kept = int(np.sum(keep_mask))
            if kept >= 2 and kept < len(self._meta):
                dropped_alt = len(self._meta) - kept
                self._meta = [m for m, k in zip(self._meta, keep_mask) if k]
                self._poses = [p for p, k in zip(self._poses, keep_mask) if k]
                print(f"Altitude filter: dropped {dropped_alt} frames outside "
                      f"median±{self.agl_tolerance_m:.1f} m (median AGL={med_agl:.1f} m)")

        print(f"Initialised {len(self._poses)} poses from GPS/IMU")
        if len(poses) < 2:
            raise ValueError(
                f"Only {len(poses)} valid poses. Check XMP RelativeAltitude tags "
                f"in your images — run: exiftool -RelativeAltitude DJI_0001.JPG")
        return poses

    # ── Step 3 (optional): bundle adjustment ─────────────────

    def refine_poses(self, max_iter=50):
        if not self._poses:
            self.initialise_poses()
        if len(self._poses) < 3:
            print("Too few poses for BA — skipping")
            return self._poses

        K = self._build_K(self._meta[0], scale=1.0)
        SSCALE = 0.1  # very small for BA feature matching
        K_s    = K.copy(); K_s[:2] *= SSCALE

        sift  = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.02)
        flann = cv2.FlannBasedMatcher({'algorithm':1,'trees':5},{'checks':50})
        feats = []
        for m in self._meta:
            img = cv2.imread(str(m.path))
            img = cv2.resize(img, (int(m.image_width*SSCALE),
                                    int(m.image_height*SSCALE)))
            kp, d = sift.detectAndCompute(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            feats.append((kp, d))
            del img; gc.collect()

        from .pose import triangulate_points
        rec = Reconstruction(cameras={}, points={}, observations=[])
        for i, (pose, m) in enumerate(zip(self._poses, self._meta)):
            dist = (self.dist_coeffs or
                    _DEFAULT_DIST.get(m.model, _DEFAULT_DIST['UNKNOWN']))
            rec.cameras[i] = CameraPose(i, pose.R, pose.t, K_s,
                                         np.array(dist), fixed=(i==0))

        pt_id = 0
        for i in range(len(self._meta)-1):
            kp1, d1 = feats[i]; kp2, d2 = feats[i+1]
            if d1 is None or d2 is None: continue
            matches = flann.knnMatch(d1, d2, k=2)
            good = [(mm.queryIdx, mm.trainIdx)
                    for mm, nn in matches if mm.distance < 0.75*nn.distance]
            if len(good) < 8: continue
            p1 = np.float32([kp1[g[0]].pt for g in good])
            p2 = np.float32([kp2[g[1]].pt for g in good])
            pts3d = triangulate_points(p1, p2,
                                        self._poses[i], self._poses[i+1], K_s)
            for pt, pp1, pp2 in zip(pts3d, p1, p2):
                if not np.isfinite(pt).all(): continue
                Xc = self._poses[i].transform_to_camera(pt.reshape(1,3))[0]
                if Xc[2] <= 0: continue
                rec.points[pt_id] = Point3D(pt_id, pt)
                rec.observations += [Observation(i,   pt_id, pp1),
                                      Observation(i+1, pt_id, pp2)]
                pt_id += 1

        if len(rec.points) < 10:
            print("Too few triangulated points — skipping BA")
            return self._poses

        rec_ref = BundleAdjuster(max_iterations=max_iter).optimize(rec)
        for i in range(len(self._poses)):
            c = rec_ref.cameras[i]
            self._poses[i] = Pose(R=c.R, t=c.t)
        return self._poses

    # ── Step 4: orthorectify ──────────────────────────────────

    def generate_orthomosaic(self) -> Tuple[np.ndarray, tuple, float]:
        if not self._poses:
            self.initialise_poses()

        meta  = self._meta
        poses = self._poses

        # Full-coverage: also use non-thinned images for gap fill
        fill_meta, fill_poses = self._get_fill_images()

        # Compute output bounds from image footprints
        e_vals, n_vals = [], []
        for m, pose in zip(meta, poses):
            cam  = self._build_camera(m)
            half_w = m.altitude_rel * (m.sensor_width_mm  / m.focal_length_mm) / 2
            half_h = m.altitude_rel * (m.sensor_height_mm / m.focal_length_mm) / 2
            C = pose.C
            e_vals += [C[0]-half_w, C[0]+half_w]
            n_vals += [C[1]-half_h, C[1]+half_h]

        PAD    = 5.0
        bounds = (min(e_vals)-PAD, min(n_vals)-PAD,
                  max(e_vals)+PAD, max(n_vals)+PAD)

        # GSD from best (lowest AGL) image
        best = min(meta, key=lambda m: m.altitude_rel if m.altitude_rel>0 else 999)
        gsd  = (self.gsd_target or
                (best.altitude_rel * (best.pixel_size_um * 1e-3)
                 / best.focal_length_mm * self.scale))

        w_m = bounds[2]-bounds[0]; h_m = bounds[3]-bounds[1]
        npx = int(w_m/gsd) * int(h_m/gsd) * 3 * 8 / 1e6
        print(f"\nOutput size : {int(w_m/gsd)}x{int(h_m/gsd)} px  "
              f"(~{npx:.1f} MB accumulators)")

        # Build DEM if requested
        dem, dem_res = self._build_dem(meta, poses, bounds, gsd)

        params = OrthoParams(gsd=gsd, output_bounds=bounds,
                             target_srs=f"EPSG:{self._utm.epsg_code}",
                             elevation_model=dem,
                             dem_resolution=dem_res)
        ort    = Orthorectifier(params)
        out_h  = ort.height; out_w = ort.width

        # Overlap refinement
        self._refine_overlap(meta, poses, ort)

        # Color harmonization: per-tile gain+offset to reference
        ref_img = cv2.imread(str(meta[0].path))
        if ref_img is not None and self.scale != 1.0:
            ref_img = cv2.resize(ref_img,
                (int(meta[0].image_width * self.scale),
                 int(meta[0].image_height * self.scale)))
        ref_mean = np.array([128., 128., 128.]) if ref_img is None \
            else cv2.mean(ref_img)[:3]
        ref_std  = np.array([40., 40., 40.]) if ref_img is None \
            else np.array([float(np.std(ref_img[:,:,c])) for c in range(3)])
        if ref_img is not None:
            del ref_img; gc.collect()

        use_multiband = (self.blending == 'multiband')

        if self.blending == 'nearest':
            nearest = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            best_w  = np.zeros((out_h, out_w),    dtype=np.float64)
            score_old = np.zeros((out_h, out_w),   dtype=np.float64)
        elif use_multiband:
            ortho_tiles = []
            weight_tiles = []
        else:
            accum = np.zeros((out_h, out_w, 3), dtype=np.float64)
            wsum  = np.zeros((out_h, out_w),    dtype=np.float64)

        print(f"\nProcessing images …")
        for i, (m, pose) in enumerate(zip(meta, poses)):
            print(f"  [{i+1}/{len(meta)}] {m.path.name}")
            img = cv2.imread(str(m.path))
            if img is None:
                print(f"    Cannot load — skipping"); continue

            if self.scale != 1.0:
                nw  = int(m.image_width  * self.scale)
                nh  = int(m.image_height * self.scale)
                img = cv2.resize(img, (nw, nh))

            cam  = self._build_camera(m)

            # Pre-undistort the image if supported
            if hasattr(cam, 'undistort_image'):
                undist_img = cam.undistort_image(img)
                pin_cam = PinholeCamera(
                    width=undist_img.shape[1],
                    height=undist_img.shape[0],
                    fx=cam.fx, fy=cam.fy,
                    cx=cam.cx, cy=cam.cy,
                    skew=cam.skew
                )
                img_use = undist_img
            else:
                img_use = img
                pin_cam = cam

            # Color harmonization on img_use
            img_f = img_use.astype(np.float64)
            for c in range(3):
                mask_c = img_f[:,:,c] > 0
                tile_mean = float(img_f[:,:,c][mask_c].mean()) if np.any(mask_c) else 128.
                tile_std  = float(np.std(img_f[:,:,c][mask_c])) if np.any(mask_c) else 40.
                if tile_std < 1.0:
                    tile_std = 40.0
                img_f[:,:,c] = (img_f[:,:,c] - tile_mean) * (ref_std[c] / tile_std) + ref_mean[c]
            img_use = np.clip(img_f, 0, 255).astype(np.uint8)
            del img_f

            ortho, weight = ort.orthorectify_image(img_use, pose, pin_cam)
            oh = min(ortho.shape[0], out_h)
            ow = min(ortho.shape[1], out_w)

            if self.blending == 'nearest':
                score_new = weight[:oh,:ow]
                mask = (score_new > score_old[:oh,:ow]) & (score_new > 0.01)
                nearest[:oh,:ow][mask] = ortho[:oh,:ow][mask]
                best_w[:oh,:ow][mask] = weight[:oh,:ow][mask]
                score_old[:oh,:ow][mask] = score_new[mask]
            elif use_multiband:
                ortho_tiles.append(ortho[:oh, :ow].copy())
                weight_tiles.append(weight[:oh, :ow].copy())
            else:
                accum[:oh,:ow] += ortho[:oh,:ow].astype(np.float64) * weight[:oh,:ow,np.newaxis]
                wsum[:oh,:ow]  += weight[:oh,:ow]
            del img, ortho, weight; gc.collect()

        # Gap fill: process non-thinned images to fill coverage holes
        if fill_meta and len(fill_meta) > len(meta):
            main_paths = {m.path for m in meta}
            fill_set = [(m, p) for m, p in zip(fill_meta, fill_poses)
                        if m.path not in main_paths]
            if fill_set:
                print(f"\nGap-fill pass ({len(fill_set)} extra images) …")
                for fi, (m, pose) in enumerate(fill_set):
                    img = cv2.imread(str(m.path))
                    if img is None:
                        continue
                    if self.scale != 1.0:
                        nw = int(m.image_width * self.scale)
                        nh = int(m.image_height * self.scale)
                        img = cv2.resize(img, (nw, nh))

                    cam = self._build_camera(m)
                    # Pre-undistort and use pinhole camera
                    if hasattr(cam, 'undistort_image'):
                        undist_img = cam.undistort_image(img)
                        pin_cam = PinholeCamera(
                            width=undist_img.shape[1],
                            height=undist_img.shape[0],
                            fx=cam.fx, fy=cam.fy,
                            cx=cam.cx, cy=cam.cy,
                            skew=cam.skew
                        )
                        img_use = undist_img
                    else:
                        img_use = img
                        pin_cam = cam

                    # Color harmonization on img_use
                    img_f = img_use.astype(np.float64)
                    for c in range(3):
                        mask_c = img_f[:,:,c] > 0
                        tile_mean = float(img_f[:,:,c][mask_c].mean()) if np.any(mask_c) else 128.
                        tile_std  = float(np.std(img_f[:,:,c][mask_c])) if np.any(mask_c) else 40.
                        if tile_std < 1.0:
                            tile_std = 40.0
                        img_f[:,:,c] = (img_f[:,:,c] - tile_mean) * (ref_std[c] / tile_std) + ref_mean[c]
                    img_use = np.clip(img_f, 0, 255).astype(np.uint8)
                    del img_f

                    ortho, weight = ort.orthorectify_image(img_use, pose, pin_cam)
                    oh = min(ortho.shape[0], out_h)
                    ow = min(ortho.shape[1], out_w)

                    if self.blending == 'nearest':
                        empty = (best_w[:oh, :ow] < 0.01) & (weight[:oh, :ow] > 0.05)
                        if np.any(empty):
                            nearest[:oh, :ow][empty] = ortho[:oh, :ow][empty]
                            best_w[:oh, :ow][empty] = weight[:oh, :ow][empty]
                    elif use_multiband:
                        ortho_tiles.append(ortho[:oh, :ow].copy())
                        weight_tiles.append(weight[:oh, :ow].copy())
                    else:
                        empty = (wsum[:oh, :ow] < 1e-6) & (weight[:oh, :ow] > 0.05)
                        if np.any(empty):
                            accum[:oh,:ow][empty] = (ortho[:oh,:ow].astype(np.float64)
                                                      * weight[:oh,:ow,np.newaxis])[empty]
                            wsum[:oh,:ow][empty] = weight[:oh,:ow][empty]
                    del img, ortho, weight; gc.collect()

        # Final blending
        print(f"\nNormalising …")
        if self.blending == 'nearest':
            result = nearest
        elif use_multiband and ortho_tiles:
            result = self._multiband_blend(ortho_tiles, weight_tiles, out_h, out_w)
        else:
            result = np.clip(accum / np.maximum(wsum[:,:,np.newaxis], 1e-8),
                             0, 255).astype(np.uint8)
        coverage = 100 * np.sum(result.any(axis=2)) / (out_h * out_w)
        print(f"Coverage: {coverage:.1f}%")
        return result, bounds, gsd

    # ── Step 5: save ──────────────────────────────────────────

    def save(self, result, bounds, gsd):
        geo_b = GeoBounds(left=bounds[0], bottom=bounds[1],
                          right=bounds[2], top=bounds[3],
                          crs=f"EPSG:{self._utm.epsg_code}")
        writer = GeoTIFFWriter(crs=geo_b.crs)
        print(f"\nWriting GeoTIFF …")
        writer.write_orthomosaic(str(self.output), result, geo_b, gsd=gsd)
        kml = self.output.with_suffix('.kml')
        writer.create_kml_overlay(str(kml), self.output.name, geo_b,
                                   name="Drone Orthomosaic")
        # JPEG output (visual only, no georeferencing)
        jpeg = self.output.with_suffix('.jpg')
        from PIL import Image
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(jpeg), quality=95, optimize=True)
        print(f"Writing JPEG …")
        # Summary
        import os
        print(f"\n{'='*55}")
        print(f"✓  DONE")
        for suf in ['.kml', '.tif', '.jpg']:
            p = self.output.with_suffix(suf)
            if p.exists():
                print(f"   {p.name:<30} {os.path.getsize(p)/1e6:.2f} MB")
        print(f"{'='*55}")

    # ── Convenience ───────────────────────────────────────────

    def run(self) -> Path:
        print("="*55)
        print("DRONE ORTHOMOSAIC - PROFESSIONAL")
        print("="*55)
        self.load_images()
        self.initialise_poses()
        if self.run_ba:
            print("\nRefining poses with bundle adjustment …")
            self.refine_poses()
        self._refine_poses_pairwise()
        result, bounds, gsd = self.generate_orthomosaic()
        self.save(result, bounds, gsd)
        return self.output

    # ── Internal ──────────────────────────────────────────────

    def _build_K(self, m: DJIImageMeta, scale=None) -> np.ndarray:
        s = self.scale if scale is None else scale
        return np.array([[m.fx_pixels*s,0,m.cx_pixels*s],
                          [0,m.fy_pixels*s,m.cy_pixels*s],
                          [0,0,1]])

    def _build_camera(self, m: DJIImageMeta):
        s    = self.scale
        dist = (self.dist_coeffs or
                _DEFAULT_DIST.get(m.model, _DEFAULT_DIST['UNKNOWN']))
        return BrownConradyCamera(
            width  = int(m.image_width  * s),
            height = int(m.image_height * s),
            fx = m.fx_pixels*s, fy = m.fy_pixels*s,
            cx = m.cx_pixels*s, cy = m.cy_pixels*s,
            k1=dist[0], k2=dist[1], p1=dist[2], p2=dist[3], k3=dist[4])

    # ── Pairwise pose refinement ──────────────────────────────

    def _refine_poses_pairwise(self):
        """SIFT+PnP pairwise pose refinement — corrects GPS drift.

        For nadir flights PnP rotation is unreliable (degenerate),
        so only the camera position is corrected.
        """
        if len(self._poses) < 3:
            return
        print("\nPairwise pose refinement (SIFT + PnP) …")

        sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02,
                                edgeThreshold=12)
        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5},
                                       {'checks': 100})

        SCALE_FEAT = 0.25
        feats = []
        for m in self._meta:
            img = cv2.imread(str(m.path))
            if img is None:
                feats.append(None)
                continue
            nw = max(128, int(m.image_width  * SCALE_FEAT))
            nh = max(128, int(m.image_height * SCALE_FEAT))
            img_s = cv2.resize(img, (nw, nh))
            gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            feats.append((kp, des, (nw, nh)))
            del img; gc.collect()

        n_corrected = 0
        for i in range(1, len(self._poses)):
            best_pose = None
            best_info = None

            for skip in [2, 1]:
                j = i - skip
                if j < 0:
                    continue
                if feats[j] is None or feats[i] is None:
                    continue

                kp_j, des_j, sz_j = feats[j]
                kp_i, des_i, sz_i = feats[i]
                if des_j is None or des_i is None:
                    continue
                if len(kp_j) < 30 or len(kp_i) < 30:
                    continue

                matches = flann.knnMatch(des_i, des_j, k=2)
                good = []
                for pair in matches:
                    if len(pair) < 2:
                        continue
                    a, b = pair
                    if a.distance < 0.75 * b.distance:
                        good.append(a)

                if len(good) < 25:
                    continue

                p_i = np.float32([kp_i[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
                p_j = np.float32([kp_j[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)

                s_i = self._meta[i].image_width / max(1, sz_i[0])
                s_j = self._meta[j].image_width / max(1, sz_j[0])
                p_i_native = p_i * s_i
                p_j_native = p_j * s_j

                K_j = self._build_K(self._meta[j], scale=1.0)
                K_i = self._build_K(self._meta[i], scale=1.0)

                pts3d = triangulate_points(p_j_native.reshape(-1, 2),
                                            p_i_native.reshape(-1, 2),
                                            self._poses[j], self._poses[i], K_j)

                valid = np.isfinite(pts3d).all(axis=1)
                pts3d = pts3d[valid]
                p_i_v = p_i_native.reshape(-1, 2)[valid]

                if len(pts3d) < 20:
                    continue
                X_cam = self._poses[i].transform_to_camera(pts3d)
                in_front = X_cam[:, 2] > 0
                pts3d = pts3d[in_front]
                p_i_v = p_i_v[in_front]
                if len(pts3d) < 15:
                    continue

                dist_arr = np.array(self.dist_coeffs or
                                    _DEFAULT_DIST.get(self._meta[i].model,
                                                      _DEFAULT_DIST['UNKNOWN']),
                                    dtype=np.float64)
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts3d.astype(np.float32), p_i_v.astype(np.float32),
                    K_i.astype(np.float32), dist_arr[:5].astype(np.float32),
                    iterationsCount=1000, reprojectionError=2.0,
                    confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)

                if not success or inliers is None or len(inliers) < 15:
                    continue

                R_new, _ = cv2.Rodrigues(rvec)
                t_new = tvec.flatten()

                # For nadir flights, PnP rotation is unreliable.
                # Keep original rotation, only correct position.
                C_new = -R_new.T @ t_new
                d_xy = np.linalg.norm(C_new[:2] - self._poses[i].C[:2])

                if d_xy > 15.0 or d_xy < 0.01:
                    continue

                R_orig = self._poses[i].R
                pose_new = Pose(R=R_orig, t=-R_orig @ C_new)

                best_pose = pose_new
                best_info = (j, d_xy, len(inliers))
                break  # prefer skip-2

            if best_pose is not None:
                self._poses[i] = best_pose
                j, d_xy, n_in = best_info
                print(f"  Pose {i} (ref {j}): corrected by "
                      f"{d_xy:.2f}m ({n_in} inliers)")
                n_corrected += 1

        print(f"Pairwise refinement done: {n_corrected}/{len(self._poses)} poses corrected")

    # ── DEM via DEMBuilder ────────────────────────────────────

    def _build_dem(self, meta, poses, bounds, gsd):
        """Build DSM using the new DEMBuilder class."""
        if not self.use_dem:
            return None, 1.0

        dem_gsd = self.dem_gsd or max(1.0, gsd * 10)
        dem_params = DEMParams(
            dem_gsd=dem_gsd,
            mode=self.dem_mode,
            stereo_scale=0.25,
            min_overlap=0.35,
            max_pairs_per_image=5,
            smooth_sigma=2.0)

        image_paths = [m.path for m in meta]
        cameras = [self._build_camera(m) for m in meta]

        # Ground Z in AGL frame is 0
        ground_z = 0.0

        # Distortion coefficients per image
        dist_list = []
        for m in meta:
            d = (self.dist_coeffs or
                 _DEFAULT_DIST.get(m.model, _DEFAULT_DIST['UNKNOWN']))
            dist_list.append(np.array(d, dtype=np.float64))

        builder = DEMBuilder(dem_params)
        dsm, dsm_gsd = builder.build(
            image_paths=image_paths,
            poses=poses,
            cameras=cameras,
            bounds=bounds,
            pipeline_scale=self.scale,
            dist_coeffs_list=dist_list,
            ground_z=ground_z)

        # Apply DEM strength (blend with flat surface)
        if self.dem_strength < 1.0:
            dsm = dsm * self.dem_strength

        return dsm, dsm_gsd

    # ── Multiband blending ────────────────────────────────────

    def _multiband_blend(self, ortho_tiles, weight_tiles, out_h, out_w):
        """Laplacian pyramid multi-band blending for seamless mosaic."""
        LEVELS = 5
        nc = 3

        def gauss_pyr(img, n):
            p = [img.astype(np.float32)]
            for _ in range(n):
                p.append(cv2.pyrDown(p[-1]))
            return p

        def lap_pyr(img, n):
            g = gauss_pyr(img, n)
            lp = []
            for i in range(n):
                up = cv2.pyrUp(g[i+1], dstsize=(g[i].shape[1], g[i].shape[0]))
                lp.append(g[i].astype(np.float64) - up.astype(np.float64))
            lp.append(g[-1].astype(np.float64))
            return lp

        result_pyrs = [np.zeros_like(lvl, dtype=np.float64)
                       for lvl in lap_pyr(
                           np.zeros((out_h, out_w, nc), dtype=np.float32), LEVELS)]
        weight_acc  = [np.zeros((lvl.shape[0], lvl.shape[1]), dtype=np.float64)
                       for lvl in result_pyrs]

        for tile, wgt in zip(ortho_tiles, weight_tiles):
            th, tw = tile.shape[:2]
            full_tile = np.zeros((out_h, out_w, nc), dtype=np.float32)
            full_tile[:th, :tw] = tile.astype(np.float32)
            full_w = np.zeros((out_h, out_w), dtype=np.float32)
            full_w[:th, :tw] = wgt.astype(np.float32)

            lp = lap_pyr(full_tile, LEVELS)
            gm = gauss_pyr(full_w, LEVELS)

            for lvl_idx in range(len(lp)):
                g = gm[lvl_idx]
                if g.ndim == 2:
                    g = g[:,:,np.newaxis]
                result_pyrs[lvl_idx] += lp[lvl_idx] * g
                weight_acc[lvl_idx]  += gm[lvl_idx]

            del full_tile, full_w, lp, gm; gc.collect()

        for lvl_idx in range(len(result_pyrs)):
            w = weight_acc[lvl_idx]
            w3 = np.maximum(w[:,:,np.newaxis] if w.ndim == 2 else w, 1e-8)
            result_pyrs[lvl_idx] /= w3

        result = result_pyrs[-1]
        for lvl in reversed(result_pyrs[:-1]):
            result = cv2.pyrUp(result, dstsize=(lvl.shape[1], lvl.shape[0]))
            result = result.astype(np.float64) + lvl.astype(np.float64)

        return np.clip(result, 0, 255).astype(np.uint8)

    # ── Overlap refinement ────────────────────────────────────

    def _refine_overlap(self, meta, poses, ort):
        """Affine (ORB+RANSAC) + phase-correlation XY shift refinement."""
        if not self.refine_overlap_shift or len(poses) < 2:
            return
        print("\nOverlap XY refinement …")
        SCALE_REF = 0.15

        feats = []
        for m in meta:
            img = cv2.imread(str(m.path))
            if img is None:
                feats.append(None)
                continue
            nw = max(64, int(m.image_width * SCALE_REF))
            nh = max(64, int(m.image_height * SCALE_REF))
            img_s = cv2.resize(img, (nw, nh))
            gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=800)
            kp, des = orb.detectAndCompute(gray, None)
            feats.append((kp, des, img_s.shape[:2]))
            del img; gc.collect()

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        for i in range(len(poses) - 1):
            if feats[i] is None or feats[i+1] is None:
                continue
            kp1, d1, sz1 = feats[i]
            kp2, d2, sz2 = feats[i+1]
            if d1 is None or d2 is None:
                continue
            matches = bf.knnMatch(d1, d2, k=2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) < 10:
                continue

            p1 = np.float32([kp1[g.queryIdx].pt for g in good])
            p2 = np.float32([kp2[g.trainIdx].pt for g in good])

            # Scale to ortho pixel coords
            s1 = self._meta[i].image_width / max(1, sz1[1])
            s2 = self._meta[i+1].image_width / max(1, sz2[1])
            p1_o = p1 * s1 * self.scale
            p2_o = p2 * s2 * self.scale

            # Compute XY shift in world coords
            M, inliers = cv2.estimateAffinePartial2D(p2_o, p1_o)
            if M is None:
                continue
            dx_px = M[0, 2]
            dy_px = M[1, 2]
            shift_m = np.sqrt(dx_px**2 + dy_px**2) * ort.gsd

            if shift_m < 0.01 or shift_m > self.max_shift_px * ort.gsd:
                continue

            # Apply shift to pose i+1
            dx_world = dx_px * ort.gsd
            dy_world = dy_px * ort.gsd
            C = poses[i+1].C.copy()
            C[0] += dx_world
            C[1] -= dy_world  # north-at-top: +row = -northing
            poses[i+1].C = C

    # ── Gap-fill images ───────────────────────────────────────

    def _get_fill_images(self):
        """Return ALL valid images (including thinned-out ones) with poses
        for full-coverage compositing."""
        if not self._utm or not self._all_meta_valid:
            return [], []
        all_meta = []
        all_poses = []
        main_paths = {m.path for m in self._meta}

        for m in self._all_meta_valid:
            if m.path in main_paths:
                idx = next(i for i, mm in enumerate(self._meta) if mm.path == m.path)
                all_meta.append(m)
                all_poses.append(self._poses[idx])
            else:
                e, n = self._utm.latlon_to_utm(m.latitude, m.longitude)
                if m.altitude_rel > 0:
                    agl = m.altitude_rel
                else:
                    agl = m.altitude_abs - self.ground_alt
                if agl <= 0:
                    continue
                gc_deg = self._utm.compute_grid_convergence(m.latitude, m.longitude)
                yaw_abs = m.gimbal_yaw if abs(m.gimbal_yaw) > 0.5 else m.flight_yaw
                yaw_corrected = yaw_abs + gc_deg
                pose = Pose.from_dji_gimbal(
                    yaw_corrected, m.gimbal_pitch, m.gimbal_roll,
                    np.array([e, n, agl]))
                all_meta.append(m)
                all_poses.append(pose)
        return all_meta, all_poses
