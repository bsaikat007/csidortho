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
from .pose           import Pose
from .camera         import BrownConradyCamera, PinholeCamera
from .georeference   import UTMTransformer, GeoTIFFWriter, GeoBounds
from .orthorectifier import Orthorectifier, OrthoParams
from .bundle_adjustment import (BundleAdjuster, Reconstruction,
                                 CameraPose, Point3D, Observation)


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
    min_baseline_m : minimum XY baseline in metres for frame thinning
    agl_tolerance_m : keep frames within this AGL distance from median AGL
    """

    def __init__(self, image_dir, output='orthomosaic.tif',
                 gsd=None, scale=0.5, ground_alt=None,
                 dist_coeffs=None, run_ba=False, blending='weighted',
                 min_baseline_m=2.0, agl_tolerance_m=15.0):
        self.image_dir   = Path(image_dir)
        self.output      = Path(output)
        self.gsd_target  = gsd
        self.scale       = scale
        self.ground_alt  = ground_alt
        self.dist_coeffs = dist_coeffs
        self.run_ba      = run_ba
        self.blending    = blending
        self.min_baseline_m = float(min_baseline_m)
        self.agl_tolerance_m = float(agl_tolerance_m)

        self._meta:  List[DJIImageMeta] = []
        self._poses: List[Pose]         = []
        self._utm:   Optional[UTMTransformer] = None

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
        coords_en = [self._utm.latlon_to_utm(m.latitude, m.longitude)
                     for m in self._meta]

        # Track-derived heading (0°=North, CW positive): much more stable for
        # nadir images where gimbal yaw can jitter near pitch=-90°.
        track_yaws = [None] * len(self._meta)
        for i in range(len(coords_en)):
            i0 = max(0, i - 1)
            i1 = min(len(coords_en) - 1, i + 1)
            de = coords_en[i1][0] - coords_en[i0][0]
            dn = coords_en[i1][1] - coords_en[i0][1]
            dxy = float(np.hypot(de, dn))
            if dxy >= 0.5:
                yaw_track = (np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0
                track_yaws[i] = float(yaw_track)

        # FIX [2]: ground altitude only needed as fallback
        if self.ground_alt is None:
            # Estimate: mean(GPS_abs - relative) across all images
            diffs = [m.altitude_abs - m.altitude_rel
                     for m in self._meta if m.altitude_rel > 0]
            self.ground_alt = float(np.mean(diffs)) if diffs else 0.0
            print(f"Ground elevation estimate: {self.ground_alt:.1f} m AMSL")

        poses = []
        skipped = 0
        for i, m in enumerate(self._meta):
            e, n = coords_en[i]

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

            # Prefer track heading for nadir, else use aircraft/camera yaw.
            if abs(m.gimbal_pitch + 90) < 2.0 and track_yaws[i] is not None:
                yaw_base = track_yaws[i]
            else:
                yaw_base = m.flight_yaw if abs(m.gimbal_pitch + 90) < 2.0 else m.gimbal_yaw

            gc_deg = self._utm.compute_grid_convergence(m.latitude, m.longitude)
            yaw_corrected = yaw_base + gc_deg

            if abs(m.gimbal_pitch + 90) < 2.0:
                # Nadir (pitch ≈ -90°)
                pose = Pose.make_nadir(e, n, agl, yaw_corrected)
            else:
                # Oblique
                pose = Pose.from_euler_angles(
                    np.radians(m.gimbal_roll),
                    np.radians(m.gimbal_pitch),
                    np.radians(yaw_corrected),
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

        # Frame thinning: remove near-static frames that mostly differ by yaw,
        # which otherwise creates swirl artefacts in the blended mosaic.
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

        # Altitude consistency filtering: remove takeoff/ascent frames that
        # are far from the mission's dominant flight altitude.
        if self.agl_tolerance_m > 0 and len(self._meta) > 3:
            agls = np.array([
                m.altitude_rel if m.altitude_rel > 0 else m.altitude_abs - self.ground_alt
                for m in self._meta
            ], dtype=np.float64)
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
        if len(self._poses) < 2:
            raise ValueError(
                f"Only {len(self._poses)} valid poses. Check XMP RelativeAltitude tags "
                f"in your images — run: exiftool -RelativeAltitude DJI_0001.JPG")
        return self._poses

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

        params = OrthoParams(gsd=gsd, output_bounds=bounds,
                             target_srs=f"EPSG:{self._utm.epsg_code}")
        ort    = Orthorectifier(params)
        out_h  = ort.height; out_w = ort.width

        # Per-image brightness for radiometric normalisation
        means = []
        for m in meta:
            img = cv2.imread(str(m.path))
            if img is not None:
                bright = img[img > 0].mean() if img.any() else 128.
            else:
                bright = 128.
            means.append(float(bright))
            del img; gc.collect()
        global_mean = float(np.mean(means)) if means else 128.

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
            gain = global_mean / max(means[i], 1e-6)
            img  = np.clip(img.astype(np.float32)*gain, 0, 255).astype(np.uint8)

            ortho, weight = ort.orthorectify_image(img, pose, cam)
            oh = min(ortho.shape[0], out_h)
            ow = min(ortho.shape[1], out_w)
            accum[:oh,:ow] += ortho[:oh,:ow].astype(np.float64) * weight[:oh,:ow,np.newaxis]
            wsum[:oh,:ow]  += weight[:oh,:ow]
            del img, ortho, weight; gc.collect()

        print(f"\nNormalising …")
        result   = np.clip(accum / np.maximum(wsum[:,:,np.newaxis], 1e-8),
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
        print("DRONE ORTHOMOSAIC - MEMORY EFFICIENT")
        print("="*55)
        self.load_images()
        self.initialise_poses()
        if self.run_ba:
            print("\nRefining poses with bundle adjustment …")
            self.refine_poses()
        result, bounds, gsd = self.generate_orthomosaic()
        self.save(result, bounds, gsd)
        return self.output

    # ── Internal ──────────────────────────────────────────────

    def _build_K(self, m: DJIImageMeta, scale=1.0) -> np.ndarray:
        s = scale * self.scale
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
