"""
Dense matching and depth estimation.

Fixes vs original:
  [1] plane_sweep: invalid projections → NaN (not -1 which corrupted NCC)
  [2] SGBM with proper stereo rectification (required for non-horizontal pairs)
  [3] disparity_to_depth: near-zero disparity marked invalid (not clamped to max_depth)
  [4] Left-right consistency check
  [5] NCC border handling with BORDER_REFLECT (not zero-padding)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import median_filter
from scipy.interpolate import griddata


@dataclass
class DepthMap:
    depth:      np.ndarray   # (H,W) metres
    confidence: np.ndarray   # (H,W) [0,1]
    mask:       np.ndarray   # (H,W) bool
    camera_id:  int


class DenseMatcher:

    def __init__(self, window_size=11, max_disparity=128,
                 min_depth=5., max_depth=500.):
        self.window_size   = window_size
        self.max_disparity = max_disparity
        self.min_depth     = min_depth
        self.max_depth     = max_depth

    # ── rectified stereo depth ────────────────────────────────

    def compute_depth_stereo(self, img_ref, img_src, pose_ref, pose_src,
                             K, dist_ref=None, dist_src=None) -> DepthMap:
        """
        FIX [2]: rectifies image pair before SGBM.
        Correct for aerial nadir imagery where epipolar lines are not horizontal.
        """
        dist_ref = dist_ref if dist_ref is not None else np.zeros(5)
        dist_src = dist_src if dist_src is not None else np.zeros(5)
        h, w = img_ref.shape[:2]

        R_rel = pose_src.R @ pose_ref.R.T
        t_rel = pose_src.t - R_rel @ pose_ref.t
        baseline = float(np.linalg.norm(t_rel))
        if baseline < 1e-6:
            raise ValueError("Zero baseline — cannot compute depth.")

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, dist_ref, K, dist_src, (w,h), R_rel, t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.)

        map1r, map2r = cv2.initUndistortRectifyMap(K, dist_ref, R1, P1, (w,h), cv2.CV_32FC1)
        map1s, map2s = cv2.initUndistortRectifyMap(K, dist_src, R2, P2, (w,h), cv2.CV_32FC1)
        rect_ref = cv2.remap(img_ref, map1r, map2r, cv2.INTER_LINEAR)
        rect_src = cv2.remap(img_src, map1s, map2s, cv2.INTER_LINEAR)

        disp, mask_disp = self._sgbm(rect_ref, rect_src)
        conf  = self._lr_consistency(rect_ref, rect_src, disp, mask_disp)

        # FIX [3]: mark near-zero disparity as invalid
        valid = mask_disp & (disp > 0.5)
        fx    = float(P1[0, 0])
        depth = np.where(valid, (baseline * fx) / np.where(valid, disp, 1.), 0.)
        depth = np.where(valid & (depth >= self.min_depth) &
                         (depth <= self.max_depth), depth, 0.)
        final_mask = (depth > 0) & (conf > 0.5)

        return DepthMap(depth=depth, confidence=conf,
                        mask=final_mask, camera_id=0)

    def _sgbm(self, img_l, img_r):
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if img_l.ndim==3 else img_l
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if img_r.ndim==3 else img_r
        stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.max_disparity,
            blockSize=5, P1=8*3*25, P2=32*3*25,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        raw  = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.
        return raw, (raw > 0)

    def _lr_consistency(self, img_l, img_r, disp_l, mask, threshold=1.):
        """FIX [4]: forward-backward consistency check."""
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if img_l.ndim==3 else img_l
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if img_r.ndim==3 else img_r
        try:
            stereo_base = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.max_disparity, blockSize=5)
            stereo_r = cv2.ximgproc.createRightMatcher(stereo_base)
            disp_r   = stereo_r.compute(gray_r, gray_l).astype(np.float32) / 16.
        except Exception:
            return mask.astype(np.float32)

        h, w = disp_l.shape
        x_idx   = np.arange(w, dtype=np.float32)
        x_right = (x_idx[np.newaxis,:] - disp_l).astype(np.int32)
        conf    = np.zeros_like(disp_l)
        valid   = mask & (x_right >= 0) & (x_right < w)
        rows    = np.arange(h, dtype=np.int32)[:,np.newaxis]
        diff    = np.abs(disp_l -
                         disp_r[rows*np.ones_like(x_right),
                                np.clip(x_right,0,w-1)])
        conf[valid] = (diff[valid] < threshold).astype(np.float32)
        return conf

    # ── NCC ───────────────────────────────────────────────────

    def compute_ncc(self, img1, img2, window_size=11):
        """FIX [5]: BORDER_REFLECT_101 — no zero-padding bias at borders."""
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        k    = np.ones((window_size,window_size), dtype=np.float32) / window_size**2
        bt   = cv2.BORDER_REFLECT_101
        m1   = cv2.filter2D(img1, -1, k, borderType=bt)
        m2   = cv2.filter2D(img2, -1, k, borderType=bt)
        v1   = np.maximum(cv2.filter2D(img1**2,-1,k,borderType=bt) - m1**2, 0.)
        v2   = np.maximum(cv2.filter2D(img2**2,-1,k,borderType=bt) - m2**2, 0.)
        cc   = cv2.filter2D(img1*img2,-1,k,borderType=bt) - m1*m2
        return np.clip(cc / (np.sqrt(v1)*np.sqrt(v2) + 1e-6), -1., 1.)

    # ── plane sweep ───────────────────────────────────────────

    def plane_sweep_depth(self, reference_img, neighbor_imgs,
                          reference_pose, neighbor_poses, K,
                          num_planes=64) -> DepthMap:
        """
        FIX [1]: NaN for invalid projections so they don't bias NCC.
        """
        h, w = reference_img.shape[:2]
        depths = np.logspace(np.log10(self.min_depth),
                             np.log10(self.max_depth), num_planes)

        ref_gray = (cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
                    if reference_img.ndim==3 else reference_img).astype(np.float32)

        u_g, v_g = np.meshgrid(np.arange(w), np.arange(h))
        pixels   = np.column_stack([u_g.ravel(), v_g.ravel()])

        cost_vol = np.full((h,w,num_planes), np.nan, dtype=np.float32)

        for d_idx, depth in enumerate(depths):
            x_n = (pixels[:,0]-K[0,2])/K[0,0]
            y_n = (pixels[:,1]-K[1,2])/K[1,1]
            X_world = reference_pose.transform_to_world(
                np.column_stack([x_n*depth, y_n*depth,
                                  np.full(len(pixels),depth)]))

            costs = []
            for nbr_img, nbr_pose in zip(neighbor_imgs, neighbor_poses):
                nbr_gray = (cv2.cvtColor(nbr_img,cv2.COLOR_BGR2GRAY)
                            if nbr_img.ndim==3 else nbr_img).astype(np.float32)
                X_nbr    = nbr_pose.transform_to_camera(X_world)
                in_front = X_nbr[:,2] > 0

                # FIX [1]: NaN for invalid pixels
                u_proj = np.full(len(pixels), np.nan, dtype=np.float32)
                v_proj = np.full(len(pixels), np.nan, dtype=np.float32)
                z_safe = np.where(in_front, X_nbr[:,2], 1.)
                u_proj[in_front] = (X_nbr[in_front,0]/z_safe[in_front]*K[0,0]+K[0,2]).astype(np.float32)
                v_proj[in_front] = (X_nbr[in_front,1]/z_safe[in_front]*K[1,1]+K[1,2]).astype(np.float32)

                valid_proj = (in_front & np.isfinite(u_proj) & np.isfinite(v_proj) &
                              (u_proj>=0)&(u_proj<w)&(v_proj>=0)&(v_proj<h))
                u_remap = np.where(valid_proj, u_proj, -1.).reshape(h,w).astype(np.float32)
                v_remap = np.where(valid_proj, v_proj, -1.).reshape(h,w).astype(np.float32)

                warped = cv2.remap(nbr_gray, u_remap, v_remap, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                cost   = 1. - self.compute_ncc(ref_gray, warped, self.window_size)
                cost[~valid_proj.reshape(h,w)] = np.nan
                costs.append(cost)

            if costs:
                cost_vol[:,:,d_idx] = np.nanmean(np.stack(costs,0), axis=0)

        inf_vol    = np.where(np.isfinite(cost_vol), cost_vol, np.inf)
        best_idx   = np.argmin(inf_vol, axis=2)
        best_depth = depths[best_idx]

        sorted_c   = np.sort(cost_vol, axis=2)
        margin     = sorted_c[:,:,1] - sorted_c[:,:,0]
        confidence = np.clip(np.where(np.isfinite(margin),margin,0.) /
                             (np.nanmax(np.where(np.isfinite(margin),margin,0.))+1e-8), 0,1)

        return DepthMap(depth=median_filter(best_depth.astype(np.float32),5),
                        confidence=confidence.astype(np.float32),
                        mask=confidence>0.3, camera_id=0)

    # ── DEM from point cloud ──────────────────────────────────

    def compute_dem(self, points_3d, resolution=0.1):
        x, y, z  = points_3d[:,0], points_3d[:,1], points_3d[:,2]
        x1d = np.arange(x.min(), x.max()+resolution, resolution)
        y1d = np.arange(y.min(), y.max()+resolution, resolution)
        xg, yg = np.meshgrid(x1d, y1d)
        dem  = griddata((x,y), z, (xg,yg), method='linear')
        nan  = np.isnan(dem)
        if nan.any():
            dem[nan] = griddata((x,y), z, (xg[nan],yg[nan]), method='nearest')
        return dem, xg, yg, (x.min(),x.max(),y.min(),y.max())

    def fuse_depth_maps(self, depth_maps, poses, K):
        all_pts = []
        for dm, pose in zip(depth_maps, poses):
            h, w = dm.depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            valid = dm.mask
            d_v   = dm.depth[valid]
            X_cam = np.column_stack([
                (u[valid]-K[0,2])*d_v/K[0,0],
                (v[valid]-K[1,2])*d_v/K[1,1], d_v])
            all_pts.append(pose.transform_to_world(X_cam))
        return np.vstack(all_pts) if all_pts else np.zeros((0,3))
