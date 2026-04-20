"""
Orthorectification: perspective → orthographic.

Fixes vs original:
  [1] float64 accumulator (was float32 → precision loss in blending)
  [2] compute_footprint uses backproject_ray (undistorts first)
  [3] Per-image radiometric gain normalisation before blending
  [4] Laplacian pyramid multi-band blending mode
  [5] Voronoi nearest-camera label map for 'nearest' blend
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import RectBivariateSpline


@dataclass
class OrthoParams:
    gsd:             float
    output_bounds:   Tuple[float, float, float, float]  # xmin ymin xmax ymax
    target_srs:      str   = "EPSG:32633"
    elevation_model: Optional[np.ndarray] = None
    dem_resolution:  float = 1.0


class Orthorectifier:

    def __init__(self, params: OrthoParams, use_gpu: bool = False):
        self.params = params
        self.x_min, self.y_min, self.x_max, self.y_max = params.output_bounds
        self.gsd = params.gsd

        self.width  = max(1, int(round((self.x_max - self.x_min) / self.gsd)))
        self.height = max(1, int(round((self.y_max - self.y_min) / self.gsd)))

        print(f"Orthorectifier: {self.width}×{self.height}px @ {self.gsd:.4f}m/px")

    # ── ground elevation ──────────────────────────────────────

    def _elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.params.elevation_model is None:
            return np.zeros(len(x))
        dem = self.params.elevation_model
        res = self.params.dem_resolution
        x_d = np.arange(dem.shape[1]) * res + self.x_min
        y_d = np.arange(dem.shape[0]) * res + self.y_min
        interp = RectBivariateSpline(y_d, x_d, dem, kx=1, ky=1)
        return interp.ev(np.clip(y, y_d[0], y_d[-1]),
                         np.clip(x, x_d[0], x_d[-1]))

    # ── project ground → image ────────────────────────────────

    def _ground_to_image(self, xg, yg, zg, pose, camera):
        X_world  = np.column_stack([xg, yg, zg])
        X_cam    = pose.transform_to_camera(X_world)
        in_front = X_cam[:, 2] > 0
        z_safe   = np.where(in_front, X_cam[:, 2], 1e-10)
        x_n = X_cam[:, 0] / z_safe
        y_n = X_cam[:, 1] / z_safe

        if hasattr(camera, 'distort_points'):
            uv  = camera.distort_points(np.column_stack([x_n, y_n]))
            u, v = uv[:, 0], uv[:, 1]
        else:
            K = camera.K()
            u = K[0,0]*x_n + K[0,1]*y_n + K[0,2]
            v =              K[1,1]*y_n + K[1,2]
        return u, v, in_front

    # ── single image ──────────────────────────────────────────

    def orthorectify_image(self, image: np.ndarray, pose, camera,
                           interp: str = 'linear'
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            ortho:  (H_out, W_out, C) uint8
            weight: (H_out, W_out)   float64 feathered mask
        """
        img_h, img_w = image.shape[:2]

        x_out = np.arange(self.width)  * self.gsd + self.x_min
        y_out = np.arange(self.height) * self.gsd + self.y_min
        xx, yy = np.meshgrid(x_out, y_out)
        xf, yf = xx.ravel(), yy.ravel()
        zf = self._elevation(xf, yf)

        u, v, in_front = self._ground_to_image(xf, yf, zf, pose, camera)
        u = u.reshape(self.height, self.width)
        v = v.reshape(self.height, self.width)
        in_front = in_front.reshape(self.height, self.width)

        valid = (in_front &
                 (u >= 0) & (u <= img_w - 1) &
                 (v >= 0) & (v <= img_h - 1))

        flag = {'nearest': cv2.INTER_NEAREST,
                'linear':  cv2.INTER_LINEAR,
                'cubic':   cv2.INTER_CUBIC}.get(interp, cv2.INTER_LINEAR)

        ortho = cv2.remap(image,
                          u.astype(np.float32), v.astype(np.float32),
                          flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Feathered weight
        dist   = cv2.distanceTransform((valid*255).astype(np.uint8),
                                        cv2.DIST_L2, 5)
        feather = max(1, self.width // 50)
        weight  = np.clip(dist / feather, 0., 1.).astype(np.float64)
        return ortho, weight

    # ── multi-image mosaic ────────────────────────────────────

    def create_orthomosaic(self, images: List[np.ndarray],
                           poses: List, cameras: List,
                           blending: str = 'weighted') -> np.ndarray:
        """
        Blend N orthorectified images into a mosaic.

        FIX [1]: float64 accumulator throughout.
        FIX [3]: per-image radiometric gain normalisation.

        blending: 'weighted' | 'nearest' | 'multiband'
        """
        if not images:
            raise ValueError("No images provided")

        nc = images[0].shape[2] if images[0].ndim == 3 else 1

        # Radiometric normalisation: match each image to global mean
        means = []
        for img in images:
            px = img[img > 0] if img.ndim == 2 else img.reshape(-1, nc)[img.reshape(-1, nc).any(axis=1)]
            means.append(float(px.mean()) if len(px) else 128.)
        global_mean = float(np.mean(means)) if means else 128.

        orthos, weights = [], []
        for i, (img, pose, cam) in enumerate(zip(images, poses, cameras)):
            gain = global_mean / max(means[i], 1e-6)
            img_c = np.clip(img.astype(np.float32)*gain, 0, 255).astype(np.uint8)
            o, w  = self.orthorectify_image(img_c, pose, cam)
            orthos.append(o); weights.append(w)
            print(f"  IMG {i+1}/{len(images)} orthorectified")

        # FIX [1]: float64 accumulators
        accum = np.zeros((self.height, self.width, nc), dtype=np.float64)
        wsum  = np.zeros((self.height, self.width),     dtype=np.float64)

        if blending == 'nearest':
            lmap = self._voronoi_labels(poses)
            for i, (o, w) in enumerate(zip(orthos, weights)):
                sel = (lmap == i) & (w > 0)
                accum[sel] = o[sel].astype(np.float64)
                wsum[sel]  = 1.0

        elif blending == 'multiband':
            for o, w in zip(orthos, weights):
                blended = self._laplacian_blend(
                    accum / np.maximum(wsum[:,:,np.newaxis], 1e-10),
                    o.astype(np.float64), w)
                wsum += w
                accum = blended * wsum[:,:,np.newaxis]
            result = np.clip(accum / np.maximum(wsum[:,:,np.newaxis], 1e-10),
                             0, 255).astype(np.uint8)
            return result

        else:  # 'weighted'
            for o, w in zip(orthos, weights):
                accum += o.astype(np.float64) * w[:,:,np.newaxis]
                wsum  += w

        result = np.clip(accum / np.maximum(wsum[:,:,np.newaxis], 1e-10),
                         0, 255).astype(np.uint8)
        return result

    # ── helpers ───────────────────────────────────────────────

    def _voronoi_labels(self, poses) -> np.ndarray:
        """FIX [5]: nearest-camera pixel assignment."""
        lmap = np.zeros((self.height, self.width), dtype=np.int32)
        x_idx = np.arange(self.width,  dtype=np.float64)
        y_idx = np.arange(self.height, dtype=np.float64)
        xx, yy = np.meshgrid(x_idx, y_idx)
        min_d   = np.full((self.height, self.width), np.inf)
        for i, pose in enumerate(poses):
            cx = (pose.C[0] - self.x_min) / self.gsd
            cy = (pose.C[1] - self.y_min) / self.gsd
            d  = (xx - cx)**2 + (yy - cy)**2
            upd = d < min_d
            lmap[upd] = i; min_d[upd] = d[upd]
        return lmap

    def _laplacian_blend(self, base, new_img, mask, levels=3):
        """FIX [4]: Laplacian pyramid blending for smooth seams."""
        nc = base.shape[2] if base.ndim == 3 else 1
        if base.ndim == 2:
            base    = base[:,:,np.newaxis]
            new_img = new_img[:,:,np.newaxis]

        def gauss_pyr(img, n):
            p = [img.astype(np.float32)]
            for _ in range(n): p.append(cv2.pyrDown(p[-1]))
            return p

        def lap_pyr(img, n):
            g = gauss_pyr(img, n)
            lp = []
            for i in range(n):
                up = cv2.pyrUp(g[i+1], dstsize=(g[i].shape[1], g[i].shape[0]))
                lp.append(g[i].astype(np.float64) - up.astype(np.float64))
            lp.append(g[-1].astype(np.float64))
            return lp

        gm  = gauss_pyr(mask.astype(np.float32), levels)
        lb  = lap_pyr(base.astype(np.float32),    levels)
        ln  = lap_pyr(new_img.astype(np.float32), levels)

        blended_pyr = []
        for l_b, l_n, gm_l in zip(lb, ln, gm):
            alpha = gm_l[:,:,np.newaxis] if l_b.ndim == 3 else gm_l
            blended_pyr.append(l_n * alpha + l_b * (1 - alpha))

        result = blended_pyr[-1]
        for lvl in reversed(blended_pyr[:-1]):
            result = cv2.pyrUp(result, dstsize=(lvl.shape[1], lvl.shape[0]))
            result = result.astype(np.float64) + lvl.astype(np.float64)

        if nc == 1: result = result[:,:,0]
        return np.clip(result, 0, 255)

    def compute_footprint(self, pose, camera,
                          ground_elevation: float = 0.) -> np.ndarray:
        """
        FIX [2]: uses camera.backproject_ray (undistorts corners first).
        """
        h, w = camera.height, camera.width
        corners = np.array([[0.,0.],[w-1.,0.],[w-1.,h-1.],[0.,h-1.]])
        rays_c = camera.backproject_ray(corners)
        rays_w = (pose.R.T @ rays_c.T).T
        C      = pose.C
        fp     = []
        for ray in rays_w:
            if abs(ray[2]) > 1e-6:
                t = (ground_elevation - C[2]) / ray[2]
                if t > 0: fp.append(C + t * ray)
        return np.array(fp) if fp else np.zeros((0, 3))
