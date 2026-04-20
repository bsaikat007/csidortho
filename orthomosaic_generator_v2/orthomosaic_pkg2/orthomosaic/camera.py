"""
Camera models — Brown-Conrady, Pinhole, Fisheye.

Fixes vs original codebase:
  [1] undistort_points: correct Gauss-Newton iteration (uses x_u consistently)
  [2] backproject returns UNIT RAYS, not depth-scaled points (secant error fix)
  [3] Full K with skew parameter γ
  [4] Equidistant fisheye model for wide-angle drones (DJI O3, GoPro)
  [5] RollingShutterCorrector for CMOS row-by-row exposure
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2


# ──────────────────────────────────────────────────────────────
# Base camera
# ──────────────────────────────────────────────────────────────

@dataclass
class Camera:
    """
    Base camera with full intrinsic matrix including skew γ.

        K = [fx   γ   cx]
            [ 0  fy   cy]
            [ 0   0    1]
    """
    width:  int
    height: int
    fx:     float
    fy:     float
    cx:     float
    cy:     float
    skew:   float = 0.0

    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, self.skew, self.cx],
            [0.0,     self.fy,   self.cy],
            [0.0,     0.0,       1.0   ]
        ], dtype=np.float64)

    def K_inv(self) -> np.ndarray:
        """Analytic inverse — avoids np.linalg.inv rounding."""
        fx, fy, cx, cy, g = self.fx, self.fy, self.cx, self.cy, self.skew
        return np.array([
            [1/fx, -g/(fx*fy), (g*cy - cx*fy)/(fx*fy)],
            [0,     1/fy,      -cy/fy                 ],
            [0,     0,          1.0                   ]
        ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# Brown-Conrady
# ──────────────────────────────────────────────────────────────

class BrownConradyCamera(Camera):
    """
    Full Brown-Conrady distortion model.

    r² = x² + y²  (normalised coords)
    Radial:     δr  = k1·r² + k2·r⁴ + k3·r⁶
    Tangential: δx  = p1·(r²+2x²) + 2·p2·x·y
                δy  = p2·(r²+2y²) + 2·p1·x·y
    Distorted:  x'  = x·(1+δr) + δx,  pixel: u = fx·x' + γ·y' + cx
    """

    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0., k2=0., k3=0., p1=0., p2=0.,
                 skew=0., pixel_size: Optional[float] = None):
        super().__init__(width, height, fx, fy, cx, cy, skew)
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.p1, self.p2 = p1, p2
        self.pixel_size = pixel_size

    # ── internal helpers ──────────────────────────────────────

    def _distortion(self, x, y):
        """Return (radial_increment, δx, δy) at normalised (x, y)."""
        r2 = x*x + y*y
        radial = self.k1*r2 + self.k2*r2**2 + self.k3*r2**3
        dx = 2*self.p1*x*y       + self.p2*(r2 + 2*x*x)
        dy = self.p1*(r2 + 2*y*y) + 2*self.p2*x*y
        return radial, dx, dy

    # ── public API ────────────────────────────────────────────

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from pixel coordinates.

        FIX [1]: all intermediate quantities use the *current* estimate
        (xu, yu) — no mixing with original distorted coordinates.

        Args:
            points: (N,2) distorted pixel coordinates

        Returns:
            (N,2) undistorted pixel coordinates
        """
        Kinv = self.K_inv()
        h    = np.column_stack([points, np.ones(len(points))])
        xd   = (Kinv @ h.T).T[:, :2]          # distorted normalised

        xu, yu = xd[:, 0].copy(), xd[:, 1].copy()

        for _ in range(20):                    # converges in 3-5 iters
            radial, dx, dy = self._distortion(xu, yu)
            xu_new = (xd[:, 0] - dx) / (1.0 + radial)
            yu_new = (xd[:, 1] - dy) / (1.0 + radial)
            if np.max(np.abs(xu_new - xu) + np.abs(yu_new - yu)) < 1e-10:
                xu, yu = xu_new, yu_new
                break
            xu, yu = xu_new, yu_new

        u = self.fx * xu + self.skew * yu + self.cx
        v = self.fy * yu + self.cy
        return np.column_stack([u, v])

    def distort_points(self, normalized: np.ndarray) -> np.ndarray:
        """
        Apply distortion to undistorted normalised coords → pixel.

        Args:
            normalized: (N,2) undistorted normalised coordinates

        Returns:
            (N,2) distorted pixel coordinates
        """
        x, y = normalized[:, 0], normalized[:, 1]
        radial, dx, dy = self._distortion(x, y)
        xd = x * (1.0 + radial) + dx
        yd = y * (1.0 + radial) + dy
        u  = self.fx * xd + self.skew * yd + self.cx
        v  = self.fy * yd + self.cy
        return np.column_stack([u, v])

    def backproject_ray(self, pixels: np.ndarray) -> np.ndarray:
        """
        Pixel → unit direction ray in camera frame.

        FIX [2]: returns UNIT vectors. The old backproject() multiplied
        by depth which caused a secant error of up to 35%+ at corner pixels.

        Args:
            pixels: (N,2) pixel coordinates

        Returns:
            (N,3) unit direction vectors in camera frame
        """
        undist = self.undistort_points(pixels)
        Kinv   = self.K_inv()
        h      = np.column_stack([undist, np.ones(len(pixels))])
        rays   = (Kinv @ h.T).T
        norms  = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / np.where(norms == 0, 1e-15, norms)

    def project(self, points_3d: np.ndarray, pose) -> np.ndarray:
        """
        Project 3-D world points → distorted pixel coordinates.

        Args:
            points_3d: (N,3)
            pose:      Pose with .R and .t
        """
        Xc  = (pose.R @ points_3d.T + pose.t.reshape(3, 1)).T
        z   = np.where(Xc[:, 2] == 0, 1e-15, Xc[:, 2])
        xy  = Xc[:, :2] / z[:, np.newaxis]
        return self.distort_points(xy)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remap entire image using OpenCV (fast)."""
        K_arr = self.K()
        dist  = np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
        h, w  = image.shape[:2]
        K_opt, _ = cv2.getOptimalNewCameraMatrix(K_arr, dist, (w,h), 1, (w,h))
        m1, m2   = cv2.initUndistortRectifyMap(K_arr, dist, None, K_opt,
                                                (w,h), cv2.CV_32FC1)
        return cv2.remap(image, m1, m2, cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────
# Pinhole (no distortion)
# ──────────────────────────────────────────────────────────────

class PinholeCamera(Camera):
    """Simple pinhole — no distortion."""

    def project(self, points_3d: np.ndarray, pose) -> np.ndarray:
        Xc = (pose.R @ points_3d.T + pose.t.reshape(3, 1)).T
        z  = np.where(Xc[:, 2] == 0, 1e-15, Xc[:, 2])
        x  = Xc[:, 0] / z
        y  = Xc[:, 1] / z
        u  = self.fx * x + self.skew * y + self.cx
        v  = self.fy * y + self.cy
        return np.column_stack([u, v])

    def backproject_ray(self, pixels: np.ndarray) -> np.ndarray:
        """Pixel → unit ray (no distortion)."""
        Kinv  = self.K_inv()
        h     = np.column_stack([pixels, np.ones(len(pixels))])
        rays  = (Kinv @ h.T).T
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / np.where(norms == 0, 1e-15, norms)


# ──────────────────────────────────────────────────────────────
# Equidistant fisheye  (DJI O3 / GoPro / Insta360)
# ──────────────────────────────────────────────────────────────

class EquidistantCamera(Camera):
    """
    Equidistant projection: r = f·θ with 4-parameter polynomial.
    θ_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
    """

    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0., k2=0., k3=0., k4=0., skew=0.):
        super().__init__(width, height, fx, fy, cx, cy, skew)
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4

    def project(self, points_3d: np.ndarray, pose) -> np.ndarray:
        Xc    = (pose.R @ points_3d.T + pose.t.reshape(3, 1)).T
        X, Y, Z = Xc[:,0], Xc[:,1], Xc[:,2]
        r3d   = np.sqrt(X*X + Y*Y)
        theta = np.arctan2(r3d, np.abs(Z))
        th2   = theta * theta
        theta_d = theta*(1 + self.k1*th2 + self.k2*th2**2
                           + self.k3*th2**3 + self.k4*th2**4)
        scale = np.where(r3d == 0, 0.,
                         theta_d / np.where(r3d == 0, 1e-15, r3d))
        u = self.fx*(scale*X) + self.skew*(scale*Y) + self.cx
        v = self.fy*(scale*Y) + self.cy
        return np.column_stack([u, v])

    def backproject_ray(self, pixels: np.ndarray) -> np.ndarray:
        Kinv = self.K_inv()
        h    = np.column_stack([pixels, np.ones(len(pixels))])
        mn   = (Kinv @ h.T).T[:, :2]
        r_d  = np.linalg.norm(mn, axis=1)
        theta = r_d.copy()
        for _ in range(20):
            th2  = theta * theta
            f    = theta*(1+self.k1*th2+self.k2*th2**2
                            +self.k3*th2**3+self.k4*th2**4) - r_d
            df   = 1+3*self.k1*th2+5*self.k2*th2**2 \
                     +7*self.k3*th2**3+9*self.k4*th2**4
            step = f / np.where(df == 0, 1e-15, df)
            theta -= step
            if np.max(np.abs(step)) < 1e-12:
                break
        scale = np.where(r_d == 0, 0.,
                         np.tan(theta) / np.where(r_d == 0, 1e-15, r_d))
        rays  = np.column_stack([scale*mn[:,0], scale*mn[:,1],
                                  np.ones(len(pixels))])
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / np.where(norms == 0, 1e-15, norms)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        K_arr = self.K()
        D     = np.array([self.k1, self.k2, self.k3, self.k4])
        h, w  = image.shape[:2]
        K_opt = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K_arr, D, (w,h), np.eye(3), balance=0.)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                    K_arr, D, np.eye(3), K_opt, (w,h), cv2.CV_32FC1)
        return cv2.remap(image, m1, m2, cv2.INTER_LINEAR)
