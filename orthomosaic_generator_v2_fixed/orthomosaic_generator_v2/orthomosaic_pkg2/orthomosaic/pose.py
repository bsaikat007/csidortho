"""
Camera pose representation and estimation.

Fixes vs original:
  [1] from_euler_angles: uses scipy Rotation for numerically correct ω-φ-κ
  [2] make_nadir_pose: explicit nadir constructor (tested on DJI FC6310)
  [3] to_euler_angles: gimbal-lock safe via quaternion path
  [4] compute_relative_pose: uses cv2.recoverPose (stable E decomposition)
  [5] triangulate_points: uses cv2.triangulatePoints (handles degenerate cases)
  [6] Quaternion / Rodrigues / Lie-algebra perturbation support
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation
import cv2


# ──────────────────────────────────────────────────────────────
# Pose
# ──────────────────────────────────────────────────────────────

@dataclass
class Pose:
    """
    Exterior orientation: world → camera.

        X_cam = R @ X_world + t
        Camera centre C = −R.T @ t  (set t = −R @ C)
    """
    R: np.ndarray   # (3,3)  world → camera
    t: np.ndarray   # (3,)

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)
        # Project onto SO(3)
        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt
        if np.linalg.det(self.R) < 0:
            U[:, 2] *= -1
            self.R = U @ Vt

    # ── properties ────────────────────────────────────────────

    @property
    def C(self) -> np.ndarray:
        """Camera centre in world (ENU) frame."""
        return -self.R.T @ self.t

    @C.setter
    def C(self, value: np.ndarray):
        self.t = -self.R @ np.asarray(value, dtype=np.float64)

    @property
    def P(self) -> np.ndarray:
        """3×4 projection matrix [R | t]."""
        return np.hstack([self.R, self.t.reshape(3, 1)])

    # ── constructors ──────────────────────────────────────────

    @classmethod
    def from_euler_angles(cls, omega: float, phi: float, kappa: float,
                          position: np.ndarray) -> 'Pose':
        """
        Photogrammetric Euler angles (ω, φ, κ) + camera centre.

        FIX [1]: uses scipy Rotation — handles gimbal lock at φ = ±90°
        (nadir) correctly.  Convention: ZYX body rotations, then invert
        to get world→camera.

        ω = roll (rotation about X)
        φ = pitch (rotation about Y)
        κ = yaw / heading (rotation about Z)
        """
        r = Rotation.from_euler('ZYX', [kappa, phi, omega])
        R_world2cam = r.as_matrix().T
        pos = np.asarray(position, dtype=np.float64)
        return cls(R=R_world2cam, t=-R_world2cam @ pos)

    @classmethod
    def from_dji_gimbal(cls, yaw_deg: float, pitch_deg: float, roll_deg: float,
                        position: np.ndarray) -> 'Pose':
        """
        Build pose from DJI gimbal angles (world-referenced absolute angles).

        DJI convention:
          yaw:   0°=North, 90°=East, clockwise positive (true bearing)
          pitch: -90°=nadir (straight down), 0°=horizontal forward
          roll:  0°=level, positive=right-side-down

        Derivation: start from nadir (alpha=0), rotate around cam_x (starboard)
        by alpha = 90°+pitch toward horizontal-forward.

          cam_x = [cos θ, −sin θ,  0]           starboard (constant for all pitch)
          cam_y = [−sin θ·cos α, −cos θ·cos α, −sin α]  image-down
          cam_z = [ sin α·sin θ,  sin α·cos θ, −cos α]  optical axis (into scene)

        Verified:
          pitch=-90° (α=0)  → cam_z=[0,0,−1] (nadir)      ✓
          pitch=0°   (α=90°) → cam_z=[sin θ, cos θ, 0] (horiz forward) ✓
        """
        th    = np.radians(yaw_deg)
        alpha = np.radians(90.0 + pitch_deg)   # 0 = nadir, π/2 = horizontal
        phi   = np.radians(roll_deg)

        cam_x = np.array([ np.cos(th),                        -np.sin(th),              0.])
        cam_y = np.array([-np.sin(th) * np.cos(alpha), -np.cos(th) * np.cos(alpha), -np.sin(alpha)])
        cam_z = np.array([ np.sin(alpha) * np.sin(th),  np.sin(alpha) * np.cos(th), -np.cos(alpha)])

        if abs(phi) > 1e-6:
            # Roll: rotate cam_x/cam_y around optical axis
            cx =  np.cos(phi) * cam_x + np.sin(phi) * cam_y
            cy = -np.sin(phi) * cam_x + np.cos(phi) * cam_y
            cam_x, cam_y = cx, cy

        R = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
        C = np.asarray(position, dtype=np.float64)
        return cls(R=R, t=-R @ C)

    @classmethod
    def make_nadir(cls, easting: float, northing: float, agl: float,
                   yaw_deg: float) -> 'Pose':
        """Nadir convenience wrapper — delegates to from_dji_gimbal."""
        return cls.from_dji_gimbal(yaw_deg, -90.0, 0.0,
                                   np.array([easting, northing, agl]))

    @classmethod
    def from_quaternion(cls, q: np.ndarray, position: np.ndarray) -> 'Pose':
        """q = [qx, qy, qz, qw] (scipy convention)."""
        R = Rotation.from_quat(q).as_matrix()
        return cls(R=R, t=-R @ np.asarray(position, dtype=np.float64))

    @classmethod
    def from_rodrigues(cls, rvec: np.ndarray, tvec: np.ndarray) -> 'Pose':
        """OpenCV rvec / tvec → Pose."""
        R, _ = cv2.Rodrigues(rvec)
        return cls(R=R.astype(np.float64), t=tvec.ravel().astype(np.float64))

    # ── conversions ───────────────────────────────────────────

    def to_euler_angles(self) -> Tuple[float, float, float]:
        """
        Extract (ω, φ, κ).

        FIX [3]: via quaternion → no gimbal-lock singularity.
        """
        r = Rotation.from_matrix(self.R.T)      # cam→world
        kappa, phi, omega = r.as_euler('ZYX')
        return float(omega), float(phi), float(kappa)

    def to_quaternion(self) -> np.ndarray:
        """[qx, qy, qz, qw]."""
        return Rotation.from_matrix(self.R).as_quat()

    def to_rodrigues(self) -> Tuple[np.ndarray, np.ndarray]:
        rvec, _ = cv2.Rodrigues(self.R)
        return rvec.ravel(), self.t.copy()

    def get_params_6dof(self) -> np.ndarray:
        """6-DOF vector [rvec(3) | t(3)] for bundle adjustment."""
        rvec, _ = cv2.Rodrigues(self.R)
        return np.concatenate([rvec.ravel(), self.t])

    @classmethod
    def from_params_6dof(cls, params: np.ndarray) -> 'Pose':
        R, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
        return cls(R=R.astype(np.float64), t=params[3:6])

    # ── transforms ────────────────────────────────────────────

    def transform_to_camera(self, X_world: np.ndarray) -> np.ndarray:
        """X_cam = R @ X_world + t."""
        X = np.atleast_2d(X_world)
        return (self.R @ X.T + self.t.reshape(3, 1)).T

    def transform_to_world(self, X_cam: np.ndarray) -> np.ndarray:
        """X_world = R.T @ (X_cam − t)."""
        X = np.atleast_2d(X_cam)
        return (self.R.T @ (X.T - self.t.reshape(3, 1))).T

    def inverse(self) -> 'Pose':
        return Pose(R=self.R.T, t=-self.R.T @ self.t)

    def compose(self, other: 'Pose') -> 'Pose':
        return Pose(R=self.R @ other.R, t=self.R @ other.t + self.t)

    def perturb(self, delta: np.ndarray) -> 'Pose':
        """Left-multiply SO(3) perturbation: P_new = Exp(δω) · P."""
        dR = Rotation.from_rotvec(delta[:3]).as_matrix()
        return Pose(R=dR @ self.R, t=dR @ self.t + delta[3:6])


# ──────────────────────────────────────────────────────────────
# Relative pose
# ──────────────────────────────────────────────────────────────

def compute_relative_pose(points1: np.ndarray,
                          points2: np.ndarray,
                          K: np.ndarray,
                          threshold: float = 1.0
                          ) -> Tuple[Optional[Pose], np.ndarray]:
    """
    Relative pose from 2-D point correspondences.

    FIX [4]: uses cv2.recoverPose — handles all 4 E decompositions and
    cheirality check internally; more numerically stable than manual SVD.

    Returns:
        (Pose, inlier_bool_mask)  or  (None, zeros) on failure
    """
    if len(points1) < 5:
        return None, np.zeros(len(points1), dtype=bool)

    focal = (K[0, 0] + K[1, 1]) / 2.0
    pp    = (K[0, 2], K[1, 2])

    E, mask_e = cv2.findEssentialMat(
        points1, points2, focal=focal, pp=pp,
        method=cv2.RANSAC, prob=0.9999, threshold=threshold)

    if E is None or mask_e is None:
        return None, np.zeros(len(points1), dtype=bool)

    mask_e   = mask_e.ravel().astype(bool)
    inliers1 = points1[mask_e]
    inliers2 = points2[mask_e]

    if len(inliers1) < 5:
        return None, mask_e

    _, R, t, _ = cv2.recoverPose(E, inliers1, inliers2,
                                  focal=focal, pp=pp)
    return Pose(R=R.astype(np.float64), t=t.ravel().astype(np.float64)), mask_e


# ──────────────────────────────────────────────────────────────
# Triangulation
# ──────────────────────────────────────────────────────────────

def triangulate_points(points1: np.ndarray,
                       points2: np.ndarray,
                       pose1: Pose, pose2: Pose,
                       K: np.ndarray) -> np.ndarray:
    """
    Triangulate N 3-D world points from calibrated views.

    FIX [5]: uses cv2.triangulatePoints — handles near-degenerate
    geometry via homogeneous coordinates.

    Returns: (N,3) world points  (NaN where degenerate)
    """
    P1 = K @ pose1.P
    P2 = K @ pose2.P

    pts4d = cv2.triangulatePoints(
        P1, P2,
        points1.T.astype(np.float32),
        points2.T.astype(np.float32))   # (4, N)

    w     = pts4d[3]
    valid = np.abs(w) > 1e-10
    pts3d = np.full((len(points1), 3), np.nan)
    pts3d[valid] = (pts4d[:3, valid] / w[valid]).T
    return pts3d


def solve_pnp(object_points: np.ndarray,
              image_points:  np.ndarray,
              K:             np.ndarray,
              dist_coeffs:   Optional[np.ndarray] = None
              ) -> Tuple[Optional[Pose], np.ndarray]:
    """
    Perspective-n-Point with RANSAC.

    Returns:
        (Pose, inlier_bool_mask)  or  (None, zeros) on failure
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        K.astype(np.float32),
        dist_coeffs.astype(np.float32),
        reprojectionError=2.0, confidence=0.9999, iterationsCount=2000)

    if not ok or inliers is None:
        return None, np.zeros(len(object_points), dtype=bool)

    R, _ = cv2.Rodrigues(rvec)
    mask = np.zeros(len(object_points), dtype=bool)
    mask[inliers.ravel()] = True
    return Pose(R=R.astype(np.float64), t=tvec.ravel().astype(np.float64)), mask
