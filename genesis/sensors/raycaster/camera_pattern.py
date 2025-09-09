import math
from dataclasses import dataclass

import numpy as np
import torch

import genesis as gs

from .base_pattern import RaycastPattern, RaycastPatternGenerator, register_pattern


@dataclass
class DepthCameraPattern(RaycastPattern):
    """Configuration for pinhole depth camera ray casting.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    fx : float | None
        Focal length in x direction (pixels). Computed from FOV if None.
    fy : float | None
        Focal length in y direction (pixels). Computed from FOV if None.
    cx : float | None
        Principal point x coordinate (pixels). Defaults to image center if None.
    cy : float | None
        Principal point y coordinate (pixels). Defaults to image center if None.
    fov_horizontal : float
        Horizontal field of view in degrees. Used to compute fx if fx is None.
    fov_vertical : float | None
        Vertical field of view in degrees. Used to compute fy if fy is None.
    """

    width: int = 128
    height: int = 96
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    fov_horizontal: float = 90.0
    fov_vertical: float | None = None

    def get_return_shape(self) -> tuple[int, ...]:
        return (self.height, self.width, 3)


@register_pattern(DepthCameraPattern)
class DepthCameraPatternGenerator(RaycastPatternGenerator):
    """Generator for pinhole depth camera ray patterns."""

    def get_ray_directions(self) -> torch.Tensor:
        """Generate ray directions for pinhole camera model.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (height, width, 3) in robotics camera frame.
        """
        W, H = int(self.cfg.width), int(self.cfg.height)

        if W <= 0 or H <= 0:
            raise ValueError("Image dimensions must be positive")

        fx, fy, cx, cy = self._compute_intrinsics(W, H)

        u = np.arange(0, W, dtype=np.float32) + 0.5
        v = np.arange(0, H, dtype=np.float32) + 0.5
        uu, vv = np.meshgrid(u, v, indexing="xy")

        # standard camera frame coordinates
        x_c = (uu - cx) / fx
        y_c = (vv - cy) / fy
        z_c = np.ones_like(x_c, dtype=np.float32)

        # transform to robotics camera frame
        x_r, y_r, z_r = z_c, -x_c, -y_c
        dirs = np.stack([x_r, y_r, z_r], axis=-1).astype(np.float32)

        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-8)

        return torch.from_numpy(dirs).to(device=gs.device, dtype=gs.tc_float)

    def _compute_intrinsics(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Compute camera intrinsics from configuration.

        Parameters
        ----------
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.

        Returns
        -------
        tuple[float, float, float, float]
            Camera intrinsics (fx, fy, cx, cy) in pixels.
        """
        fx, fy = self.cfg.fx, self.cfg.fy
        cx, cy = self.cfg.cx, self.cfg.cy

        if fx is None or fy is None:
            fx, fy = self._compute_focal_lengths(width, height)

        if cx is None:
            cx = width * 0.5
        if cy is None:
            cy = height * 0.5

        return fx, fy, cx, cy

    def _compute_focal_lengths(self, width: int, height: int) -> tuple[float, float]:
        """Compute focal lengths from field of view specifications.

        Parameters
        ----------
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.

        Returns
        -------
        tuple[float, float]
            Focal lengths (fx, fy) in pixels.
        """
        fh = self.cfg.fov_horizontal
        fv = self.cfg.fov_vertical

        if fh is not None and fv is None:
            fh_rad = math.radians(fh)
            fv_rad = 2.0 * math.atan((height / width) * math.tan(fh_rad / 2.0))
        elif fv is not None and fh is None:
            fv_rad = math.radians(fv)
            fh_rad = 2.0 * math.atan((width / height) * math.tan(fv_rad / 2.0))
        else:
            fh_rad = math.radians(fh)
            fv_rad = math.radians(fv)

        fx = width / (2.0 * math.tan(fh_rad / 2.0))
        fy = height / (2.0 * math.tan(fv_rad / 2.0))

        return fx, fy
