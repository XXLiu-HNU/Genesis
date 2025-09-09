import hashlib
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import spherical_to_cartesian

from .base_pattern import RaycastPattern, RaycastPatternGenerator, register_pattern

VLP32_VERTICAL_ANGLES_DEG = np.array(
    [
        -25.0,
        -22.5,
        -20.0,
        -15.0,
        -13.0,
        -10.0,
        -5.0,
        -3.0,
        -2.333,
        -1.0,
        -0.667,
        -0.333,
        0.0,
        0.0,
        0.333,
        0.667,
        1.0,
        1.333,
        1.667,
        2.0,
        2.333,
        2.667,
        3.0,
        3.333,
        3.667,
        4.0,
        5.0,
        7.0,
        10.0,
        15.0,
        17.0,
        20.0,
    ],
    dtype=np.float32,
)

BPEARL_VERTICAL_ANGLES_DEG = [
    89.5,
    86.6875,
    83.875,
    81.0625,
    78.25,
    75.4375,
    72.625,
    69.8125,
    67.0,
    64.1875,
    61.375,
    58.5625,
    55.75,
    52.9375,
    50.125,
    47.3125,
    44.5,
    41.6875,
    38.875,
    36.0625,
    33.25,
    30.4375,
    27.625,
    24.8125,
    22,
    19.1875,
    16.375,
    13.5625,
    10.75,
    7.9375,
    5.125,
    2.3125,
]

LIVOX_SENSOR_PARAMS = {
    "avia": {
        "laser_min_range": 0.1,
        "laser_max_range": 200.0,
        "horizontal_fov": 70.4,
        "vertical_fov": 77.2,
        "samples": 24000,
    },
    "HAP": {
        "laser_min_range": 0.1,
        "laser_max_range": 200.0,
        "samples": 45300,
        "horizontal_fov": 81.7,
        "vertical_fov": 25.1,
    },
    "horizon": {
        "laser_min_range": 0.1,
        "laser_max_range": 200.0,
        "horizontal_fov": 81.7,
        "vertical_fov": 25.1,
        "samples": 24000,
    },
    "mid40": {
        "laser_min_range": 0.1,
        "laser_max_range": 200.0,
        "horizontal_fov": 81.7,
        "vertical_fov": 25.1,
        "samples": 24000,
    },
    "mid70": {
        "laser_min_range": 0.1,
        "laser_max_range": 200.0,
        "horizontal_fov": 70.4,
        "vertical_fov": 70.4,
        "samples": 10000,
    },
    "mid360": {
        "laser_min_range": 0.1,
        "laser_max_range": 200.0,
        "horizontal_fov": 360.0,
        "vertical_fov": 59.0,
        "samples": 20000,
    },
    "tele": {
        "laser_min_range": 0.1,
        "laser_max_range": 200.0,
        "horizontal_fov": 14.5,
        "vertical_fov": 16.1,
        "samples": 24000,
    },
}

SPINNING_LIDAR_DEFAULTS = {
    "hdl64": {"n_channels": 64, "f_rot": 10.0, "sample_rate": 2.2e6, "phi_range": (-24.9, 2.0)},
    "vlp32": {"n_channels": 32, "f_rot": 10.0, "sample_rate": 1.2e6, "angles": VLP32_VERTICAL_ANGLES_DEG},
    "os128": {"n_channels": 128, "f_rot": 20.0, "sample_rate": 5.2e6, "phi_range": (-22.5, 22.5)},
}


@dataclass
class GridPattern(RaycastPattern):
    """
    Configuration for grid-based ray casting.

    Defines a 2D grid of rays in the sensor coordinate system.

    Parameters
    ----------
    resolution : float
        Grid spacing in meters.
    size : tuple[float, float]
        Grid dimensions (length, width) in meters.
    direction : tuple[float, float, float]
        Ray direction vector.
    ordering : str
        Point ordering, either "xy" or "yx".
    """

    resolution: float = 0.1
    size: tuple[float, float] = (2.0, 2.0)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    ordering: str = "xy"

    def get_return_shape(self) -> tuple[int, ...]:
        x_coords = np.arange(-self.size[0] / 2, self.size[0] / 2 + 1e-9, self.resolution)
        y_coords = np.arange(-self.size[1] / 2, self.size[1] / 2 + 1e-9, self.resolution)
        n_rays = len(x_coords) * len(y_coords)
        return (1, n_rays, 3)


@register_pattern(GridPattern)
class GridPatternGenerator(RaycastPatternGenerator):
    """Generator for 2D grid ray patterns."""

    def __init__(self, cfg: GridPattern):
        super().__init__(cfg)
        self.x_coords = np.arange(-cfg.size[0] / 2, cfg.size[0] / 2 + 1e-9, cfg.resolution)
        self.y_coords = np.arange(-cfg.size[1] / 2, cfg.size[1] / 2 + 1e-9, cfg.resolution)

    def get_ray_directions(self) -> torch.Tensor:
        dirs = torch.zeros(self.cfg.get_return_shape(), dtype=gs.tc_float, device=gs.device)
        dirs[0, :, :] = torch.tensor(self.cfg.direction, dtype=gs.tc_float, device=gs.device)
        return dirs

    def get_ray_starts(self) -> torch.Tensor:
        if self.cfg.ordering not in ["xy", "yx"]:
            raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{self.cfg.ordering}'.")
        if self.cfg.resolution <= 0:
            raise ValueError(f"Resolution must be greater than 0. Received: '{self.cfg.resolution}'.")
        if self.cfg.ordering == "xy":
            grid_x, grid_y = np.meshgrid(self.x_coords, self.y_coords, indexing="xy")
        else:
            grid_x, grid_y = np.meshgrid(self.x_coords, self.y_coords, indexing="ij")
        starts = torch.zeros(self.cfg.get_return_shape(), dtype=gs.tc_float, device=gs.device)
        starts[0, :, 0] = grid_x.flatten()
        starts[0, :, 1] = grid_y.flatten()

        return starts


@dataclass
class LidarPattern(RaycastPattern):
    """
    Configuration for multi-channel LiDAR ray casting.

    Parameters
    ----------
    channels : int
        Number of vertical scanning channels.
    vertical_fov_range : tuple[float, float]
        Vertical field of view limits in degrees (min, max).
    horizontal_fov_range : tuple[float, float]
        Horizontal field of view limits in degrees (min, max).
    horizontal_res : float
        Horizontal angular resolution in degrees.
    """

    channels: int = 32
    vertical_fov_range: tuple[float, float] = (-15.0, 15.0)
    horizontal_fov_range: tuple[float, float] = (-180.0, 180.0)
    horizontal_res: float = 1.0

    def get_return_shape(self) -> tuple[int, ...]:
        h_range = self.horizontal_fov_range[1] - self.horizontal_fov_range[0]
        num_horizontal_angles = math.ceil(h_range / self.horizontal_res)
        if abs(abs(h_range) - 360.0) < 1e-6:
            num_horizontal_angles -= 1
        return (self.channels, num_horizontal_angles, 3)


@register_pattern(LidarPattern)
class LidarPatternGenerator(RaycastPatternGenerator):
    """Generator for multi-channel LiDAR ray patterns."""

    def __init__(self, cfg: LidarPattern):
        super().__init__(cfg)
        self.h_range = cfg.horizontal_fov_range[1] - cfg.horizontal_fov_range[0]
        self.num_horizontal_angles = math.ceil(self.h_range / cfg.horizontal_res)
        if abs(abs(self.h_range) - 360.0) < 1e-6:
            self.num_horizontal_angles -= 1

    def get_ray_directions(self) -> torch.Tensor:
        """Generate spherical ray pattern for multi-channel LiDAR.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (channels, angles_per_channel, 3).
        """
        vertical_angles = np.linspace(self.cfg.vertical_fov_range[0], self.cfg.vertical_fov_range[1], self.cfg.channels)
        horizontal_angles = np.linspace(
            self.cfg.horizontal_fov_range[0], self.cfg.horizontal_fov_range[1], self.num_horizontal_angles
        )

        v_rad = np.deg2rad(vertical_angles)
        h_rad = np.deg2rad(horizontal_angles)
        v_angles, h_angles = np.meshgrid(v_rad, h_rad, indexing="ij")

        x, y, z = spherical_to_cartesian(h_angles, v_angles)
        ray_directions = np.stack([x, y, z], axis=-1).astype(np.float32)

        return torch.from_numpy(ray_directions).to(device=gs.device, dtype=gs.tc_float)


@dataclass
class BpearlPattern(RaycastPattern):
    """
    Configuration for Bpearl LiDAR pattern.

    Parameters
    ----------
    horizontal_fov : float
        Horizontal field of view in degrees.
    horizontal_res : float
        Horizontal angular resolution in degrees.
    vertical_ray_angles : list[float]
        Predefined vertical ray angles in degrees.
    """

    horizontal_fov: float = 360.0
    horizontal_res: float = 10.0
    vertical_ray_angles: list[float] = field(default_factory=lambda: BPEARL_VERTICAL_ANGLES_DEG.copy())

    def get_return_shape(self) -> tuple[int, ...]:
        h_angles = np.arange(-self.horizontal_fov / 2, self.horizontal_fov / 2, self.horizontal_res)
        return (len(self.vertical_ray_angles), len(h_angles), 3)


@register_pattern(BpearlPattern)
class BpearlPatternGenerator(RaycastPatternGenerator):
    """Generator for Bpearl LiDAR ray patterns."""

    def __init__(self, cfg: BpearlPattern):
        super().__init__(cfg)
        self.h_angles = np.arange(-cfg.horizontal_fov / 2, cfg.horizontal_fov / 2, cfg.horizontal_res)

    def get_ray_directions(self) -> torch.Tensor:
        """Generate Bpearl-specific ray pattern.

        Returns
        -------
        torch.Tensor
            Ray directions with Bpearl-specific coordinate convention.
        """
        v_angles = np.array(self.cfg.vertical_ray_angles, dtype=np.float32)
        pitch, yaw = np.meshgrid(v_angles, self.h_angles, indexing="xy")

        pitch_rad = np.deg2rad(pitch.flatten()) + np.pi / 2
        yaw_rad = np.deg2rad(yaw.flatten())

        x = np.sin(pitch_rad) * np.cos(yaw_rad)
        y = np.sin(pitch_rad) * np.sin(yaw_rad)
        z = np.cos(pitch_rad)

        ray_directions = -np.stack([x, y, z], axis=1).astype(np.float32)
        ray_directions = ray_directions.reshape(*self.cfg.get_return_shape())

        return torch.from_numpy(ray_directions).to(device=gs.device, dtype=gs.tc_float)


@dataclass
class SphericalPattern(RaycastPattern):
    """
    Configuration for spherical uniform ray pattern.

    Parameters
    ----------
    n_scan_lines : int
        Number of vertical scan lines.
    n_points_per_line : int
        Number of horizontal points per scan line.
    fov_vertical : float
        Vertical field of view in degrees.
    fov_horizontal : float
        Horizontal field of view in degrees.
    """

    n_scan_lines: int = 32
    n_points_per_line: int = 64
    fov_vertical: float = 30.0
    fov_horizontal: float = 360.0

    def get_return_shape(self) -> tuple[int, ...]:
        return (self.n_scan_lines, self.n_points_per_line, 3)


@register_pattern(SphericalPattern)
class SphericalPatternGenerator(RaycastPatternGenerator):
    """Generator for uniform spherical ray patterns."""

    def get_ray_directions(self) -> torch.Tensor:
        """Generate uniform spherical ray pattern.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (n_scan_lines, n_points_per_line, 3).
        """
        vertical_angles = np.linspace(-self.cfg.fov_vertical / 2, self.cfg.fov_vertical / 2, self.cfg.n_scan_lines)
        horizontal_angles = np.linspace(
            -self.cfg.fov_horizontal / 2, self.cfg.fov_horizontal / 2, self.cfg.n_points_per_line
        )

        v_rad = np.deg2rad(vertical_angles)
        h_rad = np.deg2rad(horizontal_angles)
        h_angles, v_angles = np.meshgrid(h_rad, v_rad)

        x, y, z = spherical_to_cartesian(h_angles, v_angles)
        ray_vectors = np.stack([x, y, z], axis=-1).astype(np.float32)

        return torch.from_numpy(ray_vectors).to(device=gs.device, dtype=gs.tc_float)


@dataclass
class SpinningLidarPattern(RaycastPattern):
    """
    Configuration for traditional spinning LiDAR sensors.

    Parameters
    ----------
    sensor_type : str
        Spinning LiDAR model (hdl64, vlp32, os128).
    f_rot : float
        Rotation frequency in Hz.
    sample_rate : float
        Sample rate in samples per second.
    n_channels : int
        Number of vertical channels.
    phi_fov : tuple[float, float]
        Vertical field of view limits in degrees (min, max).
    """

    sensor_type: str = "hdl64"
    f_rot: float = 10.0
    sample_rate: float = 2.2e6
    n_channels: int = 64
    phi_fov: tuple[float, float] = (-24.9, 2.0)

    def get_return_shape(self) -> tuple[int, ...]:
        sensor = self.sensor_type.lower()
        defaults = SPINNING_LIDAR_DEFAULTS.get(sensor, SPINNING_LIDAR_DEFAULTS["hdl64"])

        if sensor == "vlp32":
            n_channels = len(VLP32_VERTICAL_ANGLES_DEG)
        else:
            n_channels = self.n_channels or defaults["n_channels"]

        f_rot = self.f_rot or defaults["f_rot"]
        sample_rate = self.sample_rate or defaults["sample_rate"]

        n_time_steps = int(sample_rate / (f_rot * n_channels))
        n_rays = n_time_steps * n_channels
        return (1, n_rays, 3)


@register_pattern(SpinningLidarPattern)
class SpinningLidarPatternGenerator(RaycastPatternGenerator):
    """Generator for traditional spinning LiDAR patterns."""

    def __init__(self, cfg: SpinningLidarPattern):
        super().__init__(cfg)

    def get_ray_directions(self) -> torch.Tensor:
        """Generate spinning LiDAR ray pattern.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (1, n_rays, 3).
        """
        sensor = self.cfg.sensor_type.lower()
        if sensor not in SPINNING_LIDAR_DEFAULTS:
            raise ValueError(f"Unsupported spinning lidar type: {self.cfg.sensor_type}")

        defaults = SPINNING_LIDAR_DEFAULTS[sensor]

        if sensor == "vlp32":
            phi = np.deg2rad(VLP32_VERTICAL_ANGLES_DEG)
            n_channels = len(VLP32_VERTICAL_ANGLES_DEG)
        else:
            n_channels = self.cfg.n_channels or defaults["n_channels"]
            if "angles" in defaults:
                phi = np.deg2rad(defaults["angles"])
            else:
                phi_min, phi_max = np.deg2rad(defaults["phi_range"])
                phi = np.linspace(phi_min, phi_max, n_channels, dtype=np.float32)

        f_rot = self.cfg.f_rot or defaults["f_rot"]
        sample_rate = self.cfg.sample_rate or defaults["sample_rate"]

        t = np.arange(0.0, 1.0 / f_rot, n_channels / sample_rate, dtype=np.float32)[:, None]
        theta = (2.0 * np.pi * f_rot * t) % (2.0 * np.pi)

        theta_grid = theta + np.zeros((1, n_channels), dtype=np.float32)
        phi_grid = np.zeros_like(theta, dtype=np.float32) + phi

        theta_flat = theta_grid.reshape(-1)
        phi_flat = phi_grid.reshape(-1)

        x, y, z = spherical_to_cartesian(theta_flat, phi_flat)
        dirs = np.stack([x, y, z], axis=1).astype(np.float32)

        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-8)

        return torch.from_numpy(dirs.reshape(1, -1, 3)).to(device=gs.device, dtype=gs.tc_float)


@dataclass
class LivoxPattern(RaycastPattern):
    """
    Configuration for Livox solid-state LiDAR patterns.

    Parameters
    ----------
    sensor_type : str
        Livox sensor model (avia, HAP, horizon, mid40, mid70, mid360, tele).
    samples : int
        Number of ray samples per scan frame.
    downsample : int
        Downsampling factor for ray patterns.
    use_simple_grid : bool
        Whether to use simple grid pattern instead of realistic scan patterns.
    rolling_window_start : int
        Starting index for rolling window sampling.
    horizontal_line_num : int
        Number of horizontal lines for simple grid pattern.
    vertical_line_num : int
        Number of vertical lines for simple grid pattern.
    horizontal_fov_deg_min : float
        Minimum horizontal FOV for simple grid pattern in degrees.
    horizontal_fov_deg_max : float
        Maximum horizontal FOV for simple grid pattern in degrees.
    vertical_fov_deg_min : float
        Minimum vertical FOV for simple grid pattern in degrees.
    vertical_fov_deg_max : float
        Maximum vertical FOV for simple grid pattern in degrees.
    enable_dynamic_pattern : bool
        Enable dynamic ray pattern updates over time.
    pattern_rotation_speed : float
        Rotation speed for dynamic patterns.
    """

    sensor_type: str = "avia"
    samples: int = 24000
    downsample: int = 1
    use_simple_grid: bool = False
    rolling_window_start: int = 0

    # Simple grid parameters
    horizontal_line_num: int = 80
    vertical_line_num: int = 50
    horizontal_fov_deg_min: float = -180
    horizontal_fov_deg_max: float = 180
    vertical_fov_deg_min: float = -2
    vertical_fov_deg_max: float = 57

    # Dynamic pattern parameters
    enable_dynamic_pattern: bool = True
    pattern_rotation_speed: float = 0.1

    _is_dynamic: bool = True

    def get_return_shape(self) -> tuple[int, ...]:
        if self.use_simple_grid:
            n_rays = self.vertical_line_num * self.horizontal_line_num
        else:
            params = LIVOX_SENSOR_PARAMS.get(self.sensor_type, {})
            n_rays = min(self.samples, params.get("samples", self.samples))
            if self.downsample > 1:
                n_rays = n_rays // self.downsample
        return (1, n_rays, 3)


@register_pattern(LivoxPattern)
class LivoxPatternGenerator(RaycastPatternGenerator):
    """Generator for Livox solid-state LiDAR patterns.

    Supports both simple grid patterns and realistic scan patterns loaded from
    precomputed files or generated via random sampling.
    """

    _pattern_cache: dict[str, np.ndarray] = {}

    def __init__(self, cfg: LivoxPattern):
        super().__init__(cfg)
        self.current_start_index = 0
        self.generated_patterns = {}
        self._last_update_tick = None

    def get_actual_ray_count(self) -> int:
        """Get the actual number of rays after downsampling.

        Returns
        -------
        int
            Number of rays in the final pattern.
        """
        return self.cfg.get_return_shape()[1]

    def get_ray_directions(self) -> torch.Tensor:
        """Generate Livox ray pattern.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (1, n_rays, 3).
        """
        if self.cfg.use_simple_grid:
            ray_directions = self._generate_simple_grid_pattern()
        else:
            ray_directions = self._generate_livox_scan_pattern()

        return torch.from_numpy(ray_directions).to(device=gs.device, dtype=gs.tc_float)

    def _generate_simple_grid_pattern(self) -> np.ndarray:
        """Generate simple uniform grid pattern.

        Returns
        -------
        np.ndarray
            Ray directions array with shape (vertical_lines, horizontal_lines, 3).
        """
        cfg = self.cfg
        h_fov_min = math.radians(cfg.horizontal_fov_deg_min)
        h_fov_max = math.radians(cfg.horizontal_fov_deg_max)
        v_fov_min = math.radians(cfg.vertical_fov_deg_min)
        v_fov_max = math.radians(cfg.vertical_fov_deg_max)

        if cfg.vertical_line_num > 1:
            v_angles = np.linspace(v_fov_min, v_fov_max, cfg.vertical_line_num)
        else:
            v_angles = np.array([(v_fov_min + v_fov_max) / 2])

        if cfg.horizontal_line_num > 1:
            h_angles = np.linspace(h_fov_min, h_fov_max, cfg.horizontal_line_num)
        else:
            h_angles = np.array([(h_fov_min + h_fov_max) / 2])

        h_grid, v_grid = np.meshgrid(h_angles, v_angles)
        x, y, z = spherical_to_cartesian(h_grid, v_grid)

        return np.stack([x, y, z], axis=-1).astype(np.float32)

    def _generate_livox_scan_pattern(self) -> np.ndarray:
        """Generate realistic Livox scan pattern.

        Returns
        -------
        np.ndarray
            Sampled ray directions array with shape (1, n_rays, 3).
        """
        cfg = self.cfg
        if cfg.sensor_type not in LIVOX_SENSOR_PARAMS:
            raise ValueError(f"Unsupported Livox sensor type: {cfg.sensor_type}")

        params = LIVOX_SENSOR_PARAMS[cfg.sensor_type]
        cache_key = self._create_cache_key(params)

        if cache_key in self._pattern_cache:
            full_pattern = self._pattern_cache[cache_key]
        else:
            full_pattern = self._generate_pattern_angles(params)
            self._pattern_cache[cache_key] = full_pattern

        self.generated_patterns[cfg.sensor_type] = full_pattern
        return self._sample_pattern(full_pattern)

    def _create_cache_key(self, params: dict) -> str:
        """
        Create unique cache key for pattern configuration.

                Parameters
                ----------
                params : dict
                    Sensor parameters dictionary.

                Returns
                -------
                str
                    MD5 hash of the configuration parameters.
        """
        key_data = {
            "sensor_type": self.cfg.sensor_type,
            "horizontal_fov": params.get("horizontal_fov", 360.0),
            "vertical_fov": params.get("vertical_fov", 90.0),
            "total_samples": params["samples"] * 10,
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _generate_pattern_angles(self, params: dict) -> np.ndarray:
        """Load or generate Livox scan pattern angles.

        First attempts to load from precomputed .npy files, falls back to
        random generation if files are unavailable.

        Parameters
        ----------
        params : dict
            Sensor parameters dictionary.

        Returns
        -------
        np.ndarray
            Pattern angles array with shape (N, 2) containing [theta, phi] in radians.
        """
        pattern_angles = self._load_precomputed_pattern()

        if pattern_angles is None:
            pattern_angles = self._generate_random_pattern(params)

        return pattern_angles

    def _load_precomputed_pattern(self) -> np.ndarray | None:
        """Load precomputed pattern from .npy file."""
        pattern_files = {
            "avia": "avia.npy",
            "horizon": "horizon.npy",
            "HAP": "HAP.npy",
            "mid360": "mid360.npy",
            "mid40": "mid40.npy",
            "mid70": "mid70.npy",
            "tele": "tele.npy",
        }

        pattern_file = pattern_files.get(self.cfg.sensor_type)
        if pattern_file is None:
            return None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        pattern_path = os.path.join(script_dir, "patterns", pattern_file)

        if not os.path.exists(pattern_path):
            return None

        try:
            data = np.load(pattern_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                if "angles" in data:
                    data = data["angles"]
                elif "theta" in data and "phi" in data:
                    data = np.stack([data["theta"], data["phi"]], axis=-1)
                else:
                    data = data[list(data.files)[0]]

            if data.ndim == 2 and data.shape[1] >= 2:
                return data[:, :2].astype(np.float32)
        except Exception:
            pass

        return None

    def _generate_random_pattern(self, params: dict) -> np.ndarray:
        """Generate random scan pattern angles."""
        total_samples = params["samples"] * 10
        h_fov = math.radians(params.get("horizontal_fov", 360.0))
        v_fov = math.radians(params.get("vertical_fov", 90.0))

        rng = np.random.default_rng(seed=abs(hash(self.cfg.sensor_type)) % (2**32))
        pattern_angles = np.empty((total_samples, 2), dtype=np.float32)
        pattern_angles[:, 0] = rng.uniform(-0.5 * h_fov, 0.5 * h_fov, size=total_samples)
        pattern_angles[:, 1] = rng.uniform(-0.5 * v_fov, 0.5 * v_fov, size=total_samples)

        return pattern_angles

    def _sample_pattern(self, full_pattern: np.ndarray, start_index: int | None = None) -> np.ndarray:
        """Sample subset of rays from full pattern.

        Parameters
        ----------
        full_pattern : np.ndarray
            Full pattern angles with shape (N, 2).
        start_index : int | None
            Starting index for sampling. Uses cfg.rolling_window_start if None.

        Returns
        -------
        np.ndarray
            Ray directions array with shape (1, n_samples, 3).
        """
        cfg = self.cfg
        total_rays = full_pattern.shape[0]
        samples = min(cfg.samples, total_rays)

        start_idx = (start_index if start_index is not None else cfg.rolling_window_start) % total_rays

        if start_idx + samples <= total_rays:
            selected_angles = full_pattern[start_idx : start_idx + samples]
        else:
            end_samples = total_rays - start_idx
            begin_samples = samples - end_samples
            selected_angles = np.vstack([full_pattern[start_idx:], full_pattern[:begin_samples]])

        if cfg.downsample > 1:
            selected_angles = selected_angles[:: cfg.downsample]

        theta, phi = selected_angles[:, 0], selected_angles[:, 1]
        x, y, z = spherical_to_cartesian(theta, phi)

        ray_directions = np.stack([x, y, z], axis=1).astype(np.float32)
        norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
        ray_directions = ray_directions / np.maximum(norms, 1e-8)

        return ray_directions.reshape(1, -1, 3)

    def update_dynamic_pattern(self, cfg: LivoxPattern, time_step: float) -> np.ndarray | None:
        """Update dynamic pattern by advancing rolling window.

        Parameters
        ----------
        cfg : LivoxPattern
            Pattern configuration.
        time_step : float
            Current simulation time in seconds.

        Returns
        -------
        np.ndarray | None
            Updated ray directions or None if no update needed.
        """
        if not cfg.enable_dynamic_pattern or cfg.sensor_type not in self.generated_patterns:
            return None

        pattern_update_rate = 10
        current_tick = int(time_step * pattern_update_rate + 1e-6)

        if self._last_update_tick is not None and current_tick == self._last_update_tick:
            return None

        self._last_update_tick = current_tick
        full_pattern = self.generated_patterns[cfg.sensor_type]
        total_rays = full_pattern.shape[0]

        self.current_start_index = (self.current_start_index + cfg.samples) % total_rays
        return self._sample_pattern(full_pattern, start_index=self.current_start_index)
