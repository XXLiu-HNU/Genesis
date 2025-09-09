from dataclasses import dataclass

import numpy as np
import torch

import genesis as gs


@dataclass
class RaycastPattern:
    """Base configuration for a pattern."""

    _is_dynamic: bool = False

    def get_return_shape(self) -> tuple[int, ...]:
        """Get the shape of the ray vectors, e.g. [n_scan_lines, n_points_per_line, 3] or [1, n_rays, 3]"""
        raise NotImplementedError(f"{type(self).__name__} must implement `get_return_shape()`.")


class RaycastPatternGenerator:
    """Base class for raycast pattern generators."""

    def __init__(self, cfg: RaycastPattern):
        self.cfg = cfg

    def get_ray_directions(self) -> torch.Tensor:
        """Get the direction vectors of the rays."""
        raise NotImplementedError(f"{type(self).__name__} must implement `get_ray_directions()`.")

    def get_ray_starts(self) -> torch.Tensor:
        """Get the local start positions of the rays."""
        return torch.zeros(self.cfg.get_return_shape(), dtype=gs.tc_float, device=gs.device)


PATTERN_GENERATORS: dict[type[RaycastPattern], type[RaycastPatternGenerator]] = {}


def register_pattern(pattern_type: type[RaycastPattern]):
    def _impl(generator_type: type[RaycastPatternGenerator]):
        PATTERN_GENERATORS[pattern_type] = generator_type
        return generator_type

    return _impl


def create_pattern_generator(cfg: RaycastPattern) -> RaycastPatternGenerator:
    gen_cls = PATTERN_GENERATORS.get(type(cfg))
    if gen_cls is None:
        raise ValueError(f"Unsupported pattern configuration type: {type(cfg)}")
    return gen_cls(cfg)
