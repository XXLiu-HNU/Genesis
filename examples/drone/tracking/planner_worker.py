import pickle
from typing import List, Tuple, Dict, Any

import numpy as np

# Local roadmap implementation (CPU/NumPy)
from roadmap import Roadmap, _line_of_sight_free


_ROADMAP: Roadmap | None = None


def init(roadmap_bytes: bytes) -> None:
    """Initializer for worker processes: receive serialized roadmap once."""
    global _ROADMAP
    _ROADMAP = pickle.loads(roadmap_bytes)


def plan_batch(
    env_ids: List[int],
    starts_xy: np.ndarray,
    goals_xy_shared: List[Tuple[float, float]],
    k_attach: int = 8,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Plan paths for a batch of environments.

    Inputs:
      - env_ids: list of environment ids (len=B)
      - starts_xy: (B,2) float32
      - goals_xy_shared: list of candidate goals shared across the batch
    Output:
      - dict env_id -> path as list[(x,y)] (may be length>=2), or empty list if not found
    """
    assert _ROADMAP is not None, "planner worker not initialized"

    result: Dict[int, List[Tuple[float, float]]] = {}
    B = starts_xy.shape[0]
    for i in range(B):
        start_xy = (float(starts_xy[i, 0]), float(starts_xy[i, 1]))

        # Fast LOS to any goal
        los_hits = [g for g in goals_xy_shared if _line_of_sight_free(
            np.asarray(start_xy, np.float32),
            np.asarray(g, np.float32),
            _ROADMAP.obs_xy, _ROADMAP.obs_r, _ROADMAP.clearance
        )]
        if len(los_hits) > 0:
            path = np.asarray([start_xy, los_hits[0]], dtype=np.float32)
        else:
            path_np = _ROADMAP.query(start_xy, goals_xy_shared, k_attach=k_attach)
            path = path_np if path_np is not None else None

        if path is not None and path.shape[0] >= 2:
            result[env_ids[i]] = [(float(x), float(y)) for (x, y) in path.tolist()]
        else:
            result[env_ids[i]] = []

    return result


