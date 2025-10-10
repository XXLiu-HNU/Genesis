# -*- coding: utf-8 -*-
"""
path_search.py
通用路径规划 + 时间参数化跟随（与具体仿真/控制解耦）
- sample_free_xy:    随机采样一个与障碍不碰的点
- build_occupancy_grid: 保守占据（圆-方格相交）+ 可选离散膨胀
- astar:             8邻域 A* 搜索
- plan_path:         一次性完成 A* -> 世界坐标 ->（可选）捷径平滑
- PathFollower:      速度/加速度受限 + 终点减速 + 前瞻 的轨迹跟随器
"""

import math
import random
from typing import List, Tuple, Optional

import torch

# ---------------------- 基础几何/工具 ----------------------
# --- 模块级缓存 ---
_mesh_cache = {}  # key -> (X, Y, xs, ys)

def _device_key(device: torch.device) -> str:
    # 区分多 GPU；CPU 时 index 可能为 None
    if device.type == "cuda":
        idx = 0 if device.index is None else device.index
        return f"cuda:{idx}"
    return "cpu"

def _get_mesh(H, W, world_xy_min, cell, device):
    # key 里加设备、尺寸、原点、cell，避免误复用
    key = (H, W, float(world_xy_min[0]), float(world_xy_min[1]), float(cell), _device_key(device))
    if key not in _mesh_cache:
        # 明确 dtype=float32，避免后面 +0.5*cell 带来的隐式类型提升
        xs = world_xy_min[0] + (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * cell
        ys = world_xy_min[1] + (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * cell
        X, Y = torch.meshgrid(ys, xs, indexing="ij")  # X: y坐标场(HxW), Y: x坐标场(HxW)
        _mesh_cache[key] = (X, Y, xs, ys)
    return _mesh_cache[key]

def _ensure_tensors(
    obs_xy, obs_r, device=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.device]:
    """把 obs 转成 torch.Tensor（float32），并确定 device。"""
    if isinstance(obs_xy, torch.Tensor):
        xy = obs_xy
    else:
        xy = torch.tensor(obs_xy, dtype=torch.float32)
    if isinstance(obs_r, torch.Tensor):
        r = obs_r
    else:
        r = torch.tensor(obs_r, dtype=torch.float32)
    if device is None:
        device = xy.device if xy.numel() > 0 else torch.device("cpu")
    return xy.to(device), r.to(device), device

def sample_free_xy(
    world_xy_min: Tuple[float, float],
    world_xy_max: Tuple[float, float],
    obs_xy,
    obs_r,
    safe_radius: float,
    max_tries: int = 2000,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    从矩形边界内随机采样一个与任何膨胀圆(障碍)不相交的点。
    safe_radius 一般取 (drone_radius + safety_margin)
    """
    xy, r, device = _ensure_tensors(obs_xy, obs_r, device)
    for _ in range(max_tries):
        x = random.uniform(world_xy_min[0], world_xy_max[0])
        y = random.uniform(world_xy_min[1], world_xy_max[1])
        if xy.numel() == 0:
            return x, y
        p = torch.tensor([x, y], dtype=torch.float32, device=device)
        d = torch.norm(xy - p, dim=1)  # (n,)
        if torch.all(d >= (r + safe_radius)):
            return x, y
    raise RuntimeError("sample_free_xy: failed to find a collision-free sample.")

def sample_free_xy_batch(
    world_xy_min: Tuple[float, float],
    world_xy_max: Tuple[float, float],
    obs_xy: torch.Tensor,     # [K,2]
    obs_r:  torch.Tensor,     # [K]
    safe_radius: float,
    n: int,
    max_tries: int = 2000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    一次采 n 个与膨胀圆障碍不相交的点（[n,2], torch.float32）。
    如果障碍为 0，则均匀采样即可。
    """
    device = device or (obs_xy.device if isinstance(obs_xy, torch.Tensor) else "cpu")
    out = torch.empty((n, 2), dtype=torch.float32, device=device)
    ok  = torch.zeros((n,), dtype=torch.bool, device=device)

    if obs_xy.numel() == 0:
        out[:, 0] = torch.empty(n, device=device).uniform_(world_xy_min[0], world_xy_max[0])
        out[:, 1] = torch.empty(n, device=device).uniform_(world_xy_min[1], world_xy_max[1])
        ok[:] = True
        return out

    thr = (obs_r + safe_radius)**2  # [K]
    tries = 0
    while (~ok).any() and tries < max_tries:
        tries += 1
        need = (~ok).nonzero(as_tuple=False).squeeze(-1)  # indices to fill
        # 生成候选
        cand = torch.empty((need.numel(), 2), dtype=torch.float32, device=device)
        cand[:, 0].uniform_(world_xy_min[0], world_xy_max[0])
        cand[:, 1].uniform_(world_xy_min[1], world_xy_max[1])
        # 距离判定（广播）：[M,2] - [1,K,2] -> [M,K,2] -> [M,K]
        d2 = (cand[:, None, :] - obs_xy[None, :, :]).pow(2).sum(-1)
        free = torch.all(d2 >= thr[None, :], dim=1)        # [M]
        # 写回
        out[need[free]] = cand[free]
        ok[need[free]] = True

    if (~ok).any():
        raise RuntimeError("sample_free_xy_batch: failed to sample enough points.")
    return out

def sample_origin_xy(
    r_min: float = 2.0,
    r_max: float = 3.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    在以 center 为圆心的 [r_min, r_max] 环形区域内随机采样一个点。
    假设该区域已经没有障碍。
    """
    # 面积均匀的环形采样
    u = random.random()
    rho = math.sqrt(u * (r_max * r_max - r_min * r_min) + r_min * r_min)
    theta = random.random() * 2.0 * math.pi
    x = center[0] + rho * math.cos(theta)
    y = center[1] + rho * math.sin(theta)
    return x, y

def sample_origin_xy_batch(
    n: int,
    r_min: float = 2.0,
    r_max: float = 3.0,
    center=(0.0, 0.0),
    device="cpu"
) -> torch.Tensor:
    """
    批量采样 n 个起点，返回形状 (n,2) 的张量。
    """
    u = torch.rand(n, device=device)
    rho = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
    theta = 2.0 * math.pi * torch.rand(n, device=device)
    x = center[0] + rho * torch.cos(theta)
    y = center[1] + rho * torch.sin(theta)
    return torch.stack([x, y], dim=1)  # (n,2)

def sample_around_centers_batch(
    centers_xy: torch.Tensor,   # [N,2] target 位置
    r_min: float,
    r_max: float,
    obs_xy: torch.Tensor,       # [K,2]
    obs_r:  torch.Tensor,       # [K]
    safe_radius: float,
    max_tries: int = 2000,
) -> torch.Tensor:
    """
    对每个 center 采 1 个点：半径~U[r_min,r_max]、角度~U[0,2π)，并确保与障碍 (r+safe) 不相交。
    返回 [N,2]
    """
    device = centers_xy.device
    N = centers_xy.shape[0]
    out = torch.empty((N, 2), dtype=torch.float32, device=device)
    ok  = torch.zeros((N,), dtype=torch.bool, device=device)

    if obs_xy.numel() == 0:
        # 无障碍，直接采样
        theta = torch.empty(N, device=device).uniform_(0, 2*math.pi)
        rr    = torch.empty(N, device=device).uniform_(r_min, r_max)
        out[:, 0] = centers_xy[:, 0] + rr*torch.cos(theta)
        out[:, 1] = centers_xy[:, 1] + rr*torch.sin(theta)
        ok[:] = True
        return out

    thr = (obs_r + safe_radius)**2  # [K]
    tries = 0
    while (~ok).any() and tries < max_tries:
        tries += 1
        need = (~ok).nonzero(as_tuple=False).squeeze(-1)
        M = need.numel()
        theta = torch.empty(M, device=device).uniform_(0, 2*math.pi)
        rr    = torch.empty(M, device=device).uniform_(r_min, r_max)
        cand = torch.empty((M, 2), dtype=torch.float32, device=device)
        c = centers_xy[need]  # [M,2]
        cand[:, 0] = c[:, 0] + rr*torch.cos(theta)
        cand[:, 1] = c[:, 1] + rr*torch.sin(theta)
        # 与障碍的距离判定
        d2 = (cand[:, None, :] - obs_xy[None, :, :]).pow(2).sum(-1)  # [M,K]
        free = torch.all(d2 >= thr[None, :], dim=1)
        out[need[free]] = cand[free]
        ok[need[free]]  = True

    if (~ok).any():
        raise RuntimeError("sample_around_centers_batch: failed for some centers.")
    return out


def world_to_grid(
    x: float, y: float, world_xy_min: Tuple[float, float], cell: float, W: int, H: int
) -> Tuple[int, int]:
    j = int((x - world_xy_min[0]) / cell)
    i = int((y - world_xy_min[1]) / cell)
    j = max(0, min(W - 1, j))
    i = max(0, min(H - 1, i))
    return i, j

def grid_to_world(
    i: int, j: int, world_xy_min: Tuple[float, float], cell: float
) -> Tuple[float, float]:
    x = world_xy_min[0] + (j + 0.5) * cell
    y = world_xy_min[1] + (i + 0.5) * cell
    return x, y

# ---------------------- 占据栅格（保守判定 + 可选膨胀） ----------------------


# --- build_occupancy_grid：用缓存的网格，并用“距离平方”避免 sqrt ---
def build_occupancy_grid(
    world_xy_min: Tuple[float, float],
    world_xy_max: Tuple[float, float],
    cell: float,
    obs_xy,
    obs_r,
    inflation: float,
    extra_margin: float = 0.0,
    device: Optional[torch.device] = None,
):
    xy, r, device = _ensure_tensors(obs_xy, obs_r, device)

    W = int(math.ceil((world_xy_max[0] - world_xy_min[0]) / cell))
    H = int(math.ceil((world_xy_max[1] - world_xy_min[1]) / cell))
    grid = torch.zeros((H, W), dtype=torch.bool, device=device)

    # 统一从缓存拿 X, Y, xs, ys
    X, Y, xs, ys = _get_mesh(H, W, world_xy_min, cell, device)
    half = cell * 0.5

    if xy.numel() > 0:
        # 对每个圆，只更新其 AABB 覆盖的局部子块（更快）
        for i in range(xy.shape[0]):
            cx, cy = xy[i, 0].item(), xy[i, 1].item()
            rr = (r[i].item() + inflation)
            rr2 = rr * rr

            # 计算圆的 AABB 在网格中的范围（列 j ~ x，行 i ~ y）
            j_min = max(0, int((cx - rr - world_xy_min[0]) / cell))
            j_max = min(W - 1, int((cx + rr - world_xy_min[0]) / cell))
            i_min = max(0, int((cy - rr - world_xy_min[1]) / cell))
            i_max = min(H - 1, int((cy + rr - world_xy_min[1]) / cell))
            if j_min > j_max or i_min > i_max:
                continue

            subY = Y[i_min:i_max+1, j_min:j_max+1]
            subX = X[i_min:i_max+1, j_min:j_max+1]

            # 矩形最近点距离（与之前相同逻辑，但比较平方距离）
            dx = torch.clamp(torch.abs(subY - cx) - half, min=0.0)
            dy = torch.clamp(torch.abs(subX - cy) - half, min=0.0)
            dist2 = dx * dx + dy * dy
            grid[i_min:i_max+1, j_min:j_max+1] |= (dist2 <= rr2)

    # 可选：二值膨胀（离散安全裕度）
    if extra_margin > 1e-6:
        import torch.nn.functional as F
        k = int(math.ceil(extra_margin / cell))
        if k > 0:
            mask = grid.float().unsqueeze(0).unsqueeze(0)      # 1x1xHxW
            pad = (k, k, k, k)
            mask = F.max_pool2d(F.pad(mask, pad, mode="replicate"),
                                kernel_size=2 * k + 1, stride=1)
            grid = (mask[0, 0] > 0.5)

    return grid, xs, ys


# ---------------------- A* 搜索（8邻域） ----------------------

def astar(grid: torch.Tensor, start_ij: Tuple[int, int], goal_ij: Tuple[int, int]):
    import heapq
    H, W = grid.shape
    si, sj = start_ij
    gi, gj = goal_ij
    if grid[si, sj] or grid[gi, gj]:
        return None

    def h(i, j):
        return math.hypot(i - gi, j - gj)

    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    open_heap = []
    heapq.heappush(open_heap, (h(si, sj), 0.0, (si, sj)))
    came = {(si, sj): None}
    gscore = {(si, sj): 0.0}

    while open_heap:
        _, gs, (ci, cj) = heapq.heappop(open_heap)
        if (ci, cj) == (gi, gj):
            path = []
            cur = (ci, cj)
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return list(reversed(path))

        for di, dj in nbrs:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < H and 0 <= nj < W):
                continue
            if grid[ni, nj]:
                continue
            step = math.hypot(di, dj)
            cand = gs + step
            if (ni, nj) not in gscore or cand < gscore[(ni, nj)]:
                gscore[(ni, nj)] = cand
                came[(ni, nj)] = (ci, cj)
                heapq.heappush(open_heap, (cand + h(ni, nj), cand, (ni, nj)))
    return None

# ---------------------- 线段-圆碰撞 & 捷径平滑 ----------------------

def _line_collision_free(
    p: torch.Tensor, q: torch.Tensor,
    obs_xy: torch.Tensor, obs_r: torch.Tensor, inflation: float
) -> bool:
    if obs_xy.numel() == 0:
        return True
    pq = q - p
    pq2 = torch.dot(pq, pq).item()
    if pq2 == 0:
        d = torch.norm(obs_xy - p, dim=1)
        return torch.all(d > (obs_r + inflation))
    t = torch.clamp(torch.sum((obs_xy - p) * pq, dim=1) / pq2, 0.0, 1.0)
    proj = p + t.unsqueeze(1) * pq
    d = torch.norm(obs_xy - proj, dim=1)
    return torch.all(d > (obs_r + inflation))

def shortcut_smooth(
    path_xy: List[Tuple[float, float]],
    obs_xy, obs_r, inflation: float,
    max_trials: int = 200
) -> List[Tuple[float, float]]:
    if len(path_xy) <= 2:
        return path_xy
    xy, r, device = _ensure_tensors(obs_xy, obs_r, None)
    pts = [torch.tensor(p, dtype=torch.float32, device=device) for p in path_xy]
    for _ in range(max_trials):
        if len(pts) <= 2:
            break
        i = random.randint(0, len(pts) - 2)
        j = random.randint(i + 1, len(pts) - 1)
        if j == i + 1:
            continue
        if _line_collision_free(pts[i], pts[j], xy, r, inflation + 0.02):  # 稍保守
            pts = pts[: i + 1] + pts[j:]
    return [(p[0].item(), p[1].item()) for p in pts]

# ---------------------- 一次性规划接口 ----------------------

def plan_path(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    obs_xy,
    obs_r,
    world_xy_min: Tuple[float, float],
    world_xy_max: Tuple[float, float],
    cell_size: float,
    inflation: float,
    smooth: bool = True,
    extra_grid_margin: float = 0.0,
    device: Optional[torch.device] = None,
) -> Optional[List[Tuple[float, float]]]:
    """A* -> 世界坐标 ->（可选）捷径平滑；失败返回 None"""
    grid, xs, ys = build_occupancy_grid(
        world_xy_min, world_xy_max, cell_size, obs_xy, obs_r,
        inflation=inflation, extra_margin=extra_grid_margin, device=device
    )
    H, W = grid.shape
    si, sj = world_to_grid(start_xy[0], start_xy[1], world_xy_min, cell_size, W, H)
    gi, gj = world_to_grid(goal_xy[0], goal_xy[1], world_xy_min, cell_size, W, H)
    path_ij = astar(grid, (si, sj), (gi, gj))
    if path_ij is None:
        return None
    raw_path_xy = [grid_to_world(i, j, world_xy_min, cell_size) for (i, j) in path_ij]
    return shortcut_smooth(raw_path_xy, obs_xy, obs_r, inflation) if smooth else raw_path_xy

# ---------------------- 弧长插值 & 跟随器 ----------------------

def polyline_arclen(points_xy: List[Tuple[float, float]]) -> List[float]:
    if not points_xy:
        return [0.0]
    L = [0.0]
    for i in range(1, len(points_xy)):
        dx = points_xy[i][0] - points_xy[i - 1][0]
        dy = points_xy[i][1] - points_xy[i - 1][1]
        L.append(L[-1] + math.hypot(dx, dy))
    return L

def interp_along_polyline(
    points_xy: List[Tuple[float, float]], L: List[float], s: float
) -> Tuple[float, float]:
    if not points_xy:
        return (0.0, 0.0)
    s = max(0.0, min(s, L[-1]))
    lo, hi = 0, len(L) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if L[mid] <= s:
            lo = mid
        else:
            hi = mid
    if L[hi] - L[lo] < 1e-9:
        return points_xy[hi]
    r = (s - L[lo]) / (L[hi] - L[lo])
    x = points_xy[lo][0] + r * (points_xy[hi][0] - points_xy[lo][0])
    y = points_xy[lo][1] + r * (points_xy[hi][1] - points_xy[lo][1])
    return (x, y)

class PathFollower:
    """
    时间参数化路径跟随：
    - 速度上限 v_max（支持 warmup）
    - 加速度上限 a_max
    - 终点减速（v^2 <= 2 a * s_remain / k）
    - 前瞻距离：max(min_lookahead, v * lookahead_time)
    """
    def __init__(
        self, path_xy: List[Tuple[float, float]], dt: float,
        v_max: float = 0.8, a_max: float = 1.5,
        warmup_time: float = 2.0, v_init: float = 0.1,
        lookahead_time: float = 0.6, min_lookahead: float = 0.15,
        slow_down_k: float = 1.0
    ):
        self.dt = dt
        self.v_max_nominal = v_max
        self.a_max = a_max
        self.warmup_time = max(1e-6, warmup_time)
        self.v_init = max(0.02, v_init)
        self.lookahead_time = lookahead_time
        self.min_lookahead = min_lookahead
        self.slow_down_k = max(0.5, slow_down_k)
        self.reset_with_path(path_xy)

    def reset_with_path(self, path_xy: List[Tuple[float, float]]):
        self.path_xy = list(path_xy) if path_xy else [(0.0, 0.0)]
        self.L = polyline_arclen(self.path_xy)
        self.s_total = self.L[-1]
        self.s_ref = 0.0
        self.v = 0.0
        self.t = 0.0

    def _warmup_vmax(self) -> float:
        alpha = min(1.0, self.t / self.warmup_time)
        return self.v_init + alpha * (self.v_max_nominal - self.v_init)

    def step(self, cur_xy: Tuple[float, float] | None = None) -> Tuple[float, float]:
        self.t += self.dt
        vmax_now = self._warmup_vmax()
        s_remain = max(0.0, self.s_total - self.s_ref)
        v_brake = math.sqrt(max(0.0, 2.0 * self.a_max * s_remain / self.slow_down_k)) if self.a_max > 1e-9 else vmax_now
        v_des = min(vmax_now, v_brake)
        dv = max(-self.a_max * self.dt, min(self.a_max * self.dt, v_des - self.v))
        self.v += dv
        self.s_ref = min(self.s_total, self.s_ref + self.v * self.dt)
        lookahead = max(self.min_lookahead, self.v * self.lookahead_time)
        s_query = min(self.s_total, self.s_ref + lookahead)
        return interp_along_polyline(self.path_xy, self.L, s_query)

    def reached_goal(self, thresh: float = 0.12) -> bool:
        return (self.s_total - self.s_ref) <= thresh
