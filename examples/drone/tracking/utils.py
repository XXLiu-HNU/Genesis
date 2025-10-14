import torch
import genesis as gs
from typing import Literal, Tuple, Dict
import torch.nn.functional as F

# ---------- 小工具 ----------
def _to_xy(p: torch.Tensor) -> torch.Tensor:
    """将 [B,3] or [B,2] 投影到 XY -> [B,2]"""
    return p[..., :2]

def _broadcast_obstacles(obs_xy: torch.Tensor, obs_r: torch.Tensor, device=None, dtype=None):
    """将 [M,2]/[M] 升维成 [1,M,2]/[1,M]，方便广播"""
    if device is None: device = obs_xy.device
    if dtype  is None: dtype  = obs_xy.dtype
    obs_xy = obs_xy.to(device=device, dtype=dtype).unsqueeze(0)  # [1,M,2]
    obs_r  = obs_r.to(device=device,  dtype=dtype).unsqueeze(0)  # [1,M]
    return obs_xy, obs_r

def _softmin(x: torch.Tensor, beta: float = 10.0, dim: int = -1) -> torch.Tensor:
    """可导的近似 min"""
    return -torch.logsumexp(-beta * x, dim=dim) / beta

# 线段 p0->p1 到点集 c 的最小距离（2D，向量化）
def _seg_point_distance_2d(p0_xy: torch.Tensor, p1_xy: torch.Tensor, c_xy: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    p0_xy, p1_xy: [B,2]
    c_xy:         [1,M,2] 或 [B,M,2]
    返回:         [B,M]
    """
    v  = p1_xy - p0_xy                      # [B,2]
    vv = (v * v).sum(dim=-1, keepdim=True)  # [B,1]
    p0e = p0_xy.unsqueeze(1)                # [B,1,2]
    ve  = v.unsqueeze(1)                    # [B,1,2]
    t = ((c_xy - p0e) * ve).sum(dim=-1, keepdim=True) / (vv.unsqueeze(1) + eps)  # [B,M,1]
    t = t.clamp(0.0, 1.0)                   # 线段投影
    proj = p0e + t * ve                     # [B,M,2]
    d = torch.linalg.norm(proj - c_xy, dim=-1)  # [B,M]
    return d

# ---------- 1) 无人机-障碍 碰撞检查 ----------
def collision_check(tracker_pos: torch.Tensor,
                    obs_xy: torch.Tensor,
                    obs_r: torch.Tensor,
                    uav_radius: float = 0.25):
    """
    返回:
      collided:   [B] bool，是否与任一障碍相交/接触（净空 < 0）
      min_margin: [B] 最薄弱净空（>0 安全，<0 已碰撞；单位: 米）
    """
    p_u = _to_xy(tracker_pos)                      # [B,2]
    obs_xy_b, obs_r_b = _broadcast_obstacles(obs_xy, obs_r, device=p_u.device, dtype=p_u.dtype)
    # 到各圆边界的净空：|pu-oi| - (ri + uav_radius)
    margin_all = torch.linalg.norm(p_u.unsqueeze(1) - obs_xy_b, dim=-1) - (obs_r_b + uav_radius)  # [B,M]
    # 取真正的 min（而不是 softmin，碰撞判定需要严格）
    min_margin, _ = margin_all.min(dim=1)          # [B]
    collided = min_margin < 0.0
    return collided, min_margin

# ---------- 2) 目标是否被遮挡（丢失） ----------
def occlusion_check(tracker_pos: torch.Tensor,
                    target_pos: torch.Tensor,
                    obs_xy: torch.Tensor,
                    obs_r: torch.Tensor):
    """
    用“线段到圆心的最近距离 - 半径”判断遮挡。
    返回:
      occluded:    [B] bool，是否被任一圆柱遮挡（最小净空 < 0）
      min_clear:   [B] 最薄弱视线净空（>0 可见，<0 被遮挡/切断）
      blocker_idx: [B] 导致最薄弱净空的障碍索引（便于调试/可视化）
    """
    p_u = _to_xy(tracker_pos)                       # [B,2]
    p_t = _to_xy(target_pos)                        # [B,2]
    obs_xy_b, obs_r_b = _broadcast_obstacles(obs_xy, obs_r, device=p_u.device, dtype=p_u.dtype)

    # 线段 p_u -> p_t 到每个圆心的最小距离
    d_seg = _seg_point_distance_2d(p_u, p_t, obs_xy_b)    # [B,M]
    clearance = d_seg - obs_r_b                           # [B,M] 视线净空

    min_clear, idx = clearance.min(dim=1)                 # [B], [B]
    occluded = min_clear < 0.0
    blocker_idx = idx                                     # 造成最薄弱净空的障碍 id
    return occluded, min_clear, blocker_idx


def setup_random_cylindrical_obstacles(
        scene, n_obstacles=20, min_radius=0.2, max_radius=0.3,
        min_height=3.0, max_height=5.0, min_distance=1.0,
        world_bounds=(-10, 10, -10, 10), device="cpu", oversample_factor=10,
        origin_clearance=0.0  # 新增参数：原点周围清空的半径
):
    """
    Randomly generates and places non-overlapping cylindrical obstacles in a scene,
    while ensuring no obstacles are generated near the origin.
    """

    # Oversample candidates (more than needed)
    n_candidates = n_obstacles * oversample_factor
    xs = torch.empty(n_candidates, device=device).uniform_(world_bounds[0], world_bounds[1])
    ys = torch.empty(n_candidates, device=device).uniform_(world_bounds[2], world_bounds[3])
    radii = torch.empty(n_candidates, device=device).uniform_(min_radius, max_radius)
    heights = torch.empty(n_candidates, device=device).uniform_(min_height, max_height)

    # ! ---------------- filter by world bounds ----------------------------------------
    in_bounds = (
        (xs - radii >= world_bounds[0]) &
        (xs + radii <= world_bounds[1]) &
        (ys - radii >= world_bounds[2]) &
        (ys + radii <= world_bounds[3])
    )

    xs, ys, radii, heights = xs[in_bounds], ys[in_bounds], radii[in_bounds], heights[in_bounds]

    # ! ---------------- filter out near origin ----------------------------------------
    dist_origin = torch.sqrt(xs**2 + ys**2)
    not_near_origin = dist_origin > (origin_clearance + radii)
    xs, ys, radii, heights = xs[not_near_origin], ys[not_near_origin], radii[not_near_origin], heights[not_near_origin]

    # ! ---------------- filter by min distance ----------------------------------------
    obstacles = []
    obstacle_positions = []
    obstacle_radii = []

    for i in range(xs.shape[0]):
        if len(obstacles) >= n_obstacles:
            break
        x, y, r, h = xs[i].item(), ys[i].item(), radii[i].item(), heights[i].item()

        if obstacle_positions:
            pos_tensor = torch.tensor(obstacle_positions, device=device)
            rad_tensor = torch.tensor(obstacle_radii, device=device)
            dx = pos_tensor[:, 0] - x
            dy = pos_tensor[:, 1] - y
            dist = torch.sqrt(dx**2 + dy**2)
            if torch.any(dist < (rad_tensor + r + min_distance)):
                continue  # too close, skip

        # Add obstacle
        r_color = torch.empty(1, device=device).uniform_(0.3, 0.8).item()
        g_color = torch.empty(1, device=device).uniform_(0.3, 0.8).item()
        b_color = torch.empty(1, device=device).uniform_(0.3, 0.8).item()

        obstacle = scene.add_entity(
            gs.morphs.Cylinder(
                radius=r,
                height=h,
                pos=(x, y, h/2),
                fixed=True
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(r_color, g_color, b_color),
                ),
            ),
        )
        obstacles.append(obstacle)
        obstacle_positions.append((x, y))
        obstacle_radii.append(r)

    if len(obstacles) < n_obstacles:
        print(f"Warning: Only generated {len(obstacles)} obstacles (requested {n_obstacles})")

    return obstacles, obstacle_positions, obstacle_radii

# ! ------------------------------- Obstacle Features --------------------------------------

"""
    obstacle_features (vectorized, GPU-friendly)
    Compute low-dimensional obstacle features from infinite-height cylinders (2D circles).
    Everything is batch/vectorized; no Python loops on N/M/K.

    Inputs:
    - tracker_pos_w:        (N,3) or (3,)
    - tracker_quat:         (N,4) or (4,)    quaternion (xyzw or wxyz)
    - tracker_lin_vel_w:    (N,3) or (3,)
    - obs_xy_w:             (N,M,2) or (M,2) circle centers in WORLD (z ignored)
    - obs_r:                (N,M,1) or (M,1) radii

    Hyperparams:
    - range_max: float  (normalization for distances/raycast)
    - ttc_max:   float  (normalization cap for TTC)
    - K: int            number of angular sectors (0 to disable)
    - quat_format: "xyzw" or "wxyz"

    Outputs dict (per batch N):
    - d_min_norm:          (N,1)   in [0,1]
    - bearing_min_pi:      (N,1)   in [-1,1]
    - ttc_min_norm:        (N,1)   in [0,1]
    - mean_clear_norm:     (N,1)   in [0,1]
    - var_clear_norm:      (N,1)   in [0,1]
    - heading_clear_norm:  (N,1)   in [0,1]
    - sector_mins_norm:    (N,K)   in [0,1]  (only if K>0)
"""

QuatFormat = Literal["xyzw", "wxyz"]

def _ensure_batch(x: torch.Tensor, last_dim: int) -> Tuple[torch.Tensor, bool]:
    """Make tensor at least 2D: (N, last_dim). Return (tensor, was_squeezed)."""
    if x.dim() == 1:
        assert x.shape[0] == last_dim, f"Expected shape ({last_dim},), got {tuple(x.shape)}"
        return x.unsqueeze(0), True
    elif x.dim() == 2:
        assert x.shape[1] == last_dim, f"Expected shape (*,{last_dim}), got {tuple(x.shape)}"
        return x, False
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(x.shape)}")

def _split_quat(q: torch.Tensor, quat_format: QuatFormat) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (w,x,y,z) as separate tensors, independent of input convention."""
    if quat_format == "xyzw":
        x, y, z, w = q.unbind(dim=-1)
    elif quat_format == "wxyz":
        w, x, y, z = q.unbind(dim=-1)
    else:
        raise ValueError("quat_format must be 'xyzw' or 'wxyz'")
    return w, x, y, z



def quat_to_Rbw(q: torch.Tensor, quat_format: QuatFormat = "xyzw") -> torch.Tensor:
    """Quaternion(s) -> rotation matrix R_bw (world->body)."""
    q, sq = _ensure_batch(q, 4)
    eps = 1e-9
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=eps)

    w, x, y, z = _split_quat(q, quat_format)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1 - 2 * (yy + zz); r01 = 2 * (xy - wz); r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz);     r11 = 1 - 2 * (xx + zz); r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy);     r21 = 2 * (yz + wx);     r22 = 1 - 2 * (xx + yy)

    R = torch.stack([
        torch.stack([r00, r01, r02], -1),
        torch.stack([r10, r11, r12], -1),
        torch.stack([r20, r21, r22], -1),
    ], -2)  # (N,3,3)

    return R.squeeze(0) if sq else R


def world_to_body_xy(vec_w_xy: torch.Tensor, R_bw: torch.Tensor) -> torch.Tensor:
    """
    Transform planar vectors (x,y) from WORLD to BODY using R_bw (world->body).
    vec_w_xy: (N,2) or (2,)
    R_bw:     (N,3,3) or (3,3)
    return:   (N,2) or (2,)
    """
    v, vs = _ensure_batch(vec_w_xy, 2)
    R, Rs = _ensure_batch(R_bw.reshape(-1, 9), 9)
    R = R.reshape(-1, 3, 3)

    if v.shape[0] != R.shape[0]:
        if v.shape[0] == 1:
            v = v.expand(R.shape[0], -1)
        elif R.shape[0] == 1:
            R = R.expand(v.shape[0], -1, -1)
        else:
            raise ValueError("Batch mismatch in world_to_body_xy")

    v3 = torch.cat([v, torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype)], dim=-1)
    vb = torch.bmm(R, v3.unsqueeze(-1)).squeeze(-1)[..., :2]
    return vb.squeeze(0) if (vs and Rs) else vb


# ---------- core features (fully vectorized) ----------

def obstacle_features(
    tracker_pos_w: torch.Tensor,
    tracker_quat: torch.Tensor,
    tracker_lin_vel_w: torch.Tensor,
    obs_xy_w: torch.Tensor,
    obs_r: torch.Tensor,
    *,
    range_max: float = 20.0,
    ttc_max: float = 5.0,
    K: int = 0,
    quat_format: QuatFormat = "xyzw",
) -> Dict[str, torch.Tensor]:
    """
    Compute low-dim features from cylinders (2D circles). All in BODY frame.
    Everything is vectorized (no Python for loops over N/M/K).
    """
    # ---- sanitize & broadcast ----
    p_w, _ = _ensure_batch(tracker_pos_w, 3)       # (N,3)
    q, _ = _ensure_batch(tracker_quat, 4)          # (N,4)
    v_w, _ = _ensure_batch(tracker_lin_vel_w, 3)   # (N,3)

    if obs_xy_w.dim() == 2:                        # (M,2) -> (N,M,2)
        obs_xy_w = obs_xy_w.unsqueeze(0).expand(p_w.shape[0], -1, -1)
    if obs_r.dim() == 2:                           # (M,1) -> (N,M,1)
        obs_r = obs_r.unsqueeze(0).expand(p_w.shape[0], -1, -1)

    N, M, _2 = obs_xy_w.shape
    device = p_w.device
    eps = 1e-6

    # ---- rotations ----
    R_bw = quat_to_Rbw(q, quat_format=quat_format)                     # (N,3,3)

    # relative centers in WORLD -> BODY (planar)
    rel_xy_w = obs_xy_w - p_w[:, :2].unsqueeze(1)                     # (N,M,2)
    rel_xy_b = world_to_body_xy(rel_xy_w.reshape(-1, 2),
                                 R_bw.repeat_interleave(M, 0)).view(N, M, 2)

    # planar velocity in BODY
    v_xy_b = world_to_body_xy(v_w[:, :2], R_bw)                       # (N,2)

    # ---- boundary clearances ----
    center_dist = torch.linalg.norm(rel_xy_b, dim=-1)                 # (N,M)
    radii = obs_r.squeeze(-1)                                         # (N,M)
    clear = center_dist - radii                                       # (N,M)  (can be negative if overlapping)

    # nearest obstacle & bearing
    d_min, idx_min = torch.min(clear, dim=1, keepdim=True)            # (N,1)
    rel_min = torch.gather(rel_xy_b, 1, idx_min.unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)  # (N,2)
    bearing_min = torch.atan2(rel_min[:, 1], rel_min[:, 0]).unsqueeze(-1)                    # (N,1)
    bearing_min_pi = torch.clamp(bearing_min / torch.pi, -1.0, 1.0)                          # [-1,1]

    # ---- TTC to nearest (closing only) ----
    u_rad = rel_min / (torch.linalg.norm(rel_min, dim=-1, keepdim=True) + eps)               # (N,2)
    closing = torch.clamp(-torch.sum(v_xy_b * u_rad, dim=-1, keepdim=True), min=0.0)         # (N,1)
    ttc = d_min / (closing + eps)                                                             # (N,1)
    ttc_min_norm = torch.clamp(ttc, max=ttc_max) / ttc_max                                    # [0,1]

    # ---- heading ray-cast clearance (ray dir = body +x) ----
    cx, cy, rr = rel_xy_b[..., 0], rel_xy_b[..., 1], radii                                    # (N,M)
    A = torch.ones_like(cx)
    B = -2.0 * cx
    C = cx * cx + cy * cy - rr * rr
    disc = B * B - 4 * A * C                                                                  # (N,M)
    disc_clamped = torch.clamp(disc, min=0.0)
    sqrt_disc = torch.sqrt(disc_clamped)

    s1 = ( -B - sqrt_disc ) / 2.0
    s2 = ( -B + sqrt_disc ) / 2.0
    pos_s1 = (disc >= 0.0) & (s1 > 0)
    pos_s2 = (disc >= 0.0) & (s2 > 0)
    s1 = torch.where(pos_s1, s1, torch.full_like(s1, float("inf")))
    s2 = torch.where(pos_s2, s2, torch.full_like(s2, float("inf")))
    s_hit = torch.minimum(s1, s2)                                                              # (N,M)

    heading_clear = torch.min(s_hit, dim=1, keepdim=True).values                               # (N,1)
    heading_clear_norm = torch.clamp(heading_clear, max=range_max) / range_max                 # [0,1]

    # ---- global stats ----
    clear_pos = torch.clamp(clear, min=0.0, max=range_max)                                     # (N,M) in [0,range_max]
    mean_clear_norm = torch.mean(clear_pos, dim=1, keepdim=True) / range_max                   # (N,1)
    var_clear_norm  = torch.var(clear_pos / range_max, dim=1, keepdim=True, unbiased=False)    # (N,1)

    out: Dict[str, torch.Tensor] = {
        "d_min_norm":         torch.clamp(d_min / range_max, 0.0, 1.0),
        "bearing_min_pi":     bearing_min_pi,
        "ttc_min_norm":       torch.clamp(ttc_min_norm, 0.0, 1.0),
        "mean_clear_norm":    torch.clamp(mean_clear_norm, 0.0, 1.0),
        "var_clear_norm":     torch.clamp(var_clear_norm, 0.0, 1.0),
        "heading_clear_norm": torch.clamp(heading_clear_norm, 0.0, 1.0),
    }

    # ---- sector minima (vectorized) ----
    if K and K > 0:
        # angle of each obstacle in BODY in [0,2pi)
        ang = (torch.atan2(rel_xy_b[..., 1], rel_xy_b[..., 0]) + 2 * torch.pi) % (2 * torch.pi)   # (N,M)
        sector_idx = torch.clamp((ang / (2 * torch.pi) * K).long(), 0, K - 1)                     # (N,M)

        # we want per (n, sector) the min of clear_pos[n, m where sector_idx==sector]
        # Approach A: torch.scatter_reduce (PyTorch >=2.0) with amin
        sector_mins = None
        if hasattr(torch.Tensor, "scatter_reduce"):
            # init with +inf, then amin-reduce
            sector_mins = torch.full((N, K), float("inf"), device=device, dtype=clear_pos.dtype)
            # flatten (N,M) -> (N*M,)
            base = torch.arange(N, device=device).unsqueeze(1).expand(N, M) * K
            flat_indices = (base + sector_idx).reshape(-1)                 # (N*M,)
            values = clear_pos.reshape(-1)                                  # (N*M,)

            # fused reduce on a flat (N*K,) then reshape back
            flat_mins = torch.full((N * K,), float("inf"), device=device, dtype=values.dtype)
            flat_mins.scatter_reduce_(0, flat_indices, values, reduce="amin", include_self=True)
            sector_mins = flat_mins.view(N, K)
        else:
            # Approach B (fallback): one-hot mask (N,K,M), masked min
            # Create (N,M,K) mask via one-hot, then move to (N,K,M)
            oh = torch.nn.functional.one_hot(sector_idx, num_classes=K).to(clear_pos.dtype)  # (N,M,K)
            mask = oh.permute(0, 2, 1)                                                       # (N,K,M)
            # masked values: where mask=1 -> clear_pos; else +inf
            vals = torch.where(mask.bool(),
                                clear_pos.unsqueeze(1).expand(-1, K, -1),
                                torch.full((N, K, M), float("inf"), device=device, dtype=clear_pos.dtype))
            sector_mins = torch.min(vals, dim=2).values                                      # (N,K)

        # clamp & normalize
        sector_mins = torch.clamp(sector_mins, max=range_max) / range_max                    # (N,K) in [0,1]
        out["sector_mins_norm"] = sector_mins

    return out
def collision_reward(
    tracker_pos: torch.Tensor,   # [N,dim]
    tracker_vel: torch.Tensor,   # [N,dim]
    obs_centers: torch.Tensor,   # [M,dim]
    obs_radii: torch.Tensor,     # [M] or [M,1]
    d_s=0.5,                     # 可以是 float/int/tuple/list/tensor；支持标量或 [N]
    beta1: float = 1.0,
    beta2: float = 10.0,
    eps: float = 1e-9,
):
    """
    返回：
      rc:      [N] 惩罚（越大越危险）
      d_t:     [N] 最近障碍边界净空
      dtdot:   [N] 净空的时间导数（>0 远离，<0 接近）
      idx:     [N] 最近障碍索引
    """
    device = tracker_pos.device
    dtype  = tracker_pos.dtype
    N      = tracker_pos.shape[0]

    # --- 统一 obs_radii 形状 ---
    if obs_radii.ndim == 2 and obs_radii.shape[1] == 1:
        obs_radii = obs_radii.squeeze(1)  # [M]

    # --- 计算到每个圆障碍边界的净空 d_t = ||p-c|| - r ---
    R = tracker_pos.unsqueeze(1) - obs_centers.unsqueeze(0)   # [N,M,dim]
    dist = torch.linalg.norm(R, dim=-1)                       # [N,M]
    clearance = dist - obs_radii.view(1, -1)                  # [N,M]

    d_t, idx = clearance.min(dim=1)                           # [N], [N]
    R_min    = R[torch.arange(N, device=device), idx]         # [N,dim]
    dist_min = dist[torch.arange(N, device=device), idx].clamp_min(eps)  # [N]
    n_hat    = R_min / dist_min.unsqueeze(-1)                 # [N,dim] 障碍外法向

    # \dot d_t = n_hat · v   （障碍静止）
    dtdot = (n_hat * tracker_vel).sum(dim=-1)                 # [N]
    v_c   = torch.clamp(-dtdot, min=0.0)                      # 只惩罚“接近”

    # --- 统一/广播 d_s ---
    # 接受 float/int/tuple/list/tensor；把它变成与 d_t 可广播的张量
    if isinstance(d_s, (tuple, list)):
        # 常见的 (0.6,) 情况
        if len(d_s) == 1:
            d_s = d_s[0]
        else:
            d_s = torch.as_tensor(d_s, device=device, dtype=dtype)
    if not torch.is_tensor(d_s):
        d_s = torch.tensor(d_s, device=device, dtype=dtype)

    # 若 d_s 是标量，自动扩展到 [N]
    if d_s.ndim == 0:
        d_s = d_s.expand_as(d_t)           # [N]
    elif d_s.ndim == 1:
        # 允许 [N]；若是 [1] 也能广播
        assert d_s.shape[0] in (1, N), f"d_s shape {d_s.shape} must be [N] or [1] or scalar."

    # --- 缓冲项 + softplus 屏障 ---
    # 缓冲项：v_c * [max(1 - (d_t - d_s), 0)]^2
    buf = v_c * torch.relu(1.0 - (d_t - d_s))**2

    # 屏障：beta1 * softplus(beta2 * (d_s - d_t))
    barrier = beta1 * F.softplus(beta2 * (d_s - d_t), beta=1.0)

    rc = buf + barrier
    return {"rc": rc, "d_t": d_t, "dtdot": dtdot, "idx": idx}