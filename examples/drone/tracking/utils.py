import torch
import genesis as gs

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