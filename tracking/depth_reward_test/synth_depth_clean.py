# -*- coding: utf-8 -*-
# synth_depth_clean.py
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

Tensor = torch.Tensor

@dataclass
class CleanSynthCfg:
    # 相机内参（像素）
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 320.0
    cy: float = 240.0

    # 相机高度（相机在世界系的 Y=cam_height）
    cam_height: float = 1.2  # m （地面在 Y=0）

    # 圆柱数量与范围（世界系：X 左右、Z 前后、Y 上下）
    n_min: int = 3
    n_max: int = 5
    x_range: Tuple[float, float] = (-2.5, 2.5)    # m
    z_range: Tuple[float, float] = (2.0, 12.0)    # m
    r_range: Tuple[float, float] = (0.35, 0.65)   # m
    h_range: Tuple[float, float] = (1.8, 2.2)     # m  # 高度（从地面起）

    # 开关
    use_ground: bool = True
    max_depth: float = 20.0
    add_noise: bool = False
    noise_sigma: float = 0.0
    add_holes: bool = False
    hole_prob: float = 0.0

    seed: Optional[int] = None


def _meshgrid(H: int, W: int, K: Tensor, device=None):
    if device is None:
        device = K.device
    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    v, u = torch.meshgrid(ys, xs, indexing='ij')  # 图像坐标
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x_n = (u - cx) / fx     # 归一化 x
    y_n_img = (v - cy) / fy # 像素向下为正
    # 采用相机坐标 Y 向上，为了与常见机器人/三维一致，这里把像素y取反
    y_n = -y_n_img
    return x_n, y_n  # 射线方向 d = (x_n, y_n, 1)


def _sample_cylinders(cfg: CleanSynthCfg, N: int, device):
    g = torch.Generator(device=device)
    if cfg.seed is not None:
        g.manual_seed(cfg.seed)
    urand = lambda *sz: torch.rand(*sz, generator=g, device=device)

    n_obs = torch.randint(cfg.n_min, cfg.n_max + 1, (N,), generator=g, device=device)
    M = int(n_obs.max().item())

    def uni(lo, hi):
        return lo + (hi - lo) * urand(N, M)

    cx = uni(cfg.x_range[0], cfg.x_range[1])   # 圆柱中心 X
    cz = uni(cfg.z_range[0], cfg.z_range[1])   # 圆柱中心 Z
    r  = uni(cfg.r_range[0], cfg.r_range[1])   # 半径
    h  = uni(cfg.h_range[0], cfg.h_range[1])   # 高度（底面在地面）
    # 有效 mask
    idx = torch.arange(M, device=device).view(1, M).expand(N, M)
    valid = (idx < n_obs.view(N, 1))
    return cx, cz, r, h, valid


@torch.no_grad()
def render_depth_clean(
    N: int, H: int, W: int, cfg: CleanSynthCfg,
    device: Optional[torch.device] = None
) -> Tensor:
    """
    返回深度图 [N,H,W]。坐标：
      - 相机位于世界系 (0, cam_height, 0)，朝 +Z；
      - 射线方向 d = (x_n, y_n, 1)，其中 y_n 是“向上为正”的归一化像素。
      - 地面：Y=0；圆柱：轴向平行 Y，底面在 Y=0，顶面在 Y=h。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K = torch.tensor([[cfg.fx, 0., cfg.cx],
                      [0., cfg.fy, cfg.cy],
                      [0., 0., 1.]], device=device, dtype=torch.float32)

    # 网格
    x_n, y_n = _meshgrid(H, W, K, device)  # [H,W]
    x_n = x_n.view(1, 1, H, W)
    y_n = y_n.view(1, 1, H, W)

    # 随机圆柱
    cx, cz, r, h, valid = _sample_cylinders(cfg, N, device)
    N, M = cx.shape
    cx_, cz_, r_, h_, valid_ = [t.view(N, M, 1, 1) for t in (cx, cz, r, h, valid.float())]

    # 圆柱侧面的射线交（忽略高度）：
    # 在相机坐标中，相机位于原点，射线 p(s) = s * d, d = (x_n, y_n, 1)
    # 圆柱方程（轴向Y）：(X - cx)^2 + (Z - cz)^2 = r^2
    # 将 X = s*x_n, Z = s → (s*x_n - cx)^2 + (s - cz)^2 = r^2
    a = (x_n**2 + 1.0)                 # [1,1,H,W]
    b = -2.0 * (x_n * cx_ + cz_)       # [N,M,H,W]
    c = (cx_**2 + cz_**2 - r_**2)      # [N,M,H,W]
    disc = b*b - 4*a*c
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    denom = 2.0 * a
    s1 = (-b - sqrt_disc) / denom
    s2 = (-b + sqrt_disc) / denom
    s_cyl = torch.where(s1 > 0, s1,
              torch.where(s2 > 0, s2, torch.full_like(s1, float("inf"))))

    # 高度裁剪：世界Y坐标 = cam_height + s*y_n
    # 圆柱可见高度范围：Y ∈ [0, h]  （底面在地面）
    Y_at = cfg.cam_height + s_cyl * y_n   # [N,M,H,W]
    in_height = (Y_at >= 0.0) & (Y_at <= h_) & (valid_ > 0.5)
    s_cyl = torch.where(in_height, s_cyl, torch.full_like(s_cyl, float("inf")))
    s_cyl_min = s_cyl.min(dim=1).values    # [N,H,W]

    # 地面射线交：cam_height + s*y_n = 0  →  s = -cam_height / y_n
    if cfg.use_ground:
        eps = 1e-6
        s_ground = -cfg.cam_height / (y_n + eps)              # [1,1,H,W]
        ground = torch.where((s_ground > 0) & (y_n < 0),      # 注意：像素下半部分 y_n<0 才看到地面
                             s_ground, torch.full_like(s_ground, float("inf")))
        ground = ground.expand(N, 1, H, W).squeeze(1)         # [N,H,W]
        depth = torch.minimum(s_cyl_min, ground)
    else:
        depth = s_cyl_min

    # 裁到远平面
    depth = torch.minimum(depth, torch.full_like(depth, cfg.max_depth))

    # 可选噪声/空洞
    if cfg.add_noise and cfg.noise_sigma > 0:
        depth = depth + torch.randn_like(depth) * cfg.noise_sigma
    if cfg.add_holes and cfg.hole_prob > 0:
        holes = torch.rand_like(depth) < cfg.hole_prob
        depth = torch.where(holes, torch.full_like(depth, cfg.max_depth), depth)

    return depth.clamp(0.0, cfg.max_depth).contiguous()
