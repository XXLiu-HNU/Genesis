# depth_visibility_reward_uvd.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple

Tensor = torch.Tensor

def sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)

def crop_window(Z: Tensor, center_uv: Tensor, radius: int) -> Tuple[Tensor, Tensor]:
    """裁剪局部窗口并返回展平值和有效mask"""
    N, H, W = Z.shape
    device = Z.device
    ys = torch.arange(-radius, radius+1, device=device)
    xs = torch.arange(-radius, radius+1, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    u = center_uv[:,0].round().long()
    v = center_uv[:,1].round().long()
    uu = (u.view(N,1,1) + gx.view(1,2*radius+1,2*radius+1))
    vv = (v.view(N,1,1) + gy.view(1,2*radius+1,2*radius+1))
    mask = (uu>=0) & (uu<W) & (vv>=0) & (vv<H)
    idx = (vv.clamp(0,H-1) * W + uu.clamp(0,W-1))
    Z_flat = Z.view(N,-1)
    patch = torch.gather(Z_flat, 1, idx.view(N,-1))
    return patch, mask.view(N,-1)

def depth_grad_edges(Z: Tensor, tau: float) -> Tensor:
    """Sobel 边缘检测，返回 bool mask"""
    device = Z.device
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=Z.dtype, device=device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=Z.dtype, device=device).view(1,1,3,3)
    Z1 = Z.unsqueeze(1)
    Gx = F.conv2d(Z1, kx, padding=1)
    Gy = F.conv2d(Z1, ky, padding=1)
    mag = (Gx.square() + Gy.square()).sqrt().squeeze(1)
    return mag > tau

@dataclass
class RewardParams:
    delta: float = 0.15
    alpha: float = 5.0
    d_max: float = 15.0
    patch_radius: int = 8     # 局部窗口半径（像素）
    edge_tau: float = 0.02
    r_min_clip: float = -2.0
    r_max_clip: float = 2.0

class DepthVisibilityReward:
    def __init__(self, params: RewardParams):
        self.p = params

    @torch.no_grad()
    def __call__(self, Z: Tensor,
                 target_uvd: Tensor,        # [N,3]  (u,v,d)
                 target_uvd_prev: Tensor    # [N,3]
                 ) -> Dict[str, Tensor]:
        """
        Z: [N,H,W] 深度图 (meters)
        target_uvd: [N,3] 当前目标 (u,v,d)
        target_uvd_prev: [N,3] 上一帧目标 (u,v,d)
        """
        p = self.p
        N, H, W = Z.shape
        device = Z.device

        # 提取 u,v,d
        target_uv = target_uvd[:,:2]
        d_t = target_uvd[:,2]    # 直接用输入的目标深度
        target_uv_prev = target_uvd_prev[:,:2]

        # 局部窗口：找更近物体
        patch, mask = crop_window(Z, target_uv, p.patch_radius)
        big = torch.finfo(patch.dtype).max
        patch2 = patch.clone(); patch2[~mask] = big
        d_min = patch2.min(dim=1).values

        # 净空差
        c = d_t - d_min
        V_los = sigmoid(p.alpha * ((c - p.delta) / max(p.delta,1e-3)))
        V_fov = (1 - (d_t / p.d_max)).clamp(min=0.0)
        V = V_los * V_fov

        reward = V.clamp(p.r_min_clip, p.r_max_clip)

        return {"reward": reward, "V": V, "d_t": d_t, "d_min": d_min, "c": c}
