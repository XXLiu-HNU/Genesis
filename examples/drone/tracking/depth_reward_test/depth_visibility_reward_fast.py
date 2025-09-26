# depth_visibility_reward_fast_gpu.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Tensor = torch.Tensor

@dataclass
class FastRewardParams:
    f_px: float = 200.0    # 焦距像素
    R_world: float = 0.2   # 目标物理半径 (m)
    delta: float = 0.1     # 容忍深度差 (m)
    H: int = 3             # 预测步长
    dt: float = 0.1
    w_v: float = 1.0
    w_tto: float = 1.0
    v_thresh: float = 0.5  # 认为“将被遮挡”的阈值（未来可见度 < v_thresh）

class DepthVisibilityRewardFast:
    def __init__(self, p: FastRewardParams):
        self.p = p
        self._cached_shape: Optional[Tuple[int,int,torch.device]] = None
        self._X = None  # [1,1,1,W]
        self._Y = None  # [1,1,H,1]

    def _ensure_grid(self, H: int, W: int, device: torch.device):
        key = (H, W, device)
        if self._cached_shape == key:
            return
        # 只建一次整图网格（广播友好形状）
        Y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        X = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
        self._Y, self._X = Y, X
        self._cached_shape = key

    @torch.no_grad()
    def __call__(self, Z: Tensor, target_uvd: Tensor, target_uvd_prev: Tensor) -> Dict[str, Tensor]:
        """
        Z: [N,H,W] (m)
        target_uvd: [N,3] (u,v,d)
        target_uvd_prev: [N,3]
        """
        N, H, W = Z.shape
        device = Z.device
        p = self.p

        self._ensure_grid(H, W, device)

        # 当前中心/深度与上一帧
        u = target_uvd[:, 0].view(N, 1, 1, 1)   # [N,1,1,1]
        v = target_uvd[:, 1].view(N, 1, 1, 1)
        d = target_uvd[:, 2].view(N, 1, 1, 1)
        u_prev = target_uvd_prev[:, 0].view(N, 1, 1, 1)
        v_prev = target_uvd_prev[:, 1].view(N, 1, 1, 1)

        # 半径 r(d) ~ f_px * R_world / d
        r = (p.f_px * p.R_world / torch.clamp(d, min=1e-6)).clamp(min=3.0)  # 像素
        r2 = r * r

        # 整图上的“障碍布尔图”：更近于目标 (严格 < d - delta) 的像素为 True
        obstacle = (Z.view(N, 1, H, W) < (d - p.delta))

        # ---- 当前可见度 V（圆内遮挡占比） ----
        dist2_now = (self._X - u) ** 2 + (self._Y - v) ** 2            # [N,1,H,W]
        mask_now = dist2_now <= r2                                     # [N,1,H,W]
        occ_now = (obstacle & mask_now).sum(dim=(2, 3))                # [N,1]
        area_now = mask_now.sum(dim=(2, 3)).clamp(min=1)               # [N,1]
        V = 1.0 - (occ_now.float() / area_now.float())                # [N,1] in [0,1]
        V = V.squeeze(1)                                               # [N]

        # ---- 未来 H 步的中心（并行计算） ----
        v_rel_u = (u - u_prev)                                         # [N,1,1,1]
        v_rel_v = (v - v_prev)
        k = torch.arange(1, p.H + 1, device=device, dtype=torch.float32).view(1, p.H, 1, 1)
        u_next = u + v_rel_u * k                                       # [N,H,1,1]
        v_next = v + v_rel_v * k                                       # [N,H,1,1]

        # 未来每步的半径（深度不变时复用当前 r）
        r2_next = r2.expand(N, p.H, 1, 1)                              # [N,H,1,1]

        # 圆掩码并行生成
        dist2_next = (self._X - u_next) ** 2 + (self._Y - v_next) ** 2 # [N,H,H,W]
        mask_next = dist2_next <= r2_next                               # [N,H,H,W]

        # 越界处（中心跑出图外），掩码全部置 False
        out_h = (u_next < 0) | (u_next >= W) | (v_next < 0) | (v_next >= H)  # [N,H,1,1]
        if out_h.any():
            mask_next = torch.where(out_h, torch.zeros_like(mask_next, dtype=torch.bool), mask_next)

        # 复制 obstacle 到 H 维度（或用广播不复制：这里显式 expand）
        obstacle_H = obstacle.expand(N, p.H, H, W)                     # [N,H,H,W]

        occ_next = (obstacle_H & mask_next).sum(dim=(2, 3))            # [N,H]
        area_next = mask_next.sum(dim=(2, 3)).clamp(min=1)             # [N,H]
        V_future = 1.0 - (occ_next.float() / area_next.float())        # [N,H]

        # TTO：首次 V_future < v_thresh 的归一化位置（没有则=1）
        oc_mask = (V_future < p.v_thresh)                              # [N,H]
        any_oc = oc_mask.any(dim=1)
        first_idx = torch.where(
            any_oc,
            oc_mask.float().argmax(dim=1) + 1,                         # 1..H
            torch.full((N,), p.H, device=device, dtype=torch.long)
        )
        TTO = first_idx.float() / float(p.H)                           # [N]

        # 奖励
        reward = p.w_v * V + p.w_tto * TTO                             # [N]

        return {"reward": reward, "V": V, "TTO": TTO}

    # --- 可视化：当前圆 + 未来 H 步圆 + 二值遮挡图 ---
    @torch.no_grad()
    def visualize_sample(self, Z: Tensor, target_uvd: Tensor, target_uvd_prev: Tensor,
                         idx: int = 0, show_mask: bool = True, title: Optional[str] = None):
        p = self.p
        device = Z.device
        N, H, W = Z.shape
        self._ensure_grid(H, W, device)

        out = self(Z, target_uvd, target_uvd_prev)
        u, v, d = target_uvd[idx].tolist()
        u_prev, v_prev, _ = target_uvd_prev[idx].tolist()
        r_px = float(max(3.0, p.f_px * p.R_world / max(d, 1e-6)))

        Zi = Z[idx].detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2 if show_mask else 1, figsize=(10 if show_mask else 6, 5))

        if not show_mask:
            axes = [axes]

        # 左：原深度 + 圆
        ax0 = axes[0]
        im0 = ax0.imshow(Zi, cmap='viridis')
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label='Depth (m)')
        ax0.add_patch(plt.Circle((u, v), r_px, fill=False, color='cyan', lw=2, label='current ROI'))
        ax0.plot(u, v, 'yo', label='target')
        ax0.plot(u_prev, v_prev, 'go', label='prev')
        # 未来 H 步
        du, dv = (u - u_prev), (v - v_prev)
        for k in range(1, p.H + 1):
            cx, cy = u + du * k, v + dv * k
            ax0.add_patch(plt.Circle((cx, cy), r_px, fill=False, color='orange', alpha=0.35))
            ax0.plot(cx, cy, 'rx', ms=5)
        ax0.set_xlim([0, W]); ax0.set_ylim([H, 0])
        ax0.set_title(title or f"reward={out['reward'][idx]:.2f}, V={out['V'][idx]:.2f}, TTO={out['TTO'][idx]:.2f}")
        ax0.legend(loc='upper right', fontsize=9, framealpha=0.7)

        # 右：按 (Z < d - delta) 的二值遮挡图（你之前要的“更近=白”的图）
        if show_mask:
            occ_mask = (Z[idx] < (d - p.delta)).float().cpu().numpy()
            ax1 = axes[1]
            im1 = ax1.imshow(occ_mask, cmap='gray', vmin=0, vmax=1)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=f"Z < d - δ (δ={p.delta} m)")
            ax1.add_patch(plt.Circle((u, v), r_px, fill=False, color='cyan', lw=2))
            ax1.set_xlim([0, W]); ax1.set_ylim([H, 0])
            ax1.set_title("Occluders (closer than target)")

        plt.tight_layout()
        plt.show()
