# depth_visibility_reward.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon, Circle

Tensor = torch.Tensor

# ---------------------------
# 工具：姿态/投影/几何
# ---------------------------




# ---------------------------
# 图像域运算（不依赖 OpenCV/Scipy）
# ---------------------------

def crop_window(Z: Tensor, center_uv: Tensor, radius: int) -> Tensor:
    """
    Z: [N,H,W], center_uv: [N,2] (u,v) 像素, radius: int
    返回每个样本的局部窗口 [N, h', w']（不等长不好批处理；这里做 pad 然后 gather）
    我们只返回窗口内像素的张量与合法mask，用于统计分位/中位。
    """
    N, H, W = Z.shape
    device = Z.device
    r = int(max(1, radius))
    # 网格
    ys = torch.arange(-r, r+1, device=device)
    xs = torch.arange(-r, r+1, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [K,K]
    Kk = grid_x.numel()
    # 中心索引
    u = center_uv[:,0].round().long()
    v = center_uv[:,1].round().long()
    # 加偏移
    uu = (u.view(N,1,1) + grid_x.view(1,2*r+1,2*r+1))
    vv = (v.view(N,1,1) + grid_y.view(1,2*r+1,2*r+1))
    # 合法 mask
    mask = (uu>=0) & (uu<W) & (vv>=0) & (vv<H)
    # 索引映射到线性
    idx = (vv.clamp(0,H-1) * W + uu.clamp(0,W-1))
    Z_flat = Z.view(N, -1)
    patch = torch.gather(Z_flat, 1, idx.view(N, -1))  # [N, Kk]
    mask_flat = mask.view(N, -1)
    return patch, mask_flat


def percentile(patch: Tensor, mask: Tensor, q: float) -> Tensor:
    """
    带掩码分位数（近似）：把无效位置替换为 +inf/+large，使用 torch.kthvalue 近似分位。
    patch: [N,K], mask: [N,K], q in [0,100]
    """
    N, K = patch.shape
    valid = mask
    num = valid.sum(dim=1).clamp(min=1)
    k = ( (q/100.0) * (num.float()-1) ).round().long() + 1  # kthvalue 是 1-based
    large = torch.finfo(patch.dtype).max
    x = patch.clone()
    x[~valid] = large
    # 对每个样本分别取前 k 小
    # 简便起见：排序后取索引（小 N, K 下足够快；批量可改 topk）
    xs, _ = x.sort(dim=1)
    # 需要防越界
    k_idx = torch.clamp(k-1, 0, K-1).view(N,1).expand(N, K)
    gather_idx = torch.arange(K, device=patch.device).view(1,K).expand(N,K)
    # 这里其实只需 xs[range(N), k-1]，构造一下：
    out = xs[torch.arange(N, device=patch.device), (k-1).clamp(0,K-1)]
    return out


def depth_grad_edges(Z: Tensor, tau: float) -> Tensor:
    """
    简单双边 Sobel/Scharr 近似：对深度做梯度，阈值成边缘。
    Z: [N,H,W]  -> E: [N,H,W] bool
    """
    device = Z.device
    # Sobel 核
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=Z.dtype, device=device)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=Z.dtype, device=device)
    kx = kx.view(1,1,3,3)
    ky = ky.view(1,1,3,3)
    Z1 = Z.unsqueeze(1)
    Gx = torch.nn.functional.conv2d(Z1, kx, padding=1)
    Gy = torch.nn.functional.conv2d(Z1, ky, padding=1)
    mag = (Gx.square() + Gy.square()).sqrt().squeeze(1)
    return mag > tau


def visible_ratio_in_bbox(Z: Tensor, bbox_xyxy: Tensor, d_t: Tensor, delta: float) -> Tensor:
    """
    目标框内“非更近物体”比例（可见比例）
    bbox_xyxy: [N,4]: (x1,y1,x2,y2)
    """
    N, H, W = Z.shape
    device = Z.device
    x1 = bbox_xyxy[:,0].round().long().clamp(0, W-1)
    y1 = bbox_xyxy[:,1].round().long().clamp(0, H-1)
    x2 = bbox_xyxy[:,2].round().long().clamp(0, W-1)
    y2 = bbox_xyxy[:,3].round().long().clamp(0, H-1)

    ratios = []
    for i in range(N):
        xs = slice(x1[i].item(), x2[i].item()+1)
        ys = slice(y1[i].item(), y2[i].item()+1)
        roi = Z[i, ys, xs]
        if roi.numel() == 0:
            ratios.append(torch.tensor(0.0, device=device, dtype=Z.dtype))
        else:
            vis = (roi >= (d_t[i] - delta))
            ratios.append(vis.float().mean())
    return torch.stack(ratios, dim=0)


def line_strip_samples(Z: Tensor, p0: Tensor, p1: Tensor, width: int) -> Tuple[Tensor, Tensor]:
    """
    在当前帧深度图上，从 p0 到 p1 采样一条“条带”矩形，返回该区域的深度与mask。
    p0, p1: [N,2] 像素坐标；width: 条带宽度（像素）
    返回 (vals, mask)，其中 vals 拼接所有采样点 [N,K]，mask 同形状
    """
    N, H, W = Z.shape
    device = Z.device
    # 生成沿线的 K 个中心采样点（整像素步进）
    vec = (p1 - p0)                       # [N,2]
    length = vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    steps = length.round().long().clamp(min=1)  # 每像素一步
    # 对每个样本构造 0..steps_i 的参数
    max_steps = steps.max().item()
    t = torch.arange(max_steps+1, device=device).view(1,-1)  # [1,S+1]
    t = t.expand(N, -1)                                      # [N,S+1]
    step_mask = (t <= steps)                                 # [N,S+1]
    dirn = vec / length
    centers = p0.unsqueeze(1) + dirn.unsqueeze(1) * t.unsqueeze(-1)  # [N,S+1,2]

    # 宽度方向：取法向量
    nvec = torch.stack([-dirn[:,1], dirn[:,0]], dim=-1)      # 旋转90度
    half = width//2
    offs = torch.arange(-half, half+1, device=device)        # [-w/2 .. w/2]
    offs = offs.view(1,1,-1,1)                               # [1,1,W,1]
    band = centers.unsqueeze(2) + nvec.unsqueeze(1).unsqueeze(2)*offs  # [N,S+1,W,2]

    u = band[...,0].round().long()
    v = band[...,1].round().long()

    valid = (u>=0)&(u<W)&(v>=0)&(v<H)&step_mask.unsqueeze(-1)
    idx = (v.clamp(0,H-1)*W + u.clamp(0,W-1))
    Z_flat = Z.view(N, -1)
    vals = torch.gather(Z_flat, 1, idx.view(N,-1))
    return vals, valid.view(N,-1)


def sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


@dataclass
class RewardParams:
    # 净空/阈值
    delta: float = 0.15          # m, 深度噪声/目标厚度分位
    alpha: float = 5.0           # Sigmoid 斜率（净空->分数）
    # FOV / 量程
    theta_max_deg: float = 60.0
    d_max: float = 15.0
    # 条带预测
    H: int = 6                   # horizon steps
    dt: float = 0.08             # 控制周期（与像素速度定义一致）
    strip_width: int = 8
    # 视差
    kappa: float = 0.015         # 像素速率->角速率缩放的gain
    f_px: float = 800.0          # 焦距（像素），用于像素速率归一化；缺省值可粗估
    # 边缘阈值
    edge_tau: float = 0.02
    gap_norm: float = 24.0       # V_gap = tanh(dist_px / gap_norm)
    # 近障
    d_safe: float = 0.8
    # 组合权重
    w1: float = 1.0
    w2: float = 0.7
    w3: float = 0.5
    w4: float = 0.6
    w5: float = 0.5
    w6: float = 0.2
    w7: float = 0.3
    lambda_near: float = 0.5
    lambda_low: float = 0.2
    eta: float = 0.2
    r_min: float = -2.0
    r_max: float = 2.0


class DepthVisibilityReward:
    def __init__(self, params: RewardParams):
        self.p = params

    @torch.no_grad()
    def __call__(self,
                 Z: Tensor,                # [N,H,W] 深度图
                 target_uvd: Tensor,       # [N,3] (u,v,d)
                 target_uvd_prev: Tensor,  # [N,3]
                 bbox_xyxy: Optional[Tensor] = None
                 ) -> Dict[str, Tensor]:
        """
        输入:
            Z: [N,H,W] 深度(米)
            target_uvd: [N,3] 当前目标 (u,v,d)
            target_uvd_prev: [N,3] 上一帧目标 (u,v,d)
            bbox_xyxy: [N,4] 目标框 (可选)
        返回: dict，各奖励分量和总 reward
        """
        N, H, W = Z.shape
        device = Z.device
        p = self.p

        target_uv = target_uvd[:, :2]
        d_t = target_uvd[:, 2]
        target_uv_prev = target_uvd_prev[:, :2]

        # 没 bbox 就建个固定小框
        if bbox_xyxy is None:
            box_half = 16.0
            bbox_xyxy = torch.stack([
                (target_uv[:,0]-box_half).clamp(0, W-1),
                (target_uv[:,1]-box_half).clamp(0, H-1),
                (target_uv[:,0]+box_half).clamp(0, W-1),
                (target_uv[:,1]+box_half).clamp(0, H-1),
            ], dim=-1)

        # --- 单帧净空 ---
        radius = self._patch_radius_from_bbox(bbox_xyxy)
        patch, mask = crop_window(Z, target_uv, radius=radius)
        d_min = percentile(patch, mask, q=10.0)  # 更近物体分位数
        c = d_t - d_min
        V_los = sigmoid(p.alpha * ((c - p.delta) / max(p.delta, 1e-3)))
        V_fov = (1 - (d_t / p.d_max)).clamp(min=0.0)
        V = V_los * V_fov

        # --- gap: 最近边缘
        E = depth_grad_edges(Z, tau=p.edge_tau)
        E_patch, Em = crop_window(E.float(), target_uv, radius=radius)
        near_mask = patch < (d_t.view(-1,1) - p.delta)
        near_edge = (near_mask & (E_patch>0.5) & Em & mask)
        V_gap = []
        for i in range(N):
            if not torch.any(near_edge[i]):
                V_gap.append(torch.tensor(1.0, device=device))
            else:
                r = radius
                ys, xs = torch.where(near_edge[i].view(2*r+1,2*r+1))
                dy, dx = (ys-r).float(), (xs-r).float()
                dist = torch.sqrt(dx*dx+dy*dy).min()
                V_gap.append(torch.tanh(dist / p.gap_norm))
        V_gap = torch.stack(V_gap, dim=0)

        # --- 短视界 ---
        v_rel = target_uv - target_uv_prev
        tiny = (v_rel.norm(dim=-1)<1.0).float().view(N,1)
        v_rel = v_rel + tiny*torch.tensor([1.0,0.0], device=device).view(1,2)

        V_list, c_list = [], []
        for k in range(1, p.H+1):
            p_tau = target_uv + v_rel * k
            vals, vm = line_strip_samples(Z, target_uv, p_tau, width=p.strip_width)
            d_min_tau = self._masked_percentile(vals, vm, q=5.0)
            patch_tau, mask_tau = crop_window(Z, p_tau, radius=radius)
            d_t_tau = self._masked_median(patch_tau, mask_tau)
            c_tau = d_t_tau - d_min_tau
            V_tau = sigmoid(p.alpha*((c_tau-p.delta)/max(p.delta,1e-3)))
            V_list.append(V_tau); c_list.append(c_tau)

        V_list = torch.stack(V_list,1)
        c_list = torch.stack(c_list,1)
        V_avg = V_list.mean(1); V_min = V_list.min(1).values
        le_mask = (c_list<=0.0)
        first_idx = torch.where(le_mask.any(1),
                                le_mask.float().argmax(1),
                                torch.full((N,),p.H-1,device=device))
        tau_star = (first_idx.float()+1)*p.dt
        R_tto = torch.minimum(torch.ones_like(V_avg), tau_star/(p.H*p.dt))

        # --- 视差代理 ---
        speed_px = (target_uv-target_uv_prev).norm(dim=-1)/max(p.dt,1e-6)
        R_parallax = torch.tanh(p.kappa*(speed_px/max(p.f_px,1e-6)))

        # --- 框内可见比例 ---
        R_vis_area = visible_ratio_in_bbox(Z, bbox_xyxy, d_t, p.delta)

        # --- 近障惩罚 ---
        Z_flat = Z.view(N,-1)
        q05 = Z_flat.kthvalue((Z_flat.shape[1]*5)//100+1,1).values
        P_near = (q05<p.d_safe).float()

        # --- 组合 ---
        reward = (p.w1*V + p.w2*V_avg + p.w3*V_min + p.w4*R_tto +
                  p.w5*V_gap + p.w6*R_parallax + p.w7*R_vis_area
                  - p.lambda_near*P_near - p.lambda_low*(V<p.eta).float()
                 ).clamp(p.r_min,p.r_max)

        return {"reward":reward,"V":V,"V_avg":V_avg,"V_min":V_min,
                "R_tto":R_tto,"V_gap":V_gap,"R_parallax":R_parallax,
                "R_vis_area":R_vis_area,"P_near":P_near,"d_t":d_t,"c":c}
