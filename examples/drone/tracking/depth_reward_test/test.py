# demo_with_depth.py
import torch
from depth_io import load_depth_batch
from depth_visibility_reward import DepthVisibilityReward, RewardParams

# 1) 读一张深度图
Z = load_depth_batch("depth_0001.png", assume_max_depth=20.0)  # [N,H,W]
Z = Z.to('cpu')
N, H, W = Z.shape

# 2) 指定目标位置 (u,v) 和深度 d
u, v = int(W*0.3), int(H*0.35)
d = Z[0, v, u].item()   # 从深度图采样，或用你 tracker 的估计
print(f"depth at pixel (u,v)=({u},{v}): {d:.3f} m")
d = 2
print(f"target pixel (u,v)=({u},{v}), depth={d:.3f} m")
target_uvd = torch.tensor([[u, v, d]], dtype=torch.float32)

# 3) 上一帧目标位置（随便偏移一下）
target_uvd_prev = target_uvd.clone()
target_uvd_prev[:,0] -= 3.0   # u 左移 3 像素
target_uvd_prev[:,1] -= 0.5   # v 上移 0.5 像素

# 4) 构造小 bbox（只用于可视化）
bbox_size = 20
bbox = torch.stack([
    target_uvd[:,0]-bbox_size/2, target_uvd[:,1]-bbox_size/2,
    target_uvd[:,0]+bbox_size/2, target_uvd[:,1]+bbox_size/2
], dim=-1)

# 5) 计算奖励 —— 注意这里需要在 DepthVisibilityReward 里改一下：
#    如果传入 target_uvd，就直接用 d 作为目标深度
rmod = DepthVisibilityReward(RewardParams(d_max=12.0))
out = rmod(Z,
           target_uv=target_uvd[:,:2],
           target_uv_prev=target_uvd_prev[:,:2],
           bbox_xyxy=bbox,
           target_depth=target_uvd[:,2])   # <-- 新增参数

print(f"reward={out['reward'][0].item():.3f}, V={out['V'][0].item():.3f}, TTO={out['R_tto'][0].item():.2f}")

# 6) 可视化
rmod.visualize_sample(
    Z, idx=0,
    target_uv=target_uvd[:,:2],
    target_uv_prev=target_uvd_prev[:,:2],
    bbox_xyxy=bbox,
    title=f"reward={out['reward'][0].item():.3f}, V={out['V'][0].item():.3f}, TTO={out['R_tto'][0].item():.2f}"
)
