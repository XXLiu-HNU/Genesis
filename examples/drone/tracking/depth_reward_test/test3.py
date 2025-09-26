import torch
from depth_io import load_depth_batch
from depth_visibility_reward_fast import DepthVisibilityRewardFast, FastRewardParams

# 建议直接放到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Z = load_depth_batch("depth_0001.png", assume_max_depth=12.0).to(device)  # [N,H,W]
target_uvd = torch.tensor([[32.0, 44.0, 5.0]], device=device)
target_uvd_prev = torch.tensor([[30.0, 34.0, 2.0]], device=device)

rmod = DepthVisibilityRewardFast(FastRewardParams(H=3, f_px=200.0, R_world=0.2, delta=0.1))
out = rmod(Z, target_uvd, target_uvd_prev)
print(out)

# 直接可视化（含右侧“Z<d-δ”的遮挡二值图）
rmod.visualize_sample(Z, target_uvd, target_uvd_prev, idx=0)
