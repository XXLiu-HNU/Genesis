import torch
from depth_io import load_depth_batch
from depth_visibility_reward_uvd import DepthVisibilityReward, RewardParams

# 1) 读取一张深度图
Z = load_depth_batch("depth_0001.png", assume_max_depth=12.0)  # [N,H,W]
Z = Z.to('cpu')
N, H, W = Z.shape

# 2) 指定目标 (u,v,d)
u, v = int(W*0.3), int(H*0.35)
d = 2
target_uvd = torch.tensor([[u, v, d]], dtype=torch.float32)
target_uvd_prev = target_uvd.clone()
target_uvd_prev[:,:2] -= torch.tensor([[3.0, 0.5]])  # 上一帧位置

# 3) 计算奖励
rmod = DepthVisibilityReward(RewardParams(d_max=12.0))
out = rmod(Z, target_uvd=target_uvd, target_uvd_prev=target_uvd_prev)
print(out)
