import torch
from depth_visibility_2d import visibility_and_tto_2d, Params2D

# obs
obs_centers = torch.tensor([[0.,0.],[2.,1.],[-3.,-1.]])      # [M,2]
obs_radii   = torch.tensor([1.0, 0.8, 1.2])                  # [M]

# batch envs
target_pos = torch.tensor([[3.,0.], [ -2.,  3.]])            # [N,2]
target_vel = torch.tensor([[0.5,0.2],[  0., -0.6]])
tracker_pos= torch.tensor([[-3.,0.],[  3., -2.]])
tracker_vel= torch.tensor([[0.6,0.1],[ -0.3, 0.1]])

params = Params2D(alpha=8.0, dt=0.2, H=8, w_v=0.7, w_tto=0.3)
out = visibility_and_tto_2d(obs_centers, obs_radii, target_pos, target_vel, tracker_pos, tracker_vel, params)
print(out["V0"], out["TTO"], out["reward"])
