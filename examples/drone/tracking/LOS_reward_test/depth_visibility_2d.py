# Implement a GPU-friendly 2D visibility + TTO calculator with a demo and visualization.

import torch
import math
import matplotlib.pyplot as plt

# -------------------------------
# Core: batched 2D visibility/TTO
# -------------------------------

class Params2D:
    def __init__(self,
                 alpha: float = 8.0,      # sigmoid sharpness for clearance->visibility
                 delta: float = 0.00,     # extra margin (m); clearance - delta
                 dt: float = 0.1,         # step time (s)
                 H: int = 6,              # horizon steps
                 w_v: float = 1.0,        # reward weights
                 w_tto: float = 1.0):
        self.alpha = alpha
        self.delta = delta
        self.dt = dt
        self.H = H
        self.w_v = w_v
        self.w_tto = w_tto

def _segment_circle_clearance_batch(A: torch.Tensor, B: torch.Tensor,
                                    C: torch.Tensor, R: torch.Tensor):
    """
    Compute min clearance from segment AB (per env) to circles (centers C, radii R).
    A: [N,2], B: [N,2], C: [M,2], R: [M]
    Returns:
        min_clearance: [N]
        argmin_idx: [N] index of closest circle
        all_clearance: [N,M]
    """
    # Broadcast shapes
    # A -> [N,1,2], B -> [N,1,2], C -> [1,M,2]
    A_ = A.unsqueeze(1)
    B_ = B.unsqueeze(1)
    C_ = C.unsqueeze(0)

    AB = B_ - A_                               # [N,1,2]
    AB2 = (AB**2).sum(-1, keepdim=True)       # [N,1,1]
    # Avoid div by zero
    eps = 1e-9
    AB2 = AB2 + eps

    AC = C_ - A_                               # [N,M,2]
    t = (AC * AB).sum(-1, keepdim=True) / AB2  # [N,M,1]
    t = t.clamp(0.0, 1.0)
    P = A_ + t * AB                            # closest point on segment [N,M,2]
    dist = ((C_ - P)**2).sum(-1).sqrt()        # [N,M]

    R_ = R.view(1, -1)                         # [1,M]
    clearance = dist - R_                      # [N,M]

    min_clearance, idx = clearance.min(dim=1)  # [N], [N]
    return min_clearance, idx, clearance

def visibility_and_tto_2d(obs_centers: torch.Tensor,  # [M,2]
                           obs_radii: torch.Tensor,    # [M]
                           target_pos: torch.Tensor,   # [N,2]
                           target_vel: torch.Tensor,   # [N,2]
                           tracker_pos: torch.Tensor,  # [N,2]
                           tracker_vel: torch.Tensor,  # [N,2]
                           params: Params2D) -> dict:
    """
    Compute visibility (V0) and TTO for N parallel environments with M circular obstacles.
    GPU-friendly (all batched).

    Returns dict with tensors [N]: V0, clearance0, TTO, reward, plus some extras.
    """
    device = target_pos.device
    # Current visibility via minimum clearance from segment
    min_c0, idx0, all_c0 = _segment_circle_clearance_batch(tracker_pos, target_pos,
                                                           obs_centers, obs_radii)
    # Soft visibility in [0,1]
    V0 = torch.sigmoid(params.alpha * (min_c0 - params.delta))

    # Predict future positions with constant velocity
    H = params.H
    k = torch.arange(1, H+1, device=device, dtype=target_pos.dtype).view(1, H, 1)  # [1,H,1]

    A_k = tracker_pos.unsqueeze(1) + tracker_vel.unsqueeze(1) * (params.dt * k)  # [N,H,2]
    B_k = target_pos.unsqueeze(1) + target_vel.unsqueeze(1) * (params.dt * k)    # [N,H,2]

    # Compute clearance for each horizon step
    # We'll reshape to combine N and H in one batch
    NH = A_k.shape[0] * A_k.shape[1]
    A_flat = A_k.reshape(NH, 2)
    B_flat = B_k.reshape(NH, 2)

    min_ck, idxk, all_ck = _segment_circle_clearance_batch(A_flat, B_flat, obs_centers, obs_radii)
    min_ck = min_ck.view(-1, H)  # [N,H]

    # Visibility per step
    Vk = torch.sigmoid(params.alpha * (min_ck - params.delta))  # [N,H]

    # TTO: first step where clearance < 0 (intersection) or Vk < 0.5
    oc_mask = (min_ck < 0.0) | (Vk < 0.5)
    any_oc = oc_mask.any(dim=1)
    first_idx = torch.where(any_oc,
                            oc_mask.float().argmax(dim=1) + 1,  # 1..H
                            torch.full((oc_mask.shape[0],), H, device=device, dtype=torch.long))
    TTO = first_idx.float() / float(H)

    # Reward
    reward = params.w_v * V0 + params.w_tto * TTO

    return {
        "V0": V0,
        "clearance0": min_c0,
        "TTO": TTO,
        "reward": reward,
        "Vk": Vk,              # [N,H]
        "min_ck": min_ck,      # [N,H]
        "argmin0": idx0,       # [N]
    }

# -------------------------------
# Demo: random envs + visualization (first env)
# -------------------------------

def demo_random(n_envs=64, n_obs=12, seed=24, device=None, save_path="/home/xingxun/Pictures/vis2d_demo.png"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    # World bounds (meters)
    Wmin, Wmax = -10.0, 10.0

    # Obstacles (shared across envs for simplicity)
    obs_centers = (Wmin + (Wmax - Wmin) * torch.rand((n_obs, 2), device=device, generator=g))
    obs_radii = (0.5 + 1.2 * torch.rand((n_obs,), device=device, generator=g))

    # Targets and trackers
    target_pos = (Wmin + (Wmax - Wmin) * torch.rand((n_envs, 2), device=device, generator=g))
    tracker_pos = (Wmin + (Wmax - Wmin) * torch.rand((n_envs, 2), device=device, generator=g))

    # Avoid degenerate: make sure they are not the same
    same_mask = (target_pos - tracker_pos).norm(dim=1) < 1e-2
    tracker_pos[same_mask] += torch.tensor([0.5, 0.0], device=device)

    # Velocities (m/s)
    target_vel = 1.0 * (torch.randn((n_envs, 2), device=device, generator=g))
    tracker_vel = 1.5 * (torch.randn((n_envs, 2), device=device, generator=g))

    params = Params2D(alpha=8.0, delta=0.0, dt=0.2, H=8, w_v=0.7, w_tto=0.3)

    out = visibility_and_tto_2d(obs_centers, obs_radii,
                                target_pos, target_vel,
                                tracker_pos, tracker_vel,
                                params)

    # Print basic stats
    print("V0 mean:", float(out["V0"].mean().cpu()))
    print("TTO mean:", float(out["TTO"].mean().cpu()))
    print("reward mean:", float(out["reward"].mean().cpu()))

    # Visualization for first environment
    i = 0
    fig, ax = plt.subplots(figsize=(6,6))
    # Obstacles
    for j in range(n_obs):
        cx, cy = obs_centers[j].detach().cpu().tolist()
        r = float(obs_radii[j].detach().cpu())
        circ = plt.Circle((cx, cy), r, color='gray', alpha=0.3)
        ax.add_patch(circ)
        ax.plot([cx], [cy], 'k.', ms=2)

    # Current positions
    A = tracker_pos[i].detach().cpu().numpy()
    B = target_pos[i].detach().cpu().numpy()
    ax.plot(A[0], A[1], 'bo', label='tracker')
    ax.plot(B[0], B[1], 'ro', label='target')
    ax.plot([A[0], B[0]], [A[1], B[1]], 'b--', lw=1.5, label='LOS')

    # Future segments
    Hh = params.H
    dA = (tracker_vel[i] * params.dt).detach().cpu().numpy()
    dB = (target_vel[i] * params.dt).detach().cpu().numpy()
    for k in range(1, Hh+1):
        Ak = A + dA * k
        Bk = B + dB * k
        ax.plot([Ak[0], Bk[0]], [Ak[1], Bk[1]], '-', lw=1.0, alpha=max(0.15, 0.5 - 0.05*k))

    # Title with numbers
    title = f"env #{i} | V0={out['V0'][i].item():.2f}, TTO={out['TTO'][i].item():.2f}, R={out['reward'][i].item():.2f}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.set_xlim([Wmin-1, Wmax+1])
    ax.set_ylim([Wmin-1, Wmax+1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)

    return out, save_path

# Run the demo and expose an image
# out, path = demo_random(n_envs=64, n_obs=10, seed=32)
# path
