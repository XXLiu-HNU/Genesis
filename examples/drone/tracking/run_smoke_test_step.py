#!/usr/bin/env python3
import sys, os, traceback
sys.path.insert(0, os.path.join(os.getcwd(), 'examples/drone/tracking'))
import genesis as gs
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from track_env import TrackerEnv


def main():
    try:
        gs.init()
    except Exception:
        pass

    env_cfg = {
        "num_actions": 4,
        "simulate_action_latency": False,
        "episode_length_s": 5.0,
        "max_visualize_FPS": 30,
        "visualize_camera": False,
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],
        "train_mode": True,
        "clip_actions": 1.0,
        "termination_if_pitch_greater_than": 3.14,
        "termination_if_roll_greater_than": 3.14,
        "termination_if_close_to_ground": -1.0,
        "termination_if_x_greater_than": 100.0,
        "termination_if_y_greater_than": 100.0,
        "termination_if_z_greater_than": 100.0,
    }
    obs_cfg = {"num_obs": 50, "obs_scales": {"max_diff": 1.0, "max_lin": 1.0, "max_ang": 1.0}}
    reward_cfg = {"reward_scales": {"smooth": 1.0}}

    env = TrackerEnv(1, env_cfg, obs_cfg, reward_cfg, show_viewer=False)

    # do reset
    _ = env.reset()

    # step the scene multiple times to ensure sensors report valid readings
    max_steps = 8
    depths_raw = env.tracker_sensor.read_image()
    d_raw = depths_raw.detach().cpu().numpy()
    if np.all(d_raw == 0):
        print('raw all zero after reset, stepping scene to update sensors...')
        for i in range(max_steps):
            env.scene.step()
            depths_raw = env.tracker_sensor.read_image()
            d_raw = depths_raw.detach().cpu().numpy()
            print(f'  step {i+1}/{max_steps}: raw min={d_raw.min():.6f}, max={d_raw.max():.6f}, mean={d_raw.mean():.6f}')
            if not np.all(d_raw == 0):
                print('  got non-zero raw depths after step', i+1)
                break
        else:
            print('Warning: raw depths remained all zero after', max_steps, 'steps')
    else:
        print('raw depths non-zero immediately after reset')
    depths_aug_feats, depths_aug, mask = env.img_proc.process(depths_raw, training=True, seed=12345)

    outdir = os.path.join(os.getcwd(), 'examples', 'drone', 'tracking', 'output')
    os.makedirs(outdir, exist_ok=True)

    # prepare arrays
    d_raw = depths_raw.detach().cpu().numpy()
    d_aug = depths_aug.detach().cpu().numpy()
    m = mask.detach().cpu().numpy()

    maxr = float(env.img_proc.max_range)
    # if raw all zeros, map zeros to ones for visualization; but we want to see actual values now
    if np.all(d_raw == 0):
        print('raw still all zero after one step')
        d_raw_n = np.ones_like(d_raw)
    else:
        d_raw_n = np.clip(d_raw / maxr, 0.0, 1.0)
    d_aug_n = np.clip(d_aug / maxr, 0.0, 1.0)

    # plot side-by-side with mask overlay
    cmap_name = 'magma'
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    im0 = axes[0].imshow(d_raw_n[0], cmap=cmap_name, vmin=0, vmax=1)
    axes[0].set_title('Raw depth')
    axes[0].axis('off')

    im1 = axes[1].imshow(d_aug_n[0], cmap=cmap_name, vmin=0, vmax=1)
    invalid = ~m[0].astype(bool)
    if invalid.any():
        overlay = np.zeros((invalid.shape[0], invalid.shape[1], 4), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = invalid * 0.4
        axes[1].imshow(overlay, interpolation='nearest', zorder=3)  # 确保覆盖在上层
    axes[1].set_title('Augmented ')
    axes[1].axis('off')

    fig.suptitle('Depth compare')

    # 共享一个 colorbar，绑定到图像 mappable（im1 或 im0 都可）
    cbar = fig.colorbar(im1, ax=axes, location='right', fraction=0.05, pad=0.02)
    cbar.set_label('Normalized depth')

    out_path = os.path.join(outdir, 'depth_compare_step.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


    print('Saved', out_path)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
