import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from track_env import TrackerEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--run_time", type=str, default=None)   
    args = parser.parse_args()

    gs.init()

    # log_dir = f"logs/{args.exp_name}"
    base_log_dir = f"logs/{args.exp_name}"

    if args.run_time:
        log_dir = os.path.join(base_log_dir, args.run_time)
    else:
        # 自动获取最新时间戳目录
        runs = [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))]
        if not runs:
            raise RuntimeError(f"No previous runs found in {base_log_dir}")
        latest_run = sorted(runs)[-1]  # 按名字排序，取最新
        log_dir = os.path.join(base_log_dir, latest_run)

    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # mark eval mode so env can print key metrics
    env_cfg["eval_mode"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = TrackerEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename="video.mp4", fps=env_cfg["max_visualize_FPS"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/drone/hover_eval.py

# Note
If you experience slow performance or encounter other issues
during evaluation, try removing the --record option.
"""
