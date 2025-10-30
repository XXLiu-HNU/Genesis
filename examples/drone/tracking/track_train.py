import argparse
import os
import pickle
import datetime
from importlib import metadata

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


def get_train_cfg(exp_name, max_iterations, resume, resume_path):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": resume,
            "resume_path": resume_path,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 90,  # degree
        "termination_if_pitch_greater_than": 90,
        "termination_if_close_to_ground": 0.1,
        "n_obstacles": 0,
        "termination_if_x_greater_than": 10.0,
        "termination_if_y_greater_than": 10.0,
        "termination_if_z_greater_than": 2.5,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        "train_mode": True,
    }

    obs_cfg = {
        # "num_obs": 33,
        "num_obs":  161, # 128 image feats + 33 other obs
        "obs_scales": {
            "max_diff": 1 / 5.0,
            "max_lin": 1 / 3.0,
            "max_ang": 1 / 3.14159,
        },
    }

    reward_cfg = {
        "reward_scales": {
            "collision": - 1.0,
            "distance_horizontal": 5.0,
            "distance_vertical": 1.0,
            "smooth": -10,
            "crash": -20.0,
            "max_speed": -0.5,
            "visibility_dir": 1,
            "visibility_obs": 1,
        },
    }

    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=301)

    # 新增参数
    parser.add_argument("--resume", action="store_true", default=False, help="resume training from checkpoint")

    parser.add_argument("--run_time", type=str, default="", help="path to run data directory (e.g., logs/drone-hovering/20230824-152345)")

    parser.add_argument("--ckpt", type=int, default=-1, help="checkpoint number to resume from (e.g., 300)")

    args = parser.parse_args()

    gs.init(logging_level="warning")

    base_log_dir = f"logs/{args.exp_name}"

    if args.resume:
        if args.run_time:
            log_dir = os.path.join(base_log_dir, args.run_time)
        else:
            # 自动获取最新时间戳目录
            runs = [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))]
            if not runs:
                raise RuntimeError(f"No previous runs found in {base_log_dir}")
            latest_run = sorted(runs)[-1]  # 按名字排序，取最新
            log_dir = os.path.join(base_log_dir, latest_run)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(base_log_dir, timestamp)


    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg = get_cfgs()

    # 确定 resume_path
    resume_path = None
    if args.resume and args.ckpt > 0:
        resume_path = f"{log_dir}/model_{args.ckpt}.pt"

    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, args.resume, resume_path)

    if args.vis:
        env_cfg["visualize_target"] = True

    # 保存 cfgs
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = TrackerEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume and args.ckpt is not None:
        runner.load(resume_path, load_optimizer=True)
        runner.current_learning_iteration = runner.current_learning_iteration  + 1 # 

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
