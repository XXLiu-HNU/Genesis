# TODO 设计奖励函数。
# 实现目标的稳定跟踪
# 增加课程学习 

import os
import torch
import math
import copy
import yaml
import genesis as gs
from pid import PIDcontroller
from odom import Odom
from genesis.utils.geom import (
    quat_to_xyz,
    xyz_to_quat,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from utils import collision_check,occlusion_check,setup_random_cylindrical_obstacles, relative_position_body, relative_velocity_body, orientation_state, obstacle_features, quat_to_rotmat

from depth_visibility_2d import visibility_and_tto_2d, Params2D

from path_search import (
    sample_free_xy, PathFollower, sample_free_xy_batch, sample_around_centers_batch
)
from roadmap import Roadmap,_line_of_sight_free

import numpy as np

import time

class TrackerEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device
        self.od_min_sq = 1.0*1.0
        self.od_max_sq = 3.0*3.0

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])


        # ! Add target search parms
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config/search.yaml"), "r") as f:
            nav_cfg = yaml.safe_load(f)

        def cfg(path, default=None):
            # 轻量 get：path 形如 "world.xy_min"
            cur = nav_cfg
            for k in path.split("."):
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur
        
        # 基本参数
        self.drone_height = float(cfg("drone.height", 1.0))
        self.world_xy_min = tuple(cfg("world.xy_min", [-10.0, -10.0]))
        self.world_xy_max = tuple(cfg("world.xy_max", [ 10.0,  10.0]))

        self.drone_radius  = float(cfg("drone.radius", 0.12))
        self.safety_margin = float(cfg("safety.margin", 0.12))

        # 规划/鲁棒性参数（从 cfg 读取）
        self.max_replan_tries = int(cfg("planner.max_replan_tries", 8))
        self.goals_per_try    = int(cfg("planner.goals_per_try", 5))


        # inflation：默认 & 最小（派生量）
        self.inflation_default = self.drone_radius + self.safety_margin
        self.inflation_min     = self.drone_radius + float(cfg("planner.inflation_min_addon", 0.06))
        self.inflation         = self.inflation_default  # 当前使用的 inflation，可在重试中调整

        # follower 参数
        self.v_max          = float(cfg("follower.v_max", 0.6))
        self.a_max          = float(cfg("follower.a_max", 1.2))
        self.warmup_time    = float(cfg("follower.warmup_time", 2.5))
        self.v_init         = float(cfg("follower.v_init", 0.08))
        self.lookahead_time = float(cfg("follower.lookahead_time", 0.5))
        self.min_lookahead  = float(cfg("follower.min_lookahead", 0.12))
        self.goal_reach_thresh = float(cfg("follower.goal_reach_thresh", 0.12))

        # ! Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(0.0, 0.0, 8.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=80,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),  
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False)
        )

        # ! Add plane
        self.scene.add_entity(gs.morphs.Plane())

        # !Add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # ! Add Tracker
        self.tracker_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.tracker_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.tracker_inv_init_quat = inv_quat(self.tracker_init_quat)
        self.tracker = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/tracker_drone_urdf/drone.urdf"))

        # ! Add Traget
        self.target_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.target_inv_init_quat = inv_quat(self.target_init_quat)
        self.target = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/target_drone_urdf/drone.urdf"))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config/pos.yaml"), "r") as file:
                self.pos_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)
        with open(os.path.join(script_dir, "config/rate.yaml"), "r") as file:
                self.rate_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)

        # ! Add odom and controller for drone
        self._setup_imu_and_controller(self.tracker, "rate", self.rate_ctrl_config)
        self._setup_imu_and_controller(self.target, "position", self.pos_ctrl_config)
        
        # ! Add obstacles
        self.n_obstacles = int(cfg("obstacles.n", 100))
        world_bounds_xyxy = (
            self.world_xy_min[0], self.world_xy_max[0],
            self.world_xy_min[1], self.world_xy_max[1],
        )
        obs, obs_xy, obs_r = setup_random_cylindrical_obstacles(self.scene, n_obstacles=self.n_obstacles, world_bounds=world_bounds_xyxy,origin_clearance = 0.0, min_distance = 2.0)

        self.obs_xy = torch.tensor(obs_xy, dtype=torch.float32, device=self.device) if len(obs_xy) > 0 else torch.zeros((0,2), dtype=torch.float32, device=self.device)
        self.obs_r  = torch.tensor(obs_r , dtype=torch.float32, device=self.device) if len(obs_r ) > 0 else torch.zeros((0,),  dtype=torch.float32, device=self.device)
        
        # ! Build scene
        self.scene.build(n_envs=num_envs)


        # 先把 obstacle/边界搬到 CPU numpy，避免重复 to('cpu')
        self.obs_xy_cpu_np = self.obs_xy.detach().to("cpu", copy=True).numpy().astype(np.float32)
        self.obs_r_cpu_np  = self.obs_r.detach().to("cpu", copy=True).numpy().astype(np.float32)

        # ===== 引入 Roadmap =====

        # 参数可按场景微调
        self.prm_num_nodes   = 1500
        self.prm_k_neighbors = 12
        self.prm_max_edge    = 2.5
        self.prm_clearance   = self.inflation_default

        self.roadmap = Roadmap.build(
            world_min=(float(self.world_xy_min[0]), float(self.world_xy_min[1])),
            world_max=(float(self.world_xy_max[0]), float(self.world_xy_max[1])),
            obs_xy=self.obs_xy_cpu_np,
            obs_r=self.obs_r_cpu_np,
            n_nodes=self.prm_num_nodes,
            k=self.prm_k_neighbors,
            max_edge_len=self.prm_max_edge,
            clearance=self.prm_clearance,
        )

        # ! Add for path searching
        self.goal_xy = [None] * self.num_envs
        self.path_wps = [None] * self.num_envs
        self.followers = [None] * self.num_envs

        from collections import deque

        self.replan_queue = deque()   # FIFO 队列存 env_id
        self.replan_inqueue = set()   # 去重
        self.max_plan_per_step = 32   # 每帧最多处理的规划请求数（可调/自适应）


        # ! Prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # ! Initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.tracker_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.tracker_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.tracker_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.tracker_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.tracker_last_pos = torch.zeros_like(self.tracker_pos)

        self.target_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.target_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.target_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.target_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.target_last_pos = torch.zeros_like(self.target_pos)
        self.initial_angle = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def plan_new_mission(self, envs_idx):
        """
        为指定的 envs 重新规划 target 的轨迹
        """
        # 统一把 envs_idx 变成 Python 列表
        if isinstance(envs_idx, torch.Tensor):
            envs_idx = envs_idx.tolist()
        elif not isinstance(envs_idx, (list, tuple)):
            envs_idx = [int(envs_idx)]

        with torch.no_grad():

            for env_id in envs_idx:
                # 当前起点（世界坐标）
                cur = self.target.get_pos()[env_id]
                start_xy = (cur[0].item(), cur[1].item())

                path_found, chosen_goal, chosen_path = False, None, None

                # 逐步放宽（注意：使用 cached 网格时，放宽参数不会改变已缓存网格，仅改变采样与重试策略）
                for attempt in range(self.max_replan_tries):
                    # 多采几个目标点以增加成功率
                    goals_xy = []
                    for _ in range(self.goals_per_try):
                        g = sample_free_xy(
                            self.world_xy_min, self.world_xy_max,
                            self.obs_xy, self.obs_r,
                            safe_radius=self.prm_clearance,
                            device=self.device
                        )
                        if isinstance(g, torch.Tensor):
                            g = (float(g[0].item()), float(g[1].item()))
                        goals_xy.append(g)

                    # 一次性查询（命中任一目标即返回路径）
                    path_np = self.roadmap.query(start_xy, goals_xy, k_attach=8)
                    if path_np is not None and path_np.shape[0] >= 2:
                        chosen_path = [(float(x), float(y)) for (x, y) in path_np.tolist()]
                        chosen_goal = chosen_path[-1]
                        path_found = True
                        break

                if path_found:
                    # 成功：写回该 env 的目标与路径
                    self.goal_xy[env_id]  = chosen_goal
                    self.path_wps[env_id] = chosen_path

                    # follower：按 env 独立创建/重置
                    if self.followers[env_id] is None:
                        self.followers[env_id] = PathFollower(
                            chosen_path, self.dt,
                            v_max=self.v_max, a_max=self.a_max,
                            warmup_time=self.warmup_time, v_init=self.v_init,
                            lookahead_time=self.lookahead_time, min_lookahead=self.min_lookahead,
                            slow_down_k=1.2
                        )
                    else:
                        self.followers[env_id].reset_with_path(chosen_path)

                else:
                    # 失败：不要 raise，悬停 + 下次再试
                    print(f"[Planner] WARNING: env {env_id} no path found, holding position.")
                    hold_xy = start_xy
                    self.goal_xy[env_id]  = hold_xy
                    self.path_wps[env_id] = [hold_xy, hold_xy]

                    if self.followers[env_id] is None:
                        self.followers[env_id] = PathFollower(
                            self.path_wps[env_id], self.dt,
                            v_max=self.v_max, a_max=self.a_max,
                            warmup_time=self.warmup_time, v_init=self.v_init,
                            lookahead_time=self.lookahead_time, min_lookahead=self.min_lookahead,
                            slow_down_k=1.2
                        )
                    else:
                        self.followers[env_id].reset_with_path(self.path_wps[env_id])

    def _collision_detect(self):
        if self.n_obstacles > 0:
            bool,_ = collision_check(self.tracker_pos, self.obs_xy, self.obs_r)
            return bool
        else:
            return False
    
    def _loss_detect(self):
        if self.n_obstacles > 0:
            bool, _, _ = occlusion_check(self.tracker_pos, self.target_pos, self.obs_xy, self.obs_r)
            return bool
        else:
            return False


    def step(self, actions):
        """
        Steps the environment forward by one time step.

        This function is responsible for updating the state of the environment based on the provided actions.
        It handles the application of actions, collision detection, loss detection, and updating of various buffers.

        Args:
            actions (torch.Tensor): The actions to be applied to the environment.
        """
        # ! -------------------------- apply actions --------------------------
        self.actions = actions
        exec_actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # 
        tracker_prop_rpms = self.tracker.controller.step(exec_actions)       # [N,4] tensor 
        self.tracker.set_propellels_rpm(tracker_prop_rpms)

        cur_pos = self.target.get_pos()   # shape [num_envs, 3]
        cur_xy = cur_pos[:, :2]           # (N,2)

        ref_pos = torch.zeros((self.num_envs, 4), device=self.device)
        ref_pos[:, 2] = self.drone_height
        ref_pos[:, 3] = 0.0

        # 1) 收集有路径跟随器的 env 索引
        has_follower = [i for i in range(self.num_envs) if self.followers[i] is not None]
        M = len(has_follower)

        if M > 0:
            # 2) 批量调用 step
            next_list = [self.followers[i].step() for i in has_follower]   # List[(x,y)]

            # 3) 只写入 x,y，不覆盖 z
            next_xy = torch.tensor(next_list, device=self.device, dtype=gs.tc_float)  # (M,2)
            idx = torch.tensor(has_follower, device=self.device, dtype=torch.long)
            ref_pos[idx, 0:2] = next_xy

            # 4) 批量判断 reached_goal
            reached = [i for i in has_follower if self.followers[i].reached_goal(thresh=self.goal_reach_thresh)]
            for env_id in reached:
                if env_id not in self.replan_inqueue:
                    self.replan_queue.append(env_id)
                    self.replan_inqueue.add(env_id)

        # 5) 没有 follower 的 env：保持当前位置
        no_follower_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if M > 0:
            no_follower_mask[idx] = False
        if no_follower_mask.any():
            ref_pos[no_follower_mask, :3] = cur_pos[no_follower_mask]  # 保持原位置 (x,y,z)
            ref_pos[no_follower_mask, 3]  = 0.0


        target_prop_rpms = self.target.controller.step(ref_pos)
        self.target.set_propellels_rpm(target_prop_rpms)

        self.scene.step()

        # ! -------------------------- update buffers --------------------------
        self.episode_length_buf += 1
        self.tracker_last_pos[:] = self.tracker_pos[:]
        self.tracker_pos[:] = self.tracker.get_pos()

        self.target_last_pos[:] = self.target_pos[:]
        self.target_pos[:] = self.target.get_pos()


        self.tracker_quat[:] = self.tracker.get_quat()
        self.tracker_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.tracker_quat) * self.tracker_inv_init_quat,
                self.tracker_quat,
            ), rpy=True, degrees=True)
        
        inv_tracker_quat = inv_quat(self.tracker_quat)
        self.tracker_lin_vel[:] = transform_by_quat(self.tracker.get_vel(), inv_tracker_quat)
        self.tracker_ang_vel[:] = transform_by_quat(self.tracker.get_ang(), inv_tracker_quat)

        self.target_quat[:] = self.target.get_quat()
        self.target_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.target_quat) * self.target_inv_init_quat,
                self.target_quat,
            ), rpy=True, degrees=True)
        
        inv_target_quat = inv_quat(self.target_quat)
        self.target_lin_vel[:] = transform_by_quat(self.target.get_vel(), inv_target_quat)
        self.target_ang_vel[:] = transform_by_quat(self.target.get_ang(), inv_target_quat)

        # Change state to relative position and velocity in body frame
        # self.rel_pos = self.target_pos - self.tracker_pos
        self.rel_pos = relative_position_body(self.tracker_pos, self.tracker_quat,self.target_pos)
        self.rel_vel = relative_velocity_body(self.tracker_lin_vel, self.target_lin_vel, self.tracker_quat)

        
        # shape == (n_envs,7) -> [x_w(3), z_w(3), cos_upright(1)]

        # check termination and reset
        # 判断终止条件
        # 1. 无人机发生碰撞
        # 2. 目标无人机丢失
        
        # ! -------------------------- check termination and reset --------------------------
        collision_flag = self._collision_detect()

        loss_flag = self._loss_detect()

        self.crash_condition = (collision_flag
                                | loss_flag
                                | (torch.abs(self.tracker_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                                | (torch.abs(self.tracker_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
                                | (self.tracker_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
                                | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
                                | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
                                | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
                                )

        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ! -------------------------- compute reward --------------------------
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            # print(f"{name} reward: {rew.mean().item():.3f}")

        # ! -------------------------- compute observations --------------------------
        
        self.rel_pos = relative_position_body(self.tracker_pos, self.tracker_quat,self.target_pos)
        self.rel_vel = relative_velocity_body(self.tracker_lin_vel, self.target_lin_vel, self.tracker_quat)
        ori_feat = orientation_state(self.tracker_quat)

        feats = obstacle_features(
            tracker_pos_w=self.tracker_pos,
            tracker_quat=self.tracker_quat,
            tracker_lin_vel_w=self.tracker_lin_vel,
            obs_xy_w=self.obs_xy,      # (N,M,2) 或 (M,2)
            obs_r=self.obs_r,          # (N,M,1) 或 (M,1)
            range_max=20.0,
            ttc_max=5.0,
            K=8,
            quat_format="xyzw",
        )

        obs_env = torch.cat([
            feats["d_min_norm"],
            feats["bearing_min_pi"],
            feats["ttc_min_norm"],
            feats["mean_clear_norm"],
            feats["heading_clear_norm"],
            feats["sector_mins_norm"],   # 若 K>0
        ], dim=-1)                       # (N, 4+3+4+1+1+K)

        eps = 1e-6
        N = self.rel_pos.size(0)

        # 2) 正确缩放 last_actions：角速度/推力分开
        #    假定 self.last_actions: (N,4) -> [p_cmd, q_cmd, r_cmd, T_cmd]
        ang_prev = torch.clamp(self.last_actions[:, :3] , -1.0, 1.0)     # (N,3)
        thrust_prev = torch.clamp(self.last_actions[:, 3:4] , -1.0, 1.0)  # (N,1)

        # 4) 主观测拼装（与原缩放风格一致）
        self.obs_buf = torch.cat([
            torch.clamp(self.rel_pos * self.obs_scales["max_diff"], -1.0, 1.0),       # (N,3)  机体系相对位置
            torch.clamp(self.rel_vel * self.obs_scales["max_lin"], -1.0, 1.0),        # (N,3)  机体系相对速度
            torch.clamp(self.tracker_ang_vel * self.obs_scales["max_ang"], -1.0, 1.0),# (N,3)  自身角速度（机体系）
            ang_prev, thrust_prev,                                                     # (N,3)+(N,1)
            ori_feat,                                                                  # (N,7)  [x_w(3), z_w(3), cos_upright(1)]
            obs_env,                                                                   # (N, 5+K)
        ], dim=-1)

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        self._drain_replan_queue()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        """
        Resets the specified environments to their initial states.

        This function is called to reset a subset of environments identified by `envs_idx`.
        It handles the resetting of the tracker and target drones' states, including their
        positions, orientations, and velocities. For the target, it also randomizes the
        parameters of its circular trajectory for the new episode.

        Args:
            envs_idx (list or torch.Tensor): The indices of the environments to reset.
        """
        if len(envs_idx) == 0:
            return

        num_resets = len(envs_idx)
        inflation = self.inflation_default

        # ! -------------------------- reset target --------------------------
        # sample new target position
        tgt_xy = sample_free_xy_batch(
            self.world_xy_min, self.world_xy_max,
            self.obs_xy, self.obs_r,
            safe_radius=inflation,
            n=num_resets, device=self.device
        )  # [n,2] # (num_resets,2)

        new_target_pos = torch.zeros((num_resets, 3), device=self.device)
        new_target_pos[:, 0] = tgt_xy[:, 0]
        new_target_pos[:, 1] = tgt_xy[:, 1]
        new_target_pos[:, 2] = self.drone_height
        
        self.target_pos[envs_idx] = new_target_pos
        self.target_last_pos[envs_idx] = new_target_pos
        self.target_quat[envs_idx] = self.target_init_quat.repeat(num_resets, 1)
        self.target.set_pos(self.target_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target.set_quat(self.target_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target_lin_vel[envs_idx] = 0.0
        self.target_ang_vel[envs_idx] = 0.0
        self.target.zero_all_dofs_velocity(envs_idx)

        # ! -------------------------- reset tracker --------------------------
        # reset range from target
        r_min, r_max = 2.0, 2.5  
        trk_xy = sample_around_centers_batch(
            tgt_xy, r_min, r_max,
            self.obs_xy, self.obs_r,
            safe_radius=inflation
        )  # [n,2]

        new_trk_pos = torch.zeros((num_resets, 3), device=self.device, dtype=torch.float32)
        new_trk_pos[:, :2] = trk_xy
        new_trk_pos[:, 2] = self.drone_height


        self.tracker_pos[envs_idx] = new_trk_pos
        self.tracker_last_pos[envs_idx] = new_trk_pos

        # face towards target
        dir_xy = tgt_xy - trk_xy  # (N,2)
        yaw = torch.atan2(dir_xy[:, 1], dir_xy[:, 0])  # (N,)
        rpy = torch.stack([
            torch.zeros_like(yaw),   # roll
            torch.zeros_like(yaw),   # pitch
            yaw                      # yaw
        ], dim=-1)  # (N,3)
        facing_quat = xyz_to_quat(rpy, rpy=True, degrees=False)

        self.tracker_quat[envs_idx] = facing_quat
        self.tracker.set_pos(self.tracker_pos[envs_idx],  zero_velocity=True, envs_idx=envs_idx)
        self.tracker.set_quat(self.tracker_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.tracker_lin_vel[envs_idx] = 0.0
        self.tracker_ang_vel[envs_idx] = 0.0
        self.tracker.zero_all_dofs_velocity(envs_idx)
        
        # ! -------------------------- reset relative position --------------------------
        self.rel_pos[envs_idx] = self.target_pos[envs_idx] - self.tracker_pos[envs_idx]

        # ! -------------------------- reset buffers ------------------------------------
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # ! -------------------------- set extras ------------------------------------
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        for eid in envs_idx:
            if int(eid) not in self.replan_inqueue:
                self.replan_queue.append(int(eid))
                self.replan_inqueue.add(int(eid))


    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def _drain_replan_queue(self, budget=None):
        """每帧消化最多 budget 个规划请求"""
        if budget is None:
            budget = self.max_plan_per_step

        batch = []
        while budget > 0 and self.replan_queue:
            eid = self.replan_queue.popleft()
            if eid in self.replan_inqueue:
                self.replan_inqueue.remove(eid)
            batch.append(eid)
            budget -= 1

        if batch:
            # 用你已经实现的批量规划（或 roadmap.query 逐个）
            self.plan_new_mission_batch(batch, max_plan_per_step=len(batch))

    def plan_new_mission_batch(self, envs_idx, max_plan_per_step=16):
        if isinstance(envs_idx, torch.Tensor):
            envs_idx = envs_idx.tolist()
        elif not isinstance(envs_idx, (list, tuple)):
            envs_idx = [int(envs_idx)]
        if len(envs_idx) == 0:
            return

        # 节流：每步只处理部分 env，避免突刺
        envs_idx = envs_idx[:max_plan_per_step]

        # 1) 批量取起点
        cur_pos = self.target.get_pos()   # [N,3]
        starts_xy = [(float(cur_pos[i,0].item()), float(cur_pos[i,1].item())) for i in envs_idx]

        # 2) 为这一批 env 共享一套候选目标（减少采样 & KNN 次数）
        goals_xy_shared = []
        for _ in range(max(8, self.goals_per_try)):  # 稍微多采一点，共享
            g = sample_free_xy(self.world_xy_min, self.world_xy_max,
                            self.obs_xy, self.obs_r,
                            safe_radius=self.prm_clearance, device=self.device)
            if isinstance(g, torch.Tensor):
                g = (float(g[0].item()), float(g[1].item()))
            goals_xy_shared.append(g)

        # 3) 逐起点跑一次 query（但共享目标 & KNN 索引）
        for idx, env_id in enumerate(envs_idx):
            start_xy = starts_xy[idx]
            chosen_path = None
            # 先试直线可达（批量目标里命中概率很高）
            los_hits = [g for g in goals_xy_shared
                        if _line_of_sight_free(
                            np.asarray(start_xy, np.float32),
                            np.asarray(g, np.float32),
                            self.roadmap.obs_xy, self.roadmap.obs_r, self.roadmap.clearance)]
            if len(los_hits) > 0:
                chosen_path = np.asarray([start_xy, los_hits[0]], dtype=np.float32)
            else:
                # 再 PRM 查询
                chosen_path = self.roadmap.query(start_xy, goals_xy_shared, k_attach=8)

            if chosen_path is not None and chosen_path.shape[0] >= 2:
                path_list = [(float(x), float(y)) for (x, y) in chosen_path.tolist()]
                self.goal_xy[env_id]  = path_list[-1]
                self.path_wps[env_id] = path_list
                if self.followers[env_id] is None:
                    self.followers[env_id] = PathFollower(
                        self.path_wps[env_id], self.dt,
                        v_max=self.v_max, a_max=self.a_max,
                        warmup_time=self.warmup_time, v_init=self.v_init,
                        lookahead_time=self.lookahead_time, min_lookahead=self.min_lookahead,
                        slow_down_k=1.2
                    )
                else:
                    self.followers[env_id].reset_with_path(self.path_wps[env_id])
            else:
                # 兜底：悬停
                hold_xy = start_xy
                self.goal_xy[env_id]  = hold_xy
                self.path_wps[env_id] = [hold_xy, hold_xy]
                if self.followers[env_id] is None:
                    self.followers[env_id] = PathFollower(
                        self.path_wps[env_id], self.dt,
                        v_max=self.v_max, a_max=self.a_max,
                        warmup_time=self.warmup_time, v_init=self.v_init,
                        lookahead_time=self.lookahead_time, min_lookahead=self.min_lookahead,
                        slow_down_k=1.2
                    )
                else:
                    self.followers[env_id].reset_with_path(self.path_wps[env_id])
    def _setup_imu_and_controller(self, drone, controller_type, config):
        """
        Sets up the IMU and controller for the drone.

        This function sets up an odometry (Odom) and a PID controller for a given drone entity.
        It attaches the created objects as attributes to the drone entity for later access.

        Args:
            drone: The drone entity for which to set up the IMU and controller.
            controller_type (str): The type of controller to be used.
            config (dict): Configuration parameters for the controller.
        """
        # Setup Odom (IMU) for the drone
        odom = Odom(
            num_envs=self.num_envs,
            device=self.device
        )
        odom.set_drone(drone)
        setattr(drone, 'odom', odom)

        # Setup PID controller for the drone
        pid = PIDcontroller(
            num_envs=self.num_envs,
            odom=drone.odom,
            config=config,
            device=self.device,
            controller=controller_type,
        )
        pid.set_drone(drone)
        setattr(drone, 'controller', pid)

    # ! -------------------------------------------- reward functions--------------------------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_action = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        smooth_attitude = torch.sum(torch.square(self.actions[:,:3]), dim=1)
        smooth_rew = smooth_action + smooth_attitude
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(-0.5 * (yaw / 0.2)**2)  # 范围 0~1
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.tracker_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
    
    def _reward_distance_horizontal(self):
        # 计算水平距离的平方
        horizontal_dist_sq = torch.sum(torch.square(self.rel_pos[:, :2]), dim=1)
        
        # 限制在合理范围，避免数值爆炸
        horizontal_dist_sq = torch.clamp(horizontal_dist_sq, min=0.0, max=25.0)
        
        # 创建掩码：判断距离是否在[od_min_sq, od_max_sq]范围内
        in_range = (horizontal_dist_sq >= self.od_min_sq) & (horizontal_dist_sq <= self.od_max_sq)
        
        # 对于超出范围的部分计算惩罚
        # 小于最小值的惩罚
        penalty_below = torch.clamp(self.od_min_sq - horizontal_dist_sq, min=0.0)** 2
        # 大于最大值的惩罚
        penalty_above = torch.clamp(horizontal_dist_sq - self.od_max_sq, min=0.0)**2
        
        # 总惩罚（只对超出范围的部分）
        total_penalty = penalty_below + penalty_above
        
        # 初始化奖励：范围内的给予固定奖励，范围外的为负惩罚
        reward = torch.where(in_range, 
                            torch.tensor(1.0, device=horizontal_dist_sq.device),  # 固定奖励值
                            -total_penalty)  # 范围外的惩罚
        
        # 最终限幅，确保奖励在合理区间
        reward = torch.clamp(reward, min=-100.0, max=1.0)  # 最大值调整为固定奖励值
        
        return reward
    
    def _reward_distance_vertical(self):
        """
        对垂直方向的距离进行奖励。
        垂直距离为0时获得最大奖励1.0，垂直距离增加时奖励递减。
        """
        # ! 1. 获取垂直方向的距离（取绝对值）
        vertical_dist = torch.abs(self.rel_pos[:, 2])
        
        # ! 2. 使用高斯奖励函数，当垂直距离为0时获得最大奖励1.0
        # sigma控制奖励随距离衰减的速度，可以根据实际需求调整
        sigma = 0.5
        reward = torch.exp(-0.5 * (vertical_dist / sigma)**2)
        
        return reward

    def _reward_max_speed(self):
        """
        对速度超过物理极限的行为进行惩罚。
        使用指数函数对超速进行强力惩罚。
        """
        # 无人机线速度存储在 self.tracker_lin_vel 中
        speed_norm = torch.norm(self.tracker_lin_vel, dim=-1)

        # ! 定义最大允许速度，例如 5 m/s
        max_speed = 5.0
    
        exceed_speed = torch.clamp(speed_norm - max_speed, min=0.0)

        speed_penalty = (exceed_speed ** 2) * 2.0  # 惩罚系数 2.0
        return speed_penalty

    def _reward_visibility_dir(self):
        """
        奖励：机头前向 与 (目标速度方向、指向目标方向) 的对齐程度（机体系）。

        要求：
        - self.rel_pos 是机体系 (target - tracker in body)
        - self.target_lin_vel 是世界系线速度
        - self.tracker_quat 可用于 world->body 旋转
        """
        eps = 1e-6

        # 1) 机体系前向向量（常量）
        f_b = torch.tensor([1.0, 0.0, 0.0], device=self.device)\
                .expand(self.num_envs, -1)                             # (N,3)

        # 2) 目标速度转到机体系
        #    R_bw: world -> body
        R_bw = quat_to_rotmat(self.tracker_quat)                       # (N,3,3)
        v_t_b = torch.bmm(R_bw, self.target_lin_vel.unsqueeze(-1))\
                .squeeze(-1)                                         # (N,3)
        v_t_dir = v_t_b / (torch.norm(v_t_b, dim=-1, keepdim=True) + eps)

        # 3) 机体系相对位置方向（已由你预先提供）
        r_b = self.rel_pos                                             # (N,3) body-frame
        r_dir = r_b / (torch.norm(r_b, dim=-1, keepdim=True) + eps)

        # 4) 两个对齐分数（余弦，范围[-1,1]）
        align_vel = torch.sum(f_b * v_t_dir, dim=-1)                   # 与目标速度方向对齐
        align_pos = torch.sum(f_b * r_dir,   dim=-1)                   # 与目标指向对齐

        # 5) 自适应权重：目标几乎不动时，更依赖位置对齐；目标速度大时，更依赖速度对齐
        #    你也可以改回常数权重 w_direction=0.5, w_yaw=0.5
        speed = torch.norm(v_t_b, dim=-1, keepdim=True)                # (N,1)
        w_dir = torch.sigmoid( (speed - 0.5) * 3.0 )                   # 速度≈0->~0, 速度高->~1
        w_yaw = 1.0 - w_dir

        # 6) 合成奖励
        visibility_reward = (w_dir.squeeze(-1) * align_vel +
                            w_yaw.squeeze(-1) * align_pos)

        return visibility_reward
    
    def _reward_visibility_obs(self):
        """
        计算并奖励无人机朝向与目标运动方向及相对位置的对齐程度。
        """
        params = Params2D(alpha=8.0, dt=self.dt, H=8, w_v=0.7, w_tto=0.3)
        out = visibility_and_tto_2d(self.obs_xy, self.obs_r, self.target_pos[:,:2], self.target_lin_vel[:,:2], self.tracker_pos[:,:2], self.tracker_lin_vel[:,:2], params)
        # print(out["V0"], out["TTO"], out["reward"])
        return out["reward"]