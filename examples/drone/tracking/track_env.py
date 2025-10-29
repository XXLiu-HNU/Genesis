import os
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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
from utils import collision_check,occlusion_check,setup_random_cylindrical_obstacles, obstacle_features, collision_reward

from depth_visibility_2d import visibility_and_tto_2d, Params2D

from path_search import (
    sample_free_xy, PathFollower, BatchedPathFollower, sample_free_xy_batch, sample_around_centers_batch
)
from roadmap import Roadmap,_line_of_sight_free
from gpu_roadmap import GPURoadmap, sample_free_goals_gpu

import numpy as np

# from genesis.sensors.raycaster.patterns import DepthCameraPattern

class TrackerEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device
        self.od_min = 1.0                                                               # minimum distance for observation
        self.od_max = 3.0                                                               # maximum distance for observation

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01                                                                  # run in 100hz
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
            cur = nav_cfg
            for k in path.split("."):
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur
        
        # parameters from config file
        self.drone_height = float(cfg("drone.height", 1.0))
        self.world_xy_min = tuple(cfg("world.xy_min", [-10.0, -10.0]))
        self.world_xy_max = tuple(cfg("world.xy_max", [ 10.0,  10.0]))

        self.drone_radius  = float(cfg("drone.radius", 0.12))
        self.safety_margin = float(cfg("safety.margin", 0.12))

        #  planning parameters
        self.max_replan_tries = int(cfg("planner.max_replan_tries", 8))
        self.goals_per_try    = int(cfg("planner.goals_per_try", 5))


        # inflation
        self.inflation_default = self.drone_radius + self.safety_margin
        self.inflation_min     = self.drone_radius + float(cfg("planner.inflation_min_addon", 0.06))
        self.inflation         = self.inflation_default

        # follower parameters
        self.v_max          = float(cfg("follower.v_max", 0.6))
        self.a_max          = float(cfg("follower.a_max", 1.2))
        self.warmup_time    = float(cfg("follower.warmup_time", 2.5))
        self.v_init         = float(cfg("follower.v_init", 0.08))
        self.lookahead_time = float(cfg("follower.lookahead_time", 0.5))
        self.min_lookahead  = float(cfg("follower.min_lookahead", 0.12))
        self.goal_reach_thresh = float(cfg("follower.goal_reach_thresh", 0.12))

        # camera parameters
        self.width = 64 
        self.height = 48

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

        # # ! Add Tracker Sensor
        # sensor_kwargs = dict(
        #     entity_idx=self.tracker.idx,
        #     pos_offset=(0.0, 0.0, 0.0),
        #     euler_offset=(0.0, 0.0, 0.0),
        #     return_world_frame=True,
        #     draw_debug=True,
        # )
        # res = ( self.width, self.height)
        # self.tracker_sensor = self.scene.add_sensor(gs.sensors.DepthCamera(pattern=DepthCameraPattern(res=res), **sensor_kwargs))

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


        # move obstacle to CPU numpy array, because is more efficient to build roadmap on CPU
        self.obs_xy_cpu_np = self.obs_xy.detach().to("cpu", copy=True).numpy().astype(np.float32)
        self.obs_r_cpu_np  = self.obs_r.detach().to("cpu", copy=True).numpy().astype(np.float32)

        # =====  Roadmap =====

        # roadmap parameters
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

        # Cache obstacle tensors on device (used in multiple places)
        self.obs_xy_dev = self.obs_xy.to(self.device, dtype=torch.float32)
        self.obs_r_dev = self.obs_r.to(self.device, dtype=torch.float32)
        self.obs_inflated = (self.obs_r_dev + self.prm_clearance)

        # GPU roadmap (kNN + LOS on device; CPU dijkstra). Build once on GPU
        try:
            self.gpu_roadmap = GPURoadmap.build(
                world_min=(float(self.world_xy_min[0]), float(self.world_xy_min[1])),
                world_max=(float(self.world_xy_max[0]), float(self.world_xy_max[1])),
                obs_xy=self.obs_xy_dev,
                obs_r=self.obs_r_dev,
                n_nodes=self.prm_num_nodes,
                k=self.prm_k_neighbors,
                max_edge_len=self.prm_max_edge,
                clearance=self.prm_clearance,
                device=self.device,
            )
        except Exception as e:
            print(f"[Planner] GPU roadmap disabled: {e}")
            self.gpu_roadmap = None

        # No CPU async pool; use GPU roadmap exclusively
        self._planner_k_attach = 8
        self._planner_goals_per_submit = 32  # batch GPU goal sampling

        # ! Add for path searching (use batched follower on GPU)
        self.goal_xy = [None] * self.num_envs
        self.path_wps = [None] * self.num_envs
        self.batched_follower = BatchedPathFollower(
            num_envs=self.num_envs, device=self.device, dt=self.dt,
            v_max=self.v_max, a_max=self.a_max,
            warmup_time=self.warmup_time, v_init=self.v_init,
            lookahead_time=self.lookahead_time, min_lookahead=self.min_lookahead,
            slow_down_k=1.2
        )
        # Track which envs have active paths
        self.follower_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        from collections import deque

        self.replan_queue = deque()
        self.replan_inqueue = set()   # remove same env_id in queue
        self.max_plan_per_step = 32   # conservative default; large values can hurt due to overhead


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

        # Preallocate reusable buffers to avoid per-step allocation
        self.ref_pos_buf = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.ref_pos_buf[:, 2] = self.drone_height
        self.ref_pos_buf[:, 3] = 0.0

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        # self.images = torch.zeros((self.num_envs, self.height, self.width), device=gs.device, dtype=gs.tc_float)

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

        # self.images[:] = self.tracker_sensor.read_image()
        # ! -------------------------- apply actions --------------------------
        self.actions = actions
        exec_actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # 
        tracker_prop_rpms = self.tracker.controller.step(exec_actions)       # [N,4] tensor 
        self.tracker.set_propellels_rpm(tracker_prop_rpms)

        # Batched follower step on GPU (reuse preallocated ref_pos_buf)
        next_xy = self.batched_follower.step()  # (num_envs,2)
        self.ref_pos_buf[:, 0:2] = next_xy

        # Check reached goal in batch (defer CPU sync to reduce frequency)
        reached_t = self.batched_follower.reached_goal(thresh=self.goal_reach_thresh)  # (num_envs,) bool
        if torch.any(reached_t):
            reached_ids = reached_t.nonzero(as_tuple=False).squeeze(-1).tolist()
            for env_id in reached_ids:
                if env_id not in self.replan_inqueue:
                    self.replan_queue.append(env_id)
                    self.replan_inqueue.add(env_id)


        target_prop_rpms = self.target.controller.step(self.ref_pos_buf)
        self.target.set_propellels_rpm(target_prop_rpms)

        self.scene.step()

        # ! -------------------------- update buffers --------------------------
        self.episode_length_buf += 1
        self.tracker_last_pos[:] = self.tracker_pos[:]
        self.tracker_pos[:] = self.tracker.get_pos()

        self.target_last_pos[:] = self.target_pos[:]
        self.target_pos[:] = self.target.get_pos()

        self.rel_pos = self.target_pos - self.tracker_pos

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

        # check termination and reset
        # 1. if drone is in collision
        # 2. if target is lost
        # 3. if drone attitude exceeds max angle
        # 4. if drone is close to ground
        # 5. if drone is out of range
        
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
        feats = obstacle_features(
                tracker_pos_w=self.tracker_pos,
                tracker_quat=self.tracker_quat,
                tracker_lin_vel_w=self.tracker_lin_vel,
                obs_xy_w=self.obs_xy,      # (N,M,2) or (M,2)
                obs_r=self.obs_r,          # (N,M,1) or (M,1)
                range_max=20.0,
                ttc_max=5.0,
                K=8,
                quat_format="xyzw",
        )       # dict of (N,*) tensors
        obs_env = torch.cat([
            feats["d_min_norm"],
            feats["bearing_min_pi"],
            feats["ttc_min_norm"],
            feats["mean_clear_norm"],
            feats["heading_clear_norm"],
            feats["sector_mins_norm"],   # if K>0
        ], dim=-1)                       # (N, 4+3+4+1+1+K)
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["max_diff"], -1, 1),          # relative position
                self.tracker_quat,                                                      # tracker quaternion
                torch.clip(self.tracker_lin_vel * self.obs_scales["max_lin"], -1, 1),   # tracker linear velocity
                torch.clip(self.tracker_ang_vel * self.obs_scales["max_ang"], -1, 1),   # tracker angular velocity
                torch.clip(self.target_lin_vel * self.obs_scales["max_lin"], -1, 1),    # target linear velocity
                torch.clip(self.last_actions * self.obs_scales["max_lin"], -1, 1),      # last action
                obs_env,                                                                # obstacle features 
            ],
            axis=-1,
        )

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
        """Drains the replan queue and processes planning requests using GPU exclusively."""
        if budget is None:
            budget = self.max_plan_per_step

        batch = []
        while budget > 0 and self.replan_queue:
            eid = self.replan_queue.popleft()
            if eid in self.replan_inqueue:
                self.replan_inqueue.remove(eid)
            batch.append(eid)
            budget -= 1

        if not batch or self.gpu_roadmap is None:
            return

        # GPU goal sampling once per drain (use cached obstacle tensors)
        goals_xy_t = sample_free_goals_gpu(
            (float(self.world_xy_min[0]), float(self.world_xy_min[1])),
            (float(self.world_xy_max[0]), float(self.world_xy_max[1])),
            self.obs_xy_dev,
            self.obs_r_dev,
            self.prm_clearance,
            self._planner_goals_per_submit,
            self.device,
        )  # (G,2)

        # Use cached target_pos (already updated in step())
        B = len(batch)
        M = self.obs_xy.shape[0]

        # Batch gather starts on device (target_pos already on device)
        batch_idx = torch.tensor(batch, device=self.device, dtype=torch.long)
        starts_xy = self.target_pos.index_select(0, batch_idx)[:, :2]  # (B,2)

        if M == 0:
            # No obstacles: assign first goal to all (one batch download)
            starts_cpu = starts_xy.cpu().numpy()
            goal_cpu = goals_xy_t[0].cpu().numpy()
            chosen = (float(goal_cpu[0]), float(goal_cpu[1]))
            for i, eid in enumerate(batch):
                start = (float(starts_cpu[i,0]), float(starts_cpu[i,1]))
                path = [start, chosen]
                self.goal_xy[eid] = chosen
                self.path_wps[eid] = path
                self.batched_follower.reset_with_path(eid, path)
                self.follower_active[eid] = True
            return

        # Batch GPU LOS: (B,G,M) collision check (use cached obstacle tensors)
        v = goals_xy_t.unsqueeze(0) - starts_xy.unsqueeze(1)  # (B,G,2)
        vv = torch.clamp((v*v).sum(-1), min=1e-9)  # (B,G)
        w = self.obs_xy_dev.unsqueeze(0).unsqueeze(0) - starts_xy.unsqueeze(1).unsqueeze(1)  # (B,G,M,2)
        t = ((w * v.unsqueeze(2)).sum(-1) / vv.unsqueeze(-1)).clamp(0.0, 1.0)
        proj = starts_xy.unsqueeze(1).unsqueeze(1) + t.unsqueeze(-1) * v.unsqueeze(2)
        d = torch.linalg.norm(proj - self.obs_xy_dev.unsqueeze(0).unsqueeze(0), dim=-1)
        blocked = d <= self.obs_inflated.unsqueeze(0).unsqueeze(0)
        free = ~blocked.any(dim=2)  # (B,G)
        has_los = free.any(dim=1)  # (B,)
        idx_first = torch.argmax(free.int(), dim=1)  # (B,) first free goal per env
        
        # One batch download for all processing
        has_los_cpu = has_los.cpu().numpy()
        idx_first_cpu = idx_first.cpu().numpy()
        starts_cpu = starts_xy.cpu().numpy()
        goals_cpu = goals_xy_t.cpu().numpy()
        
        unresolved_indices = []
        for ii, eid in enumerate(batch):
            if has_los_cpu[ii]:
                gi = int(idx_first_cpu[ii])
                start = (float(starts_cpu[ii,0]), float(starts_cpu[ii,1]))
                chosen = (float(goals_cpu[gi,0]), float(goals_cpu[gi,1]))
                path = [start, chosen]
                self.goal_xy[eid] = chosen
                self.path_wps[eid] = path
                self.batched_follower.reset_with_path(eid, path)
                self.follower_active[eid] = True
            else:
                unresolved_indices.append(ii)
        
        # GPU PRM for unresolved
        if unresolved_indices:
            goals_list = [(float(goals_cpu[i,0]), float(goals_cpu[i,1])) for i in range(goals_cpu.shape[0])]
            for ii in unresolved_indices:
                eid = batch[ii]
                start = (float(starts_cpu[ii,0]), float(starts_cpu[ii,1]))
                path = self.gpu_roadmap.query(start, goals_list, k_attach=self._planner_k_attach)
                if path and len(path) >= 2:
                    self.goal_xy[eid] = path[-1]
                    self.path_wps[eid] = path
                else:
                    self.goal_xy[eid] = start
                    self.path_wps[eid] = [start, start]
                self.batched_follower.reset_with_path(eid, self.path_wps[eid])
                self.follower_active[eid] = True

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

    def _reward_smooth(self):
        smooth_action = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        smooth_attitude = torch.sum(torch.square(self.actions[:,:3]), dim=1)
        smooth_rew = smooth_action + smooth_attitude
        return smooth_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _reward_collision(self):

        out = collision_reward(
            tracker_pos=self.tracker_pos[:,:2],     # [N,dim]
            tracker_vel=self.tracker_lin_vel[:,:2],     # [N,dim]
            obs_centers=self.obs_xy,     # [M,dim]
            obs_radii=self.obs_r,         # [M] or [M,1]
        )
        return out["rc"]
    def _reward_distance_horizontal(self):
        """
        k: Softening factor for the boundary (larger values mean softer boundaries), recommended 10~50
        """
        k=20.0
        eps=1e-8
        horiz = self.rel_pos[..., :2]
        d = torch.sqrt(torch.sum(horiz * horiz, dim=-1) + eps)

        # softplus( k*(d_min - d) ), only penalize outside the range
        penalty_below = torch.nn.functional.softplus(k * (self.od_min - d)) / k
        penalty_above = torch.nn.functional.softplus(k * (d - self.od_max)) / k
        soft_barrier = penalty_below + penalty_above

        # use negative cost as reward
        reward = -(soft_barrier )

        # use tanh for smooth
        reward = torch.tanh(reward)
        return reward
    
    def _reward_distance_vertical(self):
        """
        Soft reward for vertical distance.
        Vertical distance of 0 gives maximum reward 1.0, and vertical distance increases reward decreases.
        """
        # ! 1. get the vertical distance (absolute value)
        vertical_dist = torch.abs(self.rel_pos[:, 2])
        
        # ! 2. use gaussian reward function, when vertical distance is 0, get maximum reward 1.0
        # sigma controls the rate of decay of reward with vertical distance, adjust as needed
        sigma = 0.5
        reward = torch.exp(-0.5 * (vertical_dist / sigma)**2)
        
        return reward

    def _reward_max_speed(self):
        """
        max speed cost: c_m = exp(max(0, ||v|| - v_max)) - 1 >= 0 
        use negative weight (e.g., -0.5) for this cost.
        """
        # ! 1. get the linear speed norm (world frame)
        speed_norm = torch.norm(self.tracker_lin_vel, dim=-1)
        v_max = 5.0

        exceed = torch.clamp(speed_norm - v_max, min=0.0)
        # ! 2. use expm1 for numerical stability：expm1(x) = exp(x) - 1
        cost = torch.expm1(exceed)   # >= 0, when speed is 0, cost is 0
        return cost

    def _reward_visibility_dir(self):
        """
        reward for visibility of direction and position.
        """
        # get the forward vector of the tracker drone in world frame
        # assume the forward direction of the tracker drone is the x axis of body frame
        forward_vec_body = torch.tensor([1.0, 0, 0], device=self.device).expand(self.num_envs, -1)
        forward_vec_world = transform_by_quat(forward_vec_body, self.tracker_quat)

        # ! 1. reward for visibility of direction
        # get the velocity vector of the target drone in world frame
        target_vel = self.target.get_vel()
        vel_norm = torch.norm(target_vel, dim=-1, keepdim=True)
        epsilon = 1e-6
        target_vel_normalized = target_vel / (vel_norm + epsilon)
        reward_direction = torch.sum(forward_vec_world * target_vel_normalized, dim=-1)

        # ! 2. reward for visibility of position
        # get the direction vector to the target drone in world frame
        direction_to_target = self.rel_pos
        pos_norm = torch.norm(direction_to_target, dim=-1, keepdim=True)
        direction_to_target_normalized = direction_to_target / (pos_norm + epsilon)
        reward_yaw = torch.sum(forward_vec_world * direction_to_target_normalized, dim=-1)

        # ! 3. define weights
        w_direction = 0.5
        w_yaw = 0.5
        visibility_reward = w_direction * reward_direction + w_yaw * reward_yaw
        
        return visibility_reward
    
    def _reward_visibility_obs(self):
        """
        reward for visibility of obstacles.
        """
        params = Params2D(alpha=8.0, dt=self.dt, H=8, w_v=0.7, w_tto=0.3)
        out = visibility_and_tto_2d(self.obs_xy, self.obs_r, self.target_pos[:,:2], self.target_lin_vel[:,:2], self.tracker_pos[:,:2], self.tracker_lin_vel[:,:2], params)
        # print(out["V0"], out["TTO"], out["reward"])
        return out["reward"]