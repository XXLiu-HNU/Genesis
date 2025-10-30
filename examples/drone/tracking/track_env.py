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
from utils import (
    collision_check, 
    occlusion_check, 
    setup_random_cylindrical_obstacles, 
    obstacle_features, 
    collision_reward
)

from depth_visibility_2d import visibility_and_tto_2d, Params2D
from image_processor import ImageProcessor

from path_search import sample_free_xy_batch, sample_around_centers_batch

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
        
        # Waypoint GPU parameters for target movement
        self.waypoint_samples = 16          # Number of candidate waypoints to sample per frame
        self.waypoint_distance = 3.0        # How far to sample waypoints (meters)
        self.waypoint_goal_dist = 5.0       # Random goal distance range (meters)


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
        sensor_kwargs = dict(
            entity_idx=self.tracker.idx,
            pos_offset=(0.2, 0.0, 0.1),
            euler_offset=(0.0, 0.0, 0.0),
            return_world_frame=True,
            draw_debug=True,
        )
        res = (self.width, self.height)
        self.tracker_sensor = self.scene.add_sensor(gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(res=res), **sensor_kwargs))

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
        

        self.scene.start_recording(
            data_func=(lambda: self.tracker_sensor.read_image()[0]) if self.num_envs > 0 else self.tracker_sensor.read_image,
            rec_options=gs.recorders.MPLImagePlot(),
        )
        # ! Build scene
        self.scene.build(n_envs=num_envs)


        # Cache obstacle tensors on device (used for collision detection and waypoint_gpu)
        self.obs_xy_dev = self.obs_xy.to(self.device, dtype=torch.float32)
        self.obs_r_dev = self.obs_r.to(self.device, dtype=torch.float32)
        self.obs_inflated = (self.obs_r_dev + self.inflation_default)

        print(f"[INFO] Target movement: WAYPOINT_GPU")
        print(f"       samples={self.waypoint_samples}, distance={self.waypoint_distance}m")


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
        
        # Waypoint GPU buffers
        self.target_waypoint_goal = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.target_waypoint_current = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.target_waypoint_pos = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # Actual smooth position
        self.target_waypoint_vel = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # Current velocity
        self.waypoint_step_counter = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.long)
        self.waypoint_timer = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # Timer for warmup
        
        # Waypoint motion parameters (gentle settings to avoid height oscillation)
        self.waypoint_v_max = 2.0      # Max speed (m/s) - reduced for stability
        self.waypoint_a_max = 3.0      # Max acceleration (m/s^2) - reduced for gentle motion  
        self.waypoint_warmup_time = 3.0  # Warmup time to reach v_max (seconds) - smooth start
        self.waypoint_v_init = 0.05    # Initial max speed during warmup (m/s) - very gentle start

        # Preallocate reusable buffers to avoid per-step allocation
        self.ref_pos_buf = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.ref_pos_buf[:, 2] = self.drone_height
        self.ref_pos_buf[:, 3] = 0.0

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        self.images = torch.zeros((self.num_envs, self.height, self.width), device=gs.device, dtype=gs.tc_float)
        # image processor: encoder + augment pipeline
        # create once and keep on the same device as the simulation
        self.img_proc = ImageProcessor(device=self.device, max_range=20.0, encoder_out_dim=128, use_mask_channel=True)

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

        # read depth images and process (augment + encode)
        depths = self.tracker_sensor.read_image()  # (N, H, W)
        depth_feats, depths_aug, mask = self.img_proc.process(depths, training=self.env_cfg.get("train_mode", True))
        # store augmented depths for visualization/recording
        self.images[:] = depths_aug
        # expose image features to extras so policy/training loop can use them
        self.extras["observations"]["image_feats"] = depth_feats
        # ! -------------------------- apply actions --------------------------
        self.actions = actions
        exec_actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # 
        tracker_prop_rpms = self.tracker.controller.step(exec_actions)       # [N,4] tensor 
        self.tracker.set_propellels_rpm(tracker_prop_rpms)

        # Target movement control (waypoint_gpu only)
        self._update_waypoint_gpu()
        self._move_to_waypoint_smooth()  # Smooth movement with velocity control
        self.ref_pos_buf[:, 0:2] = self.target_waypoint_pos
        
        # Ensure height and yaw are always set correctly (critical for stability)
        self.ref_pos_buf[:, 2] = self.drone_height
        self.ref_pos_buf[:, 3] = 0.0


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

        # ! -------------------------- eval prints (only in eval mode) --------------------------
        if self.env_cfg.get("eval_mode", False):
            # Horizontal distance (xy plane)
            horiz_vec = self.rel_pos[:, :2]
            horiz_dist = torch.linalg.norm(horiz_vec, dim=-1)
            # Vertical distance (z)
            vert_dist = torch.abs(self.rel_pos[:, 2])
            # Bearing of target in tracker body frame (x-forward, y-left). 0 deg when centered ahead
            # Transform rel_pos (world) to tracker body frame using inv(quat)
            rel_pos_body = transform_by_quat(self.rel_pos, inv_tracker_quat)
            bearing_rad = torch.atan2(rel_pos_body[:, 1], rel_pos_body[:, 0])
            bearing_deg = bearing_rad * (180.0 / math.pi)
            # Normalize to [-180, 180]
            bearing_deg = (bearing_deg + 180.0) % 360.0 - 180.0
            # Print first env values in eval
            idx = 0
            if horiz_dist.numel() > 0:
                print(f"[EVAL] step={int(self.episode_length_buf[idx].item())} horiz_dist={horiz_dist[idx].item():.3f} m, vert_dist={vert_dist[idx].item():.3f} m, target_bearing_body={bearing_deg[idx].item():.1f} deg")

        # ! -------------------------- compute reward --------------------------
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ! -------------------------- compute observations --------------------------
        # 这里是使用全局障碍物作为障碍物的观测特征，理想化处理
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
                depth_feats,                                                            # depth image features
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

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
        # Sample new target position
        tgt_xy = sample_free_xy_batch(
            self.world_xy_min, self.world_xy_max,
            self.obs_xy, self.obs_r,
            safe_radius=inflation,
            n=num_resets, device=self.device
        )  # (num_resets, 2)

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

        # Initialize waypoint GPU state for reset envs
        if len(envs_idx) > 0:
            # Sample random goals around initial position
            angles = torch.rand(num_resets, device=self.device) * 2 * 3.14159
            distances = torch.rand(num_resets, device=self.device) * self.waypoint_goal_dist + 2.0
            self.target_waypoint_goal[envs_idx, 0] = tgt_xy[:, 0] + distances * torch.cos(angles)
            self.target_waypoint_goal[envs_idx, 1] = tgt_xy[:, 1] + distances * torch.sin(angles)
            # Clamp to world bounds
            self.target_waypoint_goal[envs_idx, 0] = torch.clamp(
                self.target_waypoint_goal[envs_idx, 0], 
                self.world_xy_min[0] + 1, self.world_xy_max[0] - 1
            )
            self.target_waypoint_goal[envs_idx, 1] = torch.clamp(
                self.target_waypoint_goal[envs_idx, 1],
                self.world_xy_min[1] + 1, self.world_xy_max[1] - 1
            )
            # Initialize position, velocity, and timer (reset warmup)
            self.target_waypoint_current[envs_idx] = tgt_xy
            self.target_waypoint_pos[envs_idx] = tgt_xy
            self.target_waypoint_vel[envs_idx] = 0.0
            self.waypoint_step_counter[envs_idx] = 0
            self.waypoint_timer[envs_idx] = 0.0  # Reset timer for warmup


    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def _move_to_waypoint_smooth(self):
        """
        Smooth movement towards current waypoint with velocity and acceleration limits.
        Includes warmup phase to prevent aggressive initial acceleration.
        """
        # Increment timer for warmup
        self.waypoint_timer += self.dt
        
        # Warmup: gradually increase max speed from v_init to v_max
        # Ensure proper batch dimensions: (num_envs, 1)
        warmup_factor = torch.clamp(self.waypoint_timer / self.waypoint_warmup_time, 0.0, 1.0).unsqueeze(1)  # (num_envs, 1)
        current_v_max = self.waypoint_v_init + warmup_factor * (self.waypoint_v_max - self.waypoint_v_init)  # (num_envs, 1)
        
        # Direction to current waypoint
        to_waypoint = self.target_waypoint_current - self.target_waypoint_pos  # (num_envs, 2)
        dist_to_waypoint = torch.norm(to_waypoint, dim=1, keepdim=True)  # (num_envs, 1)
        
        # Desired velocity: towards waypoint, clamped by current max speed (with warmup)
        # Slow down when close to waypoint
        slowdown_dist = 1.0  # Start slowing down 1m from waypoint
        speed_factor = torch.clamp(dist_to_waypoint / slowdown_dist, 0.0, 1.0)  # (num_envs, 1)
        desired_speed = current_v_max * speed_factor  # (num_envs, 1)
        
        # Desired velocity direction
        direction = torch.where(
            dist_to_waypoint > 1e-6,
            to_waypoint / (dist_to_waypoint + 1e-9),
            torch.zeros_like(to_waypoint)
        )  # (num_envs, 2)
        desired_vel = direction * desired_speed  # (num_envs, 2) * (num_envs, 1) = (num_envs, 2)
        
        # Apply acceleration limit
        vel_change = desired_vel - self.target_waypoint_vel
        vel_change_norm = torch.norm(vel_change, dim=1, keepdim=True)
        max_vel_change = self.waypoint_a_max * self.dt
        
        vel_change = torch.where(
            vel_change_norm > max_vel_change,
            vel_change / (vel_change_norm + 1e-9) * max_vel_change,
            vel_change
        )
        
        # Update velocity and position
        self.target_waypoint_vel += vel_change
        self.target_waypoint_pos += self.target_waypoint_vel * self.dt
        
        # Clamp to world bounds
        self.target_waypoint_pos[:, 0] = torch.clamp(
            self.target_waypoint_pos[:, 0],
            self.world_xy_min[0], self.world_xy_max[0]
        )
        self.target_waypoint_pos[:, 1] = torch.clamp(
            self.target_waypoint_pos[:, 1],
            self.world_xy_min[1], self.world_xy_max[1]
        )

    def _update_waypoint_gpu(self):
        """
        GPU-based waypoint sampling for target movement.
        Samples candidate points, checks line-of-sight, picks best waypoint.
        """
        # Increment step counter
        self.waypoint_step_counter += 1
        
        # Check which envs need new waypoint (only when reached current waypoint)
        # Not using timer to avoid waypoint changing before reaching it (which causes unstable movement)
        dist_to_waypoint = torch.norm(self.target_waypoint_pos - self.target_waypoint_current, dim=1)
        need_update = (dist_to_waypoint < 0.3)  # Only update when close to waypoint
        
        if not torch.any(need_update):
            # No updates needed, continue moving to current waypoint
            return
        
        # For envs needing update, sample candidate waypoints
        n_update = need_update.sum().item()
        update_ids = need_update.nonzero(as_tuple=False).squeeze(-1)  # (n_update,)
        
        # Current positions of envs needing update (use smooth position)
        current_pos = self.target_waypoint_pos[update_ids]  # (n_update, 2)
        goals = self.target_waypoint_goal[update_ids]  # (n_update, 2)
        
        # Sample candidate waypoints: biased towards goal
        S = self.waypoint_samples
        candidates = torch.zeros((n_update, S, 2), device=self.device, dtype=torch.float32)
        
        for i in range(S):
            # Mix of random and goal-directed samples
            if i < S // 2:
                # Goal-directed: sample along direction to goal
                t = torch.rand(n_update, device=self.device) * 0.8 + 0.2  # bias towards goal
                candidates[:, i] = current_pos + t.unsqueeze(1) * (goals - current_pos) * (self.waypoint_distance / (torch.norm(goals - current_pos, dim=1, keepdim=True) + 1e-6))
            else:
                # Random exploration
                angles = torch.rand(n_update, device=self.device) * 2 * 3.14159
                distances = torch.rand(n_update, device=self.device) * self.waypoint_distance
                candidates[:, i, 0] = current_pos[:, 0] + distances * torch.cos(angles)
                candidates[:, i, 1] = current_pos[:, 1] + distances * torch.sin(angles)
        
        # Clamp to world bounds
        candidates[:, :, 0] = torch.clamp(candidates[:, :, 0], self.world_xy_min[0], self.world_xy_max[0])
        candidates[:, :, 1] = torch.clamp(candidates[:, :, 1], self.world_xy_min[1], self.world_xy_max[1])
        
        # GPU batch line-of-sight check (n_update, S, M)
        if self.obs_xy.numel() > 0:
            # Line segment from current_pos to each candidate
            v = candidates - current_pos.unsqueeze(1)  # (n_update, S, 2)
            vv = torch.clamp((v * v).sum(-1), min=1e-9)  # (n_update, S)
            
            # Vector from current_pos to each obstacle
            w = self.obs_xy_dev.unsqueeze(0).unsqueeze(0) - current_pos.unsqueeze(1).unsqueeze(1)  # (n_update, S, M, 2)
            t = ((w * v.unsqueeze(2)).sum(-1) / vv.unsqueeze(-1)).clamp(0.0, 1.0)  # (n_update, S, M)
            
            # Closest point on segment to each obstacle
            proj = current_pos.unsqueeze(1).unsqueeze(1) + t.unsqueeze(-1) * v.unsqueeze(2)  # (n_update, S, M, 2)
            d = torch.linalg.norm(proj - self.obs_xy_dev.unsqueeze(0).unsqueeze(0), dim=-1)  # (n_update, S, M)
            
            # Check collision
            blocked = d <= (self.obs_r_dev + self.inflation_default).unsqueeze(0).unsqueeze(0)  # (n_update, S, M)
            is_free = ~blocked.any(dim=2)  # (n_update, S)
        else:
            is_free = torch.ones((n_update, S), device=self.device, dtype=torch.bool)
        
        # Score waypoints: prefer free, far from current, close to goal
        dist_to_goal = torch.norm(candidates - goals.unsqueeze(1), dim=2)  # (n_update, S)
        dist_from_current = torch.norm(candidates - current_pos.unsqueeze(1), dim=2)  # (n_update, S)
        
        score = torch.zeros_like(dist_to_goal)
        score[is_free] = dist_from_current[is_free] * 2.0 - dist_to_goal[is_free] * 0.5  # Prefer far + towards goal
        score[~is_free] = -1e6  # Penalize blocked
        
        # Select best waypoint for each env
        best_idx = torch.argmax(score, dim=1)  # (n_update,)
        best_waypoints = candidates[torch.arange(n_update, device=self.device), best_idx]  # (n_update, 2)
        
        # Update waypoint targets
        self.target_waypoint_current[update_ids] = best_waypoints
        
        # Check if reached goal, sample new goal
        dist_to_goal_final = torch.norm(current_pos - goals, dim=1)
        reached_goal = dist_to_goal_final < 1.0
        if torch.any(reached_goal):
            reached_ids = update_ids[reached_goal]
            n_reached = reached_ids.numel()
            # Sample new random goals
            angles = torch.rand(n_reached, device=self.device) * 2 * 3.14159
            distances = torch.rand(n_reached, device=self.device) * self.waypoint_goal_dist + 2.0
            new_goals_x = current_pos[reached_goal, 0] + distances * torch.cos(angles)
            new_goals_y = current_pos[reached_goal, 1] + distances * torch.sin(angles)
            self.target_waypoint_goal[reached_ids, 0] = torch.clamp(new_goals_x, self.world_xy_min[0] + 1, self.world_xy_max[0] - 1)
            self.target_waypoint_goal[reached_ids, 1] = torch.clamp(new_goals_y, self.world_xy_min[1] + 1, self.world_xy_max[1] - 1)

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
        return out["reward"]