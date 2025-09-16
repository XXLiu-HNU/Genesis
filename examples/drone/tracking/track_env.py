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
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def setup_random_cylindrical_obstacles(scene, n_obstacles=20, n_envs=1, min_radius=0.1, max_radius=0.5,
                                     min_height=1.0, max_height=2.0, min_distance=1.0,
                                     world_bounds=(-10, 10, -10, 10), device="cpu", oversample_factor=10):
    """Randomly generates and places non-overlapping cylindrical obstacles in a scene.

    This function uses a batched tensor sampling approach to create a specified number of
    cylindrical obstacles. It employs an oversampling strategy to generate more candidates
    than needed, then filters them to ensure they are within world bounds and do not
    collide with each other, maintaining a minimum distance between them.

    Returns:
        list: A list of the created gs.Entity obstacle objects.
    """

    # Oversample candidates (more than needed)
    n_candidates = n_obstacles * oversample_factor
    xs = torch.empty(n_candidates, device=device).uniform_(world_bounds[0], world_bounds[1])
    ys = torch.empty(n_candidates, device=device).uniform_(world_bounds[2], world_bounds[3])
    radii = torch.empty(n_candidates, device=device).uniform_(min_radius, max_radius)
    heights = torch.empty(n_candidates, device=device).uniform_(min_height, max_height)

    # ! ---------------- filter by world bounds ----------------------------------------
    in_bounds = (
        (xs - radii >= world_bounds[0]) &
        (xs + radii <= world_bounds[1]) &
        (ys - radii >= world_bounds[2]) &
        (ys + radii <= world_bounds[3])
    )

    xs, ys, radii, heights = xs[in_bounds], ys[in_bounds], radii[in_bounds], heights[in_bounds]

    # ! ---------------- filter by min distance ----------------------------------------
    obstacles = []
    obstacle_positions = []
    obstacle_radii = []

    for i in range(xs.shape[0]):
        if len(obstacles) >= n_obstacles:
            break
        x, y, r, h = xs[i].item(), ys[i].item(), radii[i].item(), heights[i].item()

        if obstacle_positions:
            pos_tensor = torch.tensor(obstacle_positions, device=device)
            rad_tensor = torch.tensor(obstacle_radii, device=device)
            dx = pos_tensor[:, 0] - x
            dy = pos_tensor[:, 1] - y
            dist = torch.sqrt(dx**2 + dy**2)
            if torch.any(dist < (rad_tensor + r + min_distance)):
                continue  # too close, skip

        # Add obstacle
        r_color = torch.empty(1, device=device).uniform_(0.3, 0.8).item()
        g_color = torch.empty(1, device=device).uniform_(0.3, 0.8).item()
        b_color = torch.empty(1, device=device).uniform_(0.3, 0.8).item()

        obstacle = scene.add_entity(
            gs.morphs.Cylinder(
                radius=r,
                height=h,
                pos=(x, y, h/2),
                fixed=True
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(r_color, g_color, b_color),
                ),
            ),
        )
        obstacles.append(obstacle)
        obstacle_positions.append((x, y))
        obstacle_radii.append(r)

    if len(obstacles) < n_obstacles:
        print(f"Warning: Only generated {len(obstacles)} obstacles (requested {n_obstacles})")

    return obstacles

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

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
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

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # Add Tracker
        self.tracker_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.tracker_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.tracker_inv_init_quat = inv_quat(self.tracker_init_quat)
        self.tracker = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf"))

        # Add Traget
        self.circle_radius = torch.ones(self.num_envs, device=self.device)
        self.target_height = torch.ones(self.num_envs, device=self.device)
        self.circle_omega = torch.ones(self.num_envs, device=self.device)
        self.circle_center = torch.tensor([0.0, 0.0, 1.0], device=self.device) 

        self.target_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.target_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.target_inv_init_quat = inv_quat(self.target_init_quat)
        self.target = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf"))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config/pos.yaml"), "r") as file:
                self.pos_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)
        with open(os.path.join(script_dir, "config/rate.yaml"), "r") as file:
                self.rate_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)

        # Add odom and controller for drone
        self._setup_imu_and_controller(self.tracker, "rate", self.rate_ctrl_config)
        self._setup_imu_and_controller(self.target, "position", self.pos_ctrl_config)
        
        # build scene
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
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

    def _collision_detect(self):
        # TODO 
        return False
    
    def _loss_detect(self):
        # TODO 
        return False

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

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
    
        tracker_prop_rpms = self.tracker.controller.step(exec_actions)       # [N,4] tensor 
        self.tracker.set_propellels_rpm(tracker_prop_rpms)

        circle_traj = self._get_circle_traj()
        target_prop_rpms = self.target.controller.step(circle_traj) 
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
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["max_diff"], -1, 1),          # relative position
                self.tracker_quat,                                                      # tracker quaternion
                torch.clip(self.tracker_lin_vel * self.obs_scales["max_lin"], -1, 1),   # tracker linear velocity
                torch.clip(self.tracker_ang_vel * self.obs_scales["max_ang"], -1, 1),   # tracker angular velocity
                torch.clip(self.target_lin_vel * self.obs_scales["max_lin"], -1, 1),    # target linear velocity
                torch.clip(self.last_actions * self.obs_scales["max_lin"], -1, 1),      # last action
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

        # ! -------------------------- reset tracker --------------------------
        self.tracker_pos[envs_idx] = self.tracker_init_pos
        self.tracker_last_pos[envs_idx] = self.tracker_init_pos
        self.tracker_quat[envs_idx] = self.tracker_init_quat.repeat(num_resets, 1)
        self.tracker.set_pos(self.tracker_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.tracker.set_quat(self.tracker_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.tracker_lin_vel[envs_idx] = 0.0
        self.tracker_ang_vel[envs_idx] = 0.0
        self.tracker.zero_all_dofs_velocity(envs_idx)

        # ! -------------------------- reset target --------------------------
        # 随机化圆的参数
        self.circle_radius[envs_idx] = gs_rand_float(2.0, 4.0, (num_resets,), self.device)
        self.target_height[envs_idx] = gs_rand_float(1.0, 2.5, (num_resets,), self.device)
        self.circle_omega[envs_idx] = gs_rand_float(0.5, 1.0, (num_resets,), self.device)
        
        # 随机化初始角度
        self.initial_angle[envs_idx] = gs_rand_float(0, 2 * math.pi, (num_resets,), self.device)
        
        # 计算新的起始位置
        new_target_pos = torch.zeros((num_resets, 3), device=self.device)
        new_target_pos[:, 0] = self.circle_center[0] + self.circle_radius[envs_idx] * torch.cos(self.initial_angle[envs_idx])
        new_target_pos[:, 1] = self.circle_center[1] + self.circle_radius[envs_idx] * torch.sin(self.initial_angle[envs_idx])
        new_target_pos[:, 2] = self.target_height[envs_idx]
        
        self.target_pos[envs_idx] = new_target_pos
        self.target_last_pos[envs_idx] = new_target_pos
        self.target_quat[envs_idx] = self.target_init_quat.repeat(num_resets, 1)
        self.target.set_pos(self.target_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target.set_quat(self.target_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target_lin_vel[envs_idx] = 0.0
        self.target_ang_vel[envs_idx] = 0.0
        self.target.zero_all_dofs_velocity(envs_idx)
        
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

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def _get_circle_traj(self):        
        """
        Generates a circular trajectory for a target to follow.

        This function calculates the target's next position on a circle with a fixed radius and height.
        The angle is updated at each time step to create continuous motion along the circular path.
        """

        angle = self.initial_angle + self.episode_length_buf * self.dt * self.circle_omega
        
        x = self.circle_radius * torch.cos(angle)
        y = self.circle_radius * torch.sin(angle)
        z = self.target_height * torch.ones_like(x)
        t = torch.zeros_like(x)

        return torch.stack([x, y, z, t], dim=1)

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
        # 假设你的无人机线速度存储在 self.tracker_lin_vel 中
        speed_norm = torch.norm(self.actions, dim=-1)

        # ! 定义最大允许速度，例如 5 m/s
        max_speed = 5.0
    
        exceed_speed = torch.clamp(speed_norm - max_speed, min=0.0)

        speed_penalty = (exceed_speed ** 2) * 2.0  # 惩罚系数 2.0
        return speed_penalty

    def _reward_visibility(self):
        """
        计算并奖励无人机朝向与目标运动方向及相对位置的对齐程度。
        """
        # 获取追踪无人机在世界坐标系下的朝向向量
        # 假设无人机机头方向为 body-frame 的 x 轴
        forward_vec_body = torch.tensor([1.0, 0, 0], device=self.device).expand(self.num_envs, -1)
        forward_vec_world = transform_by_quat(forward_vec_body, self.tracker_quat)

        # ! 1. 计算第一个奖励项：运动方向对齐
        # 获取目标无人机在世界坐标系下的运动方向向量
        target_vel = self.target.get_vel()
        vel_norm = torch.norm(target_vel, dim=-1, keepdim=True)
        epsilon = 1e-6
        target_vel_normalized = target_vel / (vel_norm + epsilon)
        reward_direction = torch.sum(forward_vec_world * target_vel_normalized, dim=-1)

        # ! 2. 计算第二个奖励项：空间位置朝向对齐
        # 获取指向目标的方向向量
        direction_to_target = self.rel_pos
        pos_norm = torch.norm(direction_to_target, dim=-1, keepdim=True)
        direction_to_target_normalized = direction_to_target / (pos_norm + epsilon)
        reward_yaw = torch.sum(forward_vec_world * direction_to_target_normalized, dim=-1)

        # ! 3. 定义权重。你可以根据需要调整这些值。
        # 例如，如果运动方向的对齐更重要，可以增加 w_direction 的值。
        w_direction = 0.5
        w_yaw = 0.5
        visibility_reward = w_direction * reward_direction + w_yaw * reward_yaw
        
        return visibility_reward