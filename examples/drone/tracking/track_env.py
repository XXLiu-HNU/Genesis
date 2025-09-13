import torch
import math
import copy
import genesis as gs
from quadcopter_controller import DronePIDController
from pid import PIDcontroller
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


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

        # add tracker
        self.tracker_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.tracker_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.tracker_inv_init_quat = inv_quat(self.tracker_init_quat)
        self.tracker = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf"))

        # add traget
        self.target_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.target_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.target_inv_init_quat = inv_quat(self.target_init_quat)
        self.target = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf"))
        
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

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        pid_params = [
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [200.0, 0.0, 30.0],
        [200.0, 0.0, 30.0],
        [600.0, 0.0, 60.0],
        [100.0, 0.0, 10.0],
        [100.0, 0.0, 10.0],
        [200.0, 0.0, 10.0],
        ]
        pid_params = {
            "kp": 6500,
            "ki": 0.01,
            "kd": 0.0,
            "kf": 0.0,
            "thrust_compensate": 0.0,
            "pid_exec_freq": 60,
            "base_rpm": 62293.9641914,
            }
        base_rpm = 14468.429183500699
        # self.tracker_controller = DronePIDController(drone=self.tracker, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)
        self.tracker_controller = PIDcontroller(drone=self.tracker, config=pid_params)
        # self.target_controller = DronePIDController(drone=self.target, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)
        self.target_controller = PIDcontroller(drone=self.target, config=pid_params)


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
        
        # self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        prop_rpms = self.tracker_controller.update(self.actions)   # [N,4] tensor\
        self.tracker.set_propellels_rpm(prop_rpms)                  # 假设 set_propellels_rpm 支持 batched tensor

        
        hover_rpm = torch.ones(
            (self.num_envs, 4), device=self.device
        ) * 62293.9641914 # 59489.68 for cf2x, 39777.86 for our drone
        self.target.set_propellels_rpm(hover_rpm)

        # 14468 is hover rpm

        self.scene.step()

        # update buffers
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
            ),
            rpy=True,
            degrees=True,
        )
        inv_tracker_quat = inv_quat(self.tracker_quat)
        self.tracker_lin_vel[:] = transform_by_quat(self.tracker.get_vel(), inv_tracker_quat)
        self.tracker_ang_vel[:] = transform_by_quat(self.tracker.get_ang(), inv_tracker_quat)

        self.target_quat[:] = self.target.get_quat()
        self.target_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.target_quat) * self.target_inv_init_quat,
                self.target_quat,
            ),
            rpy=True,
            degrees=True,
        )
        inv_target_quat = inv_quat(self.target_quat)
        self.target_lin_vel[:] = transform_by_quat(self.target.get_vel(), inv_target_quat)
        self.target_ang_vel[:] = transform_by_quat(self.target.get_ang(), inv_target_quat)

        # check termination and reset
        # 判断终止条件
        # 1. 无人机发生碰撞
        # 2. 目标无人机丢失
        
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
        # print("crash_condition shape:", self.crash_condition.shape)

        # self.crash_condition = (
        #     (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
        #     | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
        #     | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
        #     | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
        #     | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
        #     | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        # )

        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            # print(f"{name} reward: {rew.mean().item():.3f}")

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["max_diff"], -1, 1),
                self.tracker_quat,
                # 新增：追踪者自身的线速度和角速度
                torch.clip(self.tracker_lin_vel * self.obs_scales["max_lin"], -1, 1),
                torch.clip(self.tracker_ang_vel * self.obs_scales["max_ang"], -1, 1),
                # 目标的速度
                torch.clip(self.target_lin_vel * self.obs_scales["max_lin"], -1, 1),
                torch.clip(self.last_actions * self.obs_scales["max_lin"], -1, 1)
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
        if len(envs_idx) == 0:
            return
        # reset tracker base
        self.tracker_pos[envs_idx] = self.tracker_init_pos
        self.tracker_last_pos[envs_idx] = self.tracker_init_pos
        self.tracker_quat[envs_idx] = self.tracker_init_quat.reshape(1, -1)
        self.tracker.set_pos(self.tracker_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.tracker.set_quat(self.tracker_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.tracker_lin_vel[envs_idx] = 0
        self.tracker_ang_vel[envs_idx] = 0
        self.tracker.zero_all_dofs_velocity(envs_idx)

        # reset target base
         # --- 重置 target (加入随机化) ---
        # 在一个范围内随机生成目标的位置
        # 例如：x,y 在 [-2, 2] 之间，z 在 [0.5, 1.5] 之间
        num_resets = len(envs_idx)
        random_pos_xy = gs_rand_float(-2.0, 2.0, (num_resets, 2), self.device)
        random_pos_z = gs_rand_float(0.5, 1.5, (num_resets, 1), self.device)
        random_offset = torch.cat([random_pos_xy, random_pos_z], dim=-1)
        
        # 将初始位置应用随机偏移
        # 注意：这里我们让 target 的初始位置在 tracker 的基础上进行随机化
        target_start_pos = self.tracker_init_pos + random_offset
        
        # 检查并确保 target 不会与 tracker 初始位置太近
        dist_sq = torch.sum(torch.square(random_offset[:, :2]), dim=1)
        too_close = dist_sq < 0.5*0.5
        target_start_pos[too_close, 0] += torch.sign(target_start_pos[too_close, 0]) * 1.0 # 如果太近，在x轴上推开
        
        self.target_pos[envs_idx] = target_start_pos
        self.target_last_pos[envs_idx] = target_start_pos
        self.target_quat[envs_idx] = self.target_init_quat.reshape(1, -1)
        self.target.set_pos(self.target_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target.set_quat(self.target_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target_lin_vel[envs_idx] = 0
        self.target_ang_vel[envs_idx] = 0
        self.target.zero_all_dofs_velocity(envs_idx)

        self.rel_pos = self.target_pos - self.tracker_pos


        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
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

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
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
        # 获取垂直方向的距离（取绝对值）
        vertical_dist = torch.abs(self.rel_pos[:, 2])
        
        # 使用高斯奖励函数，当垂直距离为0时获得最大奖励1.0
        # sigma控制奖励随距离衰减的速度，可以根据实际需求调整
        sigma = 0.5
        reward = torch.exp(-0.5 * (vertical_dist / sigma)**2)
        
        return reward
    def _reward_yaw_alignment(self):
        # 追踪者的前向向量 (在世界坐标系下)
        # 假设机头方向为 body-frame 的 x 轴
        forward_vec_body = torch.tensor([1.0, 0, 0], device=self.device).expand(self.num_envs, -1)
        forward_vec_world = transform_by_quat(forward_vec_body, self.tracker_quat)

        # 指向目标的方向向量
        direction_to_target = self.rel_pos
        direction_to_target_normalized = direction_to_target / (torch.norm(direction_to_target, dim=-1, keepdim=True) + 1e-6)

        # 计算点积，衡量对准程度。值越大越好。
        alignment_dot_product = torch.sum(forward_vec_world * direction_to_target_normalized, dim=-1)

        # 奖励值在 [-1, 1] 之间。我们希望它接近 1。
        return alignment_dot_product
    
    def _reward_max_speed(self):
        """
        对速度超过物理极限的行为进行惩罚。
        使用指数函数对超速进行强力惩罚。
        """
        # 假设你的无人机线速度存储在 self.tracker_lin_vel 中
        # 计算线速度的范数（即速度大小）
        speed_norm = torch.norm(self.actions, dim=-1)

        # 定义最大允许速度，例如 5 m/s
        # 你需要根据你的环境和无人机物理特性来设定这个值
        max_speed = 2.0
        
        # 计算超出最大速度的部分，小于等于0的部分为0
        exceed_speed = torch.clamp(speed_norm - max_speed, min=0.0)

        # 根据公式计算奖励/惩罚
        # exp(x) 增长非常快，因此这里可以根据你的实际需求调整常数
        # 这里我们使用一个简单的线性惩罚，更易于控制
        # 惩罚 = - (超速部分的平方) * 惩罚系数
        # 这样可以对轻微超速进行轻微惩罚，对严重超速进行强力惩罚
        speed_penalty = (exceed_speed ** 2) * 2.0  # 惩罚系数 2.0
        
        # 另一种更接近你提供的公式的实现，但是更难调参
        # 你提供的公式可能会导致非常大的负奖励，需要谨慎使用
        # speed_penalty = -torch.exp(exceed_speed + 1) + 1
        
        return speed_penalty