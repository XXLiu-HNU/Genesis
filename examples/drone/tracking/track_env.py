import torch
import math
import copy
import genesis as gs
from quadcopter_controller import DronePIDController
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class TrackerEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device
        self.od_min_sq = 1.0*1.0
        self.od_max_sq = 3.0*3.0

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

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
            show_FPS=False
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

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
        self.tracker = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

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

        base_rpm = 14468.429183500699
        self.tracker_controller = DronePIDController(drone=self.tracker, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)
        self.target_controller = DronePIDController(drone=self.target, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)


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
        
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        prop_rpms = self.tracker_controller.update(self.actions)   # [N,4] tensor
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

        self.target_pos[:] = self.target.get_pos()
        self.target_last_pos[:] = self.target_pos[:]


        self.rel_pos = self.target_pos - self.tracker_pos

        self.tracker_quat[:] = self.tracker.get_quat()
        self.tracker_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.tracker_quat) * self.target_inv_init_quat,
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
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.tracker_quat,
                torch.clip(self.target_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                self.last_actions,
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
        self.target_pos[envs_idx] = self.target_init_pos + torch.tensor([3.0, 0.0, 0.0], device=gs.device)
        self.target_last_pos[envs_idx] = self.target_init_pos
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
        sigma = 2.0
        reward = torch.exp(-0.5 * (vertical_dist / sigma)**2)
        
        # 对于过大的垂直距离添加额外惩罚
        max_allowed_dist = 1.5  # 最大允许的垂直距离
        penalty = torch.clamp(vertical_dist - max_allowed_dist, min=0.0)**2
        penalty_scale = 0.1  # 惩罚系数
        
        # 将惩罚项加入到最终奖励中
        reward = reward - penalty_scale * penalty
    
        # 限制奖励范围
        reward = torch.clamp(reward, min=-2.0, max=1.0)
        
        return reward