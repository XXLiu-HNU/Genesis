import torch
import math
import copy
import genesis as gs
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
        self.od_min_sq = 1.0
        self.od_max_sq = 3.0

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
        self.target = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf",pos = (1.0,0.0,0.0)))
        
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
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

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

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), gs.device)

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

        # 14468 is hover rpm
        # TODO 这里需要使用一个具体的控制器来控制无人机
        self.tracker.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)

        #  TODO 同时需要控制 target
        target_action = torch.zeros_like(exec_actions)
        self.target.set_propellels_rpm(14468.429183500699 * torch.ones((self.num_envs, 4), device=gs.device))

        # update target pos
        # 这里可能就不需要了
        # if self.target is not None:
        #     self.target.set_pos(self.commands, zero_velocity=True)

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.tracker_last_pos[:] = self.tracker_pos[:]
        self.tracker_pos[:] = self.tracker.get_pos()

        self.target_pos[:] = self.target.get_pos()
        self.target_last_pos[:] = self.target_pos[:]


        self.rel_pos = self.target_pos - self.tracker_pos

        # 这里是为了计算相对位置，这里不需要了
        # self.last_rel_pos = self.commands - self.last_base_pos
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

        # resample commands
        # 这里是为了在到达目标点之后重新采样一个目标点，这里不需要了
        # envs_idx = self._at_target()
        # self._resample_commands(envs_idx)

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
                                )

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
            print(f"{name} reward: {rew.mean().item():.3f}")

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

        for name, x in [("rel_pos", self.rel_pos), 
                ("tracker_quat", self.tracker_quat),
                ("target_lin_vel", self.target_lin_vel),
                ("last_actions", self.last_actions)]:
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"{name} contains NaN/Inf:", x)

        def check_tensor(x, name="tensor"):
            if isinstance(x, torch.Tensor):
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"{name} contains non-finite values: min={x.min().item()}, max={x.max().item()}, any_nan={torch.isnan(x).any().item()}")
                # 记录一下范围便于调试
                if x.abs().max().item() > 1e3:
                    print(f"WARNING {name} has large magnitude: max_abs={x.abs().max().item()}")
        # print(f"reward is {self.rew_buf.abs().max().item()}")
        # 使用示例
        check_tensor(self.obs_buf, "obs_buf")
        check_tensor(self.rew_buf, "reward")

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
        self.tracker_pos[envs_idx] = self.target_init_pos + torch.tensor([1.0, 0.0, 0.0], device=gs.device)
        self.target_last_pos[envs_idx] = self.target_init_pos
        self.target_quat[envs_idx] = self.target_init_quat.reshape(1, -1)
        self.target.set_pos(self.target_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target.set_quat(self.target_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target_lin_vel[envs_idx] = 0
        self.target_ang_vel[envs_idx] = 0
        self.target.zero_all_dofs_velocity(envs_idx)

        self.rel_pos = self.tracker_pos - self.tracker_pos
        # self.last_rel_pos = self.commands - self.last_base_pos


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

        self._resample_commands(envs_idx)

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
    
    def _reward_distance(self):
        # 距离平方
        dist_sq = torch.sum(torch.square(self.rel_pos), dim=1)

        # 限制在合理范围，避免立方爆炸
        dist_sq = torch.clamp(dist_sq, min=0.0, max=10.0)  # 100 只是例子，你可以调

        # 定义惩罚函数 g(x) = max(0, x)^3
        def g(x):
            return torch.clamp(x, min=0.0)

        penalty = g(self.od_min_sq - dist_sq) + g(dist_sq - self.od_max_sq)

        reward = -penalty  # 不要再 sum 了，直接保持 [num_envs] 维度即可

        # 最终再限幅
        reward = torch.clamp(reward, min=-100.0, max=0.0)

        return reward
