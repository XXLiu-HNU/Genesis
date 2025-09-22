"""

使用 path search 里的类进行配置，更符合训练
"""
import os
import yaml

import torch
import genesis as gs
from pid import PIDcontroller
from odom import Odom
from utils import setup_random_cylindrical_obstacles
from path_search import (
    sample_free_xy, plan_path, PathFollower
)

class TrackerEnv:
    def __init__(self, num_envs, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.planner_device = torch.device("cpu")  # 规划全部走 CPU，避免打断渲染/控制


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
        self.dt = 0.01
        self.drone_height = float(cfg("drone.height", 1.0))
        self.world_xy_min = tuple(cfg("world.xy_min", [-10.0, -10.0]))
        self.world_xy_max = tuple(cfg("world.xy_max", [ 10.0,  10.0]))
        self.grid_cell     = float(cfg("world.grid_cell", 0.10))

        self.drone_radius  = float(cfg("drone.radius", 0.12))
        self.safety_margin = float(cfg("safety.margin", 0.12))

        # 规划/鲁棒性参数（从 cfg 读取）
        self.max_replan_tries = int(cfg("planner.max_replan_tries", 8))
        self.goals_per_try    = int(cfg("planner.goals_per_try", 5))
        self.extra_grid_margin_default = float(cfg("world.extra_grid_margin_default", 0.02))
        self.extra_grid_margin_min     = float(cfg("world.extra_grid_margin_min", 0.00))
        self.grid_cell_default         = self.grid_cell
        self.grid_cell_max             = float(max(self.grid_cell_default, cfg("planner.grid_cell_max", 0.12)))

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

        # 重规划冷却（由秒换算成步）
        self.replan_cooldown = int(float(cfg("replan.cooldown_sec", 1.0)) / self.dt)
        self.last_replan_step = -10**9

        # 创建仿真场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=100,
                camera_pos=(0.0, 0.0, 6),
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

        # 地面
        self.scene.add_entity(gs.morphs.Plane())

        # 无人机
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf"))

        # 控制参数
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config/pos.yaml"), "r") as file:
            self.pos_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)

        self.set_drone_imu()
        self.set_drone_controller()


        
        # 构建障碍物（你的函数返回：obs实体, obs_xy(list/np), obs_r(list/np)）
        n_obs = int(cfg("obstacles.n", 100))
        world_bounds_xyxy = (
            self.world_xy_min[0], self.world_xy_max[0],
            self.world_xy_min[1], self.world_xy_max[1],
        )
        obs, obs_xy, obs_r = setup_random_cylindrical_obstacles(self.scene, n_obstacles=n_obs, world_bounds=world_bounds_xyxy)

        # 转张量保存
        self.obs_xy = torch.tensor(obs_xy, dtype=torch.float32, device=self.device) if len(obs_xy) > 0 else torch.zeros((0,2), dtype=torch.float32, device=self.device)
        self.obs_r  = torch.tensor(obs_r , dtype=torch.float32, device=self.device) if len(obs_r ) > 0 else torch.zeros((0,),  dtype=torch.float32, device=self.device)


        # 可视化目标
        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.05,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.5, 0.5),
                ),
            ),
        )
        # 构建场景
        self.scene.build(n_envs=self.num_envs)


        # 状态缓冲
        self.step_count = 0
        self.nav_step_in_task = 0
        self.current_wp_idx = 0
        self.path_wps = []   # [(x,y), ...]
        self.goal_xy = None
        self.start_xy = None

        # 初始化无人机位置（从可行点采样）
        self.inflation = self.drone_radius + self.safety_margin
        start_xy = sample_free_xy(
            self.world_xy_min, self.world_xy_max,
            self.obs_xy, self.obs_r,
            safe_radius=self.inflation,
            device=self.device
        )
        self.start_xy = start_xy
        self.drone_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drone_init_pos[:, 0] = start_xy[0]
        self.drone_init_pos[:, 1] = start_xy[1]
        self.drone_init_pos[:, 2] = self.drone_height
        self.drone_init_quat = torch.tensor([1,0,0,0], device=self.device).repeat(self.num_envs, 1)

        init_target = torch.zeros((self.num_envs, 3), device=self.device)
        init_target[:, 0] = self.drone_init_pos[:, 0]     # 初始就放在无人机正上方/当前位置
        init_target[:, 1] = self.drone_init_pos[:, 1]
        init_target[:, 2] = self.drone_height
        self.target.set_pos(init_target)

        self.drone.set_pos(self.drone_init_pos)
        self.drone.set_quat(self.drone_init_quat)

        # 生成目标与路径
        self.plan_new_mission()

    # ---------- 初始化 / 规划 ----------

    def plan_new_mission(self):
        # 起点取当前位姿（在 CPU 做规划，避免打断 GPU 渲染/控制）
        cur = self.drone.get_pos()[0]
        start_xy = (cur[0].item(), cur[1].item())

        with torch.no_grad():
            # 把障碍数据搬到规划设备（仅重规划瞬间发生一次）
            obs_xy_cpu = self.obs_xy.to(self.planner_device)
            obs_r_cpu  = self.obs_r.to(self.planner_device)

            path_found = False
            chosen_goal = None
            chosen_path = None

            # 逐步放宽：extra_margin ↓，inflation ↓，必要时 grid_cell ↑
            for attempt in range(self.max_replan_tries):
                extra_margin = max(
                    self.extra_grid_margin_default - 0.01 * attempt,
                    self.extra_grid_margin_min
                )
                inflation = max(
                    self.inflation_default - 0.02 * attempt,
                    self.inflation_min
                )
                grid_cell = self.grid_cell_default if attempt < self.max_replan_tries - 2 else self.grid_cell_max

                # 多采几个目标点以增加成功率
                for _ in range(self.goals_per_try):
                    goal_xy = sample_free_xy(
                        self.world_xy_min, self.world_xy_max,
                        obs_xy_cpu, obs_r_cpu,
                        safe_radius=inflation, device=self.planner_device
                    )

                    path = plan_path(
                        start_xy, goal_xy,
                        obs_xy_cpu, obs_r_cpu,
                        self.world_xy_min, self.world_xy_max,
                        grid_cell,
                        inflation=inflation,
                        smooth=True,
                        extra_grid_margin=extra_margin,
                        device=self.planner_device
                    )
                    if path is not None and len(path) >= 2:
                        chosen_goal, chosen_path = goal_xy, path
                        path_found = True
                        break

                if path_found:
                    break

            if path_found:
                # 成功：更新终点球 & 跟随器
                self.goal_xy  = chosen_goal
                self.path_wps = chosen_path

                if getattr(self, "target", None) is not None:
                    goal_pos = torch.tensor([[chosen_goal[0], chosen_goal[1], self.drone_height]],
                                            dtype=torch.float32, device=self.device)
                    self.target.set_pos(goal_pos)

                # 创建/重置 follower（在本函数内做，外部不要再 reset）
                if not hasattr(self, "follower"):
                    self.follower = PathFollower(
                        self.path_wps, self.dt,
                        v_max=self.v_max, a_max=self.a_max,
                        warmup_time=self.warmup_time, v_init=self.v_init,
                        lookahead_time=self.lookahead_time, min_lookahead=self.min_lookahead,
                        slow_down_k=1.2
                    )
                else:
                    self.follower.reset_with_path(self.path_wps)
                return

            # 失败：不要 raise！优雅降级——原地悬停，稍后再试
            print("[Planner] WARNING: No path found. Holding position and will retry after cooldown.")

            # 构造一个“原地”路径（两点重合），让 follower 保持当前位置
            hold_xy = (start_xy[0], start_xy[1])
            self.goal_xy  = hold_xy
            self.path_wps = [hold_xy, hold_xy]

            if getattr(self, "target", None) is not None:
                goal_pos = torch.tensor([[hold_xy[0], hold_xy[1], self.drone_height]],
                                        dtype=torch.float32, device=self.device)
                self.target.set_pos(goal_pos)

            if not hasattr(self, "follower"):
                self.follower = PathFollower(
                    self.path_wps, self.dt,
                    v_max=self.v_max, a_max=self.a_max,
                    warmup_time=self.warmup_time, v_init=self.v_init,
                    lookahead_time=self.lookahead_time, min_lookahead=self.min_lookahead,
                    slow_down_k=1.2
                )
            else:
                self.follower.reset_with_path(self.path_wps)

            # 设置下一次重试的冷却起点（避免一帧多次重试）
            self.last_replan_step = self.step_count



    # ---------- 控制/观测配置 ----------

    def set_drone_imu(self):
        odom = Odom(num_envs=self.num_envs, device=self.device)
        odom.set_drone(self.drone)
        setattr(self.drone, 'odom', odom)

    def set_drone_controller(self):
        pid = PIDcontroller(
            num_envs=self.num_envs,
            odom=self.drone.odom,
            config=self.pos_ctrl_config,
            device=self.device,
            controller="position",
        )
        pid.set_drone(self.drone)
        setattr(self.drone, 'controller', pid)

    # ---------- 主循环 ----------

    def step(self):
        self.step_count += 1

        # 当前无人机 xy（用于轻微自适应；即使不传也行）
        cur = self.drone.get_pos()[0]
        cur_xy = (cur[0].item(), cur[1].item())

        # 时间参数化推进
        ref_xy = self.follower.step(cur_xy)

        # 形成期望 3D 位置
        target = torch.zeros((self.num_envs, 4), device=self.device)
        target[:, 0] = ref_xy[0]
        target[:, 1] = ref_xy[1]
        target[:, 2] = self.drone_height
        target[:, 3] = 0.0

        # 发送给已有控制器
        rpms = self.drone.controller.step(target)
        self.drone.set_propellels_rpm(rpms)

        # 仿真步进
        self.scene.step()

        # 判断是否到达终点（按弧长）
        if self.follower.reached_goal(thresh=self.goal_reach_thresh) and \
            (self.step_count - self.last_replan_step) >= self.replan_cooldown:

            self.last_replan_step = self.step_count
            print("[Mission] Reached goal. Replan.")
            self.plan_new_mission()
        
        

# -------------------- 入口 --------------------

if __name__ == "__main__":
    gs.init()
    env = TrackerEnv(num_envs=1, show_viewer=True)
    while True:
        env.step()
