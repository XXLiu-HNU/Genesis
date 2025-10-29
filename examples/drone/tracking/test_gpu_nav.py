"""
测试 GPU 路径规划算法：目标无人机使用 GPURoadmap + BatchedPathFollower
可视化目标的运动轨迹，验证规划和跟踪是否正常工作
"""
import os
import yaml
import torch
import genesis as gs
from pid import PIDcontroller
from odom import Odom
from utils import setup_random_cylindrical_obstacles
from path_search import BatchedPathFollower
from gpu_roadmap import GPURoadmap, sample_free_goals_gpu


class GPUNavTest:
    def __init__(self, num_envs=4, show_viewer=True):
        self.num_envs = num_envs
        self.device = gs.device
        
        # 加载配置
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
        
        # 基本参数
        self.dt = 0.01
        self.drone_height = float(cfg("drone.height", 1.0))
        self.world_xy_min = tuple(cfg("world.xy_min", [-10.0, -10.0]))
        self.world_xy_max = tuple(cfg("world.xy_max", [10.0, 10.0]))
        
        self.drone_radius = float(cfg("drone.radius", 0.12))
        self.safety_margin = float(cfg("safety.margin", 0.12))
        self.inflation = self.drone_radius + self.safety_margin
        
        # Follower 参数
        self.v_max = float(cfg("follower.v_max", 0.6))
        self.a_max = float(cfg("follower.a_max", 1.2))
        self.warmup_time = float(cfg("follower.warmup_time", 2.5))
        self.v_init = float(cfg("follower.v_init", 0.08))
        self.lookahead_time = float(cfg("follower.lookahead_time", 0.5))
        self.min_lookahead = float(cfg("follower.min_lookahead", 0.12))
        self.goal_reach_thresh = float(cfg("follower.goal_reach_thresh", 0.12))
        
        # 创建场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(0.0, 0.0, 8.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=80,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, num_envs)))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        
        # 地面
        self.scene.add_entity(gs.morphs.Plane())
        
        # 目标无人机（使用实际的无人机模型而不是球体）
        self.target_drone = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/target_drone_urdf/drone.urdf")
        )
        
        # 控制器
        with open(os.path.join(script_dir, "config/pos.yaml"), "r") as file:
            self.pos_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)
        
        odom = Odom(num_envs=self.num_envs, device=self.device)
        odom.set_drone(self.target_drone)
        self.target_drone.odom = odom
        
        pid = PIDcontroller(
            num_envs=self.num_envs,
            odom=odom,
            config=self.pos_ctrl_config,
            device=self.device,
            controller="position",
        )
        pid.set_drone(self.target_drone)
        self.target_drone.controller = pid
        
        # 障碍物
        n_obs = int(cfg("obstacles.n", 100))
        world_bounds = (
            self.world_xy_min[0], self.world_xy_max[0],
            self.world_xy_min[1], self.world_xy_max[1],
        )
        obs, obs_xy, obs_r = setup_random_cylindrical_obstacles(
            self.scene, n_obstacles=n_obs, world_bounds=world_bounds,
            origin_clearance=2.0, min_distance=2.0
        )
        
        self.obs_xy = torch.tensor(obs_xy, dtype=torch.float32, device=self.device) if len(obs_xy) > 0 else torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        self.obs_r = torch.tensor(obs_r, dtype=torch.float32, device=self.device) if len(obs_r) > 0 else torch.zeros((0,), dtype=torch.float32, device=self.device)
        
        # 构建场景
        self.scene.build(n_envs=num_envs)
        
        # 构建 GPU Roadmap
        print("[GPU Nav] Building GPU roadmap...")
        self.gpu_roadmap = GPURoadmap.build(
            world_min=self.world_xy_min,
            world_max=self.world_xy_max,
            obs_xy=self.obs_xy,
            obs_r=self.obs_r,
            n_nodes=1500,
            k=12,
            max_edge_len=2.5,
            clearance=self.inflation,
            device=self.device,
        )
        print(f"[GPU Nav] Roadmap built: {self.gpu_roadmap.nodes.shape[0]} nodes")
        
        # BatchedPathFollower
        self.follower = BatchedPathFollower(
            num_envs=self.num_envs,
            device=self.device,
            dt=self.dt,
            v_max=self.v_max,
            a_max=self.a_max,
            warmup_time=self.warmup_time,
            v_init=self.v_init,
            lookahead_time=self.lookahead_time,
            min_lookahead=self.min_lookahead,
            slow_down_k=1.2,
        )
        
        # 初始化每个 env 的路径
        print("[GPU Nav] Planning initial paths for all envs...")
        self._plan_all_envs()
        
        # 状态
        self.step_count = 0
        self.replan_interval = int(5.0 / self.dt)  # 每 5 秒重新规划一次
        
    def _plan_all_envs(self):
        """为所有 env 规划初始路径"""
        # 采样起点（无人机初始位置）
        from path_search import sample_free_xy_batch
        starts_xy = sample_free_xy_batch(
            self.world_xy_min, self.world_xy_max,
            self.obs_xy, self.obs_r,
            safe_radius=self.inflation,
            n=self.num_envs,
            device=self.device
        )  # (N,2)
        
        # 设置无人机初始位置
        init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        init_pos[:, :2] = starts_xy
        init_pos[:, 2] = self.drone_height
        self.target_drone.set_pos(init_pos, zero_velocity=True)
        self.target_drone.set_quat(
            torch.tensor([1, 0, 0, 0], device=self.device).repeat(self.num_envs, 1),
            zero_velocity=True
        )
        
        # 采样目标点
        goals_xy_t = sample_free_goals_gpu(
            self.world_xy_min, self.world_xy_max,
            self.obs_xy, self.obs_r,
            self.inflation,
            16,  # 每个 env 尝试 16 个候选目标
            self.device,
        )
        
        # 批量下载
        starts_cpu = starts_xy.cpu().numpy()
        goals_cpu = goals_xy_t.cpu().numpy()
        goals_list = [(float(goals_cpu[i, 0]), float(goals_cpu[i, 1])) for i in range(goals_cpu.shape[0])]
        
        # 为每个 env 规划路径
        for env_id in range(self.num_envs):
            start = (float(starts_cpu[env_id, 0]), float(starts_cpu[env_id, 1]))
            path = self.gpu_roadmap.query(start, goals_list, k_attach=8)
            
            if path and len(path) >= 2:
                print(f"[GPU Nav] Env {env_id}: path with {len(path)} waypoints, goal={path[-1]}")
                self.follower.reset_with_path(env_id, path)
            else:
                # 保持原地
                print(f"[GPU Nav] Env {env_id}: no path found, holding position")
                self.follower.reset_with_path(env_id, [start, start])
    
    def step(self):
        """仿真一步"""
        # 从 follower 获取下一个目标位置
        next_xy = self.follower.step()  # (num_envs, 2)
        
        # 构造参考位置 [x, y, z, yaw]
        ref_pos = torch.zeros((self.num_envs, 4), device=self.device)
        ref_pos[:, :2] = next_xy
        ref_pos[:, 2] = self.drone_height
        ref_pos[:, 3] = 0.0
        
        # 控制器
        prop_rpms = self.target_drone.controller.step(ref_pos)
        self.target_drone.set_propellels_rpm(prop_rpms)
        
        # 仿真
        self.scene.step()
        self.step_count += 1
        
        # 检查是否到达目标
        reached = self.follower.reached_goal(thresh=self.goal_reach_thresh)
        if torch.any(reached):
            reached_ids = reached.nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"[GPU Nav] Step {self.step_count}: Env(s) {reached_ids} reached goal, replanning...")
            self._replan_envs(reached_ids)
    
    def _replan_envs(self, env_ids):
        """为指定 env 重新规划路径"""
        if not env_ids:
            return
        
        # 获取当前位置
        cur_pos = self.target_drone.get_pos()
        
        # 采样新目标
        goals_xy_t = sample_free_goals_gpu(
            self.world_xy_min, self.world_xy_max,
            self.obs_xy, self.obs_r,
            self.inflation,
            16,
            self.device,
        )
        
        # 批量下载
        goals_cpu = goals_xy_t.cpu().numpy()
        goals_list = [(float(goals_cpu[i, 0]), float(goals_cpu[i, 1])) for i in range(goals_cpu.shape[0])]
        
        for env_id in env_ids:
            start = (float(cur_pos[env_id, 0].item()), float(cur_pos[env_id, 1].item()))
            path = self.gpu_roadmap.query(start, goals_list, k_attach=8)
            
            if path and len(path) >= 2:
                print(f"[GPU Nav] Env {env_id}: new path with {len(path)} waypoints")
                self.follower.reset_with_path(env_id, path)
            else:
                print(f"[GPU Nav] Env {env_id}: replan failed, holding")
                self.follower.reset_with_path(env_id, [start, start])
    
    def run(self, max_steps=5000):
        """运行仿真"""
        print(f"[GPU Nav] Running {max_steps} steps...")
        for _ in range(max_steps):
            self.step()
            
            # 定期重新规划所有 env（模拟动态任务）
            if self.step_count % self.replan_interval == 0 and self.step_count > 0:
                print(f"\n[GPU Nav] === Periodic replan at step {self.step_count} ===")
                self._replan_envs(list(range(self.num_envs)))


def main():
    gs.init(backend=gs.cuda)
    
    env = GPUNavTest(num_envs=4, show_viewer=True)
    env.run(max_steps=5000)
    
    print("\n[GPU Nav] Test completed successfully!")


if __name__ == "__main__":
    main()

