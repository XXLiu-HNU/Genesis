"""
Parallel Target Navigation Environment for Large-Scale RL Training
支持大规模并行仿真的目标无人机导航环境
"""
import os
import yaml
import torch
import genesis as gs
from typing import Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.pid import PIDcontroller
from controller.odom import Odom
from utils import setup_random_cylindrical_obstacles


class VectorizedPathFollower:
    """
    批量化的路径跟随器，为每个环境维护独立的状态
    使用导航策略：朝目标移动 + 势场法避开障碍
    """
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        obs_xy: torch.Tensor,        # 障碍物位置 [M, 2]
        obs_r: torch.Tensor,          # 障碍物半径 [M]
        v_max: float = 0.6,
        a_max: float = 1.2,
        goal_reach_thresh: float = 0.3,
        obstacle_avoidance_gain: float = 2.0,    # 避障增益
        safe_distance: float = 0.5,               # 安全距离
    ):
        self.num_envs = num_envs
        self.device = device
        self.v_max = v_max
        self.a_max = a_max
        self.goal_reach_thresh = goal_reach_thresh
        self.obstacle_avoidance_gain = obstacle_avoidance_gain
        self.safe_distance = safe_distance
        
        # 障碍物信息
        self.obs_xy = obs_xy  # [M, 2]
        self.obs_r = obs_r    # [M]
        
        # 每个环境的状态
        self.goal_xy = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
        self.current_vel = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
        self.reached = torch.zeros(num_envs, device=device, dtype=torch.bool)
        
    def set_goals(self, goals: torch.Tensor):
        """设置目标位置 [num_envs, 2]"""
        self.goal_xy.copy_(goals)
        self.reached.fill_(False)
        
    def step(self, current_pos: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        批量计算所有环境的下一个目标位置（带障碍物避让）
        Args:
            current_pos: [num_envs, 2] 当前位置
            dt: 时间步长
        Returns:
            target_pos: [num_envs, 2] 目标位置
        """
        # 1. 计算朝向目标的吸引力
        to_goal = self.goal_xy - current_pos                      # [num_envs, 2]
        dist_to_goal = torch.norm(to_goal, dim=-1, keepdim=True)  # [num_envs, 1]
        goal_direction = to_goal / (dist_to_goal + 1e-6)          # [num_envs, 2]
        
        # 2. 计算障碍物的排斥力（势场法）
        repulsion = self._compute_obstacle_repulsion(current_pos)  # [num_envs, 2]
        
        # 3. 合成方向：吸引力 + 排斥力
        combined_direction = goal_direction + repulsion            # [num_envs, 2]
        combined_norm = torch.norm(combined_direction, dim=-1, keepdim=True)
        combined_direction = combined_direction / (combined_norm + 1e-6)
        
        # 4. 计算目标点距离（在接近目标时减速）
        slowdown_dist = 1.0
        speed_factor = torch.clamp(dist_to_goal / slowdown_dist, 0.0, 1.0)
        
        # 5. 目标点设置在combined_direction方向上的固定距离
        # 使用固定的前瞻距离，避免速度累积
        # 前瞻时间0.2秒可以让实际速度接近v_max（允许1.5倍超调）
        lookahead_dist = self.v_max * 0.2 * speed_factor
        target_pos = current_pos + combined_direction * lookahead_dist
        
        # 6. 更新到达状态
        self.reached = dist_to_goal.squeeze(-1) < self.goal_reach_thresh
        
        return target_pos
    
    def _compute_obstacle_repulsion(self, current_pos: torch.Tensor) -> torch.Tensor:
        """
        计算障碍物对无人机的排斥力（势场法）
        Args:
            current_pos: [num_envs, 2] 当前位置
        Returns:
            repulsion: [num_envs, 2] 排斥力方向
        """
        if self.obs_xy.shape[0] == 0:
            # 无障碍物，返回零向量
            return torch.zeros_like(current_pos)
        
        # 计算到所有障碍物的向量
        # current_pos: [num_envs, 2] -> [num_envs, 1, 2]
        # obs_xy: [M, 2] -> [1, M, 2]
        pos_expanded = current_pos.unsqueeze(1)                    # [num_envs, 1, 2]
        obs_expanded = self.obs_xy.unsqueeze(0)                    # [1, M, 2]
        
        # 到障碍物中心的向量
        to_obs = pos_expanded - obs_expanded                       # [num_envs, M, 2]
        dist_to_obs = torch.norm(to_obs, dim=-1, keepdim=True)     # [num_envs, M, 1]
        
        # 到障碍物边界的距离
        obs_r_expanded = self.obs_r.view(1, -1, 1)                # [1, M, 1]
        dist_to_boundary = dist_to_obs - obs_r_expanded            # [num_envs, M, 1]
        
        # 排斥力只在安全距离内起作用
        # 使用指数衰减：力 = gain * exp(-dist/safe_dist)
        repulsion_magnitude = self.obstacle_avoidance_gain * torch.exp(
            -torch.clamp(dist_to_boundary, min=0.0) / self.safe_distance
        )  # [num_envs, M, 1]
        
        # 只对距离小于安全距离的障碍物应用排斥力
        active_mask = (dist_to_boundary < self.safe_distance).float()  # [num_envs, M, 1]
        repulsion_magnitude = repulsion_magnitude * active_mask
        
        # 排斥力方向（远离障碍物）
        repulsion_direction = to_obs / (dist_to_obs + 1e-6)       # [num_envs, M, 2]
        
        # 单个障碍物的排斥力
        repulsion_per_obs = repulsion_direction * repulsion_magnitude  # [num_envs, M, 2]
        
        # 所有障碍物排斥力的总和
        total_repulsion = repulsion_per_obs.sum(dim=1)            # [num_envs, 2]
        
        return total_repulsion
    
    def check_reached(self) -> torch.Tensor:
        """返回已到达目标的环境mask [num_envs]"""
        return self.reached


class ParallelTargetNavEnv:
    """
    大规模并行的目标无人机导航环境
    """
    def __init__(
        self,
        num_envs: int,
        show_viewer: bool = False,
        config_path: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config/search.yaml")
        
        with open(config_path, "r") as f:
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
        self.world_xy_min = torch.tensor(cfg("world.xy_min", [-10.0, -10.0]), device=self.device)
        self.world_xy_max = torch.tensor(cfg("world.xy_max", [10.0, 10.0]), device=self.device)
        
        self.drone_radius = float(cfg("drone.radius", 0.12))
        self.safety_margin = float(cfg("safety.margin", 0.12))
        self.safe_radius = self.drone_radius + self.safety_margin
        
        # 导航参数
        self.v_max = float(cfg("follower.v_max", 0.6))
        self.a_max = float(cfg("follower.a_max", 1.2))
        self.goal_reach_thresh = float(cfg("follower.goal_reach_thresh", 0.3))
        self.obstacle_avoidance_gain = float(cfg("follower.obstacle_avoidance_gain", 2.0))
        self.safe_distance = float(cfg("follower.safe_distance", 0.5))
        self.replan_cooldown_steps = int(float(cfg("replan.cooldown_sec", 1.0)) / self.dt)
        
        # 创建仿真场景
        self.rendered_env_num = min(16, self.num_envs)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=100,
                camera_pos=(0.0, 0.0, 8),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=80,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.rendered_env_num))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False)
        )
        
        # 添加地面
        self.scene.add_entity(gs.morphs.Plane())
        
        # 添加目标无人机
        self.target_drone = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/target_drone_urdf/drone.urdf")
        )
        
        # 加载位置控制器配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config/pos.yaml"), "r") as f:
            self.pos_ctrl_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 设置IMU和控制器
        self._setup_target_drone_controller()
        
        # 添加障碍物
        n_obs = int(cfg("obstacles.n", 100))
        world_bounds_xyxy = (
            self.world_xy_min[0].item(), self.world_xy_max[0].item(),
            self.world_xy_min[1].item(), self.world_xy_max[1].item(),
        )
        obs, obs_xy, obs_r = setup_random_cylindrical_obstacles(
            self.scene, n_obstacles=n_obs, world_bounds=world_bounds_xyxy
        )
        
        # 转为GPU张量
        if len(obs_xy) > 0:
            self.obs_xy = torch.tensor(obs_xy, dtype=torch.float32, device=self.device)
            self.obs_r = torch.tensor(obs_r, dtype=torch.float32, device=self.device)
        else:
            self.obs_xy = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            self.obs_r = torch.zeros((0,), dtype=torch.float32, device=self.device)
        
        # 添加目标可视化球
        self.target_markers = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.08,
                fixed=True,
                collision=False,
                batch_fixed_verts=True,  # 允许每个环境有不同的位置
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.2, 0.2)),
            ),
        )
        
        # 构建场景
        self.scene.build(n_envs=self.num_envs)
        
        # 初始化环境状态
        self.step_count = 0
        self.last_replan_step = torch.full(
            (self.num_envs,), -10**9, dtype=torch.long, device=self.device
        )
        
        # 创建批量路径跟随器（带障碍物信息）
        self.follower = VectorizedPathFollower(
            num_envs=self.num_envs,
            device=self.device,
            obs_xy=self.obs_xy,
            obs_r=self.obs_r,
            v_max=self.v_max,
            a_max=self.a_max,
            goal_reach_thresh=self.goal_reach_thresh,
            obstacle_avoidance_gain=self.obstacle_avoidance_gain,
            safe_distance=self.safe_distance,
        )
        
        # 初始化位置
        self._reset_all_envs()
        
    def _setup_target_drone_controller(self):
        """设置目标无人机的IMU和控制器"""
        odom = Odom(num_envs=self.num_envs, device=self.device)
        odom.set_drone(self.target_drone)
        setattr(self.target_drone, 'odom', odom)
        
        pid = PIDcontroller(
            num_envs=self.num_envs,
            odom=self.target_drone.odom,
            config=self.pos_ctrl_config,
            device=self.device,
            controller="position",
        )
        pid.set_drone(self.target_drone)
        setattr(self.target_drone, 'controller', pid)
    
    def _sample_free_positions_batch(self, n: int) -> torch.Tensor:
        """
        批量采样不与障碍碰撞的位置
        Args:
            n: 采样数量
        Returns:
            positions: [n, 2] 位置张量
        """
        positions = torch.empty((n, 2), device=self.device, dtype=torch.float32)
        
        if self.obs_xy.shape[0] == 0:
            # 无障碍物，直接均匀采样
            positions[:, 0] = torch.empty(n, device=self.device).uniform_(
                self.world_xy_min[0].item(), self.world_xy_max[0].item()
            )
            positions[:, 1] = torch.empty(n, device=self.device).uniform_(
                self.world_xy_min[1].item(), self.world_xy_max[1].item()
            )
            return positions
        
        # 有障碍物，需要碰撞检测
        sampled = 0
        max_batch = 1000
        max_tries = 100
        
        for _ in range(max_tries):
            if sampled >= n:
                break
            
            # 批量采样候选点
            remaining = n - sampled
            batch_size = min(remaining * 5, max_batch)  # 过采样以提高效率
            
            candidates = torch.empty((batch_size, 2), device=self.device, dtype=torch.float32)
            candidates[:, 0] = torch.empty(batch_size, device=self.device).uniform_(
                self.world_xy_min[0].item(), self.world_xy_max[0].item()
            )
            candidates[:, 1] = torch.empty(batch_size, device=self.device).uniform_(
                self.world_xy_min[1].item(), self.world_xy_max[1].item()
            )
            
            # 向量化碰撞检测
            # candidates: [batch_size, 2], obs_xy: [M, 2]
            diff = candidates.unsqueeze(1) - self.obs_xy.unsqueeze(0)  # [batch_size, M, 2]
            dist = torch.norm(diff, dim=-1)  # [batch_size, M]
            min_dist = dist.min(dim=-1).values  # [batch_size]
            
            # 选择满足安全距离的点
            valid_mask = min_dist >= (self.obs_r.max() + self.safe_radius)
            valid_candidates = candidates[valid_mask]
            
            # 填充到结果中
            n_valid = min(valid_candidates.shape[0], remaining)
            if n_valid > 0:
                positions[sampled:sampled+n_valid] = valid_candidates[:n_valid]
                sampled += n_valid
        
        if sampled < n:
            print(f"Warning: Only sampled {sampled}/{n} valid positions")
            # 对未采样到的，使用中心点
            positions[sampled:] = (self.world_xy_min + self.world_xy_max) / 2
        
        return positions
    
    def _reset_all_envs(self):
        """重置所有环境"""
        # 采样初始位置
        init_positions = self._sample_free_positions_batch(self.num_envs)
        
        # 设置目标无人机位置
        drone_pos_3d = torch.zeros((self.num_envs, 3), device=self.device)
        drone_pos_3d[:, :2] = init_positions
        drone_pos_3d[:, 2] = self.drone_height
        self.target_drone.set_pos(drone_pos_3d)
        
        drone_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(self.num_envs, 1)
        self.target_drone.set_quat(drone_quat)
        
        # 采样目标点
        goal_positions = self._sample_free_positions_batch(self.num_envs)
        self.follower.set_goals(goal_positions)
        
        # 更新目标标记位置
        marker_pos = torch.zeros((self.num_envs, 3), device=self.device)
        marker_pos[:, :2] = goal_positions
        marker_pos[:, 2] = self.drone_height
        self.target_markers.set_pos(marker_pos)
        
        # 重置控制器
        self.target_drone.controller.reset()
        
        print(f"Reset all {self.num_envs} environments")
    
    def _replan_for_envs(self, env_mask: torch.Tensor):
        """
        为指定的环境重新规划
        Args:
            env_mask: [num_envs] bool张量，True表示需要重规划
        """
        if not env_mask.any():
            return
        
        n_replan = env_mask.sum().item()
        
        # 采样新目标
        new_goals = self._sample_free_positions_batch(n_replan)
        
        # 更新目标
        self.follower.goal_xy[env_mask] = new_goals
        self.follower.reached[env_mask] = False
        self.follower.current_vel[env_mask] = 0.0
        
        # 更新目标标记
        marker_pos = self.target_markers.get_pos()
        marker_pos[env_mask, 0] = new_goals[:, 0]
        marker_pos[env_mask, 1] = new_goals[:, 1]
        self.target_markers.set_pos(marker_pos)
        
        # 更新重规划时间戳
        self.last_replan_step[env_mask] = self.step_count
    
    def step(self):
        """执行一步仿真"""
        self.step_count += 1
        
        # 获取当前位置
        current_pos_3d = self.target_drone.get_pos()  # [num_envs, 3]
        current_pos_xy = current_pos_3d[:, :2]  # [num_envs, 2]
        
        # 计算目标位置
        target_pos_xy = self.follower.step(current_pos_xy, self.dt)
        
        # 构造3D目标
        target_3d = torch.zeros((self.num_envs, 4), device=self.device)
        target_3d[:, :2] = target_pos_xy
        target_3d[:, 2] = self.drone_height
        target_3d[:, 3] = 0.0
        
        # 执行控制
        rpms = self.target_drone.controller.step(target_3d)
        self.target_drone.set_propellels_rpm(rpms)
        
        # 仿真步进
        self.scene.step()
        
        # 检查是否需要重规划
        reached_mask = self.follower.check_reached()
        cooldown_mask = (self.step_count - self.last_replan_step) >= self.replan_cooldown_steps
        replan_mask = reached_mask & cooldown_mask
        
        if replan_mask.any():
            n_replan = replan_mask.sum().item()
            # print(f"Step {self.step_count}: Replanning for {n_replan} envs")
            self._replan_for_envs(replan_mask)
    
    def get_target_states(self) -> dict:
        """
        获取目标无人机的状态，供追踪无人机使用
        Returns:
            dict with keys:
                - pos: [num_envs, 3] 位置
                - vel: [num_envs, 3] 速度
                - quat: [num_envs, 4] 四元数
        """
        return {
            'pos': self.target_drone.get_pos(),
            'vel': self.target_drone.get_vel(),
            'quat': self.target_drone.get_quat(),
        }


# 测试函数
def test_scalability(env_sizes=[100, 500, 1000, 2000, 5000]):
    """测试不同规模环境的性能"""
    import time
    
    gs.init(backend=gs.gpu)
    
    results = []
    
    for n_envs in env_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_envs} environments")
        print(f"{'='*60}")
        
        try:
            # 创建环境
            start_time = time.time()
            env = ParallelTargetNavEnv(num_envs=n_envs, show_viewer=False)
            init_time = time.time() - start_time
            
            # 运行若干步并计时
            n_steps = 100
            start_time = time.time()
            for _ in range(n_steps):
                env.step()
            elapsed = time.time() - start_time
            
            fps = n_steps / elapsed
            steps_per_sec = fps * n_envs
            
            results.append({
                'n_envs': n_envs,
                'init_time': init_time,
                'fps': fps,
                'steps_per_sec': steps_per_sec,
            })
            
            print(f"Initialization time: {init_time:.2f}s")
            print(f"Simulation FPS: {fps:.2f}")
            print(f"Total steps/sec: {steps_per_sec:.0f}")
            
            # 清理
            del env
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed with {n_envs} envs: {e}")
            break
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("SCALABILITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Envs':>8} | {'Init(s)':>8} | {'FPS':>8} | {'Steps/s':>10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['n_envs']:>8} | {r['init_time']:>8.2f} | {r['fps']:>8.2f} | {r['steps_per_sec']:>10.0f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 运行扩展性测试
        test_scalability()
    else:
        # 正常运行
        gs.init(backend=gs.gpu)
        env = ParallelTargetNavEnv(num_envs=1000, show_viewer=True)
        
        print("Running parallel navigation with 1000 environments...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                env.step()
        except KeyboardInterrupt:
            print("\nStopped by user")
