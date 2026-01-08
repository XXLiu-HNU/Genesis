"""
基于Genesis物理引擎的无人机仿真器

直接使用Genesis的完整刚体动力学引擎，与tracking/controller保持一致。
"""

import os
import sys
import yaml
import numpy as np
import torch

import genesis as gs

# 导入tracking的controller
tracking_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tracking')
if tracking_dir not in sys.path:
    sys.path.insert(0, tracking_dir)

from controller.pid import PIDcontroller
from controller.odom import Odom


class DroneSim:
    """
    单环境无人机仿真器
    
    内部使用Genesis完整物理引擎，提供简洁的接口。
    """
    
    def __init__(self, dt=0.01, show_viewer=False, use_controller=True):
        """
        参数:
        -----
        dt : float
            时间步长
        show_viewer : bool
            是否显示可视化窗口
        use_controller : bool
            是否自动初始化PID控制器
        """
        self.dt = dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=100,
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        
        # 添加地面
        self.scene.add_entity(gs.morphs.Plane())
        
        # 添加无人机
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/target_drone_urdf/drone.urdf")
        )
        
        # 构建场景
        self.scene.build(n_envs=1)
        
        # 初始化控制器（可选）
        self.controller = None
        if use_controller:
            self._setup_controller()
    
    def _setup_controller(self):
        """设置Odom和PID控制器（直接复用tracking的实现）"""
        # Odom
        odom = Odom(num_envs=1, device=self.device)
        odom.set_drone(self.drone)
        self.drone.odom = odom
        
        # 加载PID配置
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'tracking/controller/config/pos.yaml'
        )
        with open(config_path, 'r') as f:
            pid_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # PID控制器
        controller = PIDcontroller(
            num_envs=1,
            odom=odom,
            config=pid_config,
            device=self.device,
            controller="position",
        )
        controller.set_drone(self.drone)
        self.controller = controller
    
    def reset(self, position=None, quaternion=None):
        """
        重置无人机状态
        
        参数:
        -----
        position : array-like, optional
            初始位置 [x, y, z]
        quaternion : array-like, optional
            初始四元数 [w, x, y, z]
        """
        if position is not None:
            pos = torch.tensor([position], dtype=torch.float32, device=self.device)
            self.drone.set_pos(pos)
        
        if quaternion is not None:
            quat = torch.tensor([quaternion], dtype=torch.float32, device=self.device)
            self.drone.set_quat(quat)
        else:
            # 默认四元数
            quat = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device)
            self.drone.set_quat(quat)
        
        # 重置控制器
        if self.controller is not None:
            self.controller.reset()
            self.drone.odom.reset(quat[0], None)
    
    def step_rpm(self, rpms):
        """
        执行一步仿真（直接指定RPM）
        
        参数:
        -----
        rpms : array-like
            4个电机的RPM值
        
        返回:
        -----
        state : dict
            当前状态
        """
        rpms_tensor = torch.tensor([rpms], dtype=torch.float32, device=self.device)
        self.drone.set_propellels_rpm(rpms_tensor)
        self.scene.step()
        
        return self.get_state()
    
    def step_controller(self, target_position):
        """
        执行一步仿真（使用PID控制器）
        
        参数:
        -----
        target_position : array-like
            目标位置 [x, y, z] 或 [x, y, z, yaw]
        
        返回:
        -----
        state : dict
            当前状态
        """
        if self.controller is None:
            raise RuntimeError("控制器未初始化，请设置use_controller=True")
        
        # 确保是4维
        if len(target_position) == 3:
            target = list(target_position) + [0.0]
        else:
            target = target_position
        
        target_tensor = torch.tensor([target], dtype=torch.float32, device=self.device)
        
        # 计算RPM
        rpms = self.controller.step(target_tensor)
        
        # 执行仿真
        self.drone.set_propellels_rpm(rpms)
        self.scene.step()
        
        return self.get_state()
    
    def get_state(self):
        """
        获取当前状态
        
        返回:
        -----
        state : dict
            包含position, velocity, quaternion, angular_velocity的字典
        """
        return {
            'position': self.drone.get_pos()[0].cpu().numpy(),
            'velocity': self.drone.get_vel()[0].cpu().numpy(),
            'quaternion': self.drone.get_quat()[0].cpu().numpy(),
            'angular_velocity': self.drone.get_ang()[0].cpu().numpy(),
        }


class DroneSimBatch:
    """
    批量环境无人机仿真器（支持并行仿真）
    
    与DroneSim接口一致，但支持多个环境并行。
    """
    
    def __init__(self, num_envs=10, dt=0.01, show_viewer=False, 
                 rendered_envs=None, use_controller=True):
        """
        参数:
        -----
        num_envs : int
            并行环境数量
        dt : float
            时间步长
        show_viewer : bool
            是否显示可视化窗口
        rendered_envs : int, optional
            渲染的环境数量（默认min(10, num_envs)）
        use_controller : bool
            是否自动初始化PID控制器
        """
        self.num_envs = num_envs
        self.dt = dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if rendered_envs is None:
            rendered_envs = min(10, num_envs)
        
        # 创建场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=100,
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(rendered_envs))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        
        # 添加地面
        self.scene.add_entity(gs.morphs.Plane())
        
        # 添加无人机
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/target_drone_urdf/drone.urdf")
        )
        
        # 构建场景
        self.scene.build(n_envs=num_envs)
        
        # 初始化控制器（可选）
        self.controller = None
        if use_controller:
            self._setup_controller()
    
    def _setup_controller(self):
        """设置Odom和PID控制器"""
        # Odom
        odom = Odom(num_envs=self.num_envs, device=self.device)
        odom.set_drone(self.drone)
        self.drone.odom = odom
        
        # 加载PID配置
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'tracking/controller/config/pos.yaml'
        )
        with open(config_path, 'r') as f:
            pid_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # PID控制器
        controller = PIDcontroller(
            num_envs=self.num_envs,
            odom=odom,
            config=pid_config,
            device=self.device,
            controller="position",
        )
        controller.set_drone(self.drone)
        self.controller = controller
    
    def reset(self, position=None, quaternion=None, env_idx=None):
        """
        重置无人机状态
        
        参数:
        -----
        position : array-like, optional
            初始位置，形状 (num_envs, 3) 或 (3,)
        quaternion : array-like, optional
            初始四元数，形状 (num_envs, 4) 或 (4,)
        env_idx : array-like, optional
            要重置的环境索引
        """
        if position is not None:
            pos = torch.tensor(position, dtype=torch.float32, device=self.device)
            if pos.ndim == 1:
                pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
            self.drone.set_pos(pos)
        
        if quaternion is not None:
            quat = torch.tensor(quaternion, dtype=torch.float32, device=self.device)
            if quat.ndim == 1:
                quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        else:
            quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
            quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        
        self.drone.set_quat(quat)
        
        # 重置控制器
        if self.controller is not None:
            self.controller.reset(env_idx)
            self.drone.odom.reset(quat[0] if env_idx is None else quat, env_idx)
    
    def step_rpm(self, rpms):
        """
        执行一步仿真（直接指定RPM）
        
        参数:
        -----
        rpms : array-like
            形状 (num_envs, 4) 的RPM值
        
        返回:
        -----
        states : dict
            当前状态（batch）
        """
        rpms_tensor = torch.tensor(rpms, dtype=torch.float32, device=self.device)
        self.drone.set_propellels_rpm(rpms_tensor)
        self.scene.step()
        
        return self.get_state()
    
    def step_controller(self, target_positions):
        """
        执行一步仿真（使用PID控制器）
        
        参数:
        -----
        target_positions : array-like
            形状 (num_envs, 3) 或 (num_envs, 4) 的目标位置
        
        返回:
        -----
        states : dict
            当前状态（batch）
        """
        if self.controller is None:
            raise RuntimeError("控制器未初始化，请设置use_controller=True")
        
        targets = torch.tensor(target_positions, dtype=torch.float32, device=self.device)
        
        # 确保是4维
        if targets.shape[-1] == 3:
            zeros = torch.zeros((targets.shape[0], 1), device=self.device)
            targets = torch.cat([targets, zeros], dim=1)
        
        # 计算RPM
        rpms = self.controller.step(targets)
        
        # 执行仿真
        self.drone.set_propellels_rpm(rpms)
        self.scene.step()
        
        return self.get_state()
    
    def get_state(self):
        """
        获取当前状态
        
        返回:
        -----
        states : dict
            包含position, velocity, quaternion, angular_velocity的字典（batch）
        """
        return {
            'position': self.drone.get_pos().cpu().numpy(),
            'velocity': self.drone.get_vel().cpu().numpy(),
            'quaternion': self.drone.get_quat().cpu().numpy(),
            'angular_velocity': self.drone.get_ang().cpu().numpy(),
        }
