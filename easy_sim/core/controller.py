"""
独立的PID控制器模块 - 不依赖Genesis
适用于easy_sim无人机仿真器
"""

import numpy as np
from scipy.spatial.transform import Rotation


class Odom:
    """状态估计器 - 从仿真器状态中提取和转换数据"""
    
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        
        # Body frame data (机体坐标系)
        self.body_euler = np.zeros((num_envs, 3))  # [roll, pitch, yaw]
        self.body_linear_vel = np.zeros((num_envs, 3))
        self.body_ang_vel = np.zeros((num_envs, 3))
        self.body_quat = np.zeros((num_envs, 4))  # [w, x, y, z]
        
        # World frame data (世界坐标系)
        self.world_pos = np.zeros((num_envs, 3))
        self.world_linear_vel = np.zeros((num_envs, 3))
        self.world_ang_vel = np.zeros((num_envs, 3))
        
        # Previous states for derivative computation
        self.last_body_linear_vel = np.zeros((num_envs, 3))
        self.last_world_linear_vel = np.zeros((num_envs, 3))
        self.last_world_pos = np.zeros((num_envs, 3))
        
        self.sim = None
    
    def set_sim(self, sim):
        """设置仿真器引用"""
        self.sim = sim
    
    def update(self):
        """从仿真器更新所有状态"""
        if self.sim is None:
            raise RuntimeError("Simulator not set. Call set_sim() first.")
        
        state = self.sim.get_state()
        
        # 更新世界坐标系数据
        self.last_world_pos[:] = self.world_pos
        self.last_world_linear_vel[:] = self.world_linear_vel
        
        if self.num_envs == 1:
            self.world_pos[0] = state['position']
            self.world_linear_vel[0] = state['velocity']
            self.world_ang_vel[0] = state['angular_velocity']
            self.body_quat[0] = state['quaternion']
            self.body_euler[0] = state['euler_angles']
        else:
            self.world_pos = state['position']
            self.world_linear_vel = state['velocity']
            self.world_ang_vel = state['angular_velocity']
            self.body_quat = state['quaternion']
            self.body_euler = state['euler_angles']
        
        # 转换到机体坐标系
        self._update_body_frame()
    
    def _update_body_frame(self):
        """将世界坐标系速度转换到机体坐标系"""
        for i in range(self.num_envs):
            # 使用四元数进行旋转变换
            R = Rotation.from_quat([
                self.body_quat[i, 1], self.body_quat[i, 2],
                self.body_quat[i, 3], self.body_quat[i, 0]
            ])
            # 世界坐标系速度转换到机体坐标系
            self.body_linear_vel[i] = R.inv().apply(self.world_linear_vel[i])
            self.body_ang_vel[i] = R.inv().apply(self.world_ang_vel[i])
    
    def reset(self, env_idx=None):
        """重置状态"""
        if env_idx is None:
            idx = slice(None)
        else:
            idx = env_idx
        
        self.body_euler[idx] = 0.0
        self.body_linear_vel[idx] = 0.0
        self.body_ang_vel[idx] = 0.0
        self.world_pos[idx] = 0.0
        self.world_linear_vel[idx] = 0.0
        self.world_ang_vel[idx] = 0.0
        self.last_body_linear_vel[idx] = 0.0
        self.last_world_linear_vel[idx] = 0.0
        self.last_world_pos[idx] = 0.0


class PIDController:
    """
    独立的PID控制器
    支持三种控制模式：position, angle, rate
    """
    
    def __init__(self, num_envs, odom, config, controller_type="angle"):
        """
        Parameters:
        -----------
        num_envs : int
            环境数量
        odom : Odom
            状态估计器
        config : dict
            PID参数配置
        controller_type : str
            控制器类型：'position', 'angle', 'rate'
        """
        self.num_envs = num_envs
        self.odom = odom
        self.controller_type = controller_type
        
        # 推力补偿
        self.thrust_compensate = config.get("thrust_compensate", 1.0)
        
        # 加载PID参数
        ang_cfg = config.get("ang", {})
        rat_cfg = config.get("rat", {})
        pos_cfg = config.get("pos", {})
        
        # Angular controller (angle mode)
        self.kp_a = self._get_gains(ang_cfg, "kp_r", "kp_p", "kp_y")
        self.ki_a = self._get_gains(ang_cfg, "ki_r", "ki_p", "ki_y")
        self.kd_a = self._get_gains(ang_cfg, "kd_r", "kd_p", "kd_y")
        self.P_term_a = np.zeros((num_envs, 3))
        self.I_term_a = np.zeros((num_envs, 3))
        self.D_term_a = np.zeros((num_envs, 3))
        
        # Angular rate controller (rate mode)
        self.kp_r = self._get_gains(rat_cfg, "kp_r", "kp_p", "kp_y")
        self.ki_r = self._get_gains(rat_cfg, "ki_r", "ki_p", "ki_y")
        self.kd_r = self._get_gains(rat_cfg, "kd_r", "kd_p", "kd_y")
        self.P_term_r = np.zeros((num_envs, 3))
        self.I_term_r = np.zeros((num_envs, 3))
        self.D_term_r = np.zeros((num_envs, 3))
        
        # Position controller
        self.kp_p = self._get_gains(pos_cfg, "kp_x", "kp_y", "kp_t")
        self.ki_p = self._get_gains(pos_cfg, "ki_x", "ki_y", "ki_t")
        self.kd_p = self._get_gains(pos_cfg, "kd_x", "kd_y", "kd_t")
        self.P_term_p = np.zeros((num_envs, 3))
        self.I_term_p = np.zeros((num_envs, 3))
        self.D_term_p = np.zeros((num_envs, 3))
        
        # 基础RPM
        self.base_rpm = config.get("base_rpm", 56500)
        
        # 其他状态变量
        self.throttle_command = np.zeros((num_envs,))
        self.last_body_ang_vel = np.zeros((num_envs, 3))
        self.body_set_point = np.zeros((num_envs, 3))
        self.pid_output = np.zeros((num_envs, 3))
        self.cur_setpoint_error = np.zeros((num_envs, 3))
        self.last_setpoint_error = np.zeros((num_envs, 3))
        
        # 选择控制器
        if controller_type == "position":
            self._controller_func = self._position_controller
        elif controller_type == "angle":
            self._controller_func = self._angle_controller
        elif controller_type == "rate":
            self._controller_func = self._rate_controller
        else:
            raise ValueError(f"Invalid controller type: {controller_type}")
    
    def _get_gains(self, cfg, k1, k2, k3):
        """从配置中提取PID增益"""
        return np.array([
            cfg.get(k1, 0.0),
            cfg.get(k2, 0.0),
            cfg.get(k3, 0.0)
        ]).reshape(1, 3).repeat(self.num_envs, axis=0)
    
    def step(self, action):
        """
        执行一步控制
        
        Parameters:
        -----------
        action : np.ndarray
            控制指令，形状根据控制器类型而定：
            - position: (num_envs, 4) [x, y, z, 0]
            - angle: (num_envs, 4) [roll, pitch, yaw, thrust]
            - rate: (num_envs, 4) [roll_rate, pitch_rate, yaw_rate, thrust]
        
        Returns:
        --------
        motor_rpms : np.ndarray
            4个电机的RPM，形状 (num_envs, 4)
        """
        # 更新状态
        self.odom.update()
        
        # 调用对应的控制器
        self._controller_func(action)
        
        # 混控器：将PID输出转换为电机RPM
        motor_rpms = self._mixer(action)
        
        return motor_rpms
    
    def _mixer(self, action):
        """
        混控器：将控制输出转换为4个电机的RPM
        
        根据drone.urdf的布局（X型）：
        prop0: 前右 (+x,-y) - 顺时针
        prop1: 后右 (-x,-y) - 逆时针
        prop2: 后左 (-x,+y) - 顺时针
        prop3: 前左 (+x,+y) - 逆时针
        
        混控矩阵：
        prop0: throttle - roll - pitch - yaw
        prop1: 后右 - roll + pitch + yaw
        prop2: 后左 + roll + pitch - yaw
        prop3: 前左 + roll - pitch + yaw
        """
        # 计算总推力
        throttle_rc = np.clip(self.throttle_command * 3, 0.0, 3.0) * self.base_rpm
        throttle_action = np.clip(action[:, -1] * 3 + self.thrust_compensate, 0.0, 3.0) * self.base_rpm
        throttle = (throttle_rc + throttle_action).reshape(-1, 1)
        
        # 混控
        roll = self.pid_output[:, 0:1]
        pitch = self.pid_output[:, 1:2]
        yaw = self.pid_output[:, 2:3]
        
        motor_outputs = np.concatenate([
            throttle - roll - pitch - yaw,  # prop0 (前右)
            throttle - roll + pitch + yaw,  # prop1 (后右)
            throttle + roll + pitch - yaw,  # prop2 (后左)
            throttle + roll - pitch + yaw,  # prop3 (前左)
        ], axis=1)
        
        return np.clip(motor_outputs, 1, self.base_rpm * 3.5)
    
    def _rate_controller(self, action):
        """
        角速率控制器
        输入：期望的角速率 [roll_rate, pitch_rate, yaw_rate, thrust]
        """
        self.body_set_point[:] = action[:, :3] * 15
        
        self.last_setpoint_error[:] = self.cur_setpoint_error
        self.cur_setpoint_error[:] = self.body_set_point - self.odom.body_ang_vel
        
        self.P_term_r[:] = self.cur_setpoint_error * self.kp_r
        self.I_term_r[:] = np.clip(self.I_term_r + self.cur_setpoint_error * self.ki_r, -0.5, 0.5)
        self.D_term_r[:] = (self.last_body_ang_vel - self.odom.body_ang_vel) * self.kd_r
        
        self.pid_output[:] = self.P_term_r + self.I_term_r + self.D_term_r
        self.last_body_ang_vel[:] = self.odom.body_ang_vel
    
    def _angle_controller(self, action):
        """
        姿态角控制器
        输入：期望的姿态角 [roll, pitch, yaw, thrust]
        """
        # Yaw不参与控制（设置mask）
        yaw_mask = np.array([1, 1, 0])
        self.body_set_point[:] = -self.odom.body_euler * yaw_mask + action[:, :3]
        
        self.last_setpoint_error[:] = self.cur_setpoint_error
        self.cur_setpoint_error[:] = self.body_set_point * 15 - self.odom.body_ang_vel
        
        self.P_term_a[:] = self.cur_setpoint_error * self.kp_a
        self.I_term_a[:] = np.clip(self.I_term_a + self.cur_setpoint_error * self.ki_a, -0.5, 0.5)
        self.D_term_a[:] = np.clip((self.last_body_ang_vel - self.odom.body_ang_vel) * self.kd_a, -0.5, 0.5)
        
        self.pid_output[:] = self.P_term_a + self.I_term_a + self.D_term_a
        self.last_body_ang_vel[:] = self.odom.body_ang_vel
    
    def _position_controller(self, action):
        """
        位置控制器
        输入：期望的位置 [x, y, z, 0]
        输出：姿态角指令 -> angle_controller
        """
        # 计算位置误差
        cur_pos_error = action[:, :3] - self.odom.world_pos
        self.cur_setpoint_error[:] = cur_pos_error * 5 - self.odom.world_linear_vel
        
        # 转换坐标：(x, y, z) -> (roll, pitch, throttle)
        # 注意：y方向反向（因为roll增加时y减少）
        self.cur_setpoint_error[:, 1] *= -1
        self.cur_setpoint_error = self.cur_setpoint_error[:, [1, 0, 2]]  # [roll, pitch, throttle]
        
        # PID计算
        self.P_term_p[:] = self.cur_setpoint_error * self.kp_p
        self.I_term_p[:] = np.clip(self.I_term_p + self.cur_setpoint_error * self.ki_p, -0.5, 0.5)
        self.D_term_p[:] = np.clip((self.odom.last_world_linear_vel - self.odom.world_linear_vel) * self.kd_p, -0.5, 0.5)
        
        sum_term = self.P_term_p + self.I_term_p + self.D_term_p
        
        # 提取推力命令（必须是非负值！）
        # 范围 [0.0, 0.5] 与原始Genesis版本一致
        self.throttle_command[:] = np.clip(sum_term[:, -1], 0.0, 0.5)
        
        # 生成姿态角指令
        sum_term[:, -1] = 0
        angle_action = np.clip(sum_term, -0.5, 0.5)
        
        # 调用姿态角控制器
        self._angle_controller(angle_action)
    
    def reset(self, env_idx=None):
        """重置控制器状态"""
        if env_idx is None:
            idx = slice(None)
        else:
            idx = env_idx
        
        # 重置所有PID项
        self.P_term_a[idx] = 0.0
        self.I_term_a[idx] = 0.0
        self.D_term_a[idx] = 0.0
        
        self.P_term_r[idx] = 0.0
        self.I_term_r[idx] = 0.0
        self.D_term_r[idx] = 0.0
        
        self.P_term_p[idx] = 0.0
        self.I_term_p[idx] = 0.0
        self.D_term_p[idx] = 0.0
        
        # 重置误差和输出
        self.body_set_point[idx] = 0.0
        self.pid_output[idx] = 0.0
        self.cur_setpoint_error[idx] = 0.0
        self.last_setpoint_error[idx] = 0.0
        self.last_body_ang_vel[idx] = 0.0
        self.throttle_command[idx] = 0.0


def load_pid_config(config_path=None):
    """
    加载PID配置
    如果未提供路径，返回默认配置
    """
    if config_path is None:
        # 默认配置（针对easy_sim优化，降低增益以防止振荡）
        # easy_sim使用的是简化的动力学模型，需要更温和的PID参数
        return {
            "ang": {
                "kp_r": 3000, "ki_r": 0.002, "kd_r": 0.00005,  # 降低角度PID增益
                "kp_p": 3000, "ki_p": 0.002, "kd_p": 0.00005,
                "kp_y": 3500, "ki_y": 0.0, "kd_y": 0.0,
            },
            "rat": {
                "kp_r": 3000, "ki_r": 0.005, "kd_r": 0.0,
                "kp_p": 3000, "ki_p": 0.005, "kd_p": 0.0,
                "kp_y": 3500, "ki_y": 0.0, "kd_y": 0.0,
            },
            "pos": {
                "kp_x": 0.8, "ki_x": 0.005, "kd_x": 0.03,  # 降低位置PID增益
                "kp_y": 0.8, "ki_y": 0.005, "kd_y": 0.03,
                "kp_t": 0.6, "ki_t": 0.005, "kd_t": 0.0,
            },
            "thrust_compensate": 1.0,
            "base_rpm": 595000,
        }
    else:
        # 从YAML文件加载
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
