"""
独立的四旋翼无人机动力学仿真器
完全解耦Genesis，可用于sim2sim测试

输入：4个螺旋桨的推力或RPM
输出：无人机的完整状态（位置、速度、姿态、角速度等）
"""

import numpy as np
from scipy.spatial.transform import Rotation


class QuadrotorDynamics:
    """四旋翼无人机动力学仿真器"""
    
    def __init__(self, config=None):
        """
        初始化无人机动力学参数
        
        Parameters:
        -----------
        config : dict, optional
            配置字典，可包含以下参数：
            - mass: 质量 (kg)
            - inertia: 转动惯量矩阵 3x3 (kg·m²)
            - arm_length: 臂长 (m)
            - kf: 推力系数
            - km: 力矩系数
            - gravity: 重力加速度 (m/s²)
            - dt: 仿真时间步长 (s)
        """
        if config is None:
            config = {}
        
        # 从target_drone_urdf读取的参数
        self.mass = config.get('mass', 0.5)  # kg
        self.inertia = np.array(config.get('inertia', [
            [1.4e-3, 0.0, 0.0],
            [0.0, 1.4e-3, 0.0],
            [0.0, 0.0, 1.4e-3]
        ]))  # kg·m²
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        self.arm_length = config.get('arm_length', 0.12)  # m
        self.kf = config.get('kf', 3.16e-10)  # 推力系数
        self.km = config.get('km', 7.94e-12)  # 力矩系数
        self.gravity = config.get('gravity', 9.81)  # m/s²
        self.dt = config.get('dt', 0.01)  # s
        
        # 螺旋桨位置（相对于质心）- X型布局
        # prop0: 前右 (+x, -y)
        # prop1: 后右 (-x, -y)
        # prop2: 后左 (-x, +y)
        # prop3: 前左 (+x, +y)
        L = self.arm_length / np.sqrt(2)
        self.propeller_positions = np.array([
            [L, -L, 0],   # prop0
            [-L, -L, 0],  # prop1
            [-L, L, 0],   # prop2
            [L, L, 0]     # prop3
        ])
        
        # 螺旋桨旋转方向 (顺时针为-1，逆时针为1)
        self.propeller_directions = np.array([-1, 1, -1, 1])
        
        # 状态变量
        self.reset()
    
    def reset(self, position=None, velocity=None, quaternion=None, angular_velocity=None):
        """
        重置无人机状态
        
        Parameters:
        -----------
        position : array-like (3,), optional
            初始位置 [x, y, z] (m)
        velocity : array-like (3,), optional
            初始速度 [vx, vy, vz] (m/s)
        quaternion : array-like (4,), optional
            初始姿态四元数 [w, x, y, z]
        angular_velocity : array-like (3,), optional
            初始角速度 [wx, wy, wz] (rad/s)
        """
        self.position = np.array(position if position is not None else [0.0, 0.0, 1.0])
        self.velocity = np.array(velocity if velocity is not None else [0.0, 0.0, 0.0])
        self.quaternion = np.array(quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = np.array(angular_velocity if angular_velocity is not None else [0.0, 0.0, 0.0])
        
        # 归一化四元数
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
    
    def step(self, rotor_thrusts):
        """
        执行一步动力学仿真
        
        Parameters:
        -----------
        rotor_thrusts : array-like (4,)
            4个螺旋桨的推力 (N)，顺序为 [prop0, prop1, prop2, prop3]
        
        Returns:
        --------
        state : dict
            更新后的状态字典
        """
        thrusts = np.array(rotor_thrusts)
        
        # 计算总推力和力矩
        total_thrust, torques = self._compute_forces_and_torques(thrusts)
        
        # 更新线性动力学（世界坐标系）
        self._update_linear_dynamics(total_thrust)
        
        # 更新角动力学（机体坐标系）
        self._update_angular_dynamics(torques)
        
        return self.get_state()
    
    def step_rpm(self, rotor_rpms):
        """
        使用RPM作为输入执行一步仿真
        
        Parameters:
        -----------
        rotor_rpms : array-like (4,)
            4个螺旋桨的转速 (RPM)
        
        Returns:
        --------
        state : dict
            更新后的状态字典
        """
        # 将RPM转换为推力
        # F = kf * omega^2, omega是rad/s
        rpms = np.array(rotor_rpms)
        omega = rpms * 2 * np.pi / 60  # 转换为rad/s
        thrusts = self.kf * omega**2
        
        return self.step(thrusts)
    
    def _compute_forces_and_torques(self, thrusts):
        """
        从螺旋桨推力计算总推力和力矩
        
        Parameters:
        -----------
        thrusts : np.ndarray (4,)
            4个螺旋桨的推力
        
        Returns:
        --------
        total_thrust : np.ndarray (3,)
            机体坐标系下的总推力向量
        torques : np.ndarray (3,)
            机体坐标系下的总力矩
        """
        # 总推力（机体坐标系，沿z轴正方向）
        total_thrust_z = np.sum(thrusts)
        total_thrust = np.array([0.0, 0.0, total_thrust_z])
        
        # 计算力矩
        # 由推力位置产生的力矩
        thrust_torques = np.zeros(3)
        for i in range(4):
            # 推力在z轴方向，位置产生的力矩
            r = self.propeller_positions[i]
            f = np.array([0.0, 0.0, thrusts[i]])
            thrust_torques += np.cross(r, f)
        
        # 由螺旋桨反扭矩产生的力矩（绕z轴）
        reaction_torques = np.zeros(3)
        for i in range(4):
            # 反扭矩 = km * omega^2 * direction
            # 由于输入是推力，我们用 T = km/kf * F
            reaction_torques[2] += self.propeller_directions[i] * (self.km / self.kf) * thrusts[i]
        
        torques = thrust_torques + reaction_torques
        
        return total_thrust, torques
    
    def _update_linear_dynamics(self, body_thrust):
        """
        更新线性动力学（位置和速度）
        
        Parameters:
        -----------
        body_thrust : np.ndarray (3,)
            机体坐标系下的推力
        """
        # 将机体坐标系的推力转换到世界坐标系
        R = Rotation.from_quat([self.quaternion[1], self.quaternion[2], 
                                self.quaternion[3], self.quaternion[0]])
        world_thrust = R.apply(body_thrust)
        
        # 计算加速度：a = F/m + g
        gravity_force = np.array([0.0, 0.0, -self.mass * self.gravity])
        acceleration = (world_thrust + gravity_force) / self.mass
        
        # 使用欧拉法更新速度和位置
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
    
    def _update_angular_dynamics(self, torques):
        """
        更新角动力学（姿态和角速度）
        
        Parameters:
        -----------
        torques : np.ndarray (3,)
            机体坐标系下的力矩
        """
        # 计算角加速度：alpha = I^(-1) * (tau - omega x (I * omega))
        I_omega = self.inertia @ self.angular_velocity
        gyroscopic = np.cross(self.angular_velocity, I_omega)
        angular_acceleration = self.inertia_inv @ (torques - gyroscopic)
        
        # 更新角速度
        self.angular_velocity += angular_acceleration * self.dt
        
        # 更新四元数
        # dq/dt = 0.5 * q * omega_quat
        omega_quat = np.array([0.0, self.angular_velocity[0], 
                               self.angular_velocity[1], self.angular_velocity[2]])
        q_current = np.array([self.quaternion[0], self.quaternion[1], 
                             self.quaternion[2], self.quaternion[3]])
        
        dq = 0.5 * self._quaternion_multiply(q_current, omega_quat)
        self.quaternion += dq * self.dt
        
        # 归一化四元数
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
    
    def _quaternion_multiply(self, q1, q2):
        """四元数乘法 q1 * q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def get_state(self):
        """
        获取当前状态
        
        Returns:
        --------
        state : dict
            包含所有状态信息的字典
        """
        # 计算欧拉角（用于可视化）
        R = Rotation.from_quat([self.quaternion[1], self.quaternion[2], 
                                self.quaternion[3], self.quaternion[0]])
        euler = R.as_euler('xyz', degrees=False)
        
        # 计算旋转矩阵
        rotation_matrix = R.as_matrix()
        
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'quaternion': self.quaternion.copy(),  # [w, x, y, z]
            'angular_velocity': self.angular_velocity.copy(),
            'euler_angles': euler,  # [roll, pitch, yaw] in radians
            'rotation_matrix': rotation_matrix
        }
    
    def get_observation(self):
        """
        获取观测向量（类似于TrackerEnv的观测）
        
        Returns:
        --------
        obs : np.ndarray
            观测向量
        """
        state = self.get_state()
        
        # 构造观测向量
        obs = np.concatenate([
            state['position'],           # 3
            state['velocity'],           # 3
            state['quaternion'],         # 4
            state['angular_velocity']    # 3
        ])
        
        return obs


class BatchedQuadrotorDynamics:
    """批量四旋翼无人机动力学仿真器（用于并行仿真多个环境）"""
    
    def __init__(self, num_envs, config=None):
        """
        初始化批量仿真器
        
        Parameters:
        -----------
        num_envs : int
            环境数量
        config : dict, optional
            配置字典，同QuadrotorDynamics
        """
        self.num_envs = num_envs
        
        # 创建单个仿真器获取参数
        single_sim = QuadrotorDynamics(config)
        
        # 复制参数
        self.mass = single_sim.mass
        self.inertia = single_sim.inertia
        self.inertia_inv = single_sim.inertia_inv
        self.arm_length = single_sim.arm_length
        self.kf = single_sim.kf
        self.km = single_sim.km
        self.gravity = single_sim.gravity
        self.dt = single_sim.dt
        self.propeller_positions = single_sim.propeller_positions
        self.propeller_directions = single_sim.propeller_directions
        
        # 批量状态变量 (num_envs, state_dim)
        self.reset()
    
    def reset(self, env_ids=None, position=None, velocity=None, quaternion=None, angular_velocity=None):
        """
        重置指定环境的状态
        
        Parameters:
        -----------
        env_ids : array-like, optional
            要重置的环境ID列表，None表示重置所有环境
        position : array-like (num_envs, 3) or (3,), optional
            初始位置
        velocity : array-like (num_envs, 3) or (3,), optional
            初始速度
        quaternion : array-like (num_envs, 4) or (4,), optional
            初始姿态四元数
        angular_velocity : array-like (num_envs, 3) or (3,), optional
            初始角速度
        """
        # 初始化所有环境（如果还没有初始化）
        if not hasattr(self, 'position'):
            self.position = np.zeros((self.num_envs, 3))
            self.velocity = np.zeros((self.num_envs, 3))
            self.quaternion = np.zeros((self.num_envs, 4))
            self.quaternion[:, 0] = 1.0  # 初始化为单位四元数
            self.angular_velocity = np.zeros((self.num_envs, 3))
        
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        
        env_ids = np.asarray(env_ids)
        num_reset = len(env_ids)
        
        # 设置默认值
        if position is None:
            position = np.tile([0.0, 0.0, 1.0], (num_reset, 1))
        elif position.ndim == 1:
            position = np.tile(position, (num_reset, 1))
        
        if velocity is None:
            velocity = np.zeros((num_reset, 3))
        elif velocity.ndim == 1:
            velocity = np.tile(velocity, (num_reset, 1))
        
        if quaternion is None:
            quaternion = np.tile([1.0, 0.0, 0.0, 0.0], (num_reset, 1))
        elif quaternion.ndim == 1:
            quaternion = np.tile(quaternion, (num_reset, 1))
        
        if angular_velocity is None:
            angular_velocity = np.zeros((num_reset, 3))
        elif angular_velocity.ndim == 1:
            angular_velocity = np.tile(angular_velocity, (num_reset, 1))
        
        # 更新状态
        self.position[env_ids] = position
        self.velocity[env_ids] = velocity
        self.quaternion[env_ids] = quaternion
        self.angular_velocity[env_ids] = angular_velocity
        
        # 归一化四元数
        norms = np.linalg.norm(self.quaternion[env_ids], axis=1, keepdims=True)
        self.quaternion[env_ids] = self.quaternion[env_ids] / norms
    
    def step(self, rotor_thrusts):
        """
        批量执行动力学仿真
        
        Parameters:
        -----------
        rotor_thrusts : array-like (num_envs, 4)
            每个环境的4个螺旋桨推力
        
        Returns:
        --------
        states : dict
            包含所有环境状态的字典
        """
        thrusts = np.array(rotor_thrusts)
        
        # 批量计算推力和力矩
        total_thrusts, torques = self._compute_forces_and_torques_batched(thrusts)
        
        # 批量更新动力学
        self._update_linear_dynamics_batched(total_thrusts)
        self._update_angular_dynamics_batched(torques)
        
        return self.get_state()
    
    def step_rpm(self, rotor_rpms):
        """批量使用RPM作为输入"""
        rpms = np.array(rotor_rpms)
        omega = rpms * 2 * np.pi / 60
        thrusts = self.kf * omega**2
        return self.step(thrusts)
    
    def _compute_forces_and_torques_batched(self, thrusts):
        """批量计算推力和力矩"""
        # thrusts: (num_envs, 4)
        
        # 总推力
        total_thrust_z = np.sum(thrusts, axis=1)  # (num_envs,)
        total_thrusts = np.zeros((self.num_envs, 3))
        total_thrusts[:, 2] = total_thrust_z
        
        # 推力产生的力矩
        thrust_torques = np.zeros((self.num_envs, 3))
        for i in range(4):
            r = self.propeller_positions[i]  # (3,)
            f = np.zeros((self.num_envs, 3))
            f[:, 2] = thrusts[:, i]
            # cross product for each env
            thrust_torques[:, 0] += r[1] * f[:, 2]  # r_y * f_z
            thrust_torques[:, 1] += -r[0] * f[:, 2]  # -r_x * f_z
        
        # 反扭矩
        reaction_torques = np.zeros((self.num_envs, 3))
        for i in range(4):
            reaction_torques[:, 2] += self.propeller_directions[i] * (self.km / self.kf) * thrusts[:, i]
        
        torques = thrust_torques + reaction_torques
        
        return total_thrusts, torques
    
    def _update_linear_dynamics_batched(self, body_thrusts):
        """批量更新线性动力学"""
        # 批量旋转
        world_thrusts = np.zeros_like(body_thrusts)
        for i in range(self.num_envs):
            R = Rotation.from_quat([self.quaternion[i, 1], self.quaternion[i, 2],
                                   self.quaternion[i, 3], self.quaternion[i, 0]])
            world_thrusts[i] = R.apply(body_thrusts[i])
        
        # 重力
        gravity_force = np.array([0.0, 0.0, -self.mass * self.gravity])
        
        # 加速度
        acceleration = (world_thrusts + gravity_force) / self.mass
        
        # 更新速度和位置
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
    
    def _update_angular_dynamics_batched(self, torques):
        """批量更新角动力学"""
        for i in range(self.num_envs):
            # 计算角加速度
            I_omega = self.inertia @ self.angular_velocity[i]
            gyroscopic = np.cross(self.angular_velocity[i], I_omega)
            angular_acceleration = self.inertia_inv @ (torques[i] - gyroscopic)
            
            # 更新角速度
            self.angular_velocity[i] += angular_acceleration * self.dt
            
            # 更新四元数
            omega_quat = np.array([0.0, self.angular_velocity[i, 0],
                                  self.angular_velocity[i, 1], self.angular_velocity[i, 2]])
            q_current = self.quaternion[i]
            
            dq = 0.5 * self._quaternion_multiply(q_current, omega_quat)
            self.quaternion[i] += dq * self.dt
            
            # 归一化
            self.quaternion[i] /= np.linalg.norm(self.quaternion[i])
    
    def _quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def get_state(self):
        """获取所有环境的状态"""
        # 计算欧拉角
        euler_angles = np.zeros((self.num_envs, 3))
        rotation_matrices = np.zeros((self.num_envs, 3, 3))
        
        for i in range(self.num_envs):
            R = Rotation.from_quat([self.quaternion[i, 1], self.quaternion[i, 2],
                                   self.quaternion[i, 3], self.quaternion[i, 0]])
            euler_angles[i] = R.as_euler('xyz', degrees=False)
            rotation_matrices[i] = R.as_matrix()
        
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'quaternion': self.quaternion.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'euler_angles': euler_angles,
            'rotation_matrix': rotation_matrices
        }
    
    def get_observation(self):
        """获取观测向量"""
        return np.concatenate([
            self.position,           # (num_envs, 3)
            self.velocity,           # (num_envs, 3)
            self.quaternion,         # (num_envs, 4)
            self.angular_velocity    # (num_envs, 3)
        ], axis=1)  # (num_envs, 13)
