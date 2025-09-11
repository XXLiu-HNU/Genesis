import torch
import torch.nn as nn
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


class ThrustRateToRPMController(nn.Module):
    def __init__(self, g, drone, hover_throttle=0.3) -> None:
        super().__init__()
        # --------------------------
        # 1. 基础参数初始化（与仿真器对齐）
        # --------------------------
        self.drone = drone
        self.KF = drone._KF  # 对应仿真器的KF
        self.KM = drone._KM  # 对应仿真器的KM
        self.directions = drone._propellers_spin  # 旋翼旋转方向
        self.n_rotors = drone._n_propellers  # 旋翼数量
        print(self.directions.device)
        
        # self.max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])  # 最大转速

        
        # 重力与质量（用于悬停校准）
        self.g = nn.Parameter(torch.tensor(g))  # 重力加速度
        self.mass = drone.get_mass()  # 无人机总质量（需在uav_params中添加）
        

        # --------------------------
        # 2. 悬停参数校准
        # --------------------------
        self.hover_throttle = hover_throttle  # 输入的悬停油门（如0.5，需实际测试校准）
        # 悬停时单个电机的目标推力：总推力=mg，四电机均分
        self.hover_single_thrust = (self.mass * self.g) / self.n_rotors
        # 悬停时单个电机的目标转速（由推力反推）
        self.hover_single_rpm = torch.sqrt(self.hover_single_thrust / self.KF)

         # 最大推力（由最大转速计算，与仿真器一致）
        self.max_rot_vel = self.hover_single_rpm/self.hover_throttle
        self.max_thrust = self.KF * (self.hover_single_rpm/self.hover_throttle) ** 2
        # 悬停油门对应的“推力比例”：用于修正推力→转速的零漂
        self.hover_thrust_ratio = 1.0
        
       
        
        # --------------------------
        # 3. 惯性矩阵与混合矩阵（保留，用于力矩→推力分配）
        # --------------------------
        self.inertia = nn.Parameter(
            torch.tensor([[1.4e-3, 0, 0], 
                          [0, 1.4e-3, 0], 
                          [0, 0, 1.4e-3]], device=self.directions.device, dtype=torch.float32)
        )
        # 简化混合矩阵（四旋翼X型，负责“角加速度+推力→各电机推力”）
        self.mixer = self._build_simple_mixer()
        
        # --------------------------
        # 4. 角速度控制器增益（调参用）
        # --------------------------
        self.gain_angular_rate = nn.Parameter(torch.tensor([0.52, 0.52, 0.025]))




    def _build_simple_mixer(self):
        """构建四旋翼X型简化混合矩阵：输入（角加速度+推力）→ 输出（四电机推力）"""
        # A矩阵：4行（滚转/俯仰/偏航/推力）×4列（四电机）
        A = torch.zeros(4, self.n_rotors, device=self.directions.device)
        # 滚转力矩（x轴）：对角电机反向贡献
        A[0, :] = torch.tensor([1.0, -1.0, -1.0, 1.0], device=self.directions.device) * (self.KM / self.KF)
        # 俯仰力矩（y轴）：对角电机反向贡献
        A[1, :] = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.directions.device) * (self.KM / self.KF)
        # 偏航力矩（z轴）：旋转方向决定（顺时与逆时电机反向）
        A[2, :] = self.directions * (self.KM / self.KF)
        # 总推力（z轴）：四电机均等贡献
        A[3, :] = torch.ones(self.n_rotors, device=self.directions.device)
        
        # 扩展惯性矩阵（3x3角加速度 + 1x1推力，适配4维输入）
        I_aug = torch.block_diag(self.inertia, torch.tensor([1.0], device=self.inertia.device))
        # 混合矩阵：最小二乘解（推力分配）
        mixer = A.T @ (A @ A.T).inverse() @ I_aug
        return mixer


    def forward(
        self,
        target_rate: torch.Tensor,  # 期望角速度（机体坐标系，3维：滚转/俯仰/偏航）
        target_thrust: torch.Tensor  # 期望总推力（机体坐标系z轴，1维）
    ):
        
        drone_pos  = self.drone.get_pos()
        drone_world_vel  = self.drone.get_vel()
        drone_world_ang = self.drone.get_ang()
        drone_quat = self.drone.get_quat()
        drone_quat_inv = inv_quat(drone_quat)

        drone_body_vel = transform_by_quat(drone_world_vel,drone_quat_inv)
        drone_body_ang = transform_by_quat(drone_world_ang,drone_quat_inv)

        root_state = torch.cat([drone_pos, drone_quat, drone_body_vel, drone_body_ang], dim=-1)
        # --------------------------
        # 1. 输入维度统一（批量处理兼容）
        # --------------------------
        batch_shape = root_state.shape[:-1]
        root_state = root_state.reshape(-1, 13)  # (batch, 13)：pos(3)+rot(4)+linvel(3)+angvel(3)
        target_rate = target_rate.reshape(-1, 3)  # (batch, 3)
        target_thrust = target_thrust.reshape(-1, 1)  # (batch, 1)
        target_thrust = target_thrust * (self.mass * self.g / self.hover_throttle)

        # --------------------------
        # 2. 计算当前角速度误差→期望角加速度
        # --------------------------
        _, rot, _, angvel = root_state.split([3, 4, 3, 3], dim=1)
        body_rate = angvel  # 世界坐标系→机体坐标系（当前角速度）
        rate_error = target_rate - body_rate  # 角速度误差
        acc_des = -rate_error * self.gain_angular_rate  # 期望角加速度（比例控制，消误差）

        # --------------------------
        # 3. 混合矩阵：角加速度+总推力→各电机目标推力
        # --------------------------
        # 拼接输入：(batch, 4) = (角加速度3维 + 总推力1维)
        angacc_thrust = torch.cat([acc_des, target_thrust], dim=1)
        # 推力分配：(batch, 4) → 四电机各自的目标推力
        single_thrust = (self.mixer @ angacc_thrust.T).T  # (batch, 4)

        # --------------------------
        # 4. 悬停校准：修正推力零漂（关键！）
        # --------------------------
        # 基于悬停油门调整推力：避免因机械误差导致悬停时推力不足/过量
        single_thrust = single_thrust * self.hover_thrust_ratio

        # --------------------------
        # 5. 推力→转速：物理公式反推（核心步骤）
        # --------------------------
        # 安全处理：推力不能为负（电机无法产生负推力）
        single_thrust = single_thrust.clip(min=1e-6)  # 最小推力避免sqrt(0)
        # 转速计算：omega = sqrt(F / KF)（与仿真器推力公式对应）
        motor_rpm = torch.sqrt(single_thrust / self.KF)

        # --------------------------
        # 6. 转速限幅：不超过物理最大转速
        # --------------------------
        motor_rpm = motor_rpm.clip(max=self.max_rot_vel)

        # --------------------------
        # 7. 恢复批量维度：输出与输入结构一致
        # --------------------------
        motor_rpm = motor_rpm.reshape(*batch_shape, self.n_rotors)  # (batch, 4)

        return motor_rpm  # 返回四电机的目标转速


    def process_rl_actions(self, actions: torch.Tensor):
        """可选：RL动作空间→期望角速度+总推力（适配RL训练）"""
        # 动作拆分：前3维=角速度指令，后1维=推力指令（均为[-1,1]）
        target_rate_cmd, target_thrust_cmd = actions.split([3, 1], -1)
        
        # 角速度映射：[-1,1] → [-pi, pi] rad/s（常见RL动作范围）
        target_rate = target_rate_cmd * torch.pi
        
        # 推力映射：[-1,1] → [0, 最大总推力]（最大总推力=4*最大单电机推力）
        max_total_thrust = self.n_rotors * self.max_thrust
        target_thrust = ((target_thrust_cmd + 1) / 2).clip(0.) * max_total_thrust
        
        return target_rate, target_thrust
    

    # utils
    def world_to_body_vector(self, input_tensor):
        """
        Convert body frame vector tensor to world frame.
        :param:
            input_tensor: vectors like vel, acc ...(N, 3) or quat(N, 3), where N is the number of environments.
        """
        if input_tensor.shape[-1] == 3:
            return transform_by_quat(input_tensor, self.body_quat_inv)
        elif input_tensor.shape[-1] == 4:
            return transform_quat_by_quat(input_tensor, self.body_quat_inv)
        else:
            raise ValueError("Input tensor must have shape (N, 3) or (N, 4).")
        
    def body_to_world_vector(self, input_tensor):
        """
        Convert world frame vector tensor to body frame.
        :param:
            input_tensor: vectors like vel, acc ...(N, 3) or quat(N, 4), where N is the number of environments.
        """
        if input_tensor.shape[-1] == 3:
            return transform_by_quat(input_tensor, self.body_quat)
        elif input_tensor.shape[-1] == 4:
            return transform_quat_by_quat(input_tensor, self.body_quat)
        else:
            raise ValueError("Input tensor must have shape (N, 3) or (N, 4).")