import torch
from torch._tensor import Tensor
import torch.nn as nn
from tensordict import TensorDict

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, quat_to_R


class LeePositionController(nn.Module):
    """
    Computes rotor commands for the given control target using the controller
    described in https://arxiv.org/abs/1003.2005.

    Inputs:
        * root_state: tensor of shape (13,) containing position, rotation (in quaternion),
        linear velocity, and angular velocity.
        * control_target: tensor of shape (7,) contining target position, linear velocity,
        and yaw angle.

    Outputs:
        * cmd: tensor of shape (num_rotors,) containing the computed rotor commands.
        * controller_state: empty dict.
    """
    def __init__(
        self,
        g: float,
        drone,
        n_envs,
    ) -> None:
        super().__init__()
        self.drone = drone

        I = torch.diag_embed(
            # torch.tensor([1.4e-3, 1.4e-3, 1.4e-3])
            torch.tensor([1.4e-5, 1.4e-5, 2.17e-5])
        )
        
        # 用变量代替 YAML 读取
        position_gain = [4.0, 4.0, 6.0]       # 例子，原本 YAML 里 position_gain
        velocity_gain = [2.0, 2.0, 3.0]       # 例子，原本 YAML 里 velocity_gain
        attitude_gain = [3, 3, 0.15]  # 原本 attitude_gain
        angular_rate_gain = [0.52, 0.52, 0.18]  # 原本 angular_rate_gain

        # 用 nn.Parameter 包装
        self.pos_gain = nn.Parameter(torch.as_tensor(position_gain).float())
        self.vel_gain = nn.Parameter(torch.as_tensor(velocity_gain).float())
        self.attitude_gain = nn.Parameter(
            torch.as_tensor(attitude_gain).float() @ I.inverse()
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(angular_rate_gain).float() @ I.inverse()
        )

        # 不需要训练这些参数
        self.requires_grad_(False)
        
        self.mass = torch.tensor(drone.get_mass())
        self.g = torch.tensor([0.0, 0.0, g]).abs()

        self.drone_pos  = torch.zeros(n_envs, 3)
        self.drone_world_vel  = torch.zeros(n_envs, 3)
        self.drone_world_ang = torch.zeros(n_envs, 3)
        self.drone_quat = torch.zeros(n_envs, 4)
        self.drone_quat_inv = torch.zeros(n_envs, 4)

        self.drone_body_vel = torch.zeros(n_envs, 3)
        self.drone_body_ang = torch.zeros(n_envs, 3)

        self.mixer = nn.Parameter(self.compute_parameters())

    def compute(
        self,
        target_pos: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        body_rate: bool=False
        ):
        self.drone_pos[:]  = self.drone.get_pos()
        self.drone_world_vel[:]  = self.drone.get_vel()
        self.drone_world_ang[:] = self.drone.get_ang()
        self.drone_quat[:] = self.drone.get_quat()
        self.drone_quat_inv[:] = inv_quat(self.drone_quat)

        self.drone_body_vel[:] = transform_by_quat(self.drone_world_vel, self.drone_quat_inv)
        self.drone_body_ang[:] = transform_by_quat(self.drone_world_ang, self.drone_quat_inv)

        root_state = torch.cat([self.drone_pos, self.drone_quat, self.drone_body_vel, self.drone_body_ang], dim=-1)

        batch_shape = root_state.shape[:-1]
        device = root_state.device
        if target_pos is None:
            target_pos = root_state[..., :3]
        else:
            target_pos = target_pos.expand(batch_shape+(3,))
        if target_vel is None:
            target_vel = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_vel = target_vel.expand(batch_shape+(3,))
        if target_acc is None:
            target_acc = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_acc = target_acc.expand(batch_shape+(3,))
        if target_yaw is None:
            target_yaw = quat_to_xyz(root_state[..., 3:7])[..., -1]
        else:
            if not target_yaw.shape[-1] == 1:
                target_yaw = target_yaw.unsqueeze(-1)
            target_yaw = target_yaw.expand(batch_shape+(1,))

        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_pos.reshape(-1, 3),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_yaw.reshape(-1, 1),
            body_rate
        )

        return cmd.reshape(*batch_shape, -1)

    def _compute(self, root_state, target_pos, target_vel, target_acc, target_yaw, body_rate):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            # convert angular velocity from world frame to body frame
            ang_vel = transform_by_quat(ang_vel, inv_quat(rot))

        pos_error = pos - target_pos
        vel_error = vel - target_vel
        # print(pos_error)

        acc = (
            pos_error * self.pos_gain
            + vel_error * self.vel_gain
            - self.g
            - target_acc
        )
        R = quat_to_R(rot)
        b1_des = torch.cat([
            torch.cos(target_yaw),
            torch.sin(target_yaw),
            torch.zeros_like(target_yaw)
        ],dim=-1)
        b3_des = -self.normalize(acc)
        b2_des = self.normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1),
            b2_des,
            b3_des
        ], dim=-1)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R)
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1],
            ang_error_matrix[:, 0, 2],
            ang_error_matrix[:, 1, 0]
        ],dim=-1)
        ang_rate_err = ang_vel
        ang_acc = (
            - ang_error * self.attitude_gain
            - ang_rate_err * self.ang_rate_gain
            + torch.linalg.cross(ang_vel, ang_vel)
        )
        thrust = (-self.mass * (acc * R[:, :, 2]).sum(-1, True))
        # print("thrust: ",thrust)
        # print("KF: ",self.drone._KF)
        # print("Mass: ",self.drone.get_mass())
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1)
        cmd = (self.mixer @ ang_acc_thrust.T).T
        # cmd = (cmd / self.max_thrusts) * 2 - 1
        rpm = torch.sqrt(cmd/self.drone._KF)
        return rpm

    def process_rl_actions(self, actions) -> Tensor:
        target_vel, target_yaw = actions.split([3, 1], dim=-1)
        return target_vel, target_yaw * torch.pi
    
    def compute_parameters(self):
        # l =  0.12
        l =  0.0397
        kf = self.drone._KF
        km = self.drone._KM

        mixer = torch.tensor([
            [ l, -l, -km/kf, 1],
            [-l, -l,  km/kf, 1],
            [-l,  l, -km/kf, 1],
            [ l,  l,  km/kf, 1],
        ], dtype=torch.float32)
        mixer[:, 3] = mixer[:, 3] / 4.0
        return mixer

    def normalize(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)
