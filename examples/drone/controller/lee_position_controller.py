# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
from tensordict import TensorDict

import yaml
import os.path as osp

from genesis.utils.geom import quat_to_R, xyz_to_quat
def compute_parameters(
    rotor_config,
    inertia_matrix,
):
    rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
    arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
    force_constants = torch.as_tensor(rotor_config["force_constants"])
    moment_constants = torch.as_tensor(rotor_config["moment_constants"])
    directions = torch.as_tensor(rotor_config["directions"])
    max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
    A = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles),
        ]
    )
    mixer = A.T @ (A @ A.T).inverse() @ inertia_matrix

    return mixer

class AttitudeController(nn.Module):
    r"""
    
    """
    def __init__(self, g, uav_params):
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.gain_attitude = nn.Parameter(
            torch.tensor([3., 3., 0.035]) @ I[:3, :3].inverse()
        )
        self.gain_angular_rate = nn.Parameter(
            torch.tensor([0.52, 0.52, 0.025]) @ I[:3, :3].inverse()
        )


    def forward(
        self, 
        root_state: torch.Tensor, 
        target_thrust: torch.Tensor,
        target_yaw_rate: torch.Tensor=None,
        target_roll: torch.Tensor=None,
        target_pitch: torch.Tensor=None,
    ):
        batch_shape = root_state.shape[:-1]
        device = root_state.device

        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(*batch_shape, 1, device=device)
        if target_pitch is None:
            target_pitch = torch.zeros(*batch_shape, 1, device=device)
        if target_roll is None:
            target_roll = torch.zeros(*batch_shape, 1, device=device)
        
        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_thrust.reshape(-1, 1),
            target_yaw_rate=target_yaw_rate.reshape(-1, 1),
            target_roll=target_roll.reshape(-1, 1),
            target_pitch=target_pitch.reshape(-1, 1),
        )
        return cmd.reshape(*batch_shape, -1)

    def _compute(
        self, 
        root_state: torch.Tensor,
        target_thrust: torch.Tensor, 
        target_yaw_rate: torch.Tensor, 
        target_roll: torch.Tensor,
        target_pitch: torch.Tensor
    ):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        device = pos.device

        R = quat_to_R(rot)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
        yaw_vec = torch.zeros(pos.shape[0], 3, device=device)
        yaw_vec[:, 2] = yaw.squeeze(-1)  # 只设置 yaw
        yaw_R = quat_to_R(xyz_to_quat(yaw_vec))

        # roll 四元数
        roll_vec = torch.zeros(pos.shape[0], 3, device=device)
        roll_vec[:, 0] = target_roll.squeeze(-1)
        roll_R = quat_to_R(xyz_to_quat(roll_vec))

        # pitch 四元数
        pitch_vec = torch.zeros(pos.shape[0], 3, device=device)
        pitch_vec[:, 1] = target_pitch.squeeze(-1)
        pitch_R = quat_to_R(xyz_to_quat(pitch_vec))

        # 最终期望旋转矩阵
        R_des = torch.bmm(torch.bmm(yaw_R, roll_R), pitch_R)
        angle_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )

        angle_error = torch.stack([
            angle_error_matrix[:, 2, 1], 
            angle_error_matrix[:, 0, 2], 
            torch.zeros(yaw.shape[0], device=device)
        ], dim=-1)

        angular_rate_des = torch.zeros_like(ang_vel)
        angular_rate_des[:, 2] = target_yaw_rate.squeeze(1)
        angular_rate_error = ang_vel - torch.bmm(torch.bmm(R_des.transpose(-2, -1), R), angular_rate_des.unsqueeze(2)).squeeze(2)

        angular_acc = (
            - angle_error * self.gain_attitude 
            - angular_rate_error * self.gain_angular_rate 
            + torch.cross(ang_vel, ang_vel)
        )
        angular_acc_thrust = torch.cat([angular_acc, target_thrust], dim=1)
        cmd = (self.mixer @ angular_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1

        return cmd.detach().cpu().numpy()

