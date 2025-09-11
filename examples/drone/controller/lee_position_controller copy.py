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
from torch._tensor import Tensor
import torch.nn as nn
from tensordict import TensorDict
from .controller import ControllerBase

from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    normalize,
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix
)
import yaml
import os.path as osp


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

class RateController(ControllerBase):

    def __init__(self, g, uav_params) -> None:
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.gain_angular_rate = nn.Parameter(
            torch.tensor([0.52, 0.52, 0.025]) @ I[:3, :3].inverse()
        )


    def forward(
        self,
        root_state: torch.Tensor,
        target_rate: torch.Tensor,
        target_thrust: torch.Tensor,
    ):
        assert root_state.shape[:-1] == target_rate.shape[:-1]

        batch_shape = root_state.shape[:-1]
        root_state = root_state.reshape(-1, 13)
        target_rate = target_rate.reshape(-1, 3)
        target_thrust = target_thrust.reshape(-1, 1)

        pos, rot, linvel, angvel = root_state.split([3, 4, 3, 3], dim=1)
        body_rate = quat_rotate_inverse(rot, angvel)

        rate_error = body_rate - target_rate
        acc_des = (
            - rate_error * self.gain_angular_rate
            + angvel.cross(angvel)
        )
        angacc_thrust = torch.cat([acc_des, target_thrust], dim=1)
        cmd = (self.mixer @ angacc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        cmd = cmd.reshape(*batch_shape, -1)
        return cmd

    def process_rl_actions(self, actions: torch.Tensor):
        target_rate, target_thrust = actions.split([3, 1], -1)
        target_thrust = ((target_thrust + 1) / 2).clip(0.) * self.max_thrusts
        return target_rate * torch.pi, target_thrust

