

import torch
import genesis as gs

import math
from genesis.utils.geom import transform_by_quat,inv_quat

class PIDcontroller:
    def __init__(
            self, 
            drone, 
            config, 
            device = torch.device("cuda")):
        self.drone = drone
        self.num_envs = self.drone.get_pos().shape[0]
        self.device = device


        self.config = config
        self.thrust_compensate = config.get("thrust_compensate", 0.5)  

        # Angular Rate controller (angular rate PID)
        self.kp_r = torch.tensor(self.config.get("kp_r", 0.0), device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.ki_r = torch.tensor(self.config.get("ki_r", 0.0), device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.kd_r = torch.tensor(self.config.get("kd_r", 0.0), device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.kf_r = torch.tensor(self.config.get("kf_r", 0.0), device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        self.P_term_r = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_r = torch.zeros_like(self.P_term_r)
        self.D_term_r = torch.zeros_like(self.P_term_r)
        self.F_term_r = torch.zeros_like(self.P_term_r)


        self.pid_freq = config.get("pid_exec_freq", 60)     # no use
        self.base_rpm = config.get("base_rpm", 14468.429183500699)
        self.dT = 1 / self.pid_freq                         # no use
        self.tpa_factor = 1
        self.tpa_rate = 0
        self.throttle_command = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)

        self.last_body_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.angle_rate_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.angle_error = torch.zeros_like(self.angle_rate_error)

        self.body_set_point = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.pid_output = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cur_setpoint_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_setpoint_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.cnt = 0

    def set_drone(self, drone):
        self.drone = drone


    def mixer(self, action=None) -> torch.Tensor:


        throttle_action = torch.clamp((action[:, -1] + self.thrust_compensate) * 3, min=0.1, max=3.0) * self.base_rpm
        throttle =  throttle_action

        # self.pid_output[:] = torch.clip(self.pid_output[:], -3.0, 3.0)
        motor_outputs = torch.stack([
           throttle - self.pid_output[:, 0] - self.pid_output[:, 1] - self.pid_output[:, 2],  # M1
           throttle - self.pid_output[:, 0] + self.pid_output[:, 1] + self.pid_output[:, 2],  # M2
           throttle + self.pid_output[:, 0] + self.pid_output[:, 1] - self.pid_output[:, 2],  # M3
           throttle + self.pid_output[:, 0] - self.pid_output[:, 1] + self.pid_output[:, 2],  # M4
        ], dim = 1)

        return torch.clamp(motor_outputs, min=1, max=self.base_rpm * 3.5)  # size: tensor(num_envs, 4)


    def update(self, action=None):

        self.rate_controller(action)
        cmd = self.mixer(action)
        return cmd 
        

    def rate_controller(self, action=None): 
        """
        Anglular rate controller, sequence is (roll, pitch, yaw), use previous-D-term PID controller
        :param: 
            action: torch.Size([num_envs, 4]), like [[roll, pitch, yaw, thrust]] if num_envs = 1
        """
        
        self.body_set_point[:] = action[:, :3] * 15 # max angle rate is 15 rad/s

        self.last_setpoint_error[:] = self.cur_setpoint_error
        ang_vel = self.drone.get_ang()
        body_ang_vel = transform_by_quat(ang_vel, inv_quat(self.drone.get_quat()))
        self.cur_setpoint_error[:] = self.body_set_point - body_ang_vel
        self.P_term_r[:] = (self.cur_setpoint_error * self.kp_r) * self.tpa_factor
        self.I_term_r[:] = torch.clamp(self.I_term_r + self.cur_setpoint_error * self.ki_r, -0.5, 0.5)
        self.D_term_r[:] = (self.last_body_ang_vel - body_ang_vel) * self.kd_r * self.tpa_factor    

        self.pid_output[:] = (self.P_term_r + self.I_term_r + self.D_term_r)
        self.last_body_ang_vel[:] = body_ang_vel

    def reset(self, env_idx=None):
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx
        # Reset the PID terms (P, I, D, F)
        self.P_term_a.index_fill_(0, reset_range, 0.0)
        self.I_term_a.index_fill_(0, reset_range, 0.0)
        self.D_term_a.index_fill_(0, reset_range, 0.0)
        self.F_term_a.index_fill_(0, reset_range, 0.0)

        self.P_term_r.index_fill_(0, reset_range, 0.0)
        self.I_term_r.index_fill_(0, reset_range, 0.0)
        self.D_term_r.index_fill_(0, reset_range, 0.0)
        self.F_term_r.index_fill_(0, reset_range, 0.0)

        self.P_term_p.index_fill_(0, reset_range, 0.0)
        self.I_term_p.index_fill_(0, reset_range, 0.0)
        self.D_term_p.index_fill_(0, reset_range, 0.0)
        self.F_term_p.index_fill_(0, reset_range, 0.0)

        # Reset the angle, position, and velocity errors
        self.angle_rate_error.index_fill_(0, reset_range, 0.0)
        self.angle_error.index_fill_(0, reset_range, 0.0)

        # Reset the body set points and pid output
        self.body_set_point.index_fill_(0, reset_range, 0.0)
        self.pid_output.index_fill_(0, reset_range, 0.0)
        
        # Reset the last angular velocity
        self.last_body_ang_vel.index_fill_(0, reset_range, 0.0)
        self.last_setpoint_error.index_fill_(0, reset_range, 0.0)
        self.cur_setpoint_error.index_fill_(0, reset_range, 0.0)
        # Reset the TPA factor and rate
        self.tpa_factor = 1
        self.tpa_rate = 0
        # Reset the RC command values if necessary


def random_quaternion(num_envs=1, device="cuda"):
    max_rad = math.radians(180)
    roll  = (torch.rand(num_envs, 1, device=device) * 2 - 1) * max_rad
    pitch = (torch.rand(num_envs, 1, device=device) * 2 - 1) * max_rad
    yaw   = (torch.rand(num_envs, 1, device=device) * 2 - 1) * max_rad / 10

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = torch.cat([w, x, y, z], dim=1)
    quat = quat / quat.norm(dim=1, keepdim=True)
    return quat
