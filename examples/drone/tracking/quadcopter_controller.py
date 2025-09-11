# quadcopter_controller.py（替换原文件）
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz

# Vectorized PID controller (per-env states)
class PIDController:
    def __init__(self, kp, ki, kd, n_envs, device=None, dtype=None):
        device = gs.device if device is None else device
        dtype = gs.tc_float if dtype is None else dtype

        # store gains as scalars (broadcastable)
        self.kp = torch.tensor(kp, device=device, dtype=dtype)
        self.ki = torch.tensor(ki, device=device, dtype=dtype)
        self.kd = torch.tensor(kd, device=device, dtype=dtype)

        # per-env states
        self.integral = torch.zeros((n_envs,), device=device, dtype=dtype)
        self.prev_error = torch.zeros((n_envs,), device=device, dtype=dtype)

    def update(self, error: torch.Tensor, dt: float) -> torch.Tensor:
        """
        error: [N] tensor
        returns: [N] tensor (PID output)
        """
        # ensure tensor on correct device/dtype
        error = error.to(self.integral.device).to(self.integral.dtype)

        self.integral = self.integral + error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset_idx(self, env_idx):
        if len(env_idx) == 0:
            return
        self.integral[env_idx] = 0.0
        self.prev_error[env_idx] = 0.0


class DronePIDController:
    """
    并行版 Drone PID Controller，兼容 Genesis 的 batched DroneEntity。
    - drone: DroneEntity（已 build，能返回 [N,3]/[N,4] 的 get_pos/get_vel/get_quat）
    - dt: timestep
    - base_rpm: 标量或张量（会广播）
    - pid_params: list of 9 (kp,ki,kd) tuples/lists
    """
    def __init__(self, drone, dt, base_rpm, pid_params):
        self.device = gs.device
        self.dtype = gs.tc_float
        self.drone = drone
        self.dt = dt
        # infer n_envs from drone.get_pos() shape
        pos = self.drone.get_pos()
        if not isinstance(pos, torch.Tensor):
            pos = torch.as_tensor(pos, device=self.device, dtype=self.dtype)
        self.n_envs = pos.shape[0]

        # base rpm as tensor
        self.base_rpm = torch.tensor(base_rpm, device=self.device, dtype=self.dtype)

        # pid_params: list of 9 (kp,ki,kd)
        assert len(pid_params) == 9, "pid_params must be length 9"

        # create vectorized PID controllers
        self.pid_pos_x = PIDController(*pid_params[0], n_envs=self.n_envs, device=self.device, dtype=self.dtype)
        self.pid_pos_y = PIDController(*pid_params[1], n_envs=self.n_envs, device=self.device, dtype=self.dtype)
        self.pid_pos_z = PIDController(*pid_params[2], n_envs=self.n_envs, device=self.device, dtype=self.dtype)

        self.pid_vel_x = PIDController(*pid_params[3], n_envs=self.n_envs, device=self.device, dtype=self.dtype)
        self.pid_vel_y = PIDController(*pid_params[4], n_envs=self.n_envs, device=self.device, dtype=self.dtype)
        self.pid_vel_z = PIDController(*pid_params[5], n_envs=self.n_envs, device=self.device, dtype=self.dtype)

        self.pid_att_roll = PIDController(*pid_params[6], n_envs=self.n_envs, device=self.device, dtype=self.dtype)
        self.pid_att_pitch = PIDController(*pid_params[7], n_envs=self.n_envs, device=self.device, dtype=self.dtype)
        self.pid_att_yaw = PIDController(*pid_params[8], n_envs=self.n_envs, device=self.device, dtype=self.dtype)

    # helper wrappers to fetch batched states from the DroneEntity
    def _get_pos(self):
        return self.drone.get_pos().to(self.device).to(self.dtype)  # [N,3]

    def _get_vel(self):
        return self.drone.get_vel().to(self.device).to(self.dtype)  # [N,3]

    def _get_att(self):
        quat = self.drone.get_quat().to(self.device).to(self.dtype)  # [N,4]
        # returns euler per env in degrees (matching your earlier code)
        return quat_to_xyz(quat, rpy=True, degrees=True)  # [N,3]

    def _get_ang_vel(self):
        # use local frame angular velocity
        return self.drone.get_local_ang_vel().to(self.device).to(self.dtype) # [N,3]

    def _mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel):
        """
        All inputs are [N] tensors. Returns [N,4] RPMs.
        M1..M4 logic kept same as official code but vectorized.
        """
        M1 = self.base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        M2 = self.base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        M3 = self.base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        M4 = self.base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        # stack as [N,4]
        rpms = torch.stack([M1, M2, M3, M4], dim=-1)
        # ensure finite and non-negative (optional clipping)
        rpms = torch.clamp(rpms, min=0.0)
        return rpms

    def reset_idx(self, env_idx):
        """Reset internal integrators for given env indices (tensor or list)."""
        if isinstance(env_idx, torch.Tensor):
            env_idx = env_idx.cpu().numpy().astype(int).tolist()
        # if empty, nothing to do
        if len(env_idx) == 0:
            return
        self.pid_pos_x.reset_idx(env_idx)
        self.pid_pos_y.reset_idx(env_idx)
        self.pid_pos_z.reset_idx(env_idx)
        self.pid_vel_x.reset_idx(env_idx)
        self.pid_vel_y.reset_idx(env_idx)
        self.pid_vel_z.reset_idx(env_idx)
        self.pid_att_roll.reset_idx(env_idx)
        self.pid_att_pitch.reset_idx(env_idx)
        self.pid_att_yaw.reset_idx(env_idx)

    def update(self, targets):
        """
        targets: [N,3] desired positions (or commands)
        returns: rpms: [N,4] torch tensor on gs.device
        """
        # ensure tensor on right device/dtype
        targets = targets.to(self.device).to(self.dtype)

        curr_pos = self._get_pos()        # [N,3]
        curr_vel = self._get_vel()        # [N,3]
        curr_att = self._get_att()        # [N,3] roll,pitch,yaw (deg)

        # --- position loop ---
        err_pos = targets - curr_pos      # [N,3]
        vel_des_x = self.pid_pos_x.update(err_pos[:, 0], self.dt)  # [N]
        vel_des_y = self.pid_pos_y.update(err_pos[:, 1], self.dt)
        vel_des_z = self.pid_pos_z.update(err_pos[:, 2], self.dt)

        # --- velocity loop ---
        # err_vel  = targets - curr_vel      # [N,3]
        err_vel_x = vel_des_x - curr_vel[:, 0]
        err_vel_y = vel_des_y - curr_vel[:, 1]
        err_vel_z = vel_des_z - curr_vel[:, 2]

        x_vel_del = self.pid_vel_x.update(err_vel_x, self.dt)
        y_vel_del = self.pid_vel_y.update(err_vel_y, self.dt)
        thrust_des = self.pid_vel_z.update(err_vel_z, self.dt)

        # --- attitude loop (target set to zero oriented wrt local frame) ---
        err_roll = -curr_att[:, 0]
        err_pitch = -curr_att[:, 1]
        err_yaw = -curr_att[:, 2]

        roll_del = self.pid_att_roll.update(err_roll, self.dt)
        pitch_del = self.pid_att_pitch.update(err_pitch, self.dt)
        yaw_del = self.pid_att_yaw.update(err_yaw, self.dt)

        # compute rpms
        prop_rpms = self._mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)  # [N,4]
        return prop_rpms
