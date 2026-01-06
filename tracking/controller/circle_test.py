import os
import sys
import torch
import math
import yaml
import genesis as gs

# 添加项目根目录到路径，以便导入 controller 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
tracking_dir = os.path.dirname(script_dir)
if tracking_dir not in sys.path:
    sys.path.insert(0, tracking_dir)

from controller.pid import PIDcontroller
from controller.odom import Odom


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class TrackerEnv:
    def __init__(self, num_envs, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义dt和圆形轨迹参数
        self.dt = 0.01
        self.circle_radius = 1.0
        self.circle_omega = 0.5
        self.drone_height = 1.0
        self.circle_center = torch.tensor([0.0, 0.0, self.drone_height], device=self.device)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=100,
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False)
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add drone
        self.initial_angle = gs_rand_float(0, 2 * math.pi, (self.num_envs,), self.device)
        
        # 将无人机初始位置设在圆周上，避免初始跳变
        self.drone_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drone_init_pos[:, 0] = self.circle_center[0] + self.circle_radius * torch.cos(self.initial_angle)
        self.drone_init_pos[:, 1] = self.circle_center[1] + self.circle_radius * torch.sin(self.initial_angle)
        self.drone_init_pos[:, 2] = self.circle_center[2]
        
        self.drone_init_quat = torch.tensor([1,0,0,0], device=self.device).repeat(self.num_envs, 1)

        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/target_drone_urdf/drone.urdf"))
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config/pos.yaml"), "r") as file:
            self.pos_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)

        self.set_drone_imu()
        self.set_drone_controller()

        # Build scene
        self.scene.build(n_envs=num_envs)

        # initialize buffers
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        
        # Set initial state for drone drone
        self.drone.set_pos(self.drone_init_pos)
        self.drone.set_quat(self.drone_init_quat)

    def step(self):
        # 增加步数
        self.episode_length_buf += 1
        
        # 获取圆形轨迹上的目标位置
        circle_traj = self.get_circle_traj()
        
        # 调用PID控制器计算动作
        drone_prop_rpms = self.drone.controller.step(circle_traj)
        self.drone.set_propellels_rpm(drone_prop_rpms)
        
        # 物理模拟
        self.scene.step()
        
        # 可以在这里添加一些日志或状态打印
        current_pos = self.drone.get_pos()
        print(f"Step: {self.episode_length_buf[0].item():d}, Curr Pos: ({current_pos[0,0]:.2f}, {current_pos[0,1]:.2f}, {current_pos[0,2]:.2f}), Target Pos: ({circle_traj[0,0]:.2f}, {circle_traj[0,1]:.2f}, {circle_traj[0,2]:.2f})")


    def get_circle_traj(self):
        """
        根据当前步数计算圆形轨迹上的目标位置。
        """
        # 将步数转换为弧度
        # self.episode_length_buf 是一个形状为 (num_envs,) 的张量
        angle = self.initial_angle + self.episode_length_buf.float() * self.dt * self.circle_omega
        
        # 计算新位置的x和y坐标
        x = self.circle_center[0] + self.circle_radius * torch.cos(angle)
        y = self.circle_center[1] + self.circle_radius * torch.sin(angle)
        z = self.circle_center[2] * torch.ones_like(x)
        t = torch.zeros_like(x)

        # 返回目标位置张量，形状为 (num_envs, 4)
        return torch.stack([x, y, z, t], dim=1)


    def set_drone_imu(self):
        odom = Odom(
            num_envs = self.num_envs,
            device = self.device
        )
        odom.set_drone(self.drone)
        setattr(self.drone, 'odom', odom)

    def set_drone_controller(self):
        pid = PIDcontroller(
            num_envs = self.num_envs,
            odom = self.drone.odom,
            config = self.pos_ctrl_config,
            device = self.device,
            controller = "position",
        )
        pid.set_drone(self.drone)
        setattr(self.drone, 'controller', pid)


if __name__ == "__main__":
    gs.init()
    env = TrackerEnv(num_envs=1, show_viewer=True)
    for i in range(2000): # 运行2000步
        env.step()