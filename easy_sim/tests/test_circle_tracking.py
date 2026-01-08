"""
圆形轨迹跟踪测试 - 使用PID位置控制器

这个示例展示如何：
1. 使用独立的PID控制器
2. 进行圆形轨迹跟踪
3. 可视化跟踪效果
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics
from easy_sim.core.controller import Odom, PIDController, load_pid_config


class CircleTracker:
    """圆形轨迹跟踪器"""
    
    def __init__(self, radius=1.0, omega=0.5, height=1.0, center=None):
        """
        Parameters:
        -----------
        radius : float
            圆形轨迹半径 (m)
        omega : float
            角速度 (rad/s)
        height : float
            飞行高度 (m)
        center : array-like (3,)
            圆心位置 [x, y, z]
        """
        self.radius = radius
        self.omega = omega
        self.height = height
        self.center = np.array(center if center is not None else [0.0, 0.0, height])
    
    def get_target_position(self, t, initial_angle=0.0):
        """
        获取给定时间的目标位置
        
        Parameters:
        -----------
        t : float or np.ndarray
            时间 (s)
        initial_angle : float
            初始相位角 (rad)
        
        Returns:
        --------
        position : np.ndarray
            目标位置 [x, y, z]
        """
        angle = initial_angle + self.omega * t
        
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        z = self.center[2]
        
        return np.array([x, y, z])
    
    def get_target_velocity(self, t, initial_angle=0.0):
        """获取给定时间的目标速度"""
        angle = initial_angle + self.omega * t
        
        vx = -self.radius * self.omega * np.sin(angle)
        vy = self.radius * self.omega * np.cos(angle)
        vz = 0.0
        
        return np.array([vx, vy, vz])


def run_circle_tracking(
    num_steps=2000,
    dt=0.01,
    radius=1.0,
    omega=0.5,
    height=1.0,
    visualize=True
):
    """
    运行圆形轨迹跟踪
    
    Parameters:
    -----------
    num_steps : int
        仿真步数
    dt : float
        时间步长 (s)
    radius : float
        圆形轨迹半径 (m)
    omega : float
        角速度 (rad/s)
    height : float
        飞行高度 (m)
    visualize : bool
        是否生成可视化
    """
    print("=" * 70)
    print("圆形轨迹跟踪测试")
    print("=" * 70)
    print(f"参数:")
    print(f"  轨迹半径: {radius} m")
    print(f"  角速度: {omega} rad/s")
    print(f"  周期: {2*np.pi/omega:.2f} s")
    print(f"  高度: {height} m")
    print(f"  仿真步数: {num_steps}")
    print(f"  时间步长: {dt} s")
    print(f"  总时间: {num_steps * dt} s")
    print()
    
    # 创建仿真器
    config = {
        'dt': dt,
        'mass': 0.5,
        'arm_length': 0.12,
        'kf': 3.16e-10,
        'km': 7.94e-12,
    }
    sim = QuadrotorDynamics(config)
    
    # 创建轨迹跟踪器
    tracker = CircleTracker(radius=radius, omega=omega, height=height)
    
    # 初始化无人机位置到圆周上
    initial_angle = 0.0
    initial_pos = tracker.get_target_position(0, initial_angle)
    sim.reset(position=initial_pos)
    
    # 创建状态估计器和PID控制器
    odom = Odom(num_envs=1)
    odom.set_sim(sim)
    
    pid_config = load_pid_config()
    controller = PIDController(
        num_envs=1,
        odom=odom,
        config=pid_config,
        controller_type="position"
    )
    
    print("初始化完成，开始仿真...\n")
    
    # 记录数据
    drone_positions = []
    target_positions = []
    tracking_errors = []
    drone_velocities = []
    euler_angles = []
    
    # 仿真循环
    for step in range(num_steps):
        t = step * dt
        
        # 获取目标位置
        target_pos = tracker.get_target_position(t, initial_angle)
        
        # 构造控制指令 [x, y, z, 0]
        action = np.array([[target_pos[0], target_pos[1], target_pos[2], 0.0]])
        
        # PID控制器计算电机RPM
        motor_rpms = controller.step(action)
        
        # 执行仿真
        state = sim.step_rpm(motor_rpms[0])
        
        # 记录数据
        drone_positions.append(state['position'].copy())
        target_positions.append(target_pos.copy())
        error = np.linalg.norm(state['position'] - target_pos)
        tracking_errors.append(error)
        drone_velocities.append(state['velocity'].copy())
        euler_angles.append(state['euler_angles'].copy())
        
        # 定期打印
        if step % 200 == 0:
            print(f"步数: {step:4d}, 时间: {t:6.2f}s, 跟踪误差: {error:.4f}m, "
                  f"位置: [{state['position'][0]:6.2f}, {state['position'][1]:6.2f}, {state['position'][2]:6.2f}]")
    
    # 转换为数组
    drone_positions = np.array(drone_positions)
    target_positions = np.array(target_positions)
    tracking_errors = np.array(tracking_errors)
    drone_velocities = np.array(drone_velocities)
    euler_angles = np.array(euler_angles)
    
    # 统计
    print("\n" + "=" * 70)
    print("跟踪性能统计")
    print("=" * 70)
    print(f"平均跟踪误差: {np.mean(tracking_errors):.4f} m")
    print(f"最大跟踪误差: {np.max(tracking_errors):.4f} m")
    print(f"最小跟踪误差: {np.min(tracking_errors):.4f} m")
    print(f"RMSE: {np.sqrt(np.mean(tracking_errors**2)):.4f} m")
    
    # 稳态性能（跳过前5秒的过渡期）
    steady_start = int(5.0 / dt)
    if steady_start < len(tracking_errors):
        steady_errors = tracking_errors[steady_start:]
        print(f"\n稳态性能（t > 5s）:")
        print(f"  平均误差: {np.mean(steady_errors):.4f} m")
        print(f"  最大误差: {np.max(steady_errors):.4f} m")
        print(f"  标准差: {np.std(steady_errors):.4f} m")
    
    # 可视化
    if visualize:
        visualize_results(
            drone_positions, target_positions, tracking_errors,
            drone_velocities, euler_angles, dt
        )
    
    return {
        'drone_positions': drone_positions,
        'target_positions': target_positions,
        'tracking_errors': tracking_errors,
        'mean_error': np.mean(tracking_errors),
        'max_error': np.max(tracking_errors),
        'rmse': np.sqrt(np.mean(tracking_errors**2))
    }


def visualize_results(drone_positions, target_positions, tracking_errors,
                     drone_velocities, euler_angles, dt):
    """可视化跟踪结果"""
    print("\n生成可视化图表...")
    
    time = np.arange(len(drone_positions)) * dt
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
            'g--', label='Target', linewidth=2, alpha=0.7)
    ax1.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2],
            'b-', label='Drone', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. XY平面轨迹
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(target_positions[:, 0], target_positions[:, 1], 'g--',
            label='Target', linewidth=2, alpha=0.7)
    ax2.plot(drone_positions[:, 0], drone_positions[:, 1], 'b-',
            label='Drone', linewidth=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. 跟踪误差
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time, tracking_errors, 'r-', linewidth=2)
    ax3.axhline(np.mean(tracking_errors), color='b', linestyle='--',
               label=f'Mean: {np.mean(tracking_errors):.4f}m')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tracking Error (m)')
    ax3.set_title('Tracking Error vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 各轴位置对比
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time, target_positions[:, 0], 'g--', label='Target X', alpha=0.7)
    ax4.plot(time, drone_positions[:, 0], 'b-', label='Drone X')
    ax4.plot(time, target_positions[:, 1], 'r--', label='Target Y', alpha=0.7)
    ax4.plot(time, drone_positions[:, 1], 'm-', label='Drone Y')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('X & Y Position vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 速度
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(time, drone_velocities[:, 0], label='Vx', linewidth=1.5)
    ax5.plot(time, drone_velocities[:, 1], label='Vy', linewidth=1.5)
    ax5.plot(time, drone_velocities[:, 2], label='Vz', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.set_title('Velocity vs Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 姿态角
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(time, np.degrees(euler_angles[:, 0]), label='Roll', linewidth=1.5)
    ax6.plot(time, np.degrees(euler_angles[:, 1]), label='Pitch', linewidth=1.5)
    ax6.plot(time, np.degrees(euler_angles[:, 2]), label='Yaw', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Attitude Angle (deg)')
    ax6.set_title('Attitude Angles vs Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('circle_tracking_results.png', dpi=150, bbox_inches='tight')
    print("✓ 结果图已保存到 circle_tracking_results.png")
    
    plt.show()


def test_different_radii():
    """测试不同半径的圆形轨迹"""
    print("\n" + "=" * 70)
    print("测试不同半径")
    print("=" * 70)
    
    radii = [0.5, 1.0, 1.5, 2.0]
    results = {}
    
    for radius in radii:
        print(f"\n测试半径 {radius}m...")
        result = run_circle_tracking(
            num_steps=1500,
            radius=radius,
            omega=0.5,
            visualize=False
        )
        results[radius] = result
        print(f"  平均误差: {result['mean_error']:.4f}m")
        print(f"  最大误差: {result['max_error']:.4f}m")
    
    # 绘制对比图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for radius, result in results.items():
        errors = result['tracking_errors']
        time = np.arange(len(errors)) * 0.01
        ax.plot(time, errors, label=f'R={radius}m', linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (m)')
    ax.set_title('Tracking Error Comparison for Different Radii')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('circle_radius_comparison.png', dpi=150)
    print("\n✓ 对比图已保存到 circle_radius_comparison.png")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='圆形轨迹跟踪测试')
    parser.add_argument('--steps', type=int, default=2000, help='仿真步数')
    parser.add_argument('--radius', type=float, default=1.0, help='轨迹半径 (m)')
    parser.add_argument('--omega', type=float, default=0.5, help='角速度 (rad/s)')
    parser.add_argument('--height', type=float, default=1.0, help='飞行高度 (m)')
    parser.add_argument('--no-viz', action='store_true', help='不生成可视化')
    parser.add_argument('--test-radii', action='store_true', help='测试不同半径')
    
    args = parser.parse_args()
    
    if args.test_radii:
        test_different_radii()
    else:
        run_circle_tracking(
            num_steps=args.steps,
            radius=args.radius,
            omega=args.omega,
            height=args.height,
            visualize=not args.no_viz
        )


if __name__ == "__main__":
    main()
