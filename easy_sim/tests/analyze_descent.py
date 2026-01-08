"""
分析为什么无人机会下降
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics
from easy_sim.core.controller import Odom, PIDController, load_pid_config


def analyze_forces():
    """分析推力和重力"""
    print("=" * 70)
    print("力分析")
    print("=" * 70)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    
    # 计算悬停推力
    hover_thrust_total = sim.mass * sim.gravity
    hover_thrust_per_motor = hover_thrust_total / 4.0
    
    print(f"\n重力和推力:")
    print(f"  质量: {sim.mass} kg")
    print(f"  重力: {sim.mass * sim.gravity:.4f} N (向下)")
    print(f"  悬停总推力: {hover_thrust_total:.4f} N (需要)")
    print(f"  悬停每电机: {hover_thrust_per_motor:.4f} N")
    
    # 计算RPM
    hover_omega = np.sqrt(hover_thrust_per_motor / sim.kf)
    hover_rpm = hover_omega * 60 / (2 * np.pi)
    print(f"  悬停RPM: {hover_rpm:.0f}")
    
    # 控制器base_rpm
    config = load_pid_config()
    print(f"\n控制器配置:")
    print(f"  base_rpm: {config['base_rpm']}")
    print(f"  thrust_compensate: {config['thrust_compensate']}")
    
    # 计算base_rpm产生的推力
    base_omega = config['base_rpm'] * 2 * np.pi / 60
    base_thrust_per_motor = sim.kf * base_omega**2
    base_thrust_total = base_thrust_per_motor * 4
    
    print(f"\n  base_rpm产生的推力:")
    print(f"    每电机: {base_thrust_per_motor:.4f} N")
    print(f"    总推力: {base_thrust_total:.4f} N")
    print(f"    vs 重力: {base_thrust_total:.4f} vs {hover_thrust_total:.4f}")
    print(f"    推重比: {base_thrust_total / hover_thrust_total:.3f}")
    
    # 当有姿态角时的垂直分量
    print(f"\n姿态角影响:")
    for angle_deg in [0, 5, 10, 15, 20, 30]:
        angle_rad = np.radians(angle_deg)
        vertical_component = base_thrust_total * np.cos(angle_rad)
        print(f"  姿态角 {angle_deg:2d}°: 垂直推力 {vertical_component:.4f} N, "
              f"vs 重力 {hover_thrust_total:.4f} N, "
              f"差值 {vertical_component - hover_thrust_total:+.4f} N")


def analyze_circle_forces():
    """分析圆形运动所需的向心力"""
    print("\n" + "=" * 70)
    print("圆形运动分析")
    print("=" * 70)
    
    radius = 1.0  # m
    omega = 0.5   # rad/s
    mass = 0.5    # kg
    
    # 向心加速度
    v = radius * omega  # 线速度
    a_centripetal = v**2 / radius  # 向心加速度
    F_centripetal = mass * a_centripetal  # 向心力
    
    print(f"\n圆形轨迹:")
    print(f"  半径: {radius} m")
    print(f"  角速度: {omega} rad/s")
    print(f"  线速度: {v:.3f} m/s")
    print(f"  向心加速度: {a_centripetal:.3f} m/s²")
    print(f"  所需向心力: {F_centripetal:.3f} N")
    
    # 所需的姿态倾斜角
    gravity = 9.81
    tilt_angle = np.arctan2(F_centripetal, mass * gravity)
    tilt_angle_deg = np.degrees(tilt_angle)
    
    print(f"\n所需倾斜角: {tilt_angle_deg:.2f}°")
    
    # 倾斜后的垂直推力
    hover_thrust = mass * gravity
    required_total_thrust = hover_thrust / np.cos(tilt_angle)
    
    print(f"  倾斜后所需总推力: {required_total_thrust:.3f} N")
    print(f"  vs 悬停推力: {hover_thrust:.3f} N")
    print(f"  增加: {(required_total_thrust / hover_thrust - 1) * 100:.1f}%")


def test_actual_circle():
    """实际测试圆形轨迹的前几步"""
    print("\n" + "=" * 70)
    print("实际圆形轨迹测试")
    print("=" * 70)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[1.0, 0.0, 1.0])
    
    odom = Odom(num_envs=1)
    odom.set_sim(sim)
    config = load_pid_config()
    controller = PIDController(1, odom, config, "position")
    
    radius = 1.0
    omega = 0.5
    height = 1.0
    
    print(f"\n前10步:")
    for step in range(10):
        t = step * 0.01
        angle = omega * t
        
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        target_z = height
        target = np.array([[target_x, target_y, target_z, 0.0]])
        
        motor_rpms = controller.step(target)
        state = sim.step_rpm(motor_rpms[0])
        
        # 计算总推力
        total_rpm = np.sum(motor_rpms[0])
        avg_rpm = total_rpm / 4
        
        pos = state['position']
        euler = np.degrees(state['euler_angles'])
        
        print(f"步{step}: 位置=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}], "
              f"RPM_avg={avg_rpm:.0f}, "
              f"姿态=[{euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f}]°")


if __name__ == "__main__":
    analyze_forces()
    analyze_circle_forces()
    test_actual_circle()
