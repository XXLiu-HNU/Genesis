"""
简单使用示例：独立无人机动力学仿真器
"""

import numpy as np
from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics


def example_basic_simulation():
    """基本仿真示例"""
    print("示例1: 基本仿真")
    print("-" * 50)
    
    # 创建仿真器
    sim = QuadrotorDynamics({
        'dt': 0.01,  # 10ms时间步长
        'mass': 0.5,  # 0.5kg
        'arm_length': 0.12,  # 12cm臂长
    })
    
    # 重置到初始位置
    sim.reset(position=[0, 0, 1.0])
    
    # 计算悬停推力
    hover_thrust = sim.mass * sim.gravity / 4.0
    print(f"悬停推力/螺旋桨: {hover_thrust:.4f} N")
    
    # 仿真10秒
    for i in range(1000):
        # 施加推力
        thrusts = [hover_thrust] * 4
        state = sim.step(thrusts)
        
        if i % 100 == 0:
            pos = state['position']
            print(f"t={i*0.01:.1f}s: 位置=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print()


def example_controlled_flight():
    """受控飞行示例"""
    print("示例2: 受控飞行 (爬升-悬停-下降)")
    print("-" * 50)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 0.5])
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    
    for i in range(1500):
        t = i * 0.01
        
        # 分段控制
        if t < 3.0:
            # 爬升
            thrust_scale = 1.3
        elif t < 10.0:
            # 悬停
            thrust_scale = 1.0
        elif t < 13.0:
            # 下降
            thrust_scale = 0.7
        else:
            # 悬停
            thrust_scale = 1.0
        
        thrusts = [hover_thrust * thrust_scale] * 4
        state = sim.step(thrusts)
        
        if i % 200 == 0:
            pos = state['position']
            vel = state['velocity']
            print(f"t={t:.1f}s: 高度={pos[2]:.3f}m, 垂直速度={vel[2]:.3f}m/s")
    
    print()


def example_attitude_control():
    """姿态控制示例"""
    print("示例3: 姿态控制 (滚转)")
    print("-" * 50)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    
    for i in range(500):
        t = i * 0.01
        
        # 创建滚转力矩
        if t < 3.0:
            # 施加滚转指令
            thrust_diff = 0.3
            thrusts = [
                hover_thrust - thrust_diff,  # 右前
                hover_thrust - thrust_diff,  # 右后
                hover_thrust + thrust_diff,  # 左后
                hover_thrust + thrust_diff,  # 左前
            ]
        else:
            # 保持
            thrusts = [hover_thrust] * 4
        
        state = sim.step(thrusts)
        
        if i % 50 == 0:
            euler = state['euler_angles']
            roll_deg = np.degrees(euler[0])
            print(f"t={t:.1f}s: 滚转角={roll_deg:.1f}°")
    
    print()


def example_rpm_control():
    """使用RPM控制示例"""
    print("示例4: 使用RPM控制")
    print("-" * 50)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    # 计算悬停RPM
    hover_thrust = sim.mass * sim.gravity / 4.0
    hover_omega = np.sqrt(hover_thrust / sim.kf)  # rad/s
    hover_rpm = hover_omega * 60 / (2 * np.pi)
    
    print(f"悬停RPM: {hover_rpm:.0f}")
    
    for i in range(500):
        # 使用RPM作为输入
        rpms = [hover_rpm] * 4
        state = sim.step_rpm(rpms)
        
        if i % 100 == 0:
            t = i * 0.01
            pos = state['position']
            print(f"t={t:.1f}s: 高度={pos[2]:.3f}m")
    
    print()


def example_trajectory_tracking():
    """轨迹跟踪示例（简单PD控制）"""
    print("示例5: 简单轨迹跟踪")
    print("-" * 50)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 0.5])
    
    # PD控制器参数
    kp = 5.0
    kd = 3.0
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    
    for i in range(1000):
        t = i * 0.01
        
        # 目标高度（正弦波）
        target_z = 1.0 + 0.3 * np.sin(2 * np.pi * t / 5.0)
        target_vz = 0.3 * (2 * np.pi / 5.0) * np.cos(2 * np.pi * t / 5.0)
        
        # 当前状态
        state = sim.get_state()
        current_z = state['position'][2]
        current_vz = state['velocity'][2]
        
        # PD控制
        error_z = target_z - current_z
        error_vz = target_vz - current_vz
        
        # 计算总推力
        total_thrust = sim.mass * (sim.gravity + kp * error_z + kd * error_vz)
        thrust_per_rotor = total_thrust / 4.0
        
        # 限制推力
        thrust_per_rotor = np.clip(thrust_per_rotor, 0, hover_thrust * 2)
        
        thrusts = [thrust_per_rotor] * 4
        sim.step(thrusts)
        
        if i % 100 == 0:
            print(f"t={t:.1f}s: 目标={target_z:.3f}m, 当前={current_z:.3f}m, 误差={error_z:.3f}m")
    
    print()


def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "无人机动力学仿真器使用示例" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    example_basic_simulation()
    example_controlled_flight()
    example_attitude_control()
    example_rpm_control()
    example_trajectory_tracking()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
    print("\n提示：")
    print("- 运行完整测试: python test_drone_dynamics.py")
    print("- Sim2sim评估: python example_policy_sim2sim.py --policy <path>")


if __name__ == "__main__":
    main()
