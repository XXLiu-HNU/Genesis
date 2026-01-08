"""
测试独立无人机动力学仿真器
"""

import numpy as np
import matplotlib.pyplot as plt
from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics, BatchedQuadrotorDynamics


def test_hover():
    """测试悬停稳定性"""
    print("=" * 60)
    print("测试1: 悬停稳定性")
    print("=" * 60)
    
    # 创建仿真器
    config = {
        'dt': 0.01,  # 10ms
    }
    sim = QuadrotorDynamics(config)
    
    # 重置到初始位置
    sim.reset(position=[0, 0, 1.0])
    
    # 计算悬停所需推力（每个螺旋桨承担1/4的重力）
    hover_thrust_per_rotor = sim.mass * sim.gravity / 4.0
    print(f"质量: {sim.mass} kg")
    print(f"悬停推力/螺旋桨: {hover_thrust_per_rotor:.4f} N")
    
    # 仿真
    num_steps = 500
    positions = []
    velocities = []
    
    for step in range(num_steps):
        # 施加悬停推力
        rotor_thrusts = [hover_thrust_per_rotor] * 4
        state = sim.step(rotor_thrusts)
        
        positions.append(state['position'].copy())
        velocities.append(state['velocity'].copy())
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # 验证结果
    final_z = positions[-1, 2]
    final_vz = velocities[-1, 2]
    z_drift = abs(final_z - 1.0)
    
    print(f"初始高度: {positions[0, 2]:.6f} m")
    print(f"最终高度: {final_z:.6f} m")
    print(f"高度漂移: {z_drift:.6f} m")
    print(f"最终垂直速度: {final_vz:.6f} m/s")
    
    if z_drift < 0.01 and abs(final_vz) < 0.01:
        print("✓ 悬停测试通过")
        return True
    else:
        print("✗ 悬停测试失败")
        return False


def test_climb():
    """测试爬升"""
    print("\n" + "=" * 60)
    print("测试2: 爬升")
    print("=" * 60)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    
    # 增加推力以爬升
    climb_thrust = hover_thrust * 1.2
    
    num_steps = 200
    positions = []
    
    for step in range(num_steps):
        rotor_thrusts = [climb_thrust] * 4
        state = sim.step(rotor_thrusts)
        positions.append(state['position'].copy())
    
    positions = np.array(positions)
    
    initial_z = positions[0, 2]
    final_z = positions[-1, 2]
    climb_height = final_z - initial_z
    
    print(f"初始高度: {initial_z:.4f} m")
    print(f"最终高度: {final_z:.4f} m")
    print(f"爬升高度: {climb_height:.4f} m")
    
    if climb_height > 0.1:
        print("✓ 爬升测试通过")
        return True
    else:
        print("✗ 爬升测试失败")
        return False


def test_roll():
    """测试滚转控制"""
    print("\n" + "=" * 60)
    print("测试3: 滚转控制")
    print("=" * 60)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    
    # 创建滚转力矩：右侧螺旋桨推力增加，左侧减少
    thrust_diff = 0.2
    
    num_steps = 300
    euler_angles = []
    
    for step in range(num_steps):
        # prop0和prop1在右侧（-y），prop2和prop3在左侧（+y）
        # 增加左侧推力，减少右侧推力 -> 正滚转
        rotor_thrusts = [
            hover_thrust - thrust_diff,  # prop0 (右前)
            hover_thrust - thrust_diff,  # prop1 (右后)
            hover_thrust + thrust_diff,  # prop2 (左后)
            hover_thrust + thrust_diff,  # prop3 (左前)
        ]
        state = sim.step(rotor_thrusts)
        euler_angles.append(state['euler_angles'].copy())
    
    euler_angles = np.array(euler_angles)
    
    final_roll = euler_angles[-1, 0]
    
    print(f"初始滚转角: {np.degrees(euler_angles[0, 0]):.2f}°")
    print(f"最终滚转角: {np.degrees(final_roll):.2f}°")
    
    if abs(np.degrees(final_roll)) > 5:
        print("✓ 滚转测试通过")
        return True
    else:
        print("✗ 滚转测试失败")
        return False


def test_pitch():
    """测试俯仰控制"""
    print("\n" + "=" * 60)
    print("测试4: 俯仰控制")
    print("=" * 60)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    thrust_diff = 0.2
    
    num_steps = 300
    euler_angles = []
    
    for step in range(num_steps):
        # prop0和prop3在前侧（+x），prop1和prop2在后侧（-x）
        # 增加后侧推力，减少前侧推力 -> 正俯仰
        rotor_thrusts = [
            hover_thrust - thrust_diff,  # prop0 (前右)
            hover_thrust + thrust_diff,  # prop1 (后右)
            hover_thrust + thrust_diff,  # prop2 (后左)
            hover_thrust - thrust_diff,  # prop3 (前左)
        ]
        state = sim.step(rotor_thrusts)
        euler_angles.append(state['euler_angles'].copy())
    
    euler_angles = np.array(euler_angles)
    final_pitch = euler_angles[-1, 1]
    
    print(f"初始俯仰角: {np.degrees(euler_angles[0, 1]):.2f}°")
    print(f"最终俯仰角: {np.degrees(final_pitch):.2f}°")
    
    if abs(np.degrees(final_pitch)) > 5:
        print("✓ 俯仰测试通过")
        return True
    else:
        print("✗ 俯仰测试失败")
        return False


def test_yaw():
    """测试偏航控制"""
    print("\n" + "=" * 60)
    print("测试5: 偏航控制")
    print("=" * 60)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    
    # 创建偏航力矩：调整螺旋桨推力产生净反扭矩
    # 螺旋桨方向：prop0(-), prop1(+), prop2(-), prop3(+)
    # 增加逆时针螺旋桨推力，减少顺时针螺旋桨推力
    thrust_diff = 0.15
    
    num_steps = 500
    euler_angles = []
    
    for step in range(num_steps):
        rotor_thrusts = [
            hover_thrust - thrust_diff,  # prop0 (顺时针)
            hover_thrust + thrust_diff,  # prop1 (逆时针)
            hover_thrust - thrust_diff,  # prop2 (顺时针)
            hover_thrust + thrust_diff,  # prop3 (逆时针)
        ]
        state = sim.step(rotor_thrusts)
        euler_angles.append(state['euler_angles'].copy())
    
    euler_angles = np.array(euler_angles)
    final_yaw = euler_angles[-1, 2]
    
    print(f"初始偏航角: {np.degrees(euler_angles[0, 2]):.2f}°")
    print(f"最终偏航角: {np.degrees(final_yaw):.2f}°")
    
    if abs(np.degrees(final_yaw)) > 5:
        print("✓ 偏航测试通过")
        return True
    else:
        print("✗ 偏航测试失败")
        return False


def test_rpm_input():
    """测试RPM输入接口"""
    print("\n" + "=" * 60)
    print("测试6: RPM输入接口")
    print("=" * 60)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    # 计算悬停所需RPM
    hover_thrust = sim.mass * sim.gravity / 4.0
    # F = kf * omega^2, omega = sqrt(F / kf)
    hover_omega = np.sqrt(hover_thrust / sim.kf)  # rad/s
    hover_rpm = hover_omega * 60 / (2 * np.pi)
    
    print(f"悬停RPM: {hover_rpm:.0f}")
    
    num_steps = 500
    positions = []
    
    for step in range(num_steps):
        rotor_rpms = [hover_rpm] * 4
        state = sim.step_rpm(rotor_rpms)
        positions.append(state['position'].copy())
    
    positions = np.array(positions)
    z_drift = abs(positions[-1, 2] - 1.0)
    
    print(f"高度漂移: {z_drift:.6f} m")
    
    if z_drift < 0.01:
        print("✓ RPM输入测试通过")
        return True
    else:
        print("✗ RPM输入测试失败")
        return False


def test_batched_simulation():
    """测试批量仿真"""
    print("\n" + "=" * 60)
    print("测试7: 批量仿真")
    print("=" * 60)
    
    num_envs = 10
    sim = BatchedQuadrotorDynamics(num_envs, {'dt': 0.01})
    
    # 重置所有环境
    sim.reset()
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    
    # 为每个环境施加稍微不同的推力
    rotor_thrusts = np.ones((num_envs, 4)) * hover_thrust
    rotor_thrusts[:, 0] += np.linspace(-0.1, 0.1, num_envs)
    
    num_steps = 300
    final_positions = []
    
    for step in range(num_steps):
        state = sim.step(rotor_thrusts)
    
    final_positions = state['position']
    
    print(f"环境数量: {num_envs}")
    print(f"最终位置范围:")
    print(f"  X: [{final_positions[:, 0].min():.4f}, {final_positions[:, 0].max():.4f}]")
    print(f"  Y: [{final_positions[:, 1].min():.4f}, {final_positions[:, 1].max():.4f}]")
    print(f"  Z: [{final_positions[:, 2].min():.4f}, {final_positions[:, 2].max():.4f}]")
    
    # 验证不同环境有不同的状态
    z_std = np.std(final_positions[:, 2])
    
    if z_std > 0.01:
        print("✓ 批量仿真测试通过")
        return True
    else:
        print("✗ 批量仿真测试失败")
        return False


def test_energy_conservation():
    """测试能量守恒（自由落体）"""
    print("\n" + "=" * 60)
    print("测试8: 能量守恒（自由落体）")
    print("=" * 60)
    
    sim = QuadrotorDynamics({'dt': 0.001})
    sim.reset(position=[0, 0, 5.0])
    
    num_steps = 1000
    positions = []
    velocities = []
    
    for step in range(num_steps):
        # 不施加推力，自由落体
        state = sim.step([0, 0, 0, 0])
        positions.append(state['position'].copy())
        velocities.append(state['velocity'].copy())
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # 计算机械能
    potential_energy = sim.mass * sim.gravity * positions[:, 2]
    kinetic_energy = 0.5 * sim.mass * np.sum(velocities**2, axis=1)
    total_energy = potential_energy + kinetic_energy
    
    initial_energy = total_energy[0]
    final_energy = total_energy[-1]
    energy_loss_percent = abs(final_energy - initial_energy) / initial_energy * 100
    
    print(f"初始总能量: {initial_energy:.4f} J")
    print(f"最终总能量: {final_energy:.4f} J")
    print(f"能量损失: {energy_loss_percent:.2f}%")
    
    if energy_loss_percent < 5:
        print("✓ 能量守恒测试通过")
        return True
    else:
        print("✗ 能量守恒测试失败")
        return False


def visualize_trajectory():
    """可视化轨迹"""
    print("\n" + "=" * 60)
    print("可视化: 生成仿真轨迹图")
    print("=" * 60)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    hover_thrust = sim.mass * sim.gravity / 4.0
    
    # 先悬停，然后爬升，再下降
    num_steps = 1000
    positions = []
    velocities = []
    euler_angles = []
    
    for step in range(num_steps):
        if step < 300:
            # 悬停
            thrust_scale = 1.0
        elif step < 500:
            # 爬升
            thrust_scale = 1.3
        elif step < 700:
            # 下降
            thrust_scale = 0.7
        else:
            # 恢复悬停
            thrust_scale = 1.0
        
        rotor_thrusts = [hover_thrust * thrust_scale] * 4
        state = sim.step(rotor_thrusts)
        
        positions.append(state['position'].copy())
        velocities.append(state['velocity'].copy())
        euler_angles.append(state['euler_angles'].copy())
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    euler_angles = np.array(euler_angles)
    
    time = np.arange(num_steps) * sim.dt
    
    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 位置
    axes[0].plot(time, positions[:, 0], label='X', linewidth=2)
    axes[0].plot(time, positions[:, 1], label='Y', linewidth=2)
    axes[0].plot(time, positions[:, 2], label='Z', linewidth=2)
    axes[0].set_ylabel('Position (m)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Drone Trajectory Simulation', fontsize=14, fontweight='bold')
    
    # 速度
    axes[1].plot(time, velocities[:, 0], label='Vx', linewidth=2)
    axes[1].plot(time, velocities[:, 1], label='Vy', linewidth=2)
    axes[1].plot(time, velocities[:, 2], label='Vz', linewidth=2)
    axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 姿态角
    axes[2].plot(time, np.degrees(euler_angles[:, 0]), label='Roll', linewidth=2)
    axes[2].plot(time, np.degrees(euler_angles[:, 1]), label='Pitch', linewidth=2)
    axes[2].plot(time, np.degrees(euler_angles[:, 2]), label='Yaw', linewidth=2)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Attitude Angle (deg)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drone_trajectory.png', dpi=150, bbox_inches='tight')
    print("✓ 轨迹图已保存到 drone_trajectory.png")


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "无人机动力学仿真器测试套件" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    tests = [
        test_hover,
        test_climb,
        test_roll,
        test_pitch,
        test_yaw,
        test_rpm_input,
        test_batched_simulation,
        test_energy_conservation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append(False)
    
    # 可视化
    try:
        visualize_trajectory()
    except Exception as e:
        print(f"可视化失败: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✓ 所有测试通过！仿真器工作正常。")
    else:
        print(f"\n✗ {total - passed} 个测试失败，请检查。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
