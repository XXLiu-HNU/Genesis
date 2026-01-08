"""
控制器调试 - 检查基本悬停功能
"""

import numpy as np
from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics
from easy_sim.core.controller import Odom, PIDController, load_pid_config


def test_hover():
    """测试悬停控制"""
    print("=" * 70)
    print("悬停控制测试")
    print("=" * 70)
    
    # 创建仿真器
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    # 创建控制器
    odom = Odom(num_envs=1)
    odom.set_sim(sim)
    
    config = load_pid_config()
    controller = PIDController(
        num_envs=1,
        odom=odom,
        config=config,
        controller_type="position"
    )
    
    print("\n目标：悬停在 [0, 0, 1.0]")
    print("\n开始测试...")
    
    for step in range(500):
        # 目标位置：原地悬停
        target = np.array([[0.0, 0.0, 1.0, 0.0]])
        
        # 控制器计算RPM
        motor_rpms = controller.step(target)
        
        # 执行仿真
        state = sim.step_rpm(motor_rpms[0])
        
        if step % 50 == 0:
            pos = state['position']
            vel = state['velocity']
            euler = np.degrees(state['euler_angles'])
            
            print(f"\n步数: {step}")
            print(f"  位置: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
            print(f"  速度: [{vel[0]:7.3f}, {vel[1]:7.3f}, {vel[2]:7.3f}]")
            print(f"  姿态: [{euler[0]:7.2f}, {euler[1]:7.2f}, {euler[2]:7.2f}] deg")
            print(f"  电机RPM: {motor_rpms[0]}")
            print(f"  推力命令: {controller.throttle_command[0]:.4f}")
            print(f"  PID输出: {controller.pid_output[0]}")


def test_step_by_step():
    """逐步测试控制器"""
    print("\n" + "=" * 70)
    print("逐步测试")
    print("=" * 70)
    
    # 创建仿真器
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    # 创建状态估计器
    odom = Odom(num_envs=1)
    odom.set_sim(sim)
    
    # 第一次更新
    print("\n初始状态:")
    odom.update()
    print(f"  世界位置: {odom.world_pos[0]}")
    print(f"  世界速度: {odom.world_linear_vel[0]}")
    print(f"  机体欧拉角: {np.degrees(odom.body_euler[0])}")
    print(f"  机体角速度: {odom.body_ang_vel[0]}")
    
    # 创建控制器
    config = load_pid_config()
    print(f"\nPID配置:")
    print(f"  位置控制器 kp: {config['pos']}")
    print(f"  角度控制器 kp: {config['ang']}")
    print(f"  base_rpm: {config['base_rpm']}")
    print(f"  thrust_compensate: {config['thrust_compensate']}")
    
    controller = PIDController(
        num_envs=1,
        odom=odom,
        config=config,
        controller_type="position"
    )
    
    # 测试一步控制
    print("\n执行一步控制:")
    target = np.array([[0.0, 0.0, 1.0, 0.0]])
    print(f"  目标位置: {target[0, :3]}")
    
    motor_rpms = controller.step(target)
    print(f"  电机RPM: {motor_rpms[0]}")
    print(f"  推力命令: {controller.throttle_command[0]:.4f}")
    print(f"  PID输出 (roll, pitch, yaw): {controller.pid_output[0]}")
    
    # 执行仿真
    state = sim.step_rpm(motor_rpms[0])
    print(f"\n仿真后状态:")
    print(f"  位置: {state['position']}")
    print(f"  速度: {state['velocity']}")


def test_base_rpm():
    """测试基础RPM是否正确"""
    print("\n" + "=" * 70)
    print("基础RPM测试")
    print("=" * 70)
    
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[0, 0, 1.0])
    
    # 计算悬停所需RPM
    hover_thrust = sim.mass * sim.gravity / 4.0
    hover_omega = np.sqrt(hover_thrust / sim.kf)
    hover_rpm = hover_omega * 60 / (2 * np.pi)
    
    print(f"\n理论悬停参数:")
    print(f"  质量: {sim.mass} kg")
    print(f"  推力系数 kf: {sim.kf}")
    print(f"  单个螺旋桨悬停推力: {hover_thrust:.4f} N")
    print(f"  悬停RPM: {hover_rpm:.0f}")
    
    # 测试悬停
    print(f"\n使用理论RPM悬停测试...")
    for i in range(100):
        state = sim.step_rpm([hover_rpm] * 4)
    
    print(f"100步后位置: {state['position']}")
    print(f"100步后速度: {state['velocity']}")
    
    # 测试使用base_rpm
    sim.reset(position=[0, 0, 1.0])
    config = load_pid_config()
    base_rpm = config['base_rpm']
    thrust_comp = config['thrust_compensate']
    
    print(f"\n配置中的base_rpm: {base_rpm}")
    print(f"推力补偿: {thrust_comp}")
    
    # 使用配置的RPM
    test_rpm = base_rpm * thrust_comp
    print(f"\n使用配置RPM测试: {test_rpm:.0f}")
    for i in range(100):
        state = sim.step_rpm([test_rpm] * 4)
    
    print(f"100步后位置: {state['position']}")
    print(f"100步后速度: {state['velocity']}")


if __name__ == "__main__":
    test_base_rpm()
    test_step_by_step()
    test_hover()
