"""
测试位置控制器的详细调试
"""

import numpy as np
from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics
from easy_sim.core.controller import Odom, PIDController, load_pid_config


def test_position_control_detailed():
    """详细测试位置控制"""
    print("=" * 70)
    print("位置控制器详细调试")
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
    
    print("\n测试：移动到新位置 [1, 0, 1.5]")
    print()
    
    target = np.array([[1.0, 0.0, 1.5, 0.0]])
    
    for step in range(100):
        # 更新状态
        odom.update()
        
        # 手动调用位置控制器并打印中间值
        if step % 10 == 0:
            print(f"\n步数: {step}")
            print(f"  当前位置: {odom.world_pos[0]}")
            print(f"  目标位置: {target[0, :3]}")
            print(f"  位置误差: {target[0, :3] - odom.world_pos[0]}")
            print(f"  当前速度: {odom.world_linear_vel[0]}")
            
            # 计算位置控制器的中间变量
            cur_pos_error = target[:, :3] - odom.world_pos
            setpoint_error = cur_pos_error * 5 - odom.world_linear_vel
            print(f"  速度误差: {setpoint_error[0]}")
            
            # 坐标转换
            setpoint_error[:, 1] *= -1
            setpoint_error = setpoint_error[:, [1, 0, 2]]
            print(f"  转换后误差 (roll, pitch, throttle): {setpoint_error[0]}")
            
            # PID项
            P_term = setpoint_error * controller.kp_p
            print(f"  P项: {P_term[0]}")
            print(f"  I项: {controller.I_term_p[0]}")
            
            sum_term = P_term + controller.I_term_p
            print(f"  总和: {sum_term[0]}")
            print(f"  推力命令: {np.clip(sum_term[0, -1], 0.0, 0.5)}")
        
        # 执行控制
        motor_rpms = controller.step(target)
        
        if step % 10 == 0:
            print(f"  实际推力命令: {controller.throttle_command[0]:.4f}")
            print(f"  电机RPM: {motor_rpms[0]}")
        
        # 执行仿真
        state = sim.step_rpm(motor_rpms[0])


if __name__ == "__main__":
    test_position_control_detailed()
