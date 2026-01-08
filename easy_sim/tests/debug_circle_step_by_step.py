"""
逐步调试圆形轨迹跟踪
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics
from easy_sim.core.controller import Odom, PIDController, load_pid_config


def debug_one_cycle():
    """详细调试一个控制周期"""
    print("=" * 70)
    print("详细调试一个控制周期")
    print("=" * 70)
    
    # 创建仿真器
    sim = QuadrotorDynamics({'dt': 0.01})
    sim.reset(position=[1.0, 0.0, 1.0])  # 圆周上的起点
    
    # 创建控制器
    odom = Odom(num_envs=1)
    odom.set_sim(sim)
    config = load_pid_config()
    controller = PIDController(1, odom, config, "position")
    
    # 圆形轨迹参数
    radius = 1.0
    omega = 0.5
    height = 1.0
    
    print("\n初始状态:")
    print(f"  无人机位置: {sim.get_state()['position']}")
    
    # 仿真几步
    for step in range(5):
        t = step * 0.01
        angle = omega * t
        
        # 目标位置
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        target_z = height
        target = np.array([[target_x, target_y, target_z, 0.0]])
        
        print(f"\n==== 步数 {step}, 时间 {t:.2f}s ====")
        print(f"目标位置: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")
        
        # 更新Odom
        odom.update()
        print(f"当前位置: {odom.world_pos[0]}")
        print(f"当前速度: {odom.world_linear_vel[0]}")
        
        # 手动计算位置控制器
        cur_pos_error = target[:, :3] - odom.world_pos
        print(f"位置误差: {cur_pos_error[0]}")
        
        setpoint_error = cur_pos_error * 5 - odom.world_linear_vel
        print(f"速度目标误差: {setpoint_error[0]}")
        
        # 坐标转换
        setpoint_error[:, 1] *= -1
        setpoint_error = setpoint_error[:, [1, 0, 2]]
        print(f"转换后 (roll, pitch, throttle): {setpoint_error[0]}")
        
        # PID
        P_term = setpoint_error * controller.kp_p
        I_term = controller.I_term_p
        D_term = controller.D_term_p
        sum_term = P_term + I_term + D_term
        
        print(f"P项: {P_term[0]}")
        print(f"I项: {I_term[0]}")
        print(f"D项: {D_term[0]}")
        print(f"总和: {sum_term[0]}")
        
        # throttle_command
        throttle_cmd = np.clip(sum_term[0, -1], 0.0, 0.5)
        print(f"推力命令: {throttle_cmd:.4f}")
        
        # 调用控制器
        motor_rpms = controller.step(target)
        print(f"电机RPM: {motor_rpms[0]}")
        print(f"实际推力命令: {controller.throttle_command[0]:.4f}")
        print(f"PID输出 (roll,pitch,yaw): {controller.pid_output[0]}")
        
        # 执行仿真
        state = sim.step_rpm(motor_rpms[0])
        print(f"仿真后位置: {state['position']}")
        print(f"仿真后速度: {state['velocity']}")


if __name__ == "__main__":
    debug_one_cycle()
