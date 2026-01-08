"""
测试圆形轨迹跟踪（使用Genesis物理引擎）

这个版本应该能完美工作，因为使用的是与tracking/controller一致的物理模型和PID参数。
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import genesis as gs
from genesis_sim.drone_sim import DroneSim
from genesis_sim.utils import visualize_trajectory


def test_circle_tracking(radius=1.0, omega=0.5, height=1.0, 
                         num_steps=2000, show_viewer=True):
    """
    测试圆形轨迹跟踪
    
    参数:
    -----
    radius : float
        圆形轨迹半径 (m)
    omega : float
        角速度 (rad/s)
    height : float
        飞行高度 (m)
    num_steps : int
        仿真步数
    show_viewer : bool
        是否显示可视化
    """
    print("=" * 70)
    print("圆形轨迹跟踪测试（Genesis物理引擎）")
    print("=" * 70)
    print(f"参数:")
    print(f"  轨迹半径: {radius} m")
    print(f"  角速度: {omega} rad/s")
    print(f"  周期: {2*np.pi/omega:.2f} s")
    print(f"  高度: {height} m")
    print(f"  仿真步数: {num_steps}")
    print(f"  时间步长: 0.01 s")
    print(f"  总时间: {num_steps * 0.01:.1f} s")
    print()
    
    # 初始化Genesis
    gs.init()
    
    # 创建仿真器
    sim = DroneSim(dt=0.01, show_viewer=show_viewer, use_controller=True)
    
    # 初始位置在圆周上
    initial_angle = 0.0
    initial_pos = [
        radius * np.cos(initial_angle),
        radius * np.sin(initial_angle),
        height
    ]
    
    print(f"初始位置: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f}]")
    sim.reset(position=initial_pos)
    
    print("初始化完成，开始仿真...")
    print()
    
    # 记录数据
    actual_positions = []
    target_positions = []
    tracking_errors = []
    
    # 运行仿真
    for step in range(num_steps):
        t = step * 0.01
        angle = initial_angle + omega * t
        
        # 目标位置
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        target_z = height
        target = [target_x, target_y, target_z]
        
        # 执行控制
        state = sim.step_controller(target)
        
        # 记录数据
        pos = state['position']
        actual_positions.append(pos)
        target_positions.append(target)
        
        error = np.linalg.norm(pos - np.array(target))
        tracking_errors.append(error)
        
        # 打印进度
        if step % 200 == 0:
            print(f"步数: {step:4d}, 时间: {t:6.2f}s, "
                  f"跟踪误差: {error:.4f}m, "
                  f"位置: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
    
    # 转换为数组
    actual_positions = np.array(actual_positions)
    target_positions = np.array(target_positions)
    tracking_errors = np.array(tracking_errors)
    
    # 统计信息
    print()
    print("=" * 70)
    print("跟踪性能统计")
    print("=" * 70)
    print(f"平均跟踪误差: {np.mean(tracking_errors):.4f} m")
    print(f"最大跟踪误差: {np.max(tracking_errors):.4f} m")
    print(f"最小跟踪误差: {np.min(tracking_errors):.4f} m")
    print(f"RMSE: {np.sqrt(np.mean(tracking_errors**2)):.4f} m")
    print()
    
    # 稳态性能（t > 5s）
    steady_idx = int(5.0 / 0.01)
    if len(tracking_errors) > steady_idx:
        steady_errors = tracking_errors[steady_idx:]
        print(f"稳态性能（t > 5s）:")
        print(f"  平均误差: {np.mean(steady_errors):.4f} m")
        print(f"  最大误差: {np.max(steady_errors):.4f} m")
        print(f"  标准差: {np.std(steady_errors):.4f} m")
    
    # 可视化
    visualize_trajectory(
        actual_positions,
        target_positions,
        save_path='genesis_sim_circle_tracking.png'
    )
    
    print()
    print("测试完成！")


def test_hover():
    """测试悬停控制"""
    print("=" * 70)
    print("悬停控制测试（Genesis物理引擎）")
    print("=" * 70)
    
    gs.init()
    sim = DroneSim(dt=0.01, show_viewer=True, use_controller=True)
    
    # 初始位置
    initial_pos = [0.0, 0.0, 1.0]
    sim.reset(position=initial_pos)
    
    print(f"目标：悬停在 {initial_pos}")
    print()
    
    positions = []
    
    for step in range(500):
        state = sim.step_controller(initial_pos)
        positions.append(state['position'])
        
        if step % 100 == 0:
            pos = state['position']
            error = np.linalg.norm(pos - np.array(initial_pos))
            print(f"步数: {step:3d}, 误差: {error:.4f}m, "
                  f"位置: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
    
    positions = np.array(positions)
    final_error = np.linalg.norm(positions[-1] - np.array(initial_pos))
    
    print()
    print(f"最终误差: {final_error:.4f} m")
    print("悬停测试完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试圆形轨迹跟踪')
    parser.add_argument('--test', type=str, default='circle', 
                       choices=['circle', 'hover'],
                       help='测试类型')
    parser.add_argument('--radius', type=float, default=1.0,
                       help='圆形轨迹半径 (m)')
    parser.add_argument('--omega', type=float, default=0.5,
                       help='角速度 (rad/s)')
    parser.add_argument('--height', type=float, default=1.0,
                       help='飞行高度 (m)')
    parser.add_argument('--steps', type=int, default=2000,
                       help='仿真步数')
    parser.add_argument('--no-viewer', action='store_true',
                       help='禁用可视化窗口')
    
    args = parser.parse_args()
    
    if args.test == 'circle':
        test_circle_tracking(
            radius=args.radius,
            omega=args.omega,
            height=args.height,
            num_steps=args.steps,
            show_viewer=not args.no_viewer
        )
    else:
        test_hover()
