"""
工具函数
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_trajectory(positions, target_positions=None, save_path='trajectory.png'):
    """
    可视化轨迹
    
    参数:
    -----
    positions : array-like
        实际轨迹，形状 (n_steps, 3)
    target_positions : array-like, optional
        目标轨迹，形状 (n_steps, 3)
    save_path : str
        保存路径
    """
    positions = np.array(positions)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D轨迹
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', 
             label='Actual', linewidth=2)
    if target_positions is not None:
        target_positions = np.array(target_positions)
        ax1.plot(target_positions[:, 0], target_positions[:, 1], 
                target_positions[:, 2], 'r--', label='Target', linewidth=1)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # XY平面
    ax2 = fig.add_subplot(132)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', label='Actual', linewidth=2)
    if target_positions is not None:
        ax2.plot(target_positions[:, 0], target_positions[:, 1], 
                'r--', label='Target', linewidth=1)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 高度随时间变化
    ax3 = fig.add_subplot(133)
    time = np.arange(len(positions)) * 0.01  # 假设dt=0.01
    ax3.plot(time, positions[:, 2], 'b-', label='Actual', linewidth=2)
    if target_positions is not None:
        ax3.plot(time, target_positions[:, 2], 'r--', label='Target', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Height vs Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory visualization saved to {save_path}")
    plt.close()
