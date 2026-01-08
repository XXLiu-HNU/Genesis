"""
Easy Sim - 独立无人机动力学仿真器

完全解耦Genesis的四旋翼无人机仿真器，用于sim2sim测试。

主要模块:
- core.drone_dynamics_sim: 动力学仿真器
- core.controller: PID控制器和状态估计器
- tests: 测试脚本
- examples: 使用示例
"""

from .core import (
    QuadrotorDynamics,
    BatchedQuadrotorDynamics,
    Odom,
    PIDController,
    load_pid_config,
)

__version__ = '1.0.0'
__all__ = [
    'QuadrotorDynamics',
    'BatchedQuadrotorDynamics',
    'Odom',
    'PIDController',
    'load_pid_config',
]
