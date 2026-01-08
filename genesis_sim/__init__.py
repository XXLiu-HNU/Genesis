"""
Genesis Sim - 基于Genesis物理引擎的独立仿真器包装

这个包直接使用Genesis的完整物理引擎，只是提供了更简洁的接口。
与easy_sim（简化模型）不同，这里使用的是与tracking/controller完全一致的物理模型。
"""

from .drone_sim import DroneSim, DroneSimBatch
from .utils import visualize_trajectory

__all__ = [
    'DroneSim',
    'DroneSimBatch',
    'visualize_trajectory',
]

__version__ = '1.0.0'
