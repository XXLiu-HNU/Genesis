# Genesis Sim - 基于Genesis物理引擎的独立仿真器

## 概述

Genesis Sim直接使用Genesis的**完整物理引擎**，只是提供了更简洁的接口。

### 为什么用这个版本？

与`easy_sim`（简化动力学模型）不同，`genesis_sim`：

- ✅ **使用Genesis完整物理引擎** - 8000+行Taichi编写的高性能刚体求解器
- ✅ **完全一致的物理模型** - 与`tracking/controller`使用相同的物理引擎
- ✅ **PID参数直接复用** - 无需重新调参，直接使用`tracking/controller/config/pos.yaml`
- ✅ **数值稳定** - 已经过充分验证的约束求解器和积分器
- ✅ **支持碰撞检测** - 完整的多体动力学支持
- ✅ **GPU加速** - Taichi并行计算，高性能
- ✅ **简洁的接口** - 封装了复杂的Scene创建和配置

### vs Easy Sim

| 特性 | Easy Sim | Genesis Sim |
|------|----------|-------------|
| 物理引擎 | 自己实现（简化） | Genesis完整引擎 |
| 稳定性 | 一般 | 优秀 |
| 准确性 | 一般 | 高 |
| PID控制 | 需要重新调参 | 直接复用 |
| 依赖 | NumPy | Genesis + PyTorch |
| 性能 | CPU | GPU加速 |
| 用途 | 教学/理解 | 生产使用 |

## 安装

```bash
# 已经在Genesis项目中，无需额外安装
cd genesis_sim
```

## 快速开始

### 1. 基本使用

```python
import genesis as gs
from genesis_sim import DroneSim

# 初始化Genesis
gs.init()

# 创建仿真器（自动配置PID控制器）
sim = DroneSim(dt=0.01, show_viewer=True, use_controller=True)

# 重置到初始位置
sim.reset(position=[0, 0, 1.0])

# 使用PID控制器进行位置控制
for i in range(1000):
    target = [0, 0, 1.0]  # 悬停
    state = sim.step_controller(target)
    print(f"Position: {state['position']}")
```

### 2. 圆形轨迹跟踪

```python
import numpy as np
import genesis as gs
from genesis_sim import DroneSim

gs.init()
sim = DroneSim(dt=0.01, show_viewer=True, use_controller=True)

# 初始化在圆周上
radius = 1.0
sim.reset(position=[radius, 0, 1.0])

# 圆形轨迹
omega = 0.5  # 角速度
for step in range(2000):
    t = step * 0.01
    angle = omega * t
    target = [
        radius * np.cos(angle),
        radius * np.sin(angle),
        1.0
    ]
    state = sim.step_controller(target)
```

### 3. 直接控制RPM

```python
gs.init()
sim = DroneSim(dt=0.01, show_viewer=True, use_controller=False)

sim.reset(position=[0, 0, 1.0])

# 直接设置4个电机的RPM
hover_rpm = 595000
for i in range(1000):
    rpms = [hover_rpm] * 4
    state = sim.step_rpm(rpms)
```

### 4. 批量并行仿真

```python
import genesis as gs
from genesis_sim import DroneSimBatch

gs.init()

# 创建10个并行环境
sim = DroneSimBatch(
    num_envs=10,
    dt=0.01,
    show_viewer=True,
    rendered_envs=5,  # 只渲染前5个
    use_controller=True
)

# 批量重置
positions = [[0, 0, 1.0] for _ in range(10)]
sim.reset(position=positions)

# 批量控制
for step in range(1000):
    targets = [[0, 0, 1.0] for _ in range(10)]
    states = sim.step_controller(targets)
    # states['position'] 形状: (10, 3)
```

## API文档

### DroneSim

#### 初始化
```python
DroneSim(dt=0.01, show_viewer=False, use_controller=True)
```

**参数:**
- `dt` (float): 时间步长，默认0.01s
- `show_viewer` (bool): 是否显示可视化窗口
- `use_controller` (bool): 是否自动初始化PID控制器

#### 方法

**reset(position=None, quaternion=None)**

重置无人机状态。

- `position`: 初始位置 `[x, y, z]`
- `quaternion`: 初始四元数 `[w, x, y, z]`

**step_controller(target_position)**

使用PID控制器执行一步仿真。

- `target_position`: 目标位置 `[x, y, z]` 或 `[x, y, z, yaw]`
- 返回: `state` 字典

**step_rpm(rpms)**

直接指定RPM执行一步仿真。

- `rpms`: 4个电机的RPM值 `[rpm1, rpm2, rpm3, rpm4]`
- 返回: `state` 字典

**get_state()**

获取当前状态。

- 返回: 字典，包含：
  - `position`: numpy数组 `[x, y, z]`
  - `velocity`: numpy数组 `[vx, vy, vz]`
  - `quaternion`: numpy数组 `[w, x, y, z]`
  - `angular_velocity`: numpy数组 `[wx, wy, wz]`

### DroneSimBatch

批量版本，接口与`DroneSim`一致，但所有输入/输出都是批量的。

```python
DroneSimBatch(num_envs=10, dt=0.01, show_viewer=False, 
              rendered_envs=None, use_controller=True)
```

## 测试

### 圆形轨迹跟踪

```bash
cd genesis_sim
python test_circle_tracking.py --test circle --radius 1.0 --omega 0.5 --steps 2000
```

### 悬停测试

```bash
python test_circle_tracking.py --test hover
```

### 期望结果

使用Genesis物理引擎，圆形轨迹跟踪应该能达到：
- 平均误差 < 0.1m
- 最大误差 < 0.3m
- 稳定跟踪，无发散

## 技术细节

### 物理引擎

Genesis Sim直接使用Genesis的刚体求解器（`genesis/engine/solvers/rigid/rigid_solver_decomp.py`），包括：

1. **推力计算**: `F = kf * RPM²`
2. **力矩计算**: `τ = km * RPM² * spin_direction`
3. **多体动力学**: 完整的关节约束、碰撞检测
4. **数值积分**: Newton约束求解器 + substeps
5. **并行计算**: Taichi GPU加速

### PID控制器

直接复用`tracking/controller`的实现：
- `tracking/controller/odom.py`: 状态估计
- `tracking/controller/pid.py`: PID控制器
- `tracking/controller/config/pos.yaml`: PID参数

### 与tracking/controller的一致性

Genesis Sim的实现与`tracking/controller/circle_test.py`完全一致：

```python
# tracking/controller/circle_test.py的核心逻辑
self.scene = gs.Scene(...)
self.drone = self.scene.add_entity(gs.morphs.Drone(...))
self.scene.build(n_envs=num_envs)

# 控制循环
rpms = self.drone.controller.step(target)
self.drone.set_propellels_rpm(rpms)
self.scene.step()
```

Genesis Sim只是把这个模式封装成了更简洁的接口。

## 下一步

- [ ] 测试sim2sim（使用训练好的RL策略）
- [ ] 添加更多轨迹类型（figure8, waypoint）
- [ ] 性能分析和优化
- [ ] 与easy_sim的性能对比

## 参考

- Genesis文档: https://github.com/Genesis-Embodied-AI/Genesis
- tracking/controller: 原始实现参考
- easy_sim: 简化版本（教学用）
