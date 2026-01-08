# Genesis Sim 设置指南

## 重要说明

Genesis Sim直接使用Genesis物理引擎，因此需要完整的Genesis环境。

## 环境要求

### 方案A：使用tracking/controller的环境

如果你已经能运行`tracking/controller/circle_test.py`，那么genesis_sim也能运行。

测试一下：
```bash
cd tracking/controller
python circle_test.py
```

如果上面能运行，那么genesis_sim就能用！

### 方案B：完整安装Genesis

如果Genesis还没安装，参考官方文档：
https://github.com/Genesis-Embodied-AI/Genesis

需要安装：
1. gstaichi (Genesis定制版Taichi)
2. 其他依赖

## 快速测试

### 1. 测试Genesis环境

```bash
cd /home/fast/Documents/Genesis
python -c "import genesis as gs; gs.init(); print('Genesis可用')"
```

### 2. 测试genesis_sim

```python
# test_genesis_sim_basic.py
import os
import sys
sys.path.insert(0, '/home/fast/Documents/Genesis')

import genesis as gs
from genesis_sim import DroneSim

gs.init()
print("✓ Genesis初始化成功")

sim = DroneSim(dt=0.01, show_viewer=False, use_controller=True)
print("✓ DroneSim创建成功")

sim.reset(position=[0, 0, 1.0])
print("✓ Reset成功")

for i in range(10):
    state = sim.step_controller([0, 0, 1.0])
    
print(f"✓ 仿真成功，最终位置: {state['position']}")
print("\n所有测试通过！Genesis Sim可以使用。")
```

运行：
```bash
python test_genesis_sim_basic.py
```

## 使用建议

由于Genesis Sim需要完整的Genesis环境，推荐两种使用方式：

### 方式1：直接在tracking中使用

Genesis Sim实际上是`tracking/controller/circle_test.py`的简化封装版本。

你可以直接参考`tracking/controller`的使用方式：

```python
# 在tracking目录下
import genesis as gs
gs.init()

from controller.odom import Odom
from controller.pid import PIDcontroller

# 创建scene，drone等...
```

### 方式2：独立模块（需要完整环境）

如果Genesis环境已配置好：

```python
import genesis as gs
from genesis_sim import DroneSim

gs.init()
sim = DroneSim(show_viewer=True, use_controller=True)
sim.reset(position=[0, 0, 1.0])

for step in range(1000):
    state = sim.step_controller([0, 0, 1.0])
```

## 对比

### Easy Sim vs Genesis Sim

| 特性 | Easy Sim | Genesis Sim |
|------|----------|-------------|
| **依赖** | 仅NumPy/SciPy | Genesis + PyTorch + Taichi |
| **环境** | 任何Python | 需要Genesis环境 |
| **物理** | 简化模型 | 完整物理引擎 |
| **稳定性** | 一般 | 优秀 |
| **控制效果** | 需要重新调参 | 直接复用tracking参数 |
| **用途** | 教学/理解 | 生产使用/sim2sim |

### 什么时候用哪个？

**使用Easy Sim** 当：
- ✅ 需要快速测试，无Genesis环境
- ✅ 学习基本的无人机动力学
- ✅ 不需要高精度控制
- ✅ 轻量级部署

**使用Genesis Sim** 当：
- ✅ 已有Genesis环境
- ✅ 需要高精度仿真
- ✅ Sim2Sim测试（与训练环境一致）
- ✅ 复杂场景（碰撞、多体）

## 故障排除

### 问题1：ModuleNotFoundError: No module named 'genesis'

**原因**: Genesis未安装或未在PYTHONPATH中

**解决**:
```bash
cd /home/fast/Documents/Genesis
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -c "import genesis; print('OK')"
```

### 问题2：ModuleNotFoundError: No module named 'gstaichi'

**原因**: Genesis的依赖未安装

**解决**: 按照Genesis官方文档安装完整环境

### 问题3：能运行tracking/controller，但不能运行genesis_sim

**解决**: 使用相同的方式运行

```bash
# 如果tracking是这样运行的：
cd tracking/controller
python circle_test.py

# 那么genesis_sim应该：
cd /home/fast/Documents/Genesis
python -m genesis_sim.test_circle_tracking --test hover
```

## 推荐工作流程

1. **开发阶段**: 使用Easy Sim快速迭代
2. **测试阶段**: 使用Genesis Sim验证（如果环境可用）
3. **Sim2Sim**: 必须使用Genesis Sim（与训练环境一致）

## 下一步

如果Genesis环境配置正确，运行：

```bash
cd /home/fast/Documents/Genesis
python -m genesis_sim.test_circle_tracking --test circle --steps 2000
```

应该能看到完美的圆形轨迹跟踪效果！
