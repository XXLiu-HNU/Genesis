# Easy Sim - 独立无人机动力学仿真器

完全解耦Genesis的四旋翼无人机仿真器，用于sim2sim测试。

## 📁 目录结构

```
easy_sim/
├── core/                           # 核心模块
│   ├── __init__.py
│   ├── drone_dynamics_sim.py      # 无人机动力学仿真器
│   └── controller.py              # PID控制器和状态估计器
│
├── tests/                          # 测试脚本
│   ├── test_drone_dynamics.py     # 动力学仿真器测试（8/8通过）
│   ├── test_controller_debug.py   # 控制器调试（悬停测试）
│   ├── test_circle_tracking.py    # 圆形轨迹跟踪测试
│   ├── test_position_control.py   # 位置控制详细调试
│   ├── test_mixer_debug.py        # 混控器行为分析
│   └── quick_test.py              # 快速验证测试
│
├── examples/                       # 示例代码
│   ├── example_simple_usage.py    # 基本使用示例（5个示例）
│   └── example_policy_sim2sim.py  # RL策略sim2sim测试
│
├── docs/                           # 文档
│   ├── easy_sim_guide.md          # 使用指南
│   ├── CONTROLLER_README.txt      # 控制器完整文档
│   ├── controller_quick_start.txt # 控制器快速开始
│   └── SUMMARY.txt                # 项目总结
│
├── results/                        # 测试结果图片
│   ├── drone_trajectory.png       # 仿真轨迹图
│   └── circle_tracking_results.png # 圆形跟踪结果
│
├── __init__.py                     # 包初始化
└── README.md                       # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy scipy matplotlib
```

### 2. 测试仿真器

```bash
cd easy_sim/tests
python quick_test.py              # 快速验证（4个测试）
python test_drone_dynamics.py     # 完整测试（8个测试）
```

### 3. 测试控制器

```bash
cd easy_sim/tests
python test_controller_debug.py   # 悬停测试
python test_circle_tracking.py --radius 1.0 --steps 1500  # 圆形轨迹
```

### 4. 运行示例

```bash
cd easy_sim/examples
python example_simple_usage.py    # 查看5个基本示例
```

## 💻 基本使用

### 方法1: 直接导入（推荐）

```python
from easy_sim import QuadrotorDynamics, PIDController, Odom, load_pid_config
import numpy as np

# 创建仿真器
sim = QuadrotorDynamics({'dt': 0.01})
sim.reset(position=[0, 0, 1.0])

# 创建控制器
odom = Odom(num_envs=1)
odom.set_sim(sim)
config = load_pid_config()
controller = PIDController(1, odom, config, "position")

# 控制循环
for step in range(1000):
    target = np.array([[0.0, 0.0, 1.0, 0.0]])  # 悬停
    motor_rpms = controller.step(target)
    state = sim.step_rpm(motor_rpms[0])
```

### 方法2: 从core模块导入

```python
from easy_sim.core import QuadrotorDynamics, PIDController
from easy_sim.core import Odom, load_pid_config
```

## 🎯 核心功能

### 动力学仿真器

- **QuadrotorDynamics**: 单环境仿真器
- **BatchedQuadrotorDynamics**: 批量并行仿真器
- 完整的6自由度刚体动力学
- 支持推力(N)和RPM两种输入方式

### PID控制器

- **Position Control**: 位置 → RPM
- **Angle Control**: 姿态角 → RPM
- **Rate Control**: 角速率 → RPM
- **Odom**: 状态估计和坐标系转换

## 📊 测试状态

✅ 动力学仿真: 8/8 测试通过  
✅ 悬停控制: 工作正常  
⚠️ 轨迹跟踪: 基本功能可用，需要PID调参优化  

## 📚 文档

- **完整使用指南**: `docs/easy_sim_guide.md`
- **控制器文档**: `docs/CONTROLLER_README.txt`
- **快速开始**: `docs/controller_quick_start.txt`
- **项目总结**: `docs/SUMMARY.txt`

## 🔧 三种控制模式

| 模式 | 输入 | 用途 |
|------|------|------|
| Position | `[x, y, z, 0]` | 自主导航、轨迹跟踪 |
| Angle | `[roll, pitch, yaw, thrust]` | 遥控飞行、姿态稳定 |
| Rate | `[roll_rate, pitch_rate, yaw_rate, thrust]` | 特技飞行、快速响应 |

## ⚙️ 关键参数

```python
# 动力学参数（基于target_drone_urdf）
mass: 0.5 kg
arm_length: 0.12 m
kf: 3.16e-10  # 推力系数
km: 7.94e-12  # 力矩系数

# 控制器参数
base_rpm: 595000  # 理论悬停RPM
thrust_compensate: 1.0
位置控制器 kp: [1.5, 1.5, 1.0]
姿态控制器 kp: [6500, 6500, 7000]
```

## 🛠️ 开发指南

### 运行测试

```bash
# 在easy_sim目录下
python -m pytest tests/              # 运行所有测试
python tests/test_drone_dynamics.py  # 运行特定测试
```

### 导入说明

由于新的目录结构，需要在项目根目录（Genesis/）运行脚本：

```bash
cd /path/to/Genesis
python easy_sim/tests/quick_test.py
# 或
cd easy_sim/tests
python -m easy_sim.tests.quick_test
```

## 📝 更新日志

### v1.0.0 (2025-01-07)

- ✅ 重新组织目录结构
- ✅ 核心模块分离到 `core/`
- ✅ 测试脚本分离到 `tests/`
- ✅ 示例代码分离到 `examples/`
- ✅ 文档统一到 `docs/`
- ✅ 结果图片移至 `results/`
- ✅ 更新所有导入路径

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可

MIT License

---

**注意**: 这是一个完全独立的仿真器，不依赖Genesis。适合用于快速原型开发和sim2sim验证。
