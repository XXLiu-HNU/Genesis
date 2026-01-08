==============================================================================
                    Easy Sim 无人机仿真器 - 总览
==============================================================================

这是一个完全独立的四旋翼无人机动力学仿真器，用于sim2sim测试。
所有文件都在 easy_sim/ 目录下。

快速开始
--------
cd easy_sim
python quick_test.py           # 快速验证（4个测试）
python test_drone_dynamics.py  # 完整测试（8个测试）
python example_simple_usage.py # 查看使用示例

目录内容
--------
easy_sim/
├── 核心文件
│   ├── __init__.py                    Python包初始化
│   ├── drone_dynamics_sim.py (19K)    核心仿真器
│   │   ├── QuadrotorDynamics          单环境仿真器
│   │   └── BatchedQuadrotorDynamics   批量仿真器
│
├── 测试和示例
│   ├── quick_test.py (3.9K)           快速测试（推荐）
│   ├── test_drone_dynamics.py (14K)   完整测试套件（8个测试）
│   ├── example_simple_usage.py (5.6K) 基本使用示例（5个示例）
│   └── example_policy_sim2sim.py (16K) 策略测试（需要torch）

核心功能
--------
✓ 完全独立（不依赖Genesis）
✓ 四旋翼动力学仿真（6自由度刚体）
✓ 双输入接口（推力N / RPM）
✓ 单环境 + 批量仿真
✓ 完整状态输出（位置、速度、姿态、角速度）

基本用法
--------
# 导入
from easy_sim import QuadrotorDynamics

# 创建仿真器
sim = QuadrotorDynamics({'dt': 0.01})
sim.reset(position=[0, 0, 1.0])

# 仿真
state = sim.step([1.23, 1.23, 1.23, 1.23])  # 推力输入
# 或
state = sim.step_rpm([60000, 60000, 60000, 60000])  # RPM输入

# 获取状态
position = state['position']           # [x, y, z]
velocity = state['velocity']           # [vx, vy, vz]
quaternion = state['quaternion']       # [w, x, y, z]
angular_velocity = state['angular_velocity']  # [wx, wy, wz]
euler_angles = state['euler_angles']   # [roll, pitch, yaw]

测试状态
--------
✓ 快速测试:      4/4 通过 (quick_test.py)
✓ 完整测试:      8/8 通过 (test_drone_dynamics.py)
✓ 导入测试:      通过（包导入 + 本地导入）
✓ 批量仿真:      通过（支持多环境并行）
✓ 示例运行:      5个示例全部正常

动力学参数
----------
来源: genesis/assets/urdf/drones/target_drone_urdf/drone.urdf

质量:       0.5 kg
臂长:       0.12 m
推力系数:   3.16e-10
力矩系数:   7.94e-12
转动惯量:   diag(1.4e-3, 1.4e-3, 1.4e-3) kg·m²

螺旋桨布局（X型）:
  prop0: 前右 (+x,-y) 顺时针
  prop1: 后右 (-x,-y) 逆时针  
  prop2: 后左 (-x,+y) 顺时针
  prop3: 前左 (+x,+y) 逆时针

推荐工作流
----------
1. 验证安装
   cd easy_sim && python quick_test.py

2. 学习使用
   python example_simple_usage.py

3. 集成到代码
   from easy_sim import QuadrotorDynamics
   sim = QuadrotorDynamics()

4. 测试策略（如果有训练好的模型）
   python example_policy_sim2sim.py --policy <model.pt> --visualize

进阶使用
--------
- 批量仿真: BatchedQuadrotorDynamics(num_envs=1024)
- 自定义参数: QuadrotorDynamics({'mass': 0.6, 'dt': 0.005})
- 轨迹跟踪: 参考 example_simple_usage.py 中的示例5


==============================================================================
准备就绪！开始使用: cd easy_sim && python quick_test.py
==============================================================================
