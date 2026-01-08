==============================================================================
                    Easy Sim 控制器集成 - 使用说明
==============================================================================

已完成的工作
------------
✓ 创建独立的PID控制器模块（controller.py）
✓ 支持三种控制模式：position, angle, rate
✓ 创建Odom状态估计器
✓ 完全解耦Genesis依赖
✓ 创建圆形轨迹跟踪测试（test_circle_tracking.py）
✓ 创建控制器调试工具（test_controller_debug.py）

文件列表
--------
easy_sim/
├── controller.py                 PID控制器和Odom模块
├── test_circle_tracking.py       圆形轨迹跟踪测试
├── test_controller_debug.py      控制器调试工具
├── test_position_control.py      位置控制详细调试
└── test_mixer_debug.py           混控器行为测试

基本使用
--------

1. 导入控制器

from controller import Odom, PIDController, load_pid_config
from drone_dynamics_sim import QuadrotorDynamics

2. 创建仿真器和控制器

# 创建仿真器
sim = QuadrotorDynamics({'dt': 0.01})
sim.reset(position=[0, 0, 1.0])

# 创建状态估计器
odom = Odom(num_envs=1)
odom.set_sim(sim)

# 加载PID配置
config = load_pid_config()

# 创建控制器
controller = PIDController(
    num_envs=1,
    odom=odom,
    config=config,
    controller_type="position"  # 或 "angle", "rate"
)

3. 控制循环

for step in range(1000):
    # 目标位置 [x, y, z, 0]
    target = np.array([[0.0, 0.0, 1.0, 0.0]])
    
    # 控制器计算电机RPM
    motor_rpms = controller.step(target)
    
    # 执行仿真
    state = sim.step_rpm(motor_rpms[0])


控制器模式
----------

1. Position Control (位置控制)
   输入：[x, y, z, 0]
   内部：位置PID -> 姿态角指令 -> 姿态角PID -> RPM
   用途：自主导航、轨迹跟踪

2. Angle Control (姿态角控制)
   输入：[roll, pitch, yaw, thrust]
   内部：姿态角PID -> RPM
   用途：遥控飞行、姿态稳定

3. Rate Control (角速率控制)
   输入：[roll_rate, pitch_rate, yaw_rate, thrust]
   内部：角速率PID -> RPM
   用途：特技飞行、快速响应


当前状态
--------
✓ 悬停控制：工作正常
✓ 基本位置控制：有效但需要调参
⚠ 圆形轨迹跟踪：存在高度下降问题
⚠ PID参数：需要针对easy_sim优化

已知问题
--------

1. 圆形轨迹跟踪时高度持续下降
   原因：PID参数可能不适配easy_sim的动力学模型
   解决：需要重新调参或检查混控器逻辑

2. 位置控制器的推力命令范围需要优化
   当前：throttle_command ∈ [-0.3, 0.3]
   问题：负值会被clip，导致无法减少推力

3. 混控器参数需要验证
   base_rpm: 595000（基于物理计算）
   thrust_compensate: 1.0
   需要验证这些参数在实际控制中的表现


测试命令
--------

1. 悬停测试（工作正常）
python test_controller_debug.py

2. 圆形轨迹跟踪
python test_circle_tracking.py --steps 1500 --radius 1.0 --omega 0.5

3. 位置控制详细调试
python test_position_control.py

4. 不同半径对比
python test_circle_tracking.py --test-radii


PID参数配置
------------

默认配置（基于tracking/controller/config/pos.yaml）：

Angular Controller (angle mode):
  kp: [6500, 6500, 7000]  # roll, pitch, yaw
  ki: [0.005, 0.005, 0.0]
  kd: [0.0001, 0.0001, 0.0]

Rate Controller:
  kp: [6500, 6500, 7000]
  ki: [0.01, 0.01, 0.0]
  kd: [0.0, 0.0, 0.0]

Position Controller:
  kp: [1.5, 1.5, 1.0]  # x, y, throttle
  ki: [0.01, 0.01, 0.01]
  kd: [0.05, 0.05, 0.0]

Motor Settings:
  base_rpm: 595000
  thrust_compensate: 1.0


调参建议
--------

如果遇到控制问题，可以尝试：

1. 降低位置控制器增益
   kp_t: 1.0 -> 0.5
   kp_x, kp_y: 1.5 -> 1.0

2. 调整推力命令范围
   在 _position_controller 中修改：
   self.throttle_command[:] = np.clip(sum_term[:, -1], -0.3, 0.3)

3. 调整base_rpm
   如果推力整体偏大或偏小，调整 base_rpm

4. 修改混控器逻辑
   如果需要更灵活的推力控制，可以修改 _mixer 函数


与Genesis版本的差异
--------------------

1. 数据类型：NumPy vs PyTorch
2. 坐标系转换：使用SciPy的Rotation
3. 批处理：简化的批处理实现
4. 动力学模型：使用easy_sim的四旋翼模型

注意：由于动力学模型的差异，Genesis中调好的PID参数
      可能需要在easy_sim中重新调整。


下一步工作
----------

1. ☐ 修复圆形轨迹跟踪的高度问题
2. ☐ 优化PID参数以适应easy_sim
3. ☐ 添加更多轨迹类型（8字形、螺旋等）
4. ☐ 实现批量环境的控制器
5. ☐ 添加控制器性能分析工具
6. ☐ 创建参数自动调优工具


参考资料
--------
- 原始控制器：tracking/controller/pid.py
- 配置文件：tracking/controller/config/pos.yaml
- Circle测试：tracking/controller/circle_test.py
- 无人机参数：genesis/assets/urdf/drones/target_drone_urdf/


联系方式
--------
如有问题，请检查：
1. 仿真器是否正确初始化
2. PID参数是否合理
3. 目标轨迹是否可达
4. 混控器输出是否在合理范围

==============================================================================
                    控制器已集成，基本功能可用
                    高级功能需要进一步调试和优化
==============================================================================
