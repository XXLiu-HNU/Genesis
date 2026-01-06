# 并行目标导航环境

高性能的目标无人机导航环境，支持大规模并行仿真，专为强化学习训练设计。

## 性能指标

在 RTX 5060 Ti (16GB) 上的实测结果：

| 环境数 | FPS | 吞吐量 | 步进时间 |
|--------|-----|--------|----------|
| 1000   | 323 | 323K steps/s | 3.1ms |

## 快速开始

### 1. 基础测试

```bash
cd /home/fast/Documents/Genesis
conda activate genesis
python tracking/target_planning/test_basic.py
```

### 2. 障碍物避让测试

```bash
# 可视化观察避障效果（显示前10个环境）
python tracking/target_planning/test_obstacle_avoidance.py --mode visual

# 压力测试（500个环境，1000步）
python tracking/target_planning/test_obstacle_avoidance.py --mode stress
```

### 3. 运行示例

```bash
# 100个环境可视化
python tracking/target_planning/quick_start.py --example 1

# 1000个环境性能测试
python tracking/target_planning/quick_start.py --example 3

# 5000个环境大规模测试
python tracking/target_planning/quick_start.py --example 4
```

### 3. 扩展性测试

```bash
# 单一规模测试
python tracking/target_planning/test_parallel_scalability.py --mode single --n_envs 1000 --n_steps 100

# 完整扩展性测试 (100到10000个环境)
python tracking/target_planning/test_parallel_scalability.py --mode scalability
```

## 使用方式

### 基本用法

```python
import genesis as gs
from target_planning.parallel_nav_env import ParallelTargetNavEnv

gs.init(backend=gs.gpu)

# 创建环境
env = ParallelTargetNavEnv(num_envs=1000, show_viewer=True)

# 运行
for step in range(10000):
    env.step()
    
    # 获取目标状态
    states = env.get_target_states()
    # states['pos']:  [1000, 3] 位置
    # states['vel']:  [1000, 3] 速度
    # states['quat']: [1000, 4] 四元数
```

### 集成到追踪训练

```python
class TrackingEnv:
    def __init__(self, num_envs):
        # 创建目标导航环境
        self.target_env = ParallelTargetNavEnv(num_envs=num_envs, show_viewer=False)
        
        # 创建追踪无人机
        self.tracker = scene.add_entity(gs.morphs.Drone(...))
        
    def step(self, action):
        # 目标移动
        self.target_env.step()
        target_states = self.target_env.get_target_states()
        
        # 追踪器执行动作
        self.tracker.set_propellels_rpm(action)
        scene.step()
        
        # 计算观测和奖励
        obs = self._compute_obs(target_states)
        reward = self._compute_reward(target_states)
        
        return obs, reward, done, info
```

## 核心改进

### 对比原版 (nav.py)

| 特性 | 原版 | 新版 | 提升 |
|------|------|------|------|
| 并行度 | 所有环境共享单个目标 | 每个环境独立目标 | ✅ 真并行 |
| 计算设备 | CPU串行规划 | GPU向量化 | ✅ 10-20x |
| 状态管理 | Python对象 | 连续GPU张量 | ✅ 内存高效 |
| 重规划 | 全局同步 | 选择性批量 | ✅ 灵活 |

### 关键实现

**1. 批量化状态**
```python
# 每个环境独立的目标和速度
self.goal_xy = torch.zeros((num_envs, 2), device=device)
self.current_vel = torch.zeros((num_envs, 2), device=device)
self.reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
```

**2. 向量化计算**
```python
# 所有环境同时计算，无Python循环
to_goal = self.goal_xy - current_pos              # [num_envs, 2]
dist = torch.norm(to_goal, dim=-1, keepdim=True)  # [num_envs, 1]
direction = to_goal / (dist + 1e-6)               # [num_envs, 2]
target_pos = current_pos + velocity * lookahead   # [num_envs, 2]
```

**3. 批量采样**
```python
# GPU批量位置采样和碰撞检测
candidates = torch.empty((batch, 2), device=device)
candidates[:, 0].uniform_(x_min, x_max)
diff = candidates.unsqueeze(1) - obs_xy.unsqueeze(0)  # [batch, M, 2]
dist = torch.norm(diff, dim=-1)                       # [batch, M]
valid = candidates[dist.min(dim=-1).values >= safe_radius]
```

**4. 选择性重规划**
```python
# 只为到达目标的环境重新规划
reached = self.follower.check_reached()  # [num_envs]
if reached.any():
    self._replan_for_envs(reached)
```

## API文档

### ParallelTargetNavEnv

**初始化**
```python
ParallelTargetNavEnv(
    num_envs: int,              # 并行环境数量
    show_viewer: bool = False,  # 是否显示可视化
    config_path: str = None     # 配置文件路径
)
```

**方法**
- `step()`: 执行一步仿真
- `get_target_states()`: 获取所有目标状态
  - 返回: `dict{'pos': [N,3], 'vel': [N,3], 'quat': [N,4]}`

### VectorizedPathFollower

**初始化**
```python
VectorizedPathFollower(
    num_envs: int,
    device: torch.device,
    v_max: float = 0.6,              # 最大速度
    a_max: float = 1.2,              # 最大加速度
    goal_reach_thresh: float = 0.3   # 到达阈值
)
```

**方法**
- `set_goals(goals)`: 设置目标 [num_envs, 2]
- `step(current_pos, dt)`: 计算下一位置 [num_envs, 2]
- `check_reached()`: 返回到达mask [num_envs]

## 配置说明

### config/search.yaml

```yaml
drone:
  height: 1.5          # 飞行高度
  radius: 0.12         # 无人机半径

safety:
  margin: 0.12         # 安全余量

follower:
  v_max: 1.0           # 最大速度 (m/s)
  a_max: 2.0           # 最大加速度 (m/s²)
  goal_reach_thresh: 0.3  # 到达阈值 (m)

obstacles:
  n: 20                # 障碍物数量
```

### config/pos.yaml

PID控制器参数，通常不需要修改。

## 文件结构

```
target_planning/
├── README.md                          # 本文件
├── parallel_nav_env.py                # 主环境实现
├── test_basic.py                      # 基础功能测试
├── test_parallel_scalability.py      # 扩展性测试
├── quick_start.py                     # 快速入门示例
├── nav.py                             # 原始实现(参考)
└── config/
    ├── search.yaml                    # 导航参数
    └── pos.yaml                       # PID参数
```

## 系统要求

- Python >= 3.8
- PyTorch >= 2.0
- Genesis >= 0.3.11
- CUDA-capable GPU (推荐)
- 显存需求:
  - 1000个环境: ~5GB
  - 5000个环境: ~15GB

## 常见问题

**Q: 显存不足怎么办？**

减少环境数量或障碍物数量：
```python
env = ParallelTargetNavEnv(num_envs=500)  # 减少环境数
```

或在配置文件中减少障碍物：
```yaml
obstacles:
  n: 10  # 从20减少到10
```

**Q: 如何确保使用GPU？**

```python
gs.init(backend=gs.gpu)  # 明确指定GPU
```

**Q: 导航策略可以自定义吗？**

可以。修改 `VectorizedPathFollower.step()` 方法实现自定义导航逻辑。

**Q: 支持动态障碍吗？**

当前版本支持静态障碍。动态障碍需要修改 `_sample_free_positions_batch()` 和碰撞检测逻辑。

## 下一步

1. ✅ 基础测试通过
2. ✅ 性能验证完成
3. 🔲 集成到追踪训练环境 (见 `track_env.py`)
4. 🔲 调整导航参数优化轨迹
5. 🔲 添加更复杂的避障策略

## 许可证

遵循 Genesis 项目许可证。
