# 轨迹编码器 - 快速开始

## 一、功能概述

为RL目标追踪任务增加了**轨迹编码器**，使用单层GRU将目标过去10帧的运动轨迹编码为32维特征向量。

**核心特性**:
- 🎯 **轻量化**: 仅3,648个参数
- 🛡️ **鲁棒性**: 自动处理目标遮挡情况
- ⚡ **高效**: GPU加速，~0.01ms/env
- 🔧 **即插即用**: 无缝集成到现有环境

## 二、关键修改

### 1. 观测空间变化

```python
# ⚠️ 必须更新 num_obs
obs_cfg = {
    "num_obs": 原始维度 + 32,  # 增加32维轨迹特征
    ...
}
```

### 2. 新增模块

```python
# tracking/track_env.py 中新增:

class TrajectoryEncoder(nn.Module):
    """单层GRU轨迹编码器"""
    # input: (batch, 10, 4) -> output: (batch, 32)

class TrackerEnv:
    def __init__(...):
        # 新增属性
        self.trajectory_encoder           # GRU编码器
        self.target_trajectory_history    # (N, 10, 4) 历史buffer
        self.target_visibility            # (N,) 可见性标签
        
    def _update_trajectory_history(self):
        # 每步自动更新历史轨迹
```

## 三、使用方法

### 基本使用

```python
from tracking.track_env import TrackerEnv

# 1. 配置环境 (更新num_obs)
obs_cfg["num_obs"] = 原始obs维度 + 32

# 2. 创建环境
env = TrackerEnv(num_envs, env_cfg, obs_cfg, reward_cfg)

# 3. 正常训练 (自动使用轨迹特征)
obs, _ = env.reset()
for step in range(max_steps):
    action = policy(obs)  # obs中包含32维轨迹特征
    obs, rew, done, _ = env.step(action)
```

### 观测空间结构

```python
obs = [
    rel_pos (3),           # 相对位置
    tracker_quat (4),      # 追踪器姿态
    tracker_vel (3+3),     # 追踪器速度
    target_vel (3),        # 目标速度
    last_actions (3),      # 上一步动作
    obs_features (21),     # 障碍物特征
    trajectory_feat (32),  # 🆕 轨迹编码特征
]
```

## 四、工作原理

### 轨迹历史格式

```
每帧: [相对x, 相对y, 相对z, 可见性v]

可见性标签:
  v = 1.0  目标可见
  v = 0.0  目标被遮挡
```

### 遮挡处理

```python
# 每一步自动处理
if 目标可见:
    记录当前相对位置
    v = 1.0
else:  # 被遮挡
    使用上一帧可见位置
    v = 0.0
    
# GRU会学习识别遮挡模式
```

### 处理流程

```
每一步 step():
  1. 更新tracker和target状态
  2. 检测目标可见性 (occlusion_check)
  3. 更新历史轨迹 (FIFO队列)
  4. GRU编码 -> 32维特征
  5. 拼接到观测空间
```

## 五、测试验证

### 运行单元测试

```bash
conda activate genesis
cd /home/fast/Documents/Genesis

# 测试编码器功能
python tracking/test_trajectory_encoder.py

# 生成可视化分析
python tracking/visualize_trajectory_encoding.py
```

### 预期输出

```
✓ 编码器参数量: 3,648
✓ 输出维度: (batch_size, 32)
✓ 能够区分遮挡和非遮挡情况
```

### 可视化结果

```
tracking/results/
├── trajectory_encoding_patterns.png    # 8种轨迹模式对比
└── trajectory_encoding_similarity.png  # 特征相似度矩阵
```

## 六、配置示例

### 完整配置参考

```python
# track_train.py 或 track_eval.py

obs_cfg = {
    # ⚠️ 更新此值!
    "num_obs": 3 + 4 + 3 + 3 + 3 + 3 + 21 + 32,  # 总计72维
    #          ↑   ↑   ↑   ↑   ↑   ↑   ↑    ↑
    #          |   |   |   |   |   |   |    └─ 轨迹特征(新增)
    #          |   |   |   |   |   |   └────── 障碍物特征
    #          |   |   |   |   |   └────────── 上一步动作
    #          |   |   |   |   └────────────── 目标速度
    #          |   |   |   └────────────────── 追踪器角速度
    #          |   |   └────────────────────── 追踪器线速度
    #          |   └────────────────────────── 追踪器姿态
    #          └────────────────────────────── 相对位置
    
    "obs_scales": {
        "max_diff": 1.0,
        "max_lin": 1.0,
        "max_ang": 1.0,
    }
}

env_cfg = {...}  # 保持不变
reward_cfg = {...}  # 保持不变
```

## 七、常见问题

### Q1: 如何调整历史窗口长度?

```python
# tracking/track_env.py, 第50行
self.trajectory_history_length = 10  # 修改此值 (例如改为15)
```

### Q2: 如何查看轨迹特征?

```python
# 在训练循环中
obs, _, _, _ = env.step(action)
trajectory_features = obs[:, -32:]  # 提取最后32维
print(trajectory_features)
```

### Q3: 模型权重如何保存?

```python
# 保存 (自动包含在PPO checkpoint中)
# 如需单独保存:
torch.save(env.trajectory_encoder.state_dict(), 'traj_encoder.pth')

# 加载
env.trajectory_encoder.load_state_dict(torch.load('traj_encoder.pth'))
```

### Q4: 可见性检测不准确怎么办?

可见性基于 `occlusion_check()` 函数，如需调整:
- 检查障碍物半径是否正确
- 调整 `inflation` 参数
- 可在 `_update_trajectory_history()` 中添加额外逻辑

## 八、性能指标

| 指标 | 数值 |
|------|------|
| 参数量 | 3,648 |
| 输入维度 | (N, 10, 4) |
| 输出维度 | (N, 32) |
| 前向时间 | ~0.01ms/env (GPU) |
| 内存占用 | ~640KB (4096 envs) |

## 九、文件清单

```
tracking/
├── track_env.py                              # ✏️ 已修改 (主要文件)
├── test_trajectory_encoder.py                # 🆕 新增 (单元测试)
├── visualize_trajectory_encoding.py          # 🆕 新增 (可视化)
├── TRAJECTORY_ENCODER_USAGE.md               # 🆕 新增 (详细文档)
├── TRAJECTORY_ENCODER_QUICKSTART.md          # 🆕 新增 (本文档)
└── results/
    ├── trajectory_encoding_patterns.png      # 🆕 生成
    └── trajectory_encoding_similarity.png    # 🆕 生成
```

## 十、下一步

1. **更新配置**: 修改训练脚本中的 `num_obs`
2. **运行测试**: 验证功能正常
3. **开始训练**: 使用新的观测空间训练策略
4. **监控性能**: 观察轨迹特征是否提升追踪效果

---

**实现日期**: 2026-01-08  
**遵循**: 《代码整洁之道》原则  
**兼容性**: Genesis 最新版本

如有问题，请查看详细文档: `TRAJECTORY_ENCODER_USAGE.md`
