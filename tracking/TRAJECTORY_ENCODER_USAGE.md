# 目标轨迹编码器 (Trajectory Encoder) 使用文档

## 概述

为强化学习的目标追踪任务新增了**轨迹编码模块**，用于将目标过去N帧的运动轨迹编码为特征向量，作为RL状态空间的一部分。该模块使用轻量级的GRU网络，适合嵌入式部署。

## 技术规格

### TrajectoryEncoder 模块

- **网络结构**: 单层 `nn.GRU`
- **输入维度**: 4 (相对坐标 x, y, z + 可见性标签 v)
- **隐藏层维度**: 32 (轻量化设计)
- **序列长度**: 10 帧 (可配置)
- **参数量**: 3,648
- **输出**: 32维特征向量

### 输入数据格式

```
输入形状: (Batch_size, Sequence_length, Input_size)
         = (num_envs, 10, 4)

每帧数据: [相对x, 相对y, 相对z, 可见性标签v]
  - 相对坐标: 目标相对于追踪器的位置
  - 可见性标签 v:
    * v = 1.0: 目标可见
    * v = 0.0: 目标被遮挡
```

## 鲁棒性处理

### 目标遮挡处理

当目标被障碍物遮挡时（v=0），系统会自动采取以下策略：

1. **位置填充**: 使用上一帧的可见位置填充当前帧的相对坐标
2. **可见性标记**: 保持 v=0 标签，让GRU学习识别遮挡情况
3. **状态保持**: 保存最后一次可见的位置，用于后续遮挡帧

这种设计使得模型能够：
- 区分目标真实运动和遮挡情况
- 学习预测遮挡后的目标位置
- 提供更鲁棒的追踪能力

## 集成说明

### 1. 观测空间维度变化

**重要**: 观测空间增加了32维轨迹特征

```python
# 修改前
num_obs = original_dimensions

# 修改后
num_obs = original_dimensions + 32  # 增加轨迹编码特征
```

### 2. 配置文件更新

在训练/评估配置中更新 `obs_cfg`:

```python
obs_cfg = {
    "num_obs": 3 + 4 + 3 + 3 + 3 + 3 + 21 + 32,  # 原始 + 轨迹特征
    "obs_scales": {
        "max_diff": 1.0,
        "max_lin": 1.0,
        "max_ang": 1.0,
    }
}
```

### 3. 新增的环境属性

```python
class TrackerEnv:
    # 轨迹编码器
    self.trajectory_encoder: TrajectoryEncoder  # GRU编码器模块
    
    # 轨迹历史buffer
    self.target_trajectory_history: Tensor  # (num_envs, 10, 4)
    
    # 可见性追踪
    self.target_visibility: Tensor  # (num_envs,)
    self.last_visible_rel_pos: Tensor  # (num_envs, 3)
```

## 使用示例

### 基本使用

```python
from tracking.track_env import TrackerEnv

# 配置环境
env_cfg = {...}
obs_cfg = {
    "num_obs": original_num_obs + 32,  # ⚠️ 增加32维
    "obs_scales": {...}
}
reward_cfg = {...}

# 创建环境
env = TrackerEnv(
    num_envs=4096,
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg
)

# 训练循环
obs, extras = env.reset()
for step in range(max_steps):
    # 策略网络会自动接收包含轨迹特征的观测
    action = policy(obs)
    obs, reward, done, extras = env.step(action)
```

### 观测空间结构

```python
obs_buf = torch.cat([
    rel_pos,              # (3,)  相对位置
    tracker_quat,         # (4,)  追踪器四元数
    tracker_lin_vel,      # (3,)  追踪器线速度
    tracker_ang_vel,      # (3,)  追踪器角速度
    target_lin_vel,       # (3,)  目标线速度
    last_actions,         # (3,)  上一步动作
    obs_env,              # (21,) 障碍物特征
    trajectory_features,  # (32,) 🆕 目标轨迹编码特征
], dim=-1)
```

## 工作原理

### 每一步的处理流程

```
1. 检测可见性
   └─> occlusion_check() -> 判断目标是否被遮挡

2. 更新轨迹历史
   ├─> 如果可见 (v=1): 记录当前相对位置
   └─> 如果遮挡 (v=0): 使用上一帧可见位置

3. FIFO队列更新
   └─> 删除最旧帧，添加最新帧

4. GRU编码
   └─> trajectory_history -> GRU -> 32维特征向量

5. 拼接到观测空间
   └─> obs_buf = [..., trajectory_features]
```

### 遮挡情况示例

```python
# 时间序列 (10帧历史)
Frame:  0    1    2    3    4    5    6    7    8    9
Vis:   [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
       └────────────┘   └──遮挡──┘   └──────────────────┘
         可见区间        使用上一帧      恢复可见

RelPos:
  x:   [1.2, 1.3, 1.4, 1.4, 1.4, 1.4, 1.8, 1.9, 2.0, 2.1]
                          ↑────────↑
                        保持frame3的位置
```

## 测试验证

### 运行测试脚本

```bash
cd /home/fast/Documents/Genesis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate genesis
python tracking/test_trajectory_encoder.py
```

### 预期输出

```
============================================================
测试 TrajectoryEncoder 模块
============================================================

配置:
  - 批次大小: 4
  - 序列长度: 10
  - 输入维度: 4 (相对坐标x,y,z + 可见性标签v)
  - 隐藏层维度: 32
  - 设备: cuda

编码器参数量: 3648

场景1: 目标始终可见
  输出特征形状: torch.Size([4, 32])
  特征范围: [-0.64, 0.68]

场景2: 目标在第4-6帧被遮挡
  输出特征形状: torch.Size([4, 32])
  特征范围: [-0.62, 0.67]

两个场景的特征差异 (L2范数): 0.1022
  -> 编码器能够区分遮挡和非遮挡情况

✓ 测试完成!
```

## 性能考虑

### 计算开销

- **GRU前向传播**: ~0.01ms per environment (GPU)
- **参数量**: 3,648 (非常轻量)
- **内存占用**: 
  - 历史buffer: `num_envs × 10 × 4 × 4字节`
  - 示例: 4096环境 ≈ 640KB

### 优化建议

1. **训练初期**: GRU使用随机初始化权重
2. **微调**: 可以先冻结GRU，训练主策略网络
3. **端到端**: 解冻GRU，端到端微调整个系统

## 常见问题

### Q1: 为什么选择10帧历史？

**A**: 10帧 × 0.01s = 0.1s 的历史窗口，足以捕捉目标的短期运动模式，同时保持轻量化。可根据需要调整 `trajectory_history_length`。

### Q2: 如何处理环境重置？

**A**: `reset_idx()` 会自动清零历史buffer：
```python
self.target_trajectory_history[envs_idx] = 0.0
self.target_visibility[envs_idx] = 1.0
```

### Q3: GRU权重如何保存/加载？

**A**: `trajectory_encoder` 是 `nn.Module`，可以单独保存：
```python
# 保存
torch.save(env.trajectory_encoder.state_dict(), 'trajectory_encoder.pth')

# 加载
env.trajectory_encoder.load_state_dict(torch.load('trajectory_encoder.pth'))
```

### Q4: 如何可视化轨迹特征？

**A**: 可以使用t-SNE/PCA降维可视化：
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 收集特征
features = []
for _ in range(100):
    obs, _, _, _ = env.step(action)
    traj_feat = obs[:, -32:]  # 提取最后32维
    features.append(traj_feat.cpu())

# 降维可视化
features = torch.cat(features, dim=0).numpy()
tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(features)
plt.scatter(embedded[:, 0], embedded[:, 1])
plt.show()
```

## 未来改进方向

1. **注意力机制**: 在GRU基础上增加注意力层，关注关键历史帧
2. **多目标追踪**: 扩展到同时编码多个目标的轨迹
3. **预测模块**: 增加轨迹预测分支，预测未来N帧位置
4. **可变序列长度**: 根据环境复杂度动态调整历史窗口

## 文件清单

```
tracking/
├── track_env.py                        # 主环境文件 (已修改)
├── test_trajectory_encoder.py          # 测试脚本 (新增)
└── TRAJECTORY_ENCODER_USAGE.md         # 本文档 (新增)
```

## 版本信息

- **实现日期**: 2026-01-08
- **Genesis版本**: 最新版本
- **PyTorch版本**: ≥1.9.0
- **CUDA支持**: 是

---

**作者备注**: 这个轨迹编码器设计遵循《代码整洁之道》原则，保持简洁、高效、可维护。
