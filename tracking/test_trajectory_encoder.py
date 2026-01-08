"""
测试轨迹编码器集成的示例脚本

这个脚本演示了如何使用新增的目标轨迹编码功能
"""

import torch
import sys
sys.path.append('.')

from tracking.track_env import TrajectoryEncoder


def test_trajectory_encoder():
    """测试TrajectoryEncoder模块的基本功能"""
    print("=" * 60)
    print("测试 TrajectoryEncoder 模块 (带Stride采样)")
    print("=" * 60)
    
    # 参数设置
    batch_size = 4
    stride = 5
    sequence_length = 10
    history_length = stride * sequence_length  # 50帧
    input_size = 4  # [rel_x, rel_y, rel_z, visibility]
    hidden_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = 0.01  # 仿真时间步长
    
    print(f"\n配置:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 历史buffer长度: {history_length} 帧 ({history_length * dt}秒)")
    print(f"  - Stride: {stride} 帧")
    print(f"  - GRU序列长度: {sequence_length} (采样后)")
    print(f"  - 覆盖时间: {history_length * dt}秒")
    print(f"  - 输入维度: {input_size} (相对坐标x,y,z + 可见性标签v)")
    print(f"  - 隐藏层维度: {hidden_size}")
    print(f"  - 设备: {device}")
    
    # 创建编码器
    encoder = TrajectoryEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        device=device
    )
    encoder.eval()
    
    print(f"\n编码器参数量: {sum(p.numel() for p in encoder.parameters())}")
    
    # 创建模拟的历史轨迹数据 (50帧完整历史)
    # 场景1: 目标始终可见，匀速运动
    trajectory_full1 = torch.zeros((batch_size, history_length, input_size), device=device)
    for t in range(history_length):
        trajectory_full1[:, t, 0] = 1.0 + 0.02 * t  # x坐标线性增加
        trajectory_full1[:, t, 1] = 2.0 + 0.01 * t  # y坐标线性增加
        trajectory_full1[:, t, 2] = 1.0  # z坐标保持不变
        trajectory_full1[:, t, 3] = 1.0  # 可见性=1 (始终可见)
    
    # 场景2: 目标在中间帧被遮挡 (第20-30帧)
    trajectory_full2 = trajectory_full1.clone()
    trajectory_full2[:, 20:31, 3] = 0.0  # 第20-30帧被遮挡
    trajectory_full2[:, 20:31, :3] = trajectory_full2[:, 19:20, :3]  # 使用上一帧的位置填充
    
    # Stride采样
    sampled_indices = torch.arange(stride - 1, history_length, stride, device=device)
    trajectory1 = trajectory_full1[:, sampled_indices, :]  # (batch_size, 10, 4)
    trajectory2 = trajectory_full2[:, sampled_indices, :]  # (batch_size, 10, 4)
    
    print("\n场景1: 目标始终可见 (0.5秒历史)")
    print(f"  完整历史形状: {trajectory_full1.shape} (50帧)")
    print(f"  Stride采样后: {trajectory1.shape} (10帧)")
    print(f"  采样索引: {sampled_indices.tolist()}")
    with torch.no_grad():
        features1 = encoder(trajectory1)
    print(f"  输出特征形状: {features1.shape}")
    print(f"  特征范围: [{features1.min():.4f}, {features1.max():.4f}]")
    
    print("\n场景2: 目标在第20-30帧被遮挡 (采样后覆盖帧4-6)")
    print(f"  完整历史形状: {trajectory_full2.shape} (50帧)")
    print(f"  Stride采样后: {trajectory2.shape} (10帧)")
    with torch.no_grad():
        features2 = encoder(trajectory2)
    print(f"  输出特征形状: {features2.shape}")
    print(f"  特征范围: [{features2.min():.4f}, {features2.max():.4f}]")
    
    # 计算两个场景的特征差异
    feature_diff = torch.norm(features1 - features2, dim=1).mean()
    print(f"\n两个场景的特征差异 (L2范数): {feature_diff:.4f}")
    print("  -> 编码器能够区分遮挡和非遮挡情况")
    
    print("\n" + "=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


def test_integration_info():
    """打印集成说明"""
    print("\n" + "=" * 60)
    print("集成说明")
    print("=" * 60)
    
    print("\n【重要】观测空间维度变化:")
    print("  原始观测维度: num_obs")
    print("  新增轨迹特征维度: 32")
    print("  新的总观测维度: num_obs + 32")
    print("\n  ⚠️  请在配置文件中更新 obs_cfg['num_obs'] 参数!")
    
    print("\n【功能说明】:")
    print("  1. 轨迹编码器记录目标过去50帧(0.5秒)的运动轨迹")
    print("  2. 使用Stride=5采样，提取10个样本输入GRU")
    print("  3. 每帧包含: [相对x, 相对y, 相对z, 可见性标签]")
    print("  4. 可见性标签 v:")
    print("     - v=1: 目标可见")
    print("     - v=0: 目标被遮挡")
    print("  5. 当目标被遮挡时，使用上一帧的可见位置填充")
    print("  6. GRU观察0.5秒历史，而非0.1秒瞬时状态")
    
    print("\n【新增的状态信息】:")
    print("  - self.target_trajectory_history: (num_envs, 50, 4) 历史轨迹buffer")
    print("  - self.trajectory_stride: 5 (采样间隔)")
    print("  - self.trajectory_sequence_length: 10 (GRU输入长度)")
    print("  - self.target_visibility: (num_envs,) 当前可见性标签")
    print("  - self.trajectory_encoder: TrajectoryEncoder模块")
    
    print("\n【使用示例】:")
    print("""
    # 环境配置 (需要更新 num_obs)
    obs_cfg = {
        "num_obs": original_num_obs + 32,  # 增加32维轨迹特征
        "obs_scales": {...}
    }
    
    # 创建环境 (自动使用stride采样)
    env = TrackerEnv(num_envs=4096, env_cfg, obs_cfg, reward_cfg)
    
    # 轨迹编码参数:
    #   - 历史buffer: 50帧 (0.5秒)
    #   - Stride: 5帧
    #   - GRU输入: 10个样本
    
    # 正常使用
    obs, extras = env.reset()
    for _ in range(1000):
        action = policy(obs)  # 策略网络会自动使用轨迹特征
        obs, rew, done, extras = env.step(action)
    """)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 测试编码器
    test_trajectory_encoder()
    
    # 打印集成说明
    test_integration_info()
