"""
示例：使用训练好的策略在独立动力学仿真器中进行sim2sim测试

这个脚本展示如何：
1. 加载训练好的RL策略
2. 使用独立的无人机动力学仿真器
3. 运行策略并评估性能
"""

import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics, BatchedQuadrotorDynamics


class PolicyWrapper:
    """策略包装器，用于加载和运行训练好的策略"""
    
    def __init__(self, policy_path, device='cuda'):
        """
        加载策略
        
        Parameters:
        -----------
        policy_path : str
            策略模型文件路径（.pt文件）
        device : str
            运行设备
        """
        self.device = device
        
        # 加载模型
        checkpoint = torch.load(policy_path, map_location=device)
        
        # 提取actor网络
        if 'model_state_dict' in checkpoint:
            self.actor_state_dict = checkpoint['model_state_dict']
        else:
            self.actor_state_dict = checkpoint
        
        # 提取normalization参数（如果有）
        self.obs_mean = None
        self.obs_std = None
        if 'obs_mean' in checkpoint:
            self.obs_mean = checkpoint['obs_mean']
            self.obs_std = checkpoint['obs_std']
        
        print(f"✓ 策略已加载: {policy_path}")
        
        # 需要根据实际的网络结构来构建actor
        # 这里需要知道obs_dim和action_dim
        self.actor = None
        self._build_actor_from_checkpoint()
    
    def _build_actor_from_checkpoint(self):
        """从checkpoint重建actor网络"""
        # 这里需要根据训练时的配置重建网络
        # 简化版本：假设使用标准的MLP结构
        try:
            # 尝试从state_dict推断网络结构
            from rsl_rl.modules import ActorCritic
            
            # 默认配置（与track_train.py中的配置一致）
            obs_dim = 33
            action_dim = 4
            
            # 构建ActorCritic
            actor_critic_cfg = {
                'class_name': 'ActorCritic',
                'init_noise_std': 1.0,
                'actor_hidden_dims': [128, 128],
                'critic_hidden_dims': [128, 128],
                'activation': 'tanh',
            }
            
            self.actor = ActorCritic(
                num_actor_obs=obs_dim,
                num_critic_obs=obs_dim,
                num_actions=action_dim,
                **actor_critic_cfg
            ).to(self.device)
            
            # 加载权重
            self.actor.load_state_dict(self.actor_state_dict)
            self.actor.eval()
            
            print(f"✓ Actor网络已构建: obs_dim={obs_dim}, action_dim={action_dim}")
            
        except Exception as e:
            print(f"警告: 无法自动构建actor网络: {e}")
            print("请手动设置actor网络")
    
    def predict(self, obs, deterministic=True):
        """
        预测动作
        
        Parameters:
        -----------
        obs : np.ndarray
            观测向量
        deterministic : bool
            是否使用确定性策略
        
        Returns:
        --------
        action : np.ndarray
            动作
        """
        if self.actor is None:
            raise RuntimeError("Actor网络未初始化")
        
        # 转换为tensor
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        
        # 归一化
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        
        # 预测
        with torch.no_grad():
            if deterministic:
                action = self.actor.act_inference(obs)
            else:
                action = self.actor.act(obs)
        
        # 转换回numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        return action


def rpm_to_thrust(rpm, kf):
    """将RPM转换为推力"""
    omega = rpm * 2 * np.pi / 60  # rad/s
    return kf * omega**2


def thrust_to_rpm(thrust, kf):
    """将推力转换为RPM"""
    omega = np.sqrt(thrust / kf)
    return omega * 60 / (2 * np.pi)


def action_to_rpm(action, base_rpm, max_rpm):
    """
    将策略输出的动作转换为RPM
    
    假设动作范围为[-1, 1]，映射到[0, max_rpm]
    
    Parameters:
    -----------
    action : np.ndarray (4,)
        策略输出的动作
    base_rpm : float
        基础RPM（悬停RPM）
    max_rpm : float
        最大RPM
    
    Returns:
    --------
    rpm : np.ndarray (4,)
        4个螺旋桨的RPM
    """
    # 将[-1, 1]映射到[0, max_rpm]
    rpm = base_rpm + action * (max_rpm - base_rpm) / 2
    rpm = np.clip(rpm, 0, max_rpm)
    return rpm


def compute_target_observation(drone_state, target_position, target_velocity):
    """
    计算相对于目标的观测
    
    Parameters:
    -----------
    drone_state : dict
        无人机状态
    target_position : np.ndarray (3,)
        目标位置
    target_velocity : np.ndarray (3,)
        目标速度
    
    Returns:
    --------
    obs : np.ndarray
        观测向量（与训练环境一致）
    """
    # 位置差
    pos_diff = target_position - drone_state['position']
    
    # 速度差
    vel_diff = target_velocity - drone_state['velocity']
    
    # 构造观测
    obs = np.concatenate([
        pos_diff / 5.0,  # 位置差归一化
        vel_diff / 3.0,  # 速度差归一化
        drone_state['velocity'] / 3.0,  # 无人机速度归一化
        drone_state['quaternion'],  # 姿态四元数
        drone_state['angular_velocity'] / 3.14159,  # 角速度归一化
        target_velocity / 3.0,  # 目标速度归一化
        target_position / 5.0,  # 目标位置归一化（相对某个参考点）
    ])
    
    return obs


def run_single_episode(policy, target_trajectory, max_steps=1500, visualize=False):
    """
    运行单个episode
    
    Parameters:
    -----------
    policy : PolicyWrapper
        策略
    target_trajectory : callable
        目标轨迹函数 target_trajectory(t) -> (position, velocity)
    max_steps : int
        最大步数
    visualize : bool
        是否可视化
    
    Returns:
    --------
    metrics : dict
        性能指标
    """
    # 创建仿真器
    config = {
        'dt': 0.01,  # 10ms
        'mass': 0.5,
        'arm_length': 0.12,
        'kf': 3.16e-10,
        'km': 7.94e-12,
    }
    sim = QuadrotorDynamics(config)
    
    # 重置
    initial_pos = target_trajectory(0)[0]
    sim.reset(position=initial_pos)
    
    # 计算悬停RPM
    hover_thrust = sim.mass * sim.gravity / 4.0
    hover_rpm = thrust_to_rpm(hover_thrust, sim.kf)
    max_rpm = hover_rpm * 2.0
    
    print(f"悬停RPM: {hover_rpm:.0f}, 最大RPM: {max_rpm:.0f}")
    
    # 记录数据
    drone_positions = []
    target_positions = []
    tracking_errors = []
    actions_history = []
    
    # 运行episode
    for step in range(max_steps):
        t = step * sim.dt
        
        # 获取目标
        target_pos, target_vel = target_trajectory(t)
        
        # 获取无人机状态
        drone_state = sim.get_state()
        
        # 计算观测
        obs = compute_target_observation(drone_state, target_pos, target_vel)
        
        # 预测动作
        action = policy.predict(obs, deterministic=True)
        if action.ndim > 1:
            action = action[0]
        
        # 转换为RPM并执行
        rpms = action_to_rpm(action, hover_rpm, max_rpm)
        next_state = sim.step_rpm(rpms)
        
        # 记录
        drone_positions.append(drone_state['position'].copy())
        target_positions.append(target_pos.copy())
        tracking_error = np.linalg.norm(drone_state['position'] - target_pos)
        tracking_errors.append(tracking_error)
        actions_history.append(action.copy())
        
        # 打印进度
        if step % 100 == 0:
            print(f"步数: {step:4d}, 跟踪误差: {tracking_error:.4f} m")
    
    # 转换为数组
    drone_positions = np.array(drone_positions)
    target_positions = np.array(target_positions)
    tracking_errors = np.array(tracking_errors)
    actions_history = np.array(actions_history)
    
    # 计算指标
    metrics = {
        'mean_tracking_error': np.mean(tracking_errors),
        'max_tracking_error': np.max(tracking_errors),
        'final_tracking_error': tracking_errors[-1],
        'rmse': np.sqrt(np.mean(tracking_errors**2)),
    }
    
    print("\n性能指标:")
    print(f"  平均跟踪误差: {metrics['mean_tracking_error']:.4f} m")
    print(f"  最大跟踪误差: {metrics['max_tracking_error']:.4f} m")
    print(f"  最终跟踪误差: {metrics['final_tracking_error']:.4f} m")
    print(f"  RMSE: {metrics['rmse']:.4f} m")
    
    # 可视化
    if visualize:
        visualize_episode(drone_positions, target_positions, tracking_errors, 
                         actions_history, sim.dt)
    
    return metrics


def visualize_episode(drone_positions, target_positions, tracking_errors, 
                     actions, dt):
    """可视化episode结果"""
    time = np.arange(len(drone_positions)) * dt
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
            'g--', label='Target', linewidth=2, alpha=0.7)
    ax1.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2],
            'b-', label='Drone', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # XY平面轨迹
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(target_positions[:, 0], target_positions[:, 1], 'g--', 
            label='Target', linewidth=2, alpha=0.7)
    ax2.plot(drone_positions[:, 0], drone_positions[:, 1], 'b-',
            label='Drone', linewidth=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 跟踪误差
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time, tracking_errors, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tracking Error (m)')
    ax3.set_title('Tracking Error')
    ax3.grid(True, alpha=0.3)
    
    # 位置对比
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time, target_positions[:, 2], 'g--', label='Target Altitude', linewidth=2)
    ax4.plot(time, drone_positions[:, 2], 'b-', label='Drone Altitude', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Altitude Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 动作
    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(4):
        ax5.plot(time, actions[:, i], label=f'Action {i}', linewidth=1.5, alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Action Value')
    ax5.set_title('Control Actions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 误差统计
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(tracking_errors, bins=50, alpha=0.7, edgecolor='black')
    ax6.axvline(np.mean(tracking_errors), color='r', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(tracking_errors):.4f}')
    ax6.set_xlabel('Tracking Error (m)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Error Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sim2sim_results.png', dpi=150, bbox_inches='tight')
    print("✓ 结果图已保存到 sim2sim_results.png")


# 预定义的目标轨迹
def hovering_trajectory(t):
    """悬停轨迹"""
    target_pos = np.array([0.0, 0.0, 1.0])
    target_vel = np.array([0.0, 0.0, 0.0])
    return target_pos, target_vel


def circular_trajectory(t, radius=2.0, height=1.5, period=10.0):
    """圆形轨迹"""
    omega = 2 * np.pi / period
    target_pos = np.array([
        radius * np.cos(omega * t),
        radius * np.sin(omega * t),
        height
    ])
    target_vel = np.array([
        -radius * omega * np.sin(omega * t),
        radius * omega * np.cos(omega * t),
        0.0
    ])
    return target_pos, target_vel


def figure8_trajectory(t, scale=2.0, height=1.5, period=15.0):
    """8字形轨迹"""
    omega = 2 * np.pi / period
    target_pos = np.array([
        scale * np.sin(omega * t),
        scale * np.sin(omega * t) * np.cos(omega * t),
        height
    ])
    target_vel = np.array([
        scale * omega * np.cos(omega * t),
        scale * omega * (np.cos(2 * omega * t)),
        0.0
    ])
    return target_pos, target_vel


def main():
    parser = argparse.ArgumentParser(description='Sim2Sim测试：在独立仿真器中评估策略')
    parser.add_argument('--policy', type=str, required=True,
                       help='策略文件路径，例如: logs/drone-hovering/20250107-120000/model_300.pt')
    parser.add_argument('--trajectory', type=str, default='hovering',
                       choices=['hovering', 'circular', 'figure8'],
                       help='目标轨迹类型')
    parser.add_argument('--steps', type=int, default=1500,
                       help='仿真步数')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化图表')
    parser.add_argument('--device', type=str, default='cuda',
                       help='运行设备')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" " * 20 + "Sim2Sim 策略测试")
    print("=" * 70)
    print(f"策略文件: {args.policy}")
    print(f"轨迹类型: {args.trajectory}")
    print(f"仿真步数: {args.steps}")
    print(f"设备: {args.device}")
    print("=" * 70)
    print()
    
    # 加载策略
    try:
        policy = PolicyWrapper(args.policy, device=args.device)
    except Exception as e:
        print(f"错误: 无法加载策略: {e}")
        print("\n注意: 请确保:")
        print("1. 策略文件路径正确")
        print("2. 已安装rsl-rl-lib==2.2.4")
        print("3. 策略文件格式正确")
        return
    
    # 选择轨迹
    if args.trajectory == 'hovering':
        trajectory_fn = hovering_trajectory
    elif args.trajectory == 'circular':
        trajectory_fn = circular_trajectory
    elif args.trajectory == 'figure8':
        trajectory_fn = figure8_trajectory
    else:
        raise ValueError(f"未知轨迹类型: {args.trajectory}")
    
    # 运行测试
    print("\n开始sim2sim测试...\n")
    metrics = run_single_episode(
        policy,
        trajectory_fn,
        max_steps=args.steps,
        visualize=args.visualize
    )
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    # 如果直接运行（不带参数），显示使用示例
    import sys
    if len(sys.argv) == 1:
        print("=" * 70)
        print("使用示例:")
        print("=" * 70)
        print("\n基本用法:")
        print("  python example_policy_sim2sim.py --policy logs/drone-hovering/20250107-120000/model_300.pt")
        print("\n指定轨迹:")
        print("  python example_policy_sim2sim.py --policy <path> --trajectory circular")
        print("\n生成可视化:")
        print("  python example_policy_sim2sim.py --policy <path> --visualize")
        print("\n完整示例:")
        print("  python example_policy_sim2sim.py \\")
        print("    --policy logs/drone-hovering/20250107-120000/model_300.pt \\")
        print("    --trajectory figure8 \\")
        print("    --steps 2000 \\")
        print("    --visualize \\")
        print("    --device cuda")
        print()
    else:
        main()
