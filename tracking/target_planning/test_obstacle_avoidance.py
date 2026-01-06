"""
障碍物避让测试：验证目标无人机是否能避开障碍物
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs
from target_planning.parallel_nav_env import ParallelTargetNavEnv


def test_obstacle_avoidance_visual():
    """可视化测试：观察无人机是否避开障碍物"""
    print("="*70)
    print(" 障碍物避让可视化测试")
    print("="*70)
    print("观察红色目标球，看无人机是否绕过障碍物而不是直接穿过")
    print("按 Ctrl+C 停止")
    print("="*70)
    
    gs.init(backend=gs.gpu)
    
    # 创建环境（100个环境，显示前10个）
    env = ParallelTargetNavEnv(num_envs=100, show_viewer=True)
    
    collision_count = 0
    step_count = 0
    
    try:
        while True:
            env.step()
            step_count += 1
            
            if step_count % 50 == 0:
                # 检查碰撞
                states = env.get_target_states()
                pos_xy = states['pos'][:, :2]  # [100, 2]
                
                # 计算到障碍物的最小距离
                pos_expanded = pos_xy.unsqueeze(1)  # [100, 1, 2]
                obs_expanded = env.obs_xy.unsqueeze(0)  # [1, M, 2]
                dist = torch.norm(pos_expanded - obs_expanded, dim=-1)  # [100, M]
                
                # 到障碍物边界的距离
                obs_r_expanded = env.obs_r.unsqueeze(0)  # [1, M]
                clearance = dist - obs_r_expanded  # [100, M]
                min_clearance = clearance.min(dim=-1).values  # [100]
                
                # 统计碰撞（净空<0.1m算碰撞）
                collisions = (min_clearance < 0.1).sum().item()
                collision_count += collisions
                
                avg_clearance = min_clearance.mean().item()
                min_clearance_val = min_clearance.min().item()
                
                print(f"Step {step_count}: "
                      f"Avg clearance: {avg_clearance:.3f}m, "
                      f"Min clearance: {min_clearance_val:.3f}m, "
                      f"Collisions: {collisions}/100, "
                      f"Total collisions: {collision_count}")
                
                if min_clearance_val < 0:
                    print(f"  ⚠️  WARNING: Collision detected! Clearance = {min_clearance_val:.3f}m")
    
    except KeyboardInterrupt:
        print(f"\n{'='*70}")
        print(f"测试停止")
        print(f"总步数: {step_count}")
        print(f"总碰撞次数: {collision_count}")
        print(f"碰撞率: {collision_count/(step_count*100)*100:.2f}%")
        print(f"{'='*70}")


def test_obstacle_avoidance_stress():
    """压力测试：密集障碍物环境"""
    print("="*70)
    print(" 障碍物避让压力测试 (密集障碍)")
    print("="*70)
    
    gs.init(backend=gs.gpu)
    
    # 创建更多障碍物的环境
    env = ParallelTargetNavEnv(num_envs=500, show_viewer=False)
    
    # 运行1000步
    n_steps = 1000
    collision_count = 0
    
    print(f"运行 {n_steps} 步...")
    for step in range(n_steps):
        env.step()
        
        if (step + 1) % 100 == 0:
            # 检查碰撞
            states = env.get_target_states()
            pos_xy = states['pos'][:, :2]
            
            pos_expanded = pos_xy.unsqueeze(1)
            obs_expanded = env.obs_xy.unsqueeze(0)
            dist = torch.norm(pos_expanded - obs_expanded, dim=-1)
            clearance = dist - env.obs_r.unsqueeze(0)
            min_clearance = clearance.min(dim=-1).values
            
            collisions = (min_clearance < 0.05).sum().item()
            collision_count += collisions
            
            avg_clearance = min_clearance.mean().item()
            print(f"  Step {step+1}/{n_steps}: "
                  f"Avg clearance: {avg_clearance:.3f}m, "
                  f"Collisions this check: {collisions}/500")
    
    collision_rate = collision_count / (n_steps * 500) * 100
    
    print(f"\n{'='*70}")
    print("测试结果:")
    print(f"  总步数: {n_steps}")
    print(f"  环境数: 500")
    print(f"  总碰撞次数: {collision_count}")
    print(f"  碰撞率: {collision_rate:.3f}%")
    
    if collision_rate < 1.0:
        print(f"  ✓ 优秀！碰撞率低于1%")
    elif collision_rate < 5.0:
        print(f"  ✓ 良好！碰撞率低于5%")
    elif collision_rate < 10.0:
        print(f"  ⚠️  尚可，碰撞率在5-10%之间")
    else:
        print(f"  ✗ 需要调整避障参数，碰撞率超过10%")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="障碍物避让测试")
    parser.add_argument('--mode', type=str, default='visual', 
                        choices=['visual', 'stress'],
                        help='测试模式: visual=可视化观察, stress=压力测试')
    args = parser.parse_args()
    
    if args.mode == 'visual':
        test_obstacle_avoidance_visual()
    elif args.mode == 'stress':
        test_obstacle_avoidance_stress()
