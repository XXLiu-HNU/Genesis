"""
速度限制测试：验证速度不会无限增长
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs
from target_planning.parallel_nav_env import ParallelTargetNavEnv


def test_speed_limit():
    """测试速度是否被正确限制"""
    print("="*70)
    print(" 速度限制测试")
    print("="*70)
    
    gs.init(backend=gs.gpu)
    
    # 创建环境
    env = ParallelTargetNavEnv(num_envs=100, show_viewer=False)
    
    max_speed_recorded = 0.0
    speed_history = []
    
    print(f"运行5000步，监控速度...")
    print(f"配置的最大速度: {env.v_max} m/s")
    print()
    
    for step in range(5000):
        env.step()
        
        if (step + 1) % 100 == 0:
            # 获取速度
            states = env.get_target_states()
            vel = states['vel'][:, :2]  # [100, 2]
            speed = torch.norm(vel, dim=-1)  # [100]
            
            avg_speed = speed.mean().item()
            max_speed = speed.max().item()
            
            speed_history.append(avg_speed)
            max_speed_recorded = max(max_speed_recorded, max_speed)
            
            print(f"Step {step+1:4d}: "
                  f"平均速度 = {avg_speed:.2f} m/s, "
                  f"最大速度 = {max_speed:.2f} m/s")
            
            # 检查是否超过限制
            if max_speed > env.v_max * 1.1:  # 允许10%误差
                print(f"  ⚠️  警告：速度超过限制！ {max_speed:.2f} > {env.v_max}")
    
    print()
    print("="*70)
    print("测试结果:")
    print(f"  配置的最大速度: {env.v_max} m/s")
    print(f"  记录的最大速度: {max_speed_recorded:.2f} m/s")
    print(f"  最后500步平均速度: {sum(speed_history[-5:])/5:.2f} m/s")
    
    # 检查速度是否稳定
    recent_speeds = speed_history[-10:]
    speed_variation = max(recent_speeds) - min(recent_speeds)
    
    print(f"  最后1000步速度变化: {speed_variation:.2f} m/s")
    
    if max_speed_recorded <= env.v_max * 1.1:
        print(f"  ✓ 优秀！速度被正确限制在 {env.v_max} m/s 以内")
    else:
        print(f"  ✗ 失败：速度超过限制")
    
    if speed_variation < 0.5:
        print(f"  ✓ 优秀！速度已稳定")
    else:
        print(f"  ⚠️  速度仍在变化，可能未完全稳定")
    
    print("="*70)


if __name__ == "__main__":
    test_speed_limit()
