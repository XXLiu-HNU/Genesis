"""
快速入门示例：展示如何使用并行目标导航环境
"""
import os
import sys
import genesis as gs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from target_planning.parallel_nav_env import ParallelTargetNavEnv


def example_1_basic():
    """示例1：基本使用 - 100个环境"""
    print("\n" + "="*70)
    print("示例1：基本使用 - 100个并行环境")
    print("="*70)
    
    gs.init(backend=gs.gpu)
    env = ParallelTargetNavEnv(num_envs=100, show_viewer=True)
    
    print("运行1500步...")
    for i in range(15000):
        env.step()
        
        if (i + 1) % 100 == 0:
            # 获取目标状态
            states = env.get_target_states()
            pos = states['pos']
            vel = states['vel']
            
            # 计算平均速度
            speed = torch.norm(vel[:, :2], dim=-1).mean().item()
            print(f"  Step {i+1}: 平均速度 = {speed:.2f} m/s")
    
    print("✓ 完成")


def example_2_with_tracker():
    """示例2：与追踪无人机集成"""
    print("\n" + "="*70)
    print("示例2：目标无人机 + 追踪无人机")
    print("="*70)
    
    gs.init(backend=gs.gpu)
    
    # 创建目标环境
    target_env = ParallelTargetNavEnv(num_envs=100, show_viewer=True)
    
    print("模拟追踪场景...")
    for i in range(500):
        # 目标无人机移动
        target_env.step()
        
        # 获取目标状态供追踪器使用
        target_states = target_env.get_target_states()
        
        # 这里可以添加追踪无人机的逻辑
        # tracker_action = policy(obs, target_states['pos'])
        # tracker.step(tracker_action)
        
        if (i + 1) % 100 == 0:
            pos = target_states['pos']
            print(f"  Step {i+1}: 目标位置范围 "
                  f"X:[{pos[:, 0].min():.1f}, {pos[:, 0].max():.1f}] "
                  f"Y:[{pos[:, 1].min():.1f}, {pos[:, 1].max():.1f}]")
    
    print("✓ 完成")


def example_3_performance():
    """示例3：性能测试"""
    import time
    import torch
    
    print("\n" + "="*70)
    print("示例3：性能测试 - 1000个环境")
    print("="*70)
    
    gs.init(backend=gs.gpu)
    env = ParallelTargetNavEnv(num_envs=20000, show_viewer=False)
    
    # 预热
    print("预热中...")
    for _ in range(20):
        env.step()
    
    # 测试
    print("性能测试中...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    n_steps = 15000
    for _ in range(n_steps):
        env.step()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    fps = n_steps / elapsed
    throughput = fps * 1000
    
    print(f"  仿真步数: {n_steps}")
    print(f"  总时间: {elapsed:.2f}秒")
    print(f"  平均FPS: {fps:.2f}")
    print(f"  吞吐量: {throughput:.0f} env-steps/sec")
    print("✓ 完成")


def example_4_large_scale():
    """示例4：大规模仿真"""
    import time
    import torch
    
    print("\n" + "="*70)
    print("示例4：大规模仿真 - 5000个环境")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("警告：未检测到GPU，大规模仿真可能很慢")
    
    gs.init(backend=gs.gpu)
    
    print("创建环境...")
    start = time.time()
    env = ParallelTargetNavEnv(num_envs=5000, show_viewer=False)
    init_time = time.time() - start
    print(f"  初始化耗时: {init_time:.2f}秒")
    
    # 预热
    print("预热中...")
    for _ in range(20):
        env.step()
    
    # 测试
    print("运行仿真...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    n_steps = 100
    for i in range(n_steps):
        env.step()
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            fps = (i + 1) / elapsed
            print(f"  进度: {i+1}/{n_steps}, FPS: {fps:.2f}")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.time() - start
    
    fps = n_steps / total_time
    throughput = fps * 5000
    
    print(f"\n结果:")
    print(f"  环境数: 5000")
    print(f"  仿真步数: {n_steps}")
    print(f"  总时间: {total_time:.2f}秒")
    print(f"  平均FPS: {fps:.2f}")
    print(f"  吞吐量: {throughput:.0f} env-steps/sec")
    
    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  GPU显存: {mem_gb:.2f} GB")
        print(f"  每环境显存: {mem_gb*1024/5000:.2f} MB")
    
    print("✓ 完成")


if __name__ == "__main__":
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="并行目标导航环境示例")
    parser.add_argument('--example', type=int, default=1, choices=[1, 2, 3, 4],
                        help='选择示例: 1=基本使用, 2=追踪集成, 3=性能测试, 4=大规模仿真')
    args = parser.parse_args()
    
    print("="*70)
    print(" 并行目标导航环境 - 快速入门示例")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("GPU: 未检测到（将使用CPU）")
    
    try:
        if args.example == 1:
            example_1_basic()
        elif args.example == 2:
            example_2_with_tracker()
        elif args.example == 3:
            example_3_performance()
        elif args.example == 4:
            example_4_large_scale()
    except KeyboardInterrupt:
        print("\n\n✓ 用户中断")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
