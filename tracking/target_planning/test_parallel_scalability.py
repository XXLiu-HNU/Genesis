"""
大规模并行性能测试脚本
Test parallel navigation environment scalability
"""
import os
import sys
import time
import torch
import genesis as gs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from target_planning.parallel_nav_env import ParallelTargetNavEnv


def benchmark_env(n_envs: int, n_steps: int = 200, warmup_steps: int = 20):
    """
    对给定规模的环境进行性能测试
    
    Args:
        n_envs: 环境数量
        n_steps: 测试步数
        warmup_steps: 预热步数（不计入统计）
    
    Returns:
        dict: 性能指标
    """
    print(f"\n{'='*70}")
    print(f"Testing {n_envs} parallel environments")
    print(f"{'='*70}")
    
    try:
        # 测量初始化时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        init_start = time.time()
        
        env = ParallelTargetNavEnv(num_envs=n_envs, show_viewer=False)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        init_time = time.time() - init_start
        
        print(f"✓ Environment initialized in {init_time:.2f}s")
        
        # 预热（编译CUDA kernels等）
        print(f"Warming up for {warmup_steps} steps...")
        for _ in range(warmup_steps):
            env.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # 实际测试
        print(f"Running benchmark for {n_steps} steps...")
        step_times = []
        
        for i in range(n_steps):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_start = time.time()
            
            env.step()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            if (i + 1) % 50 == 0:
                avg_time = sum(step_times[-50:]) / 50
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"  Step {i+1}/{n_steps}: {avg_time*1000:.2f}ms/step, {fps:.1f} FPS")
        
        # 计算统计信息
        step_times_tensor = torch.tensor(step_times)
        mean_time = step_times_tensor.mean().item()
        std_time = step_times_tensor.std().item()
        min_time = step_times_tensor.min().item()
        max_time = step_times_tensor.max().item()
        p50_time = step_times_tensor.median().item()
        p95_time = step_times_tensor.quantile(0.95).item()
        p99_time = step_times_tensor.quantile(0.99).item()
        
        fps_mean = 1.0 / mean_time if mean_time > 0 else 0
        throughput = fps_mean * n_envs
        
        # 获取内存使用
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            mem_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
        else:
            mem_allocated = 0
            mem_reserved = 0
        
        results = {
            'n_envs': n_envs,
            'init_time': init_time,
            'mean_step_time': mean_time,
            'std_step_time': std_time,
            'min_step_time': min_time,
            'max_step_time': max_time,
            'p50_step_time': p50_time,
            'p95_step_time': p95_time,
            'p99_step_time': p99_time,
            'fps': fps_mean,
            'throughput': throughput,
            'mem_allocated_gb': mem_allocated,
            'mem_reserved_gb': mem_reserved,
        }
        
        # 打印结果
        print(f"\n{'─'*70}")
        print("RESULTS:")
        print(f"{'─'*70}")
        print(f"  Init time:           {init_time:.2f}s")
        print(f"  Mean step time:      {mean_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
        print(f"  Step time range:     [{min_time*1000:.2f}ms, {max_time*1000:.2f}ms]")
        print(f"  Step time (p50):     {p50_time*1000:.2f}ms")
        print(f"  Step time (p95):     {p95_time*1000:.2f}ms")
        print(f"  Step time (p99):     {p99_time*1000:.2f}ms")
        print(f"  Average FPS:         {fps_mean:.2f}")
        print(f"  Throughput:          {throughput:.0f} env-steps/sec")
        if torch.cuda.is_available():
            print(f"  GPU memory (alloc):  {mem_allocated:.2f} GB")
            print(f"  GPU memory (total):  {mem_reserved:.2f} GB")
            print(f"  Memory per env:      {mem_allocated*1024/n_envs:.2f} MB")
        print(f"{'─'*70}")
        
        # 清理
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return results
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_scalability():
    """测试扩展性：从小到大测试不同规模"""
    
    print("="*70)
    print(" PARALLEL NAVIGATION ENVIRONMENT - SCALABILITY TEST")
    print("="*70)
    
    # 检测GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("GPU: Not available (using CPU)")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Genesis version: {gs.__version__}")
    
    # 测试不同规模
    env_sizes = [100, 500, 1000, 2000, 5000, 10000]
    
    results_list = []
    
    for n_envs in env_sizes:
        result = benchmark_env(n_envs, n_steps=200, warmup_steps=20)
        if result is not None:
            results_list.append(result)
        else:
            print(f"Stopping at {n_envs} environments due to failure")
            break
        
        # 等待一下让系统稳定
        time.sleep(1.0)
    
    # 打印汇总表格
    if results_list:
        print(f"\n{'='*70}")
        print(" SCALABILITY SUMMARY")
        print(f"{'='*70}")
        print(f"{'Envs':>8} | {'Init(s)':>8} | {'Step(ms)':>10} | {'FPS':>8} | {'Throughput':>12} | {'Mem(GB)':>8}")
        print(f"{'-'*70}")
        
        for r in results_list:
            print(f"{r['n_envs']:>8} | "
                  f"{r['init_time']:>8.2f} | "
                  f"{r['mean_step_time']*1000:>10.2f} | "
                  f"{r['fps']:>8.2f} | "
                  f"{r['throughput']:>12.0f} | "
                  f"{r['mem_allocated_gb']:>8.2f}")
        
        print(f"{'='*70}")
        
        # 计算扩展效率
        if len(results_list) >= 2:
            print("\nScalability Analysis:")
            base = results_list[0]
            for r in results_list[1:]:
                scale_factor = r['n_envs'] / base['n_envs']
                throughput_ratio = r['throughput'] / base['throughput']
                efficiency = (throughput_ratio / scale_factor) * 100
                print(f"  {r['n_envs']:>5} envs: {efficiency:.1f}% parallel efficiency "
                      f"({throughput_ratio:.2f}x throughput for {scale_factor:.1f}x envs)")
    
    print("\n✓ Scalability test completed!")


def test_visual_demo():
    """可视化演示：运行一个小规模环境并显示viewer"""
    print("="*70)
    print(" VISUAL DEMO - 100 parallel environments")
    print("="*70)
    print("Press Ctrl+C to stop")
    
    gs.init(backend=gs.gpu)
    env = ParallelTargetNavEnv(num_envs=100, show_viewer=True)
    
    step = 0
    try:
        while True:
            env.step()
            step += 1
            
            if step % 100 == 0:
                states = env.get_target_states()
                pos = states['pos']
                vel = states['vel']
                
                # 计算统计信息
                avg_speed = torch.norm(vel[:, :2], dim=-1).mean().item()
                print(f"Step {step}: Avg speed = {avg_speed:.2f} m/s")
    
    except KeyboardInterrupt:
        print("\n✓ Demo stopped by user")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test parallel navigation environment")
    parser.add_argument('--mode', type=str, default='scalability', 
                        choices=['scalability', 'demo', 'single'],
                        help='Test mode')
    parser.add_argument('--n_envs', type=int, default=1000,
                        help='Number of environments for single test')
    parser.add_argument('--n_steps', type=int, default=200,
                        help='Number of steps for single test')
    
    args = parser.parse_args()
    
    # 初始化Genesis
    gs.init(backend=gs.gpu)
    
    if args.mode == 'scalability':
        test_scalability()
    elif args.mode == 'demo':
        test_visual_demo()
    elif args.mode == 'single':
        benchmark_env(args.n_envs, args.n_steps)
