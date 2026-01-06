"""
基础功能测试：验证代码可以正常运行
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")
try:
    import genesis as gs
    print("✓ Genesis imported")
except ImportError as e:
    print(f"✗ Failed to import Genesis: {e}")
    sys.exit(1)

try:
    from controller.pid import PIDcontroller
    from controller.odom import Odom
    print("✓ Controllers imported")
except ImportError as e:
    print(f"✗ Failed to import controllers: {e}")
    sys.exit(1)

try:
    from utils import setup_random_cylindrical_obstacles
    print("✓ Utils imported")
except ImportError as e:
    print(f"✗ Failed to import utils: {e}")
    sys.exit(1)

try:
    from target_planning.parallel_nav_env import ParallelTargetNavEnv, VectorizedPathFollower
    print("✓ Parallel nav env imported")
except ImportError as e:
    print(f"✗ Failed to import parallel nav env: {e}")
    sys.exit(1)

print("\nTesting VectorizedPathFollower...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试障碍物
    obs_xy = torch.tensor([[2.0, 2.0], [5.0, 5.0]], device=device)
    obs_r = torch.tensor([0.3, 0.3], device=device)
    
    follower = VectorizedPathFollower(
        num_envs=10, 
        device=device,
        obs_xy=obs_xy,
        obs_r=obs_r
    )
    
    # 测试设置目标
    goals = torch.rand((10, 2), device=device) * 10
    follower.set_goals(goals)
    assert follower.goal_xy.shape == (10, 2), "Goal shape mismatch"
    
    # 测试步进
    current_pos = torch.rand((10, 2), device=device) * 10
    target_pos = follower.step(current_pos, dt=0.01)
    assert target_pos.shape == (10, 2), "Target pos shape mismatch"
    
    # 测试到达检查
    reached = follower.check_reached()
    assert reached.shape == (10,), "Reached mask shape mismatch"
    assert reached.dtype == torch.bool, "Reached mask dtype mismatch"
    
    print("✓ VectorizedPathFollower works correctly (with obstacle avoidance)")
except Exception as e:
    print(f"✗ VectorizedPathFollower test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting ParallelTargetNavEnv creation...")
try:
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
    
    # 创建小规模环境
    env = ParallelTargetNavEnv(num_envs=10, show_viewer=False)
    print("✓ Environment created with 10 envs")
    
    # 测试步进
    for i in range(10):
        env.step()
    print("✓ Environment stepped 10 times")
    
    # 测试获取状态
    states = env.get_target_states()
    assert 'pos' in states, "Missing 'pos' in states"
    assert 'vel' in states, "Missing 'vel' in states"
    assert 'quat' in states, "Missing 'quat' in states"
    assert states['pos'].shape == (10, 3), "Position shape mismatch"
    assert states['vel'].shape == (10, 3), "Velocity shape mismatch"
    assert states['quat'].shape == (10, 4), "Quaternion shape mismatch"
    print("✓ State retrieval works correctly")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\n代码验证成功！可以运行大规模测试了。")
    print("\n运行命令:")
    print("  python tracking/target_planning/test_parallel_scalability.py --mode scalability")
    print("  python tracking/target_planning/quick_start.py --example 1")
    
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
