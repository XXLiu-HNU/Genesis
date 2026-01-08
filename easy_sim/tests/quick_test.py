"""
快速测试脚本 - 验证easy_sim所有功能
"""

import sys
import os

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试1: 导入模块")
    print("=" * 60)
    try:
        from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics, BatchedQuadrotorDynamics
        print("✓ 本地导入成功")
    except Exception as e:
        print(f"✗ 本地导入失败: {e}")
        return False
    
    # 测试从父目录导入
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from easy_sim import QuadrotorDynamics as QD, BatchedQuadrotorDynamics as BQD
        print("✓ 包导入成功")
    except Exception as e:
        print(f"✗ 包导入失败: {e}")
        return False
    
    return True


def test_basic_sim():
    """测试基本仿真"""
    print("\n" + "=" * 60)
    print("测试2: 基本仿真")
    print("=" * 60)
    
    try:
        from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics
        
        sim = QuadrotorDynamics({'dt': 0.01})
        sim.reset(position=[0, 0, 1.0])
        
        # 仿真100步
        for i in range(100):
            state = sim.step([1.23, 1.23, 1.23, 1.23])
        
        pos = state['position']
        print(f"✓ 仿真成功")
        print(f"  最终位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        return True
    except Exception as e:
        print(f"✗ 仿真失败: {e}")
        return False


def test_batch_sim():
    """测试批量仿真"""
    print("\n" + "=" * 60)
    print("测试3: 批量仿真")
    print("=" * 60)
    
    try:
        from easy_sim.core.drone_dynamics_sim import BatchedQuadrotorDynamics
        import numpy as np
        
        num_envs = 10
        sim = BatchedQuadrotorDynamics(num_envs, {'dt': 0.01})
        sim.reset()
        
        # 仿真100步
        thrusts = np.ones((num_envs, 4)) * 1.23
        for i in range(100):
            state = sim.step(thrusts)
        
        pos = state['position']
        print(f"✓ 批量仿真成功")
        print(f"  环境数量: {num_envs}")
        print(f"  位置范围: Z=[{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ 批量仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rpm_input():
    """测试RPM输入"""
    print("\n" + "=" * 60)
    print("测试4: RPM输入")
    print("=" * 60)
    
    try:
        from easy_sim.core.drone_dynamics_sim import QuadrotorDynamics
        
        sim = QuadrotorDynamics({'dt': 0.01})
        sim.reset(position=[0, 0, 1.0])
        
        # 使用RPM控制
        for i in range(100):
            state = sim.step_rpm([60000, 60000, 60000, 60000])
        
        pos = state['position']
        print(f"✓ RPM输入成功")
        print(f"  最终位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        return True
    except Exception as e:
        print(f"✗ RPM输入失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 18 + "Easy Sim 快速测试" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    results.append(test_imports())
    results.append(test_basic_sim())
    results.append(test_batch_sim())
    results.append(test_rpm_input())
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.0f}%")
    
    if passed == total:
        print("\n✓ 所有测试通过！easy_sim已准备就绪。")
    else:
        print(f"\n✗ {total - passed} 个测试失败。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
