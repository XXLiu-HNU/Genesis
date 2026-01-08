"""
测试混控器的行为
"""

import numpy as np

# 模拟混控器
def mixer(throttle_command, action_thrust, thrust_compensate, base_rpm, pid_output):
    """
    模拟混控器
    """
    throttle_rc = np.clip(throttle_command * 3, 0.0, 3.0) * base_rpm
    throttle_action = np.clip(action_thrust * 3 + thrust_compensate, 0.0, 3.0) * base_rpm
    throttle = throttle_rc + throttle_action
    
    roll, pitch, yaw = pid_output
    
    motor_outputs = np.array([
        throttle - roll - pitch - yaw,  # M1
        throttle - roll + pitch + yaw,  # M2
        throttle + roll + pitch - yaw,  # M3
        throttle + roll - pitch + yaw,  # M4
    ])
    
    return np.clip(motor_outputs, 1, base_rpm * 3.5)


print("=" * 70)
print("混控器行为测试")
print("=" * 70)

base_rpm = 595000
thrust_compensate = 1.0

print(f"\nbase_rpm = {base_rpm}")
print(f"thrust_compensate = {thrust_compensate}")
print()

# 测试1: 理想悬停
print("测试1: 理想悬停")
print("  throttle_command = 0, action_thrust = 0")
rpms = mixer(0, 0, thrust_compensate, base_rpm, [0, 0, 0])
print(f"  输出RPM: {rpms}")
print(f"  期望: ~595000 (悬停RPM)")
print()

# 测试2: 增加推力
print("测试2: 增加推力")
print("  throttle_command = 0.3, action_thrust = 0")
rpms = mixer(0.3, 0, thrust_compensate, base_rpm, [0, 0, 0])
print(f"  输出RPM: {rpms}")
print()

# 测试3: 减少推力（throttle_command负数会被clip）
print("测试3: 减少推力")
print("  throttle_command = -0.3, action_thrust = 0")
rpms = mixer(-0.3, 0, thrust_compensate, base_rpm, [0, 0, 0])
print(f"  输出RPM: {rpms}")
print(f"  注意: throttle_command被clip到0，无法减少推力！")
print()

# 测试4: 通过降低thrust_compensate减少推力
print("测试4: 通过调整action_thrust减少推力")
print("  throttle_command = 0, action_thrust = -0.3")
rpms = mixer(0, -0.3, thrust_compensate, base_rpm, [0, 0, 0])
print(f"  输出RPM: {rpms}")
print(f"  注意: 也被clip到base_rpm了！")
print()

print("=" * 70)
print("结论：")
print("  1. throttle_command和action_thrust都不能为负")
print("  2. 无法通过这两个参数减少推力到悬停以下")
print("  3. thrust_compensate应该设为悬停点")
print("  4. 位置控制器的throttle_command应该是相对调整")
print("=" * 70)

# 新方案测试
print("\n新方案：将thrust_compensate设为悬停倍数")
print()

# 计算悬停所需的倍数
# 理论悬停RPM = 595K
# base_rpm = 595K
# 悬停时需要：throttle = base_rpm * 1.0 = 595K
# 所以 thrust_compensate 应该能产生595K

# 但公式是：
# throttle = (throttle_cmd * 3 + action * 3 + thrust_comp) * base_rpm
# 悬停时 throttle_cmd = 0, action = 0
# 所以：595K = thrust_comp * 595K
# thrust_comp = 1.0 ✓

print("当前设置正确！问题可能在其他地方...")
print()

# 检查位置控制器是否正确更新throttle_command
print("可能的问题：")
print("  1. 位置控制器的PID增益不对")
print("  2. throttle_command范围限制过大或过小")
print("  3. 姿态控制器的输出导致不稳定")
