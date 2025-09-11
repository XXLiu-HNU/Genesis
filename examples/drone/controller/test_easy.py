import numpy as np

def quad_mixer(acc_des, ang_acc_des, mass, g, l, KF, KM):
    """
    计算四旋翼螺旋桨转速 (简化模型)
    
    参数:
        acc_des: 期望加速度 [ax, ay, az]
        ang_acc_des: 期望角加速度 [p_dot, q_dot, r_dot]
        mass: 无人机质量
        g: 重力加速度
        l: 桨臂长度 (质心到桨的水平距离)
        KF: 推力系数
        KM: 力矩系数
    返回:
        omegas: 4个螺旋桨角速度 [ω1, ω2, ω3, ω4]
    """

    # 1. 总推力需求
    Fz = mass * (g + acc_des[2])  

    # 2. 力矩需求 (这里假设 I = diag(Ix, Iy, Iz)，只保留 yaw 的近似)
    tau_x = ang_acc_des[0]   # roll torque
    tau_y = ang_acc_des[1]   # pitch torque
    tau_z = ang_acc_des[2]   # yaw torque

    # 3. 分配矩阵 (Fz, tau_x, tau_y, tau_z) -> (w1^2, w2^2, w3^2, w4^2)
    alloc = np.array([
        [ KF,  KF,  KF,  KF],
        [ 0,   l*KF,  0,  -l*KF],
        [-l*KF,  0,  l*KF,   0],
        [ KM, -KM,  KM, -KM]
    ])

    # 4. 解线性方程
    desired = np.array([Fz, tau_x, tau_y, tau_z])
    w2 = np.linalg.solve(alloc, desired)

    # 5. 限制 & 开方
    w2 = np.clip(w2, 0, None)  # 避免负数
    omegas = np.sqrt(w2)

    return omegas


if __name__ == "__main__":
    mass = 1.0   # kg
    g = 9.81     # m/s^2
    l = 0.25     # m
    KF = 3e-6    # N/(rad/s)^2
    KM = 1e-7    # Nm/(rad/s)^2

    # 目标: 悬停 (只需要抵消重力)，无角加速度
    acc_des = np.array([0.0, 0.0, 0.0])      # 悬停不加速
    ang_acc_des = np.array([0.0, 0.0, 0.0])  # 不转动

    omegas = quad_mixer(acc_des, ang_acc_des, mass, g, l, KF, KM)
    print("螺旋桨角速度:", omegas)