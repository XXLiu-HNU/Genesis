import genesis as gs
from genesis.engine.entities.drone_entity import DroneEntity

import numpy as np

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


def easy_fly(drone: DroneEntity, scene: gs.Scene):
    """

    """

    # 1. 总推力需求
    acc_des = [0.0, 0.0, 0.0]  # 悬停不加速
    ang_acc_des = [0.0, 0.0, 0.0]  # 不转动
    mass = drone.get_mass()
    g = 9.81
    KF = drone._KF
    KM = drone._KM
    _propellers_spin = drone._propellers_spin
    
    l = 0.12     # m
    step = 0
    while step < 1000:
        omegas = quad_mixer(acc_des, ang_acc_des, mass, g, l, KF, KM)
        M1, M2, M3, M4 = omegas
        
        drone.set_propellels_rpm([M1, M2, M3, M4])
        scene.step()
        
        step += 1


def main():
    gs.init(backend=gs.gpu)

    ##### scene #####
    scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01))
    ##### entities #####
    plane = scene.add_entity(morph=gs.morphs.Plane())

    drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf", pos=(0, 0, 0.2)))

    cam = scene.add_camera(pos=(1, 1, 1), GUI=False, res=(640, 480), fov=30)

    ##### build #####

    scene.build()
    cam.start_recording()

    points = [(1, 1, 2), (-1, 2, 1), (0, 0, 0.5)]

    for point in points:
        
        easy_fly(drone, scene)


if __name__ == "__main__":
    main()
