import genesis as gs
import math
from pid import PIDcontroller
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.vis.camera import Camera
import torch

base_rpm = 62293.9641914


def fly_to_point(drone: DroneEntity, target, controller: PIDcontroller, scene: gs.Scene, cam: Camera):
    drone = drone
    step = 0
    x = target[0] - drone.get_pos()[0]
    y = target[1] - drone.get_pos()[1]
    z = target[2] - drone.get_pos()[2]

    distance = math.sqrt(x**2 + y**2 + z**2)

    while  step < 1000:
        target = torch.randn([1,4])
        controller.step(target)
        
        scene.step()
        cam.render()
        # print("point =", drone.get_pos())
        
        step += 1


def main():
    gs.init(backend=gs.gpu)

    ##### scene #####
    scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01),show_FPS=False)

    ##### entities #####
    plane = scene.add_entity(morph=gs.morphs.Plane())

    drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf", pos=(0, 0, 0.2)))

    # parameters are tuned such that the
    # drone can fly, not optimized
    pid_params = {
        "kp": 6500,
        "ki": 0.01,
        "kd": 0.0,
        "kf": 0.0,
        "thrust_compensate": 0.0,
        "pid_exec_freq": 60,
        "base_rpm": base_rpm,
    }
    controller = PIDcontroller(drone=drone, config=pid_params)

    cam = scene.add_camera(pos=(1, 1, 1),  GUI=False, res=(640, 480), fov=30)

    ##### build #####

    scene.build()


    points = [(1, 1, 2), (-1, 2, 1), (0, 0, 0.5)]

    for point in points:
        fly_to_point(drone, point, controller, scene, cam)

if __name__ == "__main__":
    main()
