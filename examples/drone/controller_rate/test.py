import genesis as gs
from genesis.engine.entities.drone_entity import DroneEntity

import numpy as np
import torch

from easy_controller import ThrustRateToRPMController
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

    controller = ThrustRateToRPMController(9.81, drone)
    cam.start_recording()

    points = [(1, 1, 2), (-1, 2, 1), (0, 0, 0.5)]

    step = 0
    while step < 1000:


        target_rate = torch.tensor([0, 0, 30])
        target_thrust = torch.tensor([0.3])
        cmd = controller(target_rate, target_thrust)
        # print(cmd)
        drone.set_propellels_rpm(cmd)
        scene.step()
        
        step += 1


if __name__ == "__main__":
    main()
