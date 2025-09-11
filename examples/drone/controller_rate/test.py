import math
import genesis as gs
from genesis.engine.entities.drone_entity import DroneEntity

import numpy as np
import torch

def fly_to_point(target, controller, scene, drone, tol=0.1, max_steps=1000):
    # Ensure target is numpy float array
    target = np.asarray(target, dtype=float).reshape(3,)
    device = controller.drone_pos.device

    # single-batch target on the controller/device (shape (1,3))
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device).view(1, 3)

    step = 0
    while step < max_steps:
        # ----- read current position (tensor) and convert to numpy -----
        drone_pos_t = drone.get_pos()                # likely a tensor (maybe on cuda)
        drone_pos = drone_pos_t.detach().cpu().numpy().ravel()  # shape (3,)

        # compute distance as plain float
        distance = float(np.linalg.norm(target - drone_pos))
        print(f"[step {step}] distance = {distance:.4f}, drone_pos = {drone_pos}")

        if distance <= tol:
            print("Reached target.")
            break

        # ----- compute controller command -----
        # (we keep target_tensor constant since same target for whole loop)
        cmd = controller.compute(target_pos=target_tensor)  # expected shape (1, num_rotors)

        # debug checks
        if torch.isnan(cmd).any():
            print("ERROR: NaN in cmd, aborting.")
            break
        if torch.isinf(cmd).any():
            print("ERROR: Inf in cmd, aborting.")
            break

        # clamp to avoid negatives (and tiny eps to avoid sqrt issues)
        cmd = torch.clamp(cmd, min=1e-8)

        # send to drone (keep device same as cmd)
        drone.set_propellels_rpm(cmd.squeeze(0))

        # step simulation
        scene.step()
        step += 1

    if step >= max_steps:
        print("Warning: max_steps reached before reaching target.")

# from Genesis.examples.drone.controller_rate.rate_controller import ThrustRateToRPMController
from position_controller import LeePositionController
def main():
    gs.init(backend=gs.gpu)

    ##### scene #####
    scene = gs.Scene(
        show_viewer=True, 
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
        show_FPS=False)
    ##### entities #####
    plane = scene.add_entity(morph=gs.morphs.Plane())

    # drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf", pos=(0, 0, 2)))
    drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 2)))
    

    
    cam = scene.add_camera(pos=(1, 1, 1), GUI=False, res=(640, 480), fov=30)

    ##### build #####

    scene.build()

    controller = LeePositionController(g=9.81, drone = drone, n_envs=1)

    points = [(1, 1, 2), (-1, 2, 1), (0, 0, 0.5)]

    # for point in points:
    #     fly_to_point(point, controller, scene, drone)

    while True:

        cmd = controller.compute(target_pos=torch.tensor([1, 1, 3]))  # expected shape (1, num_rotors)
        # send to drone (keep device same as cmd)
        drone.set_propellels_rpm(cmd.squeeze(0))
        scene.step()

if __name__ == "__main__":
    main()
