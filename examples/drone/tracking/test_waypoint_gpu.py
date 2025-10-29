"""
Test script to visualize waypoint_gpu target movement.
Shows how the target drone navigates around obstacles using GPU-based waypoint sampling.
"""
import os
import yaml
import torch
import genesis as gs
from pid import PIDcontroller
from odom import Odom
from utils import setup_random_cylindrical_obstacles

class WaypointTestEnv:
    def __init__(self, show_viewer=True, show_markers=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = 1  # Single environment for testing
        self.show_markers = show_markers
        
        # Load config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config/search.yaml"), "r") as f:
            nav_cfg = yaml.safe_load(f)
        
        def cfg(path, default=None):
            cur = nav_cfg
            for k in path.split("."):
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur
        
        # Basic parameters
        self.dt = 0.01
        self.drone_height = float(cfg("drone.height", 1.0))
        self.world_xy_min = tuple(cfg("world.xy_min", [-10.0, -10.0]))
        self.world_xy_max = tuple(cfg("world.xy_max", [ 10.0,  10.0]))
        self.drone_radius  = float(cfg("drone.radius", 0.12))
        self.safety_margin = float(cfg("safety.margin", 0.12))
        self.inflation_default = self.drone_radius + self.safety_margin
        
        # Waypoint GPU parameters (same as in track_env.py)
        self.waypoint_samples = 16
        self.waypoint_distance = 3.0
        self.waypoint_update_freq = 10
        self.waypoint_goal_dist = 5.0
        
        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(0.0, 0.0, 15.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=60,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False)
        )
        
        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane())
        
        # Add target drone
        self.target_drone = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/target_drone_urdf/drone.urdf")
        )
        
        # Add visual markers for waypoints and goals (optional)
        if self.show_markers:
            # Goal marker (red sphere)
            self.goal_marker = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=(100, 100, self.drone_height),
                    radius=0.3,
                    fixed=True,
                ),
                surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0, 1.0))  # Red
            )
            
            # Current waypoint marker (green sphere)
            self.waypoint_marker = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=(100, 100, self.drone_height),
                    radius=0.2,
                    fixed=True,
                ),
                surface=gs.surfaces.Rough(color=(0.0, 1.0, 0.0, 1.0))  # Green
            )
        else:
            self.goal_marker = None
            self.waypoint_marker = None
        
        # Load controller config and make it more stable for testing
        with open(os.path.join(script_dir, "config/pos.yaml"), "r") as file:
            self.pos_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)
        
        # Override PID parameters for more stable height control
        if 'pos' in self.pos_ctrl_config:
            # Increase height control gains for better stability
            self.pos_ctrl_config['pos']['kp_t'] = 1.5  # Increased from 1.0
            self.pos_ctrl_config['pos']['kd_t'] = 0.1  # Add damping
            # Reduce horizontal gains to minimize aggressive tilting
            self.pos_ctrl_config['pos']['kp_x'] = 1.2  # Reduced from 1.5
            self.pos_ctrl_config['pos']['kp_y'] = 1.2  # Reduced from 1.5
            self.pos_ctrl_config['pos']['kd_x'] = 0.08  # Increased damping
            self.pos_ctrl_config['pos']['kd_y'] = 0.08  # Increased damping
        
        # Setup obstacles
        n_obstacles = int(cfg("obstacles.n", 100))
        world_bounds_xyxy = (
            self.world_xy_min[0], self.world_xy_max[0],
            self.world_xy_min[1], self.world_xy_max[1],
        )
        obs, obs_xy, obs_r = setup_random_cylindrical_obstacles(
            self.scene, 
            n_obstacles=n_obstacles, 
            world_bounds=world_bounds_xyxy,
            origin_clearance=3.0,
            min_distance=2.0
        )
        
        self.obs_xy = torch.tensor(obs_xy, dtype=torch.float32, device=self.device)
        self.obs_r = torch.tensor(obs_r, dtype=torch.float32, device=self.device)
        
        # Build scene
        self.scene.build(n_envs=self.num_envs)
        
        # Cache obstacle tensors
        self.obs_xy_dev = self.obs_xy.to(self.device, dtype=torch.float32)
        self.obs_r_dev = self.obs_r.to(self.device, dtype=torch.float32)
        
        # Initialize target position BEFORE setting up controller
        self.target_pos = torch.tensor([[0.0, 0.0, self.drone_height]], device=self.device)
        self.target_drone.set_pos(self.target_pos[0], zero_velocity=True)
        
        # Setup controller for target drone (after setting initial position)
        self._setup_controller(self.target_drone, "position", self.pos_ctrl_config)
        
        # Waypoint GPU state buffers
        self.target_waypoint_goal = torch.zeros((1, 2), device=self.device, dtype=torch.float32)
        self.target_waypoint_current = torch.zeros((1, 2), device=self.device, dtype=torch.float32)
        self.target_waypoint_pos = torch.zeros((1, 2), device=self.device, dtype=torch.float32)  # Smooth position
        self.target_waypoint_vel = torch.zeros((1, 2), device=self.device, dtype=torch.float32)  # Velocity
        self.waypoint_step_counter = torch.zeros((1,), device=self.device, dtype=torch.long)
        
        # Motion parameters (gentle settings to avoid height oscillation)
        self.waypoint_v_max = 0.4      # Max speed (m/s) - reduced for stability
        self.waypoint_a_max = 0.6      # Max acceleration (m/s^2) - reduced for gentle motion
        self.waypoint_warmup_time = 3.0  # Warmup time (seconds) - increased for smoother start
        self.waypoint_v_init = 0.05    # Initial max speed during warmup (m/s) - very gentle start
        self.waypoint_timer = torch.zeros((1,), device=self.device, dtype=torch.float32)  # Timer for warmup
        
        # Initialize goal and waypoint
        self.target_waypoint_goal[0, 0] = 8.0
        self.target_waypoint_goal[0, 1] = 8.0
        self.target_waypoint_current[0] = self.target_pos[0, :2]
        self.target_waypoint_pos[0] = self.target_pos[0, :2]
        self.target_waypoint_vel[0] = 0.0
        
        self.ref_pos_buf = torch.zeros((1, 4), device=self.device, dtype=torch.float32)
        self.ref_pos_buf[:, 2] = self.drone_height
        self.ref_pos_buf[:, 3] = 0.0
        
        self.step_count = 0
        
        # Track last marker positions to reduce update frequency (if markers enabled)
        if self.show_markers:
            self.last_goal_marker_pos = torch.zeros(3, device=self.device)
            self.last_waypoint_marker_pos = torch.zeros(3, device=self.device)
            self.last_waypoint_value = torch.zeros(2, device=self.device)
            
            # Initialize marker positions
            init_goal_pos = torch.tensor([8.0, 8.0, self.drone_height], device=self.device)
            init_waypoint_pos = torch.tensor([0.0, 0.0, self.drone_height], device=self.device)
            self.goal_marker.set_pos(init_goal_pos)
            self.waypoint_marker.set_pos(init_waypoint_pos)
            self.last_goal_marker_pos[:] = init_goal_pos
            self.last_waypoint_marker_pos[:] = init_waypoint_pos
            self.last_waypoint_value[:] = 0.0
        
        print("=" * 60)
        print("WAYPOINT GPU TEST ENVIRONMENT")
        print("=" * 60)
        print(f"World bounds: {self.world_xy_min} to {self.world_xy_max}")
        print(f"Obstacles: {n_obstacles}")
        print(f"Waypoint samples: {self.waypoint_samples}")
        print(f"Update frequency: every {self.waypoint_update_freq} steps")
        print(f"Initial goal: {self.target_waypoint_goal[0].cpu().numpy()}")
        print("=" * 60)
        print("\nWATCH:")
        print("- Red sphere = Final goal")
        print("- Green sphere = Current waypoint")
        print("- Blue drone = Target moving with waypoint_gpu")
        print("\nTarget will navigate around obstacles to reach goals!")
        print("=" * 60)
        
        # Stabilize drone for longer before starting (critical for stable flight)
        print("\nStabilizing drone (this may take a few seconds)...")
        print("Monitoring height stability...")
        
        max_height_error = 0.0
        for i in range(300):  # Increased to 300 steps (~3 seconds) for full stabilization
            self.ref_pos_buf[:, 0:2] = self.target_pos[0, :2]
            self.ref_pos_buf[:, 2] = self.drone_height
            self.ref_pos_buf[:, 3] = 0.0
            target_rpms = self.target_drone.controller.step(self.ref_pos_buf)
            self.target_drone.set_propellels_rpm(target_rpms)
            self.scene.step()
            self.target_pos[:] = self.target_drone.get_pos()
            
            # Track maximum height error
            actual_height = self.target_pos[0, 2].item()
            height_error = abs(actual_height - self.drone_height)
            max_height_error = max(max_height_error, height_error)
            
            # Print progress periodically
            if i % 100 == 99:
                print(f"  Step {i+1}/300: height={actual_height:.3f}m, error={height_error:.3f}m, max_error={max_height_error:.3f}m")
        
        actual_height = self.target_pos[0, 2].item()
        height_error = abs(actual_height - self.drone_height)
        print(f"✓ Drone stabilized at height {actual_height:.3f}m (error: {height_error:.3f}m, max_error: {max_height_error:.3f}m)")
        
        if max_height_error > 0.2:
            print(f"[WARNING] Large height oscillation detected during stabilization!")
            print(f"[HINT] Consider reducing waypoint_v_max or increasing warmup_time")
        
        # Update waypoint positions after stabilization
        self.target_waypoint_pos[0] = self.target_pos[0, :2]
        self.target_waypoint_current[0] = self.target_pos[0, :2]
        self.waypoint_timer[0] = 0.0  # Reset timer to start warmup from beginning
        
        print("Starting waypoint navigation...\n")
    
    def _setup_controller(self, drone, controller_type, config):
        """Setup IMU and PID controller for drone."""
        odom = Odom(num_envs=self.num_envs, device=self.device)
        odom.set_drone(drone)
        setattr(drone, 'odom', odom)
        
        pid = PIDcontroller(
            num_envs=self.num_envs,
            odom=drone.odom,
            config=config,
            device=self.device,
            controller=controller_type,
        )
        pid.set_drone(drone)
        setattr(drone, 'controller', pid)
    
    def _move_to_waypoint_smooth(self):
        """Smooth movement towards current waypoint with velocity limits and warmup."""
        # Increment timer for warmup
        self.waypoint_timer += self.dt
        
        # Warmup: gradually increase max speed from v_init to v_max
        warmup_factor = torch.clamp(self.waypoint_timer / self.waypoint_warmup_time, 0.0, 1.0)
        current_v_max = self.waypoint_v_init + warmup_factor * (self.waypoint_v_max - self.waypoint_v_init)
        
        # Direction to current waypoint
        to_waypoint = self.target_waypoint_current - self.target_waypoint_pos
        dist_to_waypoint = torch.norm(to_waypoint, dim=1, keepdim=True)
        
        # Desired velocity with slowdown near waypoint
        slowdown_dist = 1.0
        speed_factor = torch.clamp(dist_to_waypoint / slowdown_dist, 0.0, 1.0)
        desired_speed = current_v_max * speed_factor
        
        # Direction and desired velocity
        direction = torch.where(
            dist_to_waypoint > 1e-6,
            to_waypoint / (dist_to_waypoint + 1e-9),
            torch.zeros_like(to_waypoint)
        )
        desired_vel = direction * desired_speed
        
        # Apply acceleration limit
        vel_change = desired_vel - self.target_waypoint_vel
        vel_change_norm = torch.norm(vel_change, dim=1, keepdim=True)
        max_vel_change = self.waypoint_a_max * self.dt
        
        vel_change = torch.where(
            vel_change_norm > max_vel_change,
            vel_change / (vel_change_norm + 1e-9) * max_vel_change,
            vel_change
        )
        
        # Update velocity and position
        self.target_waypoint_vel += vel_change
        self.target_waypoint_pos += self.target_waypoint_vel * self.dt
        
        # Clamp to world bounds
        self.target_waypoint_pos[:, 0] = torch.clamp(
            self.target_waypoint_pos[:, 0],
            self.world_xy_min[0], self.world_xy_max[0]
        )
        self.target_waypoint_pos[:, 1] = torch.clamp(
            self.target_waypoint_pos[:, 1],
            self.world_xy_min[1], self.world_xy_max[1]
        )

    def _update_waypoint_gpu(self):
        """
        GPU-based waypoint sampling for target movement.
        (Simplified version of the one in track_env.py)
        """
        # Increment step counter
        self.waypoint_step_counter += 1
        
        # Check if need new waypoint (only when reached, not on timer)
        # This prevents waypoint from changing before reaching it
        dist_to_waypoint = torch.norm(self.target_waypoint_pos - self.target_waypoint_current, dim=1)
        need_update = (dist_to_waypoint < 0.3)  # Only update when reached
        
        if not torch.any(need_update):
            return
        
        # Sample candidate waypoints
        current_pos = self.target_waypoint_pos[0].unsqueeze(0)  # (1, 2)
        goal = self.target_waypoint_goal[0].unsqueeze(0)        # (1, 2)
        
        S = self.waypoint_samples
        candidates = torch.zeros((1, S, 2), device=self.device, dtype=torch.float32)
        
        # Sample candidates
        for i in range(S):
            if i < S // 2:
                # Goal-directed samples
                t = torch.rand(1, device=self.device) * 0.8 + 0.2
                direction = goal - current_pos
                dist_to_goal = torch.norm(direction, dim=1, keepdim=True) + 1e-6
                candidates[:, i] = current_pos + t.unsqueeze(1) * direction * (self.waypoint_distance / dist_to_goal)
            else:
                # Random exploration
                angles = torch.rand(1, device=self.device) * 2 * 3.14159
                distances = torch.rand(1, device=self.device) * self.waypoint_distance
                candidates[:, i, 0] = current_pos[:, 0] + distances * torch.cos(angles)
                candidates[:, i, 1] = current_pos[:, 1] + distances * torch.sin(angles)
        
        # Clamp to world bounds
        candidates[:, :, 0] = torch.clamp(candidates[:, :, 0], self.world_xy_min[0], self.world_xy_max[0])
        candidates[:, :, 1] = torch.clamp(candidates[:, :, 1], self.world_xy_min[1], self.world_xy_max[1])
        
        # GPU batch line-of-sight check
        if self.obs_xy.numel() > 0:
            v = candidates - current_pos.unsqueeze(1)  # (1, S, 2)
            vv = torch.clamp((v * v).sum(-1), min=1e-9)  # (1, S)
            
            w = self.obs_xy_dev.unsqueeze(0).unsqueeze(0) - current_pos.unsqueeze(1).unsqueeze(1)  # (1, S, M, 2)
            t = ((w * v.unsqueeze(2)).sum(-1) / vv.unsqueeze(-1)).clamp(0.0, 1.0)  # (1, S, M)
            
            proj = current_pos.unsqueeze(1).unsqueeze(1) + t.unsqueeze(-1) * v.unsqueeze(2)  # (1, S, M, 2)
            d = torch.linalg.norm(proj - self.obs_xy_dev.unsqueeze(0).unsqueeze(0), dim=-1)  # (1, S, M)
            
            blocked = d <= (self.obs_r_dev + self.inflation_default).unsqueeze(0).unsqueeze(0)  # (1, S, M)
            is_free = ~blocked.any(dim=2)  # (1, S)
        else:
            is_free = torch.ones((1, S), device=self.device, dtype=torch.bool)
        
        # Score waypoints
        dist_to_goal = torch.norm(candidates - goal.unsqueeze(1), dim=2)  # (1, S)
        dist_from_current = torch.norm(candidates - current_pos.unsqueeze(1), dim=2)  # (1, S)
        
        score = torch.zeros_like(dist_to_goal)
        score[is_free] = dist_from_current[is_free] * 2.0 - dist_to_goal[is_free] * 0.5
        score[~is_free] = -1e6
        
        # Select best waypoint
        best_idx = torch.argmax(score, dim=1)  # (1,)
        best_waypoint = candidates[0, best_idx[0]]  # (2,)
        
        self.target_waypoint_current[0] = best_waypoint
        
        # Check if reached goal, sample new goal
        dist_to_goal_final = torch.norm(current_pos[0] - goal[0])
        if dist_to_goal_final < 1.0:
            angles = torch.rand(1, device=self.device) * 2 * 3.14159
            distances = torch.rand(1, device=self.device) * self.waypoint_goal_dist + 2.0
            new_goal_x = current_pos[0, 0] + distances * torch.cos(angles)
            new_goal_y = current_pos[0, 1] + distances * torch.sin(angles)
            self.target_waypoint_goal[0, 0] = torch.clamp(new_goal_x, self.world_xy_min[0] + 1, self.world_xy_max[0] - 1)
            self.target_waypoint_goal[0, 1] = torch.clamp(new_goal_y, self.world_xy_min[1] + 1, self.world_xy_max[1] - 1)
            print(f"\n[Step {self.step_count}] 🎯 Reached goal! New goal: {self.target_waypoint_goal[0].cpu().numpy()}")
    
    def step(self):
        """Main step function."""
        self.step_count += 1
        
        # Update waypoint using GPU sampling
        self._update_waypoint_gpu()
        
        # Smooth movement towards waypoint
        self._move_to_waypoint_smooth()
        
        # Set reference position to smooth position (ensure height is correct)
        self.ref_pos_buf[:, 0:2] = self.target_waypoint_pos
        self.ref_pos_buf[:, 2] = self.drone_height  # Explicitly set height every step
        self.ref_pos_buf[:, 3] = 0.0  # Yaw
        
        # Control target drone
        target_rpms = self.target_drone.controller.step(self.ref_pos_buf)
        self.target_drone.set_propellels_rpm(target_rpms)
        
        # Step simulation
        self.scene.step()
        
        # Update target position
        self.target_pos[:] = self.target_drone.get_pos()
        
        # Update visual markers ONLY when waypoint value actually changes (if enabled)
        if self.show_markers:
            # Check if waypoint changed (avoid updating every frame to prevent flickering)
            waypoint_changed = torch.norm(self.target_waypoint_current[0] - self.last_waypoint_value) > 0.01
            
            if waypoint_changed:
                # Waypoint changed - update green marker
                waypoint_pos = torch.zeros(3, device=self.device)
                waypoint_pos[:2] = self.target_waypoint_current[0]
                waypoint_pos[2] = self.drone_height
                self.waypoint_marker.set_pos(waypoint_pos)
                self.last_waypoint_marker_pos[:] = waypoint_pos
                self.last_waypoint_value[:] = self.target_waypoint_current[0]
                
                if self.step_count > 300:  # Skip logging during stabilization
                    print(f"[Step {self.step_count}] 🟢 New waypoint: {self.target_waypoint_current[0].cpu().numpy()}")
            
            # Update goal marker only when goal changes
            goal_changed = torch.norm(self.target_waypoint_goal[0][:2] - self.last_goal_marker_pos[:2]) > 0.01
            if goal_changed:
                goal_pos = torch.zeros(3, device=self.device)
                goal_pos[:2] = self.target_waypoint_goal[0]
                goal_pos[2] = self.drone_height
                self.goal_marker.set_pos(goal_pos)
                self.last_goal_marker_pos[:] = goal_pos
        
        # Print status every 100 steps
        if self.step_count % 100 == 0:
            current = self.target_pos[0].cpu().numpy()
            smooth_pos = self.target_waypoint_pos[0].cpu().numpy()
            velocity = self.target_waypoint_vel[0].cpu().numpy()
            speed = torch.norm(self.target_waypoint_vel[0]).item()
            waypoint = self.target_waypoint_current[0].cpu().numpy()
            goal = self.target_waypoint_goal[0].cpu().numpy()
            dist_to_goal = torch.norm(self.target_waypoint_pos[0] - self.target_waypoint_goal[0]).item()
            
            # Calculate current warmup progress
            warmup_factor = torch.clamp(self.waypoint_timer / self.waypoint_warmup_time, 0.0, 1.0).item()
            current_v_max = self.waypoint_v_init + warmup_factor * (self.waypoint_v_max - self.waypoint_v_init)
            
            print(f"\n[Step {self.step_count}]")
            print(f"  Target pos:    ({current[0]:.2f}, {current[1]:.2f}, {current[2]:.2f})")
            print(f"  Smooth pos:    ({smooth_pos[0]:.2f}, {smooth_pos[1]:.2f})")
            print(f"  Velocity:      ({velocity[0]:.2f}, {velocity[1]:.2f}) m/s")
            print(f"  Speed:         {speed:.2f} m/s (current_max: {current_v_max:.2f}, warmup: {warmup_factor*100:.0f}%)")
            print(f"  Waypoint:      ({waypoint[0]:.2f}, {waypoint[1]:.2f})")
            print(f"  Goal:          ({goal[0]:.2f}, {goal[1]:.2f})")
            print(f"  Dist to goal:  {dist_to_goal:.2f}m")
            print(f"  Ref height:    {self.ref_pos_buf[0, 2].item():.2f}m")


# -------------------- Entry point --------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test waypoint_gpu target movement")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--no-markers", action="store_true", help="Disable visual markers (for debugging)")
    args = parser.parse_args()
    
    gs.init()
    
    print("\n" + "=" * 60)
    print("Starting Waypoint GPU Test...")
    print("Press Ctrl+C to stop")
    if args.no_markers:
        print("Visual markers disabled")
    print("=" * 60 + "\n")
    
    env = WaypointTestEnv(show_viewer=not args.headless, show_markers=not args.no_markers)
    
    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
        print(f"Total steps: {env.step_count}")
        print("=" * 60)

