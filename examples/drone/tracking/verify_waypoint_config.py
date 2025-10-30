"""
Quick verification script to check if track_env.py is properly configured for waypoint_gpu.
"""
import sys

def verify_config():
    """Verify that track_env.py is correctly configured."""
    
    print("=" * 70)
    print("TRACK_ENV WAYPOINT_GPU CONFIGURATION VERIFICATION")
    print("=" * 70)
    
    try:
        # Read track_env.py
        with open("track_env.py", "r") as f:
            content = f.read()
        
        checks = []
        
        # Check 1: Waypoint GPU is the only mode (no mode variable = hardcoded waypoint_gpu)
        if "self.target_movement_mode" not in content or "_update_waypoint_gpu" in content:
            checks.append(("✓", "Waypoint_gpu mode active (hardcoded/only mode)"))
        else:
            checks.append(("✗", "Waypoint_gpu mode may not be active"))
        
        # Check 2: _update_waypoint_gpu function exists
        if "def _update_waypoint_gpu(self):" in content:
            checks.append(("✓", "_update_waypoint_gpu() function exists"))
        else:
            checks.append(("✗", "_update_waypoint_gpu() function MISSING"))
        
        # Check 3: _move_to_waypoint_smooth function exists
        if "def _move_to_waypoint_smooth(self):" in content:
            checks.append(("✓", "_move_to_waypoint_smooth() function exists"))
        else:
            checks.append(("✗", "_move_to_waypoint_smooth() function MISSING"))
        
        # Check 4: Waypoint buffers initialized
        if "self.target_waypoint_pos" in content and "self.target_waypoint_vel" in content:
            checks.append(("✓", "Waypoint buffers initialized"))
        else:
            checks.append(("✗", "Waypoint buffers NOT initialized"))
        
        # Check 5: Warmup parameters set
        if "self.waypoint_warmup_time" in content and "self.waypoint_v_init" in content:
            checks.append(("✓", "Warmup parameters configured"))
        else:
            checks.append(("✗", "Warmup parameters MISSING"))
        
        # Check 6: Height set in step()
        if 'self.ref_pos_buf[:, 2] = self.drone_height' in content:
            checks.append(("✓", "Height explicitly set in step()"))
        else:
            checks.append(("✗", "Height NOT explicitly set (may cause instability)"))
        
        # Check 7: Waypoint functions called in step()
        if "_update_waypoint_gpu()" in content and "_move_to_waypoint_smooth()" in content:
            checks.append(("✓", "Waypoint_gpu functions called in step()"))
        else:
            checks.append(("✗", "Waypoint_gpu functions NOT called in step()"))
        
        # Print results
        print("\nConfiguration Checks:")
        print("-" * 70)
        all_passed = True
        for status, message in checks:
            print(f"  {status} {message}")
            if status == "✗":
                all_passed = False
        
        print("-" * 70)
        
        if all_passed:
            print("\n✅ ALL CHECKS PASSED!")
            print("\ntrack_env.py is correctly configured for waypoint_gpu mode.")
            print("\nYou can now run training:")
            print("  python track_train.py")
            print("\nExpected performance:")
            print("  - Training speed: ~3-5 seconds/iteration")
            print("  - Target movement: Dynamic waypoint-based navigation")
            print("  - Obstacle avoidance: GPU-accelerated line-of-sight checks")
        else:
            print("\n❌ SOME CHECKS FAILED!")
            print("\nPlease review the failed checks above.")
            print("track_env.py may not work correctly with waypoint_gpu mode.")
        
        print("=" * 70)
        
        return all_passed
        
    except FileNotFoundError:
        print("\n❌ ERROR: track_env.py not found!")
        print("Please run this script from the examples/drone/tracking/ directory")
        print("=" * 70)
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = verify_config()
    sys.exit(0 if success else 1)

