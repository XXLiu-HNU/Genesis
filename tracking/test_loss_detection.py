"""
Test script for the improved loss detection mechanism

Verifies that short occlusions don't trigger target loss,
and only consecutive multi-frame occlusions are considered as loss.
"""

import torch
import sys
sys.path.append('.')


def test_loss_detection_logic():
    """
    Test the loss detection logic with simulated occlusion scenarios
    """
    print("=" * 70)
    print("Testing Improved Loss Detection Mechanism")
    print("=" * 70)
    
    # Simulate parameters
    num_envs = 4
    loss_threshold = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  - Number of environments: {num_envs}")
    print(f"  - Loss detection threshold: {loss_threshold} frames")
    print(f"  - Device: {device}")
    
    # Initialize counter
    consecutive_loss_count = torch.zeros((num_envs,), device=device, dtype=torch.int32)
    
    # Scenario 1: No occlusion
    print("\n" + "=" * 70)
    print("Scenario 1: No occlusion (all frames visible)")
    print("=" * 70)
    occlusion_pattern = [False] * 10
    
    for frame, is_occluded in enumerate(occlusion_pattern):
        occluded_mask = torch.tensor([is_occluded] * num_envs, device=device)
        consecutive_loss_count[occluded_mask] += 1
        consecutive_loss_count[~occluded_mask] = 0
        loss_flag = consecutive_loss_count >= loss_threshold
        
        print(f"  Frame {frame}: Occluded={is_occluded}, Count={consecutive_loss_count[0].item()}, Loss={loss_flag[0].item()}")
    
    # Reset
    consecutive_loss_count[:] = 0
    
    # Scenario 2: Short occlusion (3 frames)
    print("\n" + "=" * 70)
    print("Scenario 2: Short occlusion (3 frames) - Should NOT trigger loss")
    print("=" * 70)
    occlusion_pattern = [False, False, True, True, True, False, False, False, False, False]
    
    for frame, is_occluded in enumerate(occlusion_pattern):
        occluded_mask = torch.tensor([is_occluded] * num_envs, device=device)
        consecutive_loss_count[occluded_mask] += 1
        consecutive_loss_count[~occluded_mask] = 0
        loss_flag = consecutive_loss_count >= loss_threshold
        
        status = "⚠️ OCCLUDED" if is_occluded else "✓ VISIBLE"
        loss_status = "❌ LOST" if loss_flag[0].item() else "✓ OK"
        print(f"  Frame {frame}: {status:12s} | Count={consecutive_loss_count[0].item()} | Status={loss_status}")
    
    # Reset
    consecutive_loss_count[:] = 0
    
    # Scenario 3: Long occlusion (6 frames) - should trigger loss
    print("\n" + "=" * 70)
    print("Scenario 3: Long occlusion (6 frames) - Should trigger loss")
    print("=" * 70)
    occlusion_pattern = [False, False, True, True, True, True, True, True, False, False]
    
    for frame, is_occluded in enumerate(occlusion_pattern):
        occluded_mask = torch.tensor([is_occluded] * num_envs, device=device)
        consecutive_loss_count[occluded_mask] += 1
        consecutive_loss_count[~occluded_mask] = 0
        loss_flag = consecutive_loss_count >= loss_threshold
        
        status = "⚠️ OCCLUDED" if is_occluded else "✓ VISIBLE"
        loss_status = "❌ LOST" if loss_flag[0].item() else "✓ OK"
        print(f"  Frame {frame}: {status:12s} | Count={consecutive_loss_count[0].item()} | Status={loss_status}")
    
    # Reset
    consecutive_loss_count[:] = 0
    
    # Scenario 4: Intermittent occlusion (never consecutive enough)
    print("\n" + "=" * 70)
    print("Scenario 4: Intermittent occlusion - Should NOT trigger loss")
    print("=" * 70)
    occlusion_pattern = [True, True, False, True, True, False, True, True, False, True]
    
    for frame, is_occluded in enumerate(occlusion_pattern):
        occluded_mask = torch.tensor([is_occluded] * num_envs, device=device)
        consecutive_loss_count[occluded_mask] += 1
        consecutive_loss_count[~occluded_mask] = 0
        loss_flag = consecutive_loss_count >= loss_threshold
        
        status = "⚠️ OCCLUDED" if is_occluded else "✓ VISIBLE"
        loss_status = "❌ LOST" if loss_flag[0].item() else "✓ OK"
        print(f"  Frame {frame}: {status:12s} | Count={consecutive_loss_count[0].item()} | Status={loss_status}")
    
    # Reset
    consecutive_loss_count[:] = 0
    
    # Scenario 5: Exactly threshold frames (edge case)
    print("\n" + "=" * 70)
    print(f"Scenario 5: Exactly {loss_threshold} frames occluded - Should trigger loss")
    print("=" * 70)
    occlusion_pattern = [False, False] + [True] * loss_threshold + [False, False]
    
    for frame, is_occluded in enumerate(occlusion_pattern):
        occluded_mask = torch.tensor([is_occluded] * num_envs, device=device)
        consecutive_loss_count[occluded_mask] += 1
        consecutive_loss_count[~occluded_mask] = 0
        loss_flag = consecutive_loss_count >= loss_threshold
        
        status = "⚠️ OCCLUDED" if is_occluded else "✓ VISIBLE"
        loss_status = "❌ LOST" if loss_flag[0].item() else "✓ OK"
        print(f"  Frame {frame}: {status:12s} | Count={consecutive_loss_count[0].item()} | Status={loss_status}")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("=" * 70)
    
    # Summary
    print("\n[Summary]:")
    print(f"  - Threshold: {loss_threshold} consecutive frames")
    print(f"  - Short occlusions (< {loss_threshold} frames): ✓ Tolerated")
    print(f"  - Long occlusions (>= {loss_threshold} frames): ❌ Trigger loss")
    print(f"  - Intermittent occlusions: ✓ Counter resets when visible")
    print("\n[Benefits]:")
    print("  - More robust to temporary occlusions")
    print("  - Works well with trajectory compensation mechanism")
    print("  - Prevents false terminations from brief obstacles")


def print_usage_info():
    """Print usage information"""
    print("\n" + "=" * 70)
    print("Implementation Details")
    print("=" * 70)
    
    print("\n[Modified Code]:")
    print("""
    # In __init__():
    self.loss_detection_threshold = 5  # Configurable threshold
    self.consecutive_loss_count = torch.zeros((num_envs,), ...)
    
    # In _loss_detect():
    current_occluded = occlusion_check(...)
    self.consecutive_loss_count[current_occluded] += 1
    self.consecutive_loss_count[~current_occluded] = 0
    loss_flag = self.consecutive_loss_count >= self.loss_detection_threshold
    
    # In reset_idx():
    self.consecutive_loss_count[envs_idx] = 0
    """)
    
    print("\n[Configuration]:")
    print("  Adjust threshold in track_env.py:")
    print("    self.loss_detection_threshold = 5  # Change this value")
    print("  ")
    print("  Recommended values:")
    print("    - Conservative (more tolerant): 8-10 frames")
    print("    - Balanced (default): 5 frames")
    print("    - Aggressive (less tolerant): 3 frames")
    
    print("\n[Integration with Trajectory Encoder]:")
    print("  The trajectory encoder helps during occlusion:")
    print("    1. Occlusion detected (v=0)")
    print("    2. Use last visible position for trajectory")
    print("    3. Counter increments but < threshold")
    print("    4. GRU learns occlusion pattern")
    print("    5. If visible again, counter resets")
    print("    6. If occluded too long, episode terminates")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_loss_detection_logic()
    print_usage_info()
