"""
Trajectory encoder visualization script

Demonstrates feature encoding effects for different motion patterns and occlusion scenarios
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append('.')

from tracking.track_env import TrajectoryEncoder


def create_trajectory_patterns(history_len=50, stride=5, device='cuda'):
    """
    Create different trajectory patterns for demonstration
    
    Args:
        history_len: Full history length (50 frames = 0.5s)
        stride: Sampling stride (5 frames)
        device: torch device
    
    Returns:
        dict: {pattern_name: (full_trajectory, sampled_trajectory)}
    """
    patterns = {}
    seq_len = history_len // stride
    
    # Pattern 1: Uniform linear motion (always visible)
    traj_full = torch.zeros((history_len, 4), device=device)
    for t in range(history_len):
        traj_full[t, 0] = 2.0 + 0.02 * t  # x increases linearly
        traj_full[t, 1] = 1.0  # y constant
        traj_full[t, 2] = 1.0  # z constant
        traj_full[t, 3] = 1.0  # always visible
    sampled_indices = torch.arange(stride - 1, history_len, stride, device=device)
    traj_sampled = traj_full[sampled_indices]
    patterns['Uniform Linear'] = (traj_full, traj_sampled)
    
    # Pattern 2: Circular motion (always visible)
    traj_full = torch.zeros((history_len, 4), device=device)
    radius = 2.0
    for t in range(history_len):
        angle = 2 * np.pi * t / history_len
        traj_full[t, 0] = radius * np.cos(angle)
        traj_full[t, 1] = radius * np.sin(angle)
        traj_full[t, 2] = 1.0
        traj_full[t, 3] = 1.0
    traj_sampled = traj_full[sampled_indices]
    patterns['Circular'] = (traj_full, traj_sampled)
    
    # Pattern 3: Accelerated motion (always visible)
    traj_full = torch.zeros((history_len, 4), device=device)
    for t in range(history_len):
        traj_full[t, 0] = 1.0 + 0.0004 * t * t  # acceleration
        traj_full[t, 1] = 1.0
        traj_full[t, 2] = 1.0
        traj_full[t, 3] = 1.0
    traj_sampled = traj_full[sampled_indices]
    patterns['Accelerated'] = (traj_full, traj_sampled)
    
    # Pattern 4: Uniform linear + short occlusion (frames 20-25 in full history)
    traj_full, _ = patterns['Uniform Linear']
    traj_full = traj_full.clone()
    traj_full[20:26, 3] = 0.0  # occluded
    traj_full[20:26, :3] = traj_full[19:20, :3]  # use previous frame position
    traj_sampled = traj_full[sampled_indices]
    patterns['Linear+ShortOcclusion'] = (traj_full, traj_sampled)
    
    # Pattern 5: Uniform linear + long occlusion (frames 15-40 in full history)
    traj_full, _ = patterns['Uniform Linear']
    traj_full = traj_full.clone()
    traj_full[15:41, 3] = 0.0  # long occlusion
    traj_full[15:41, :3] = traj_full[14:15, :3]
    traj_sampled = traj_full[sampled_indices]
    patterns['Linear+LongOcclusion'] = (traj_full, traj_sampled)
    
    # Pattern 6: Circular motion + intermittent occlusion
    traj_full, _ = patterns['Circular']
    traj_full = traj_full.clone()
    traj_full[10:15, 3] = 0.0
    traj_full[10:15, :3] = traj_full[9:10, :3]
    traj_full[25:30, 3] = 0.0
    traj_full[25:30, :3] = traj_full[24:25, :3]
    traj_sampled = traj_full[sampled_indices]
    patterns['Circular+IntermittentOcclusion'] = (traj_full, traj_sampled)
    
    # Pattern 7: Stationary target (always visible)
    traj_full = torch.zeros((history_len, 4), device=device)
    traj_full[:, 0] = 2.0  # fixed position
    traj_full[:, 1] = 1.5
    traj_full[:, 2] = 1.0
    traj_full[:, 3] = 1.0
    traj_sampled = traj_full[sampled_indices]
    patterns['Stationary'] = (traj_full, traj_sampled)
    
    # Pattern 8: Zigzag motion (always visible)
    traj_full = torch.zeros((history_len, 4), device=device)
    for t in range(history_len):
        traj_full[t, 0] = 1.0 + 0.02 * t
        traj_full[t, 1] = 1.0 + 0.2 * ((-1) ** (t // 5))  # oscillating
        traj_full[t, 2] = 1.0
        traj_full[t, 3] = 1.0
    traj_sampled = traj_full[sampled_indices]
    patterns['Zigzag'] = (traj_full, traj_sampled)
    
    return patterns


def visualize_trajectories(patterns, encoder):
    """
    Visualize different trajectory patterns and their encoded features
    """
    n_patterns = len(patterns)
    pattern_names = list(patterns.keys())
    
    # Compute features (using sampled trajectories)
    features = {}
    for name, (traj_full, traj_sampled) in patterns.items():
        with torch.no_grad():
            feat = encoder(traj_sampled.unsqueeze(0)).squeeze(0)  # (32,)
        features[name] = feat.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, n_patterns, figure=fig, hspace=0.4, wspace=0.3)
    
    # First row: XY trajectory
    for i, name in enumerate(pattern_names):
        ax = fig.add_subplot(gs[0, i])
        traj_full, traj_sampled = patterns[name]
        traj_full_np = traj_full.cpu().numpy()
        traj_sampled_np = traj_sampled.cpu().numpy()
        
        # Plot full trajectory (lighter)
        visible = traj_full_np[:, 3] > 0.5
        ax.plot(traj_full_np[visible, 0], traj_full_np[visible, 1], 'b-', alpha=0.3, linewidth=1, label='Full (Visible)')
        if not visible.all():
            ax.plot(traj_full_np[~visible, 0], traj_full_np[~visible, 1], 'r-', alpha=0.3, linewidth=1, label='Full (Occluded)')
        
        # Plot sampled points (emphasized)
        visible_sampled = traj_sampled_np[:, 3] > 0.5
        ax.scatter(traj_sampled_np[visible_sampled, 0], traj_sampled_np[visible_sampled, 1], 
                  c='blue', s=50, marker='o', label='Sampled (Visible)', zorder=4, edgecolor='black')
        if not visible_sampled.all():
            ax.scatter(traj_sampled_np[~visible_sampled, 0], traj_sampled_np[~visible_sampled, 1], 
                      c='red', s=50, marker='x', label='Sampled (Occluded)', zorder=4)
        
        # Mark start and end points
        ax.scatter(traj_full_np[0, 0], traj_full_np[0, 1], c='green', s=100, marker='s', 
                  label='Start', zorder=5, edgecolor='black')
        ax.scatter(traj_full_np[-1, 0], traj_full_np[-1, 1], c='red', s=100, marker='^', 
                  label='End', zorder=5, edgecolor='black')
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.axis('equal')
    
    # Second row: Z coordinate and visibility time series
    for i, name in enumerate(pattern_names):
        ax = fig.add_subplot(gs[1, i])
        traj_full, traj_sampled = patterns[name]
        traj_full_np = traj_full.cpu().numpy()
        
        time_steps = np.arange(len(traj_full_np))
        
        # Plot Z coordinate
        ax.plot(time_steps, traj_full_np[:, 2], 'g-', alpha=0.5, linewidth=1, label='Z Position')
        ax.set_ylabel('Z Position (m)', color='g')
        ax.tick_params(axis='y', labelcolor='g')
        
        # Create second y-axis for visibility
        ax2 = ax.twinx()
        visibility = traj_full_np[:, 3]
        ax2.fill_between(time_steps, 0, visibility, alpha=0.3, color='blue', label='Visibility')
        ax2.plot(time_steps, visibility, 'b-', linewidth=2)
        ax2.set_ylabel('Visibility', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim(-0.1, 1.1)
        
        # Mark sampled frames with vertical lines
        sampled_frames = np.arange(4, len(traj_full_np), 5)
        for sf in sampled_frames:
            ax.axvline(sf, color='orange', alpha=0.3, linestyle='--', linewidth=1)
        
        ax.set_xlabel('Time Step (Frame)')
        ax.grid(True, alpha=0.3)
        ax.set_title('Height & Visibility (Stride=5)', fontsize=10)
    
    # Third row: Feature vector heatmap
    feature_matrix = np.array([features[name] for name in pattern_names])
    
    ax = fig.add_subplot(gs[2, :])
    im = ax.imshow(feature_matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-1.0, vmax=1.0, interpolation='nearest')
    ax.set_yticks(range(n_patterns))
    ax.set_yticklabels(pattern_names)
    ax.set_xlabel('Feature Dimension (32-dim GRU Hidden State)', fontsize=12)
    ax.set_ylabel('Trajectory Pattern', fontsize=12)
    ax.set_title('Trajectory Encoding Feature Comparison (Darker = Larger Feature Value)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, aspect=40)
    cbar.set_label('Feature Value', fontsize=10)
    
    plt.suptitle('Target Trajectory Encoder - Multi-Pattern Visualization Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def analyze_feature_similarity(patterns, encoder):
    """
    Analyze feature similarity between different patterns
    """
    pattern_names = list(patterns.keys())
    n = len(pattern_names)
    
    # Compute features (using sampled trajectories)
    features = []
    for name in pattern_names:
        _, traj_sampled = patterns[name]
        with torch.no_grad():
            feat = encoder(traj_sampled.unsqueeze(0)).squeeze(0)
        features.append(feat)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            fi = features[i]
            fj = features[j]
            # Cosine similarity
            similarity = torch.sum(fi * fj) / (torch.norm(fi) * torch.norm(fj) + 1e-8)
            similarity_matrix[i, j] = similarity.cpu().item()
    
    # Visualize similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pattern_names, rotation=45, ha='right')
    ax.set_yticklabels(pattern_names)
    
    # Display values in each cell
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Trajectory Pattern Feature Similarity Matrix (Cosine Similarity)', 
                fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='Similarity')
    plt.tight_layout()
    
    return fig


def main():
    """Main function"""
    print("=" * 70)
    print("Target Trajectory Encoder - Visualization Analysis")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create encoder
    encoder = TrajectoryEncoder(input_size=4, hidden_size=32, device=device)
    encoder.eval()
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    
    # Create trajectory patterns
    print("\nGenerating trajectory patterns...")
    print("  - Full history: 50 frames (0.5s)")
    print("  - Stride: 5 frames")
    print("  - Sampled sequence: 10 frames")
    patterns = create_trajectory_patterns(history_len=50, stride=5, device=device)
    print(f"Generated {len(patterns)} different trajectory patterns")
    
    # Visualize trajectories and features
    print("\nPlotting trajectory visualization...")
    fig1 = visualize_trajectories(patterns, encoder)
    fig1.savefig('tracking/results/trajectory_encoding_patterns.png', dpi=150, bbox_inches='tight')
    print("  Saved to: tracking/results/trajectory_encoding_patterns.png")
    
    # Analyze feature similarity
    print("\nAnalyzing feature similarity...")
    fig2 = analyze_feature_similarity(patterns, encoder)
    fig2.savefig('tracking/results/trajectory_encoding_similarity.png', dpi=150, bbox_inches='tight')
    print("  Saved to: tracking/results/trajectory_encoding_similarity.png")
    
    print("\n" + "=" * 70)
    print("✓ Visualization completed!")
    print("=" * 70)
    
    # Print some statistics
    print("\n[Feature Statistics]:")
    for name, (traj_full, traj_sampled) in patterns.items():
        with torch.no_grad():
            feat = encoder(traj_sampled.unsqueeze(0)).squeeze(0)
        print(f"  {name:30s}: mean={feat.mean():.4f}, std={feat.std():.4f}, "
              f"range=[{feat.min():.4f}, {feat.max():.4f}]")
    
    print("\nNote: Uncomment plt.show() to display figures")
    plt.show()


if __name__ == "__main__":
    # Ensure results directory exists
    import os
    os.makedirs('tracking/results', exist_ok=True)
    
    main()
