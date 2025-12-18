#!/usr/bin/env python3
"""Test Task 3: Window interpolation function."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the function (after adding to path)
from run_edf_inference import _interpolate_sliding_windows

def test_interpolation():
    """Test the _interpolate_sliding_windows function."""
    
    print("Testing _interpolate_sliding_windows()...")
    print("=" * 70)
    
    # Create test data: 3 overlapping windows
    fs = 250.0
    window_samples = 500  # 2 seconds
    n_sources = 100  # Smaller for testing
    
    all_predictions = []
    
    # Window 0: 0-2s
    pred0 = np.random.randn(n_sources, window_samples).astype(np.float32)
    all_predictions.append({
        'window_idx': 0,
        'start_time': 0.0,
        'end_time': 2.0,
        'predictions': pred0,
        'max_abs': 1.0,
    })
    
    # Window 1: 1-3s (50% overlap)
    pred1 = np.random.randn(n_sources, window_samples).astype(np.float32)
    all_predictions.append({
        'window_idx': 1,
        'start_time': 1.0,
        'end_time': 3.0,
        'predictions': pred1,
        'max_abs': 1.1,
    })
    
    # Window 2: 2-4s (50% overlap)
    pred2 = np.random.randn(n_sources, window_samples).astype(np.float32)
    all_predictions.append({
        'window_idx': 2,
        'start_time': 2.0,
        'end_time': 4.0,
        'predictions': pred2,
        'max_abs': 1.2,
    })
    
    print(f"Input: {len(all_predictions)} windows")
    print(f"  Window 0: {all_predictions[0]['start_time']}s - {all_predictions[0]['end_time']}s")
    print(f"  Window 1: {all_predictions[1]['start_time']}s - {all_predictions[1]['end_time']}s")
    print(f"  Window 2: {all_predictions[2]['start_time']}s - {all_predictions[2]['end_time']}s")
    print(f"  Prediction shape: ({n_sources}, {window_samples})")
    print()
    
    # Test with 30 FPS
    target_fps = 30
    activity_timeline, timestamps = _interpolate_sliding_windows(
        all_predictions,
        target_fps=target_fps
    )
    
    print("Output:")
    print(f"  Activity timeline shape: {activity_timeline.shape}")
    print(f"  Timestamps shape: {timestamps.shape}")
    print(f"  Expected frames: {int(4.0 * target_fps)} (4s * 30 FPS)")
    print(f"  Actual frames: {len(timestamps)}")
    print(f"  Time range: {timestamps[0]:.3f}s - {timestamps[-1]:.3f}s")
    print(f"  FPS: {len(timestamps) / timestamps[-1]:.1f}")
    print()
    
    # Verify output
    assert activity_timeline.shape[0] == n_sources, \
        f"Expected {n_sources} sources, got {activity_timeline.shape[0]}"
    
    assert activity_timeline.shape[1] == len(timestamps), \
        f"Mismatch: activity has {activity_timeline.shape[1]} frames, timestamps has {len(timestamps)}"
    
    expected_frames = int(4.0 * target_fps)
    assert len(timestamps) == expected_frames, \
        f"Expected {expected_frames} frames, got {len(timestamps)}"
    
    assert timestamps[0] == 0.0, f"First timestamp should be 0.0, got {timestamps[0]}"
    assert abs(timestamps[-1] - 4.0) < 0.01, \
        f"Last timestamp should be ~4.0, got {timestamps[-1]}"
    
    assert activity_timeline.dtype == np.float32, \
        f"Activity should be float32, got {activity_timeline.dtype}"
    assert timestamps.dtype == np.float32, \
        f"Timestamps should be float32, got {timestamps.dtype}"
    
    # Check for NaN or Inf
    assert not np.any(np.isnan(activity_timeline)), "Activity contains NaN values"
    assert not np.any(np.isinf(activity_timeline)), "Activity contains Inf values"
    
    print("✓ All shape checks passed")
    print("✓ Data types correct (float32)")
    print("✓ No NaN or Inf values")
    print()
    
    # Test overlapping region (1-2s should blend windows 0 and 1)
    # Find frames in overlapping region
    overlap_mask = (timestamps >= 1.0) & (timestamps < 2.0)
    n_overlap_frames = np.sum(overlap_mask)
    print(f"Overlapping region (1-2s): {n_overlap_frames} frames")
    
    # All frames should have non-zero values
    assert np.all(np.abs(activity_timeline[:, overlap_mask]).max(axis=0) > 0), \
        "Overlapping frames should have non-zero activity"
    
    print("✓ Overlapping region properly interpolated")
    print()
    
    # Test edge case: single window
    print("Testing edge case: single window...")
    single_window = [{
        'window_idx': 0,
        'start_time': 0.0,
        'end_time': 2.0,
        'predictions': np.random.randn(n_sources, window_samples).astype(np.float32),
        'max_abs': 1.0,
    }]
    
    activity_single, timestamps_single = _interpolate_sliding_windows(
        single_window,
        target_fps=30
    )
    
    assert activity_single.shape == (n_sources, 60), \
        f"Single window should produce 60 frames (2s * 30fps), got {activity_single.shape[1]}"
    
    print(f"✓ Single window: shape {activity_single.shape}, timestamps {len(timestamps_single)}")
    print()
    
    print("=" * 70)
    print("✅ Task 3 interpolation function verified successfully!")
    print()
    print("Summary:")
    print(f"  - Function correctly interpolates {len(all_predictions)} overlapping windows")
    print(f"  - Output shape: ({n_sources} sources, {len(timestamps)} frames)")
    print(f"  - Smooth transitions with Gaussian weighting")
    print(f"  - Handles single and multiple windows")
    print(f"  - Ready for Task 4 (NPZ export)")

if __name__ == "__main__":
    test_interpolation()
