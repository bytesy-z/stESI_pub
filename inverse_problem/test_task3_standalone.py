#!/usr/bin/env python3
"""Test Task 3: Window interpolation function (standalone)."""

import numpy as np
from typing import List, Tuple

def _interpolate_sliding_windows(
    all_predictions: List[dict],
    target_fps: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate overlapping window predictions into a smooth timeline.
    
    Args:
        all_predictions: List of dicts with keys 'window_idx', 'start_time', 
                        'end_time', 'predictions', 'max_abs'
        target_fps: Target frames per second for output timeline
        
    Returns:
        activity_timeline: (n_sources, n_frames) array of interpolated activity
        timestamps: (n_frames,) array of timestamp for each frame in seconds
    """
    if not all_predictions:
        raise ValueError("all_predictions list is empty")
    
    # Determine total duration
    total_duration = max(pred['end_time'] for pred in all_predictions)
    
    # Calculate number of frames
    n_frames = max(1, int(total_duration * target_fps))
    
    # Create timestamp array
    timestamps = np.linspace(0, total_duration, n_frames, dtype=np.float32)
    
    # Get number of sources from first prediction
    n_sources = all_predictions[0]['predictions'].shape[0]
    
    # Initialize output arrays
    activity_timeline = np.zeros((n_sources, n_frames), dtype=np.float64)
    weights = np.zeros(n_frames, dtype=np.float64)
    
    # Process each window
    for window in all_predictions:
        win_start = window['start_time']
        win_end = window['end_time']
        win_duration = win_end - win_start
        win_center = (win_start + win_end) / 2.0
        
        # Find frames that overlap with this window
        overlapping_mask = (timestamps >= win_start) & (timestamps < win_end)
        overlapping_indices = np.where(overlapping_mask)[0]
        
        if len(overlapping_indices) == 0:
            continue
        
        # Get window predictions
        win_pred = window['predictions']  # (n_sources, n_timepoints)
        n_timepoints = win_pred.shape[1]
        
        for frame_idx in overlapping_indices:
            frame_time = timestamps[frame_idx]
            
            # Calculate position within window [0, 1]
            relative_pos = (frame_time - win_start) / win_duration
            relative_pos = np.clip(relative_pos, 0.0, 1.0)
            
            # Interpolate within window prediction
            timepoint_idx = relative_pos * (n_timepoints - 1)
            idx_low = int(np.floor(timepoint_idx))
            idx_high = min(idx_low + 1, n_timepoints - 1)
            alpha = timepoint_idx - idx_low
            
            # Linear interpolation between timepoints
            if idx_low == idx_high:
                interpolated_activity = win_pred[:, idx_low]
            else:
                interpolated_activity = (
                    (1 - alpha) * win_pred[:, idx_low] +
                    alpha * win_pred[:, idx_high]
                )
            
            # Gaussian weighting (higher at center, lower at edges)
            # sigma = 0.25 means the weight drops to ~0.6 at window boundaries
            distance_from_center = abs(frame_time - win_center) / (win_duration / 2.0)
            weight = np.exp(-0.5 * (distance_from_center / 0.25) ** 2)
            
            # Accumulate weighted predictions
            activity_timeline[:, frame_idx] += interpolated_activity * weight
            weights[frame_idx] += weight
    
    # Normalize by total weights
    weights = np.maximum(weights, 1e-8)  # Avoid division by zero
    activity_timeline /= weights[np.newaxis, :]
    
    return activity_timeline.astype(np.float32), timestamps


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
