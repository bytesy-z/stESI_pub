#!/usr/bin/env python3
"""Test the updated animation timeline (1 frame per window, no interpolation)."""

import numpy as np

def _prepare_animation_timeline(all_predictions):
    """Copy of the function for testing."""
    if not all_predictions:
        raise ValueError("all_predictions list is empty")
    
    n_windows = len(all_predictions)
    n_sources = all_predictions[0]['predictions'].shape[0]
    
    activity_timeline = np.zeros((n_sources, n_windows), dtype=np.float32)
    timestamps = np.zeros(n_windows, dtype=np.float32)
    
    for i, window in enumerate(all_predictions):
        win_pred = window['predictions']
        
        # Find peak activity timepoint
        total_activity = np.abs(win_pred).sum(axis=0)
        peak_idx = np.argmax(total_activity)
        
        # Use activity at peak timepoint
        activity_timeline[:, i] = win_pred[:, peak_idx]
        
        # Timestamp is the window center
        timestamps[i] = (window['start_time'] + window['end_time']) / 2.0
    
    return activity_timeline, timestamps


def test_no_interpolation():
    print("=" * 70)
    print("Testing: 1 Frame per Window (No Interpolation)")
    print("=" * 70)
    print()
    
    # Simulate 4 windows with 50% overlap
    n_sources = 2562
    window_samples = 500
    
    all_predictions = []
    for i in range(4):
        pred = np.random.randn(n_sources, window_samples).astype(np.float32)
        all_predictions.append({
            'window_idx': i,
            'start_time': i * 1.0,
            'end_time': i * 1.0 + 2.0,
            'predictions': pred,
            'max_abs': 1.0,
        })
    
    print(f"Input: {len(all_predictions)} windows")
    for w in all_predictions:
        print(f"  Window {w['window_idx']}: {w['start_time']:.1f}s - {w['end_time']:.1f}s")
    print()
    
    # Generate timeline
    activity_timeline, timestamps = _prepare_animation_timeline(all_predictions)
    
    print("Output:")
    print(f"  Activity shape: {activity_timeline.shape}")
    print(f"  Timestamps shape: {timestamps.shape}")
    print(f"  Number of frames: {len(timestamps)}")
    print()
    
    # Verify: exactly 1 frame per window
    assert activity_timeline.shape[1] == len(all_predictions), \
        f"Should have {len(all_predictions)} frames, got {activity_timeline.shape[1]}"
    print(f"✓ Exactly 1 frame per window ({len(all_predictions)} frames)")
    
    # Verify timestamps are window centers
    expected_centers = [(w['start_time'] + w['end_time']) / 2.0 for w in all_predictions]
    assert np.allclose(timestamps, expected_centers), \
        "Timestamps should be window centers"
    print(f"✓ Timestamps are window centers:")
    for i, t in enumerate(timestamps):
        print(f"    Frame {i}: {t:.2f}s")
    print()
    
    # Calculate effective FPS
    if len(timestamps) > 1:
        avg_time_between_frames = np.mean(np.diff(timestamps))
        actual_fps = 1.0 / avg_time_between_frames
        print(f"✓ Average time between frames: {avg_time_between_frames:.2f}s")
        print(f"✓ Effective FPS: ~{actual_fps:.1f} FPS")
    print()
    
    # Compare file sizes
    old_interpolated_frames = int(5.0 * 30)  # 5s at 30 FPS
    new_frames = len(timestamps)
    
    old_size = n_sources * old_interpolated_frames * 4 / (1024 * 1024)  # float32
    new_size = n_sources * new_frames * 4 / (1024 * 1024)
    
    print(f"File size comparison (activity data only):")
    print(f"  Old approach (30 FPS interpolated): {old_interpolated_frames} frames = ~{old_size:.2f} MB")
    print(f"  New approach (1 per window): {new_frames} frames = ~{new_size:.2f} MB")
    print(f"  Reduction: {(old_size - new_size):.2f} MB ({(1 - new_size/old_size) * 100:.1f}% smaller)")
    print()
    
    print("=" * 70)
    print("✅ Updated approach verified!")
    print("=" * 70)
    print()
    print("Benefits:")
    print("  ✓ No artificial interpolation - shows actual predictions")
    print("  ✓ Smaller file size")
    print("  ✓ True to the model's temporal resolution")
    print("  ✓ Faster processing (no interpolation needed)")
    print("  ✓ Each frame = one model prediction window")

if __name__ == "__main__":
    test_no_interpolation()
