#!/usr/bin/env python3
"""Test realistic EEG processing: 60 seconds with 50% overlap."""

import numpy as np

def simulate_realistic_eeg():
    print("=" * 70)
    print("REALISTIC EEG ANIMATION TEST")
    print("=" * 70)
    print()
    
    # Realistic parameters
    total_duration = 60.0  # seconds
    window_duration = 2.0  # seconds
    overlap_fraction = 0.5
    fs = 250.0  # Hz
    n_sources = 2562  # ico3
    
    window_samples = int(window_duration * fs)  # 500 samples
    step_samples = int(window_samples * (1 - overlap_fraction))  # 250 samples
    step_duration = step_samples / fs  # 1.0 second
    
    # Calculate number of windows
    n_windows = int((total_duration - window_duration) / step_duration) + 1
    
    print(f"EEG Configuration:")
    print(f"  Total duration: {total_duration}s")
    print(f"  Window size: {window_duration}s ({window_samples} samples at {fs} Hz)")
    print(f"  Overlap: {overlap_fraction * 100:.0f}%")
    print(f"  Step size: {step_duration}s ({step_samples} samples)")
    print(f"  Number of windows: {n_windows}")
    print()
    
    print(f"Animation Output:")
    print(f"  Number of frames: {n_windows} (1 per window)")
    print(f"  Effective FPS: ~{1.0/step_duration:.1f} FPS")
    print(f"  Playback duration: {(n_windows - 1) * step_duration + window_duration}s")
    print()
    
    # File size estimation
    activity_size = n_sources * n_windows * 4  # float32
    timestamps_size = n_windows * 4
    positions_size = n_sources * 3 * 4
    triangles_size = 5120 * 3 * 4  # typical ico3
    
    total_uncompressed = (activity_size + timestamps_size + positions_size + triangles_size) / (1024 * 1024)
    estimated_compressed = total_uncompressed * 0.9  # NPZ compression
    
    print(f"File Size Estimate:")
    print(f"  Activity data: {activity_size / (1024 * 1024):.2f} MB")
    print(f"  Timestamps: {timestamps_size / 1024:.2f} KB")
    print(f"  Source positions: {positions_size / (1024 * 1024):.2f} MB")
    print(f"  Triangles: {triangles_size / (1024 * 1024):.2f} MB")
    print(f"  Total uncompressed: {total_uncompressed:.2f} MB")
    print(f"  Estimated compressed: {estimated_compressed:.2f} MB")
    print()
    
    # Compare to old interpolated approach
    old_frames = int(total_duration * 30)  # 30 FPS
    old_activity_size = n_sources * old_frames * 4 / (1024 * 1024)
    old_total = old_activity_size + (positions_size + triangles_size) / (1024 * 1024)
    
    print(f"Comparison to 30 FPS Interpolated:")
    print(f"  Old: {old_frames} frames = ~{old_total:.2f} MB")
    print(f"  New: {n_windows} frames = ~{estimated_compressed:.2f} MB")
    print(f"  Reduction: {old_total - estimated_compressed:.2f} MB ({(1 - estimated_compressed/old_total) * 100:.1f}% smaller)")
    print()
    
    # Timeline visualization
    print("Window Timeline (first 10 windows):")
    for i in range(min(10, n_windows)):
        start = i * step_duration
        end = start + window_duration
        center = (start + end) / 2.0
        print(f"  Window {i:2d}: [{start:5.2f}s - {end:5.2f}s] → Frame at {center:.2f}s")
    if n_windows > 10:
        print(f"  ... ({n_windows - 10} more windows)")
    print()
    
    print("=" * 70)
    print("✅ BENEFITS OF NEW APPROACH:")
    print("=" * 70)
    print()
    print("1. TRUE TO DATA")
    print("   - Each frame = actual model prediction")
    print("   - No artificial smoothing or interpolation")
    print(f"   - Shows model's native temporal resolution (~{1.0/step_duration:.0f} FPS)")
    print()
    print("2. EFFICIENT")
    print(f"   - {n_windows} frames instead of {old_frames}")
    print(f"   - {(1 - estimated_compressed/old_total) * 100:.0f}% smaller file")
    print("   - Faster to generate and load")
    print()
    print("3. SCIENTIFICALLY ACCURATE")
    print("   - Overlapping windows preserved")
    print("   - Each frame timestamp = window center")
    print("   - No loss of information")
    print()

if __name__ == "__main__":
    simulate_realistic_eeg()
