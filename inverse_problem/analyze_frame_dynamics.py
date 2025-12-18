#!/usr/bin/env python3
"""
Detailed flash analysis - examining frame-to-frame variation and normalization effects.

This script specifically looks at:
1. Frame-to-frame power jumps (which cause visual "flashes" even without outliers)
2. Per-frame vs global normalization effects
3. The min/max range per frame that drives visual appearance
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from models.cnn_1d import CNN1Dpl
from plot_1dcnn_inference_heatmap import (
    _ensure_within_repo,
    _find_default_model_checkpoint,
    _infer_model,
)


def analyze_frame_dynamics(
    edf_path: Path,
    annotation_csv: Optional[Path] = None,
    simu_name: str = "mes_debug",
    subject: str = "fsaverage",
    orientation: str = "constrained",
    electrode_montage: str = "standard_1020",
    source_space: str = "ico3",
    model_path: Optional[Path] = None,
    inter_layer: int = 4096,
    kernel_size: int = 5,
    train_loss: str = "cosine",
    window_seconds: Optional[float] = None,
    overlap_fraction: float = 0.5,
    max_windows: Optional[int] = None,
    output_dir: Optional[Path] = None,
):
    """Analyze frame-to-frame dynamics to understand flash behavior."""
    
    repo_root = REPO_ROOT
    
    # Setup paths and load config
    simu_root = repo_root / "simulation" / subject
    config_path = (
        simu_root / orientation / electrode_montage / source_space
        / "simu" / simu_name / f"{simu_name}{source_space}_config.json"
    )
    
    with config_path.open() as f:
        general_config = json.load(f)
    
    general_config["simu_name"] = simu_name
    general_config["electrode_space"]["electrode_montage"] = electrode_montage
    general_config.setdefault("eeg_snr", "infdb")
    
    # Load head model
    folders = FolderStructure(str(simu_root), general_config)
    electrode_space_obj = HeadModel.ElectrodeSpace(folders, general_config)
    source_space_obj = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space_obj, source_space_obj, folders, subject)
    
    # Setup window parameters
    fs = float(general_config["rec_info"]["fs"])
    if window_seconds is None:
        window_samples = int(general_config["rec_info"]["n_times"])
        window_seconds = window_samples / fs
    else:
        window_samples = max(1, int(round(window_seconds * fs)))
    
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_fraction))))
    
    # Load model
    if model_path:
        model_checkpoint = Path(model_path)
    else:
        token = f"{simu_name}{source_space}_"
        model_checkpoint = _find_default_model_checkpoint(token, inter_layer)
    
    from diagnose_flashes import _load_model, _rename_channels_to_target, _build_full_montage_raw, _segment_signal
    
    model = _load_model(
        model_checkpoint,
        head_model.electrode_space.n_electrodes,
        head_model.source_space.n_sources,
        inter_layer,
        kernel_size,
    )
    
    # Load EDF
    print(f"Loading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick(picks="eeg")
    _rename_channels_to_target(raw, head_model.electrode_space.info.ch_names)
    
    try:
        raw.set_montage("standard_1020", match_case=False, verbose=False, on_missing="warn")
    except RuntimeError:
        pass
    
    raw.set_eeg_reference("average", projection=False, verbose=False)
    
    if not math.isclose(raw.info["sfreq"], fs, rel_tol=1e-6):
        raw.resample(fs, npad="auto", verbose=False)
    
    full_raw, _ = _build_full_montage_raw(
        raw, head_model.electrode_space.info, head_model.electrode_space.info.ch_names
    )
    full_raw.set_eeg_reference("average", projection=False, verbose=False)
    data = full_raw.get_data()
    
    leadfield = torch.from_numpy(head_model.fwd["sol"]["data"]).float()
    
    # Process windows
    all_predictions = []
    
    for seg_idx, (start_sample, segment) in enumerate(
        _segment_signal(data, window_samples, step_samples, pad=True, max_windows=max_windows)
    ):
        max_abs = float(np.abs(segment).max())
        if max_abs <= 0:
            continue
        
        eeg_tensor = torch.from_numpy((segment / max_abs).astype(np.float32))
        pred = _infer_model(
            model, eeg_tensor, train_loss,
            max_src_val=1.0, max_eeg_val=max_abs, leadfield=leadfield
        )
        
        all_predictions.append({
            'window_idx': seg_idx,
            'start_time': start_sample / fs,
            'end_time': (start_sample + window_samples) / fs,
            'predictions': pred.detach().cpu().numpy(),
            'max_abs': max_abs,
        })
    
    print(f"Processed {len(all_predictions)} windows")
    
    # Extract per-frame statistics
    n_windows = len(all_predictions)
    n_sources = all_predictions[0]['predictions'].shape[0]
    
    # Arrays to store per-frame metrics
    frame_max = np.zeros(n_windows)
    frame_min = np.zeros(n_windows)
    frame_mean = np.zeros(n_windows)
    frame_std = np.zeros(n_windows)
    frame_power = np.zeros(n_windows)
    frame_max_abs = np.zeros(n_windows)  # EEG max_abs used for normalization
    timestamps = np.zeros(n_windows)
    
    activity_timeline = np.zeros((n_sources, n_windows), dtype=np.float32)
    
    for i, window in enumerate(all_predictions):
        win_pred = window['predictions']
        total_activity = np.abs(win_pred).sum(axis=0)
        peak_idx = np.argmax(total_activity)
        
        frame_activity = win_pred[:, peak_idx]
        activity_timeline[:, i] = frame_activity
        
        frame_max[i] = np.max(frame_activity)
        frame_min[i] = np.min(frame_activity)
        frame_mean[i] = np.mean(frame_activity)
        frame_std[i] = np.std(frame_activity)
        frame_power[i] = np.sum(frame_activity ** 2)
        frame_max_abs[i] = window['max_abs']
        timestamps[i] = (window['start_time'] + window['end_time']) / 2.0
    
    # Calculate frame-to-frame jumps
    power_jumps = np.abs(np.diff(frame_power))
    max_jumps = np.abs(np.diff(frame_max))
    
    # Dynamic range per frame (what visualization sees)
    frame_range = frame_max - frame_min
    
    # Normalize power to see relative variation
    norm_power = frame_power / np.mean(frame_power)
    
    print("\n" + "="*60)
    print("FRAME-TO-FRAME DYNAMICS ANALYSIS")
    print("="*60)
    
    print(f"\nSource Power Statistics:")
    print(f"  Mean:   {np.mean(frame_power):.6f}")
    print(f"  Std:    {np.std(frame_power):.6f}")
    print(f"  Min:    {np.min(frame_power):.6f}")
    print(f"  Max:    {np.max(frame_power):.6f}")
    print(f"  Ratio (max/min): {np.max(frame_power)/np.min(frame_power):.2f}x")
    
    print(f"\nNormalized Power (relative to mean):")
    print(f"  Min:   {np.min(norm_power):.3f}x")
    print(f"  Max:   {np.max(norm_power):.3f}x")
    
    print(f"\nFrame-to-Frame Power Jumps:")
    print(f"  Mean jump:   {np.mean(power_jumps):.6f}")
    print(f"  Max jump:    {np.max(power_jumps):.6f}")
    print(f"  Frames with >50% jump: {np.sum(power_jumps > 0.5 * np.mean(frame_power))}")
    print(f"  Frames with >100% jump: {np.sum(power_jumps > np.mean(frame_power))}")
    
    # Check EEG normalization factor variation
    print(f"\nEEG Normalization Factor (max_abs) Variation:")
    print(f"  Mean:   {np.mean(frame_max_abs):.6f}")
    print(f"  Std:    {np.std(frame_max_abs):.6f}")
    print(f"  Ratio (max/min): {np.max(frame_max_abs)/np.min(frame_max_abs):.2f}x")
    
    # Check correlation between EEG max_abs and source power
    corr_maxabs_power = np.corrcoef(frame_max_abs, frame_power)[0, 1]
    print(f"  Correlation with source power: {corr_maxabs_power:.4f}")
    
    # Identify potential "flash" frames based on rapid changes
    # A "flash" might be a sudden jump in power followed by a drop
    jump_threshold = np.mean(power_jumps) + 2 * np.std(power_jumps)
    large_jump_frames = np.where(power_jumps > jump_threshold)[0]
    
    print(f"\nFrames with Large Power Jumps (>{jump_threshold:.4f}):")
    print(f"  Count: {len(large_jump_frames)}")
    if len(large_jump_frames) > 0:
        print(f"  Times (first 10): {timestamps[large_jump_frames[:10]]}")
    
    # Detect "spike-and-return" patterns (flash followed by normalization)
    spike_frames = []
    for i in range(1, len(frame_power) - 1):
        prev_power = frame_power[i-1]
        curr_power = frame_power[i]
        next_power = frame_power[i+1]
        
        # Check if this frame spikes up and then returns
        if curr_power > prev_power * 1.5 and curr_power > next_power * 1.5:
            spike_frames.append(i)
    
    print(f"\nSpike-and-Return Patterns (frame significantly higher than neighbors):")
    print(f"  Count: {len(spike_frames)}")
    if spike_frames:
        print(f"  Times: {timestamps[spike_frames[:10]]}")
    
    # Setup output
    if output_dir is None:
        output_dir = repo_root / "results" / "flash_diagnostic" / edf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot detailed dynamics
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    
    # 1. Source power over time
    ax = axes[0]
    ax.plot(timestamps, frame_power, 'b-', linewidth=0.8)
    ax.scatter(timestamps[large_jump_frames], frame_power[large_jump_frames], 
               c='r', s=30, zorder=5, label='Large jumps')
    if spike_frames:
        ax.scatter(timestamps[spike_frames], frame_power[spike_frames], 
                   c='orange', s=50, marker='^', zorder=6, label='Spike patterns')
    ax.set_ylabel('Source Power')
    ax.set_title('Source Power Over Time')
    ax.legend()
    
    # 2. Normalized power (relative to mean)
    ax = axes[1]
    ax.plot(timestamps, norm_power, 'g-', linewidth=0.8)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(1.5, color='r', linestyle='--', alpha=0.5, label='1.5x mean')
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='0.5x mean')
    ax.set_ylabel('Relative Power')
    ax.set_title('Normalized Source Power (1.0 = mean)')
    ax.legend()
    
    # 3. Frame-to-frame power jumps
    ax = axes[2]
    ax.plot(timestamps[1:], power_jumps, 'm-', linewidth=0.8)
    ax.axhline(jump_threshold, color='r', linestyle='--', label=f'Threshold ({jump_threshold:.4f})')
    ax.set_ylabel('Power Jump')
    ax.set_title('Frame-to-Frame Power Jumps')
    ax.legend()
    
    # 4. EEG normalization factor
    ax = axes[3]
    ax.plot(timestamps, frame_max_abs, 'c-', linewidth=0.8)
    ax.set_ylabel('EEG max_abs')
    ax.set_title('EEG Normalization Factor Per Window')
    
    # 5. Dynamic range per frame
    ax = axes[4]
    ax.plot(timestamps, frame_range, 'k-', linewidth=0.8)
    ax.set_ylabel('Max - Min')
    ax.set_xlabel('Time (s)')
    ax.set_title('Source Activity Dynamic Range Per Frame')
    
    plt.tight_layout()
    plot_path = output_dir / 'frame_dynamics.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved frame dynamics plot to: {plot_path}")
    
    # Plot comparison: global vs per-frame normalization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top row: what visualization sees with GLOBAL normalization
    global_min = np.min(activity_timeline)
    global_max = np.max(activity_timeline)
    
    # Create a simple visualization: sum of absolute activity per source
    source_means = np.abs(activity_timeline).mean(axis=1)
    
    # Show a few frames with global normalization
    sample_frames = [0, len(timestamps)//4, len(timestamps)//2, 3*len(timestamps)//4, -1]
    
    ax = axes[0, 0]
    for i, frame_idx in enumerate(sample_frames):
        frame_data = activity_timeline[:, frame_idx]
        ax.hist(frame_data, bins=50, alpha=0.3, label=f't={timestamps[frame_idx]:.1f}s')
    ax.set_xlabel('Source Activity')
    ax.set_ylabel('Count')
    ax.set_title('Source Activity Distribution (Sample Frames)')
    ax.legend()
    
    # Per-frame vs global max
    ax = axes[0, 1]
    per_frame_max = np.max(np.abs(activity_timeline), axis=0)
    ax.plot(timestamps, per_frame_max, 'b-', label='Per-frame max', linewidth=0.8)
    ax.axhline(global_max, color='r', linestyle='--', label=f'Global max ({global_max:.4f})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Max Absolute Activity')
    ax.set_title('Per-Frame Max vs Global Max')
    ax.legend()
    
    # Effect of per-frame normalization
    ax = axes[1, 0]
    # Simulate per-frame normalization (what happens in visualization)
    per_frame_normalized = np.zeros_like(activity_timeline)
    for i in range(n_windows):
        frame = activity_timeline[:, i]
        fmax = np.max(np.abs(frame))
        if fmax > 0:
            per_frame_normalized[:, i] = frame / fmax
    
    per_frame_norm_power = np.sum(per_frame_normalized ** 2, axis=0)
    ax.plot(timestamps, per_frame_norm_power, 'r-', linewidth=0.8, label='Per-frame normalized')
    ax.plot(timestamps, frame_power / np.max(frame_power), 'b-', alpha=0.5, linewidth=0.8, label='Global normalized')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Power')
    ax.set_title('Effect of Per-Frame vs Global Normalization')
    ax.legend()
    
    # Coefficient of variation per frame
    ax = axes[1, 1]
    cv_per_frame = frame_std / (np.abs(frame_mean) + 1e-10)
    ax.plot(timestamps, cv_per_frame, 'g-', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CV (std/|mean|)')
    ax.set_title('Coefficient of Variation Per Frame')
    
    plt.tight_layout()
    norm_plot_path = output_dir / 'normalization_effects.png'
    plt.savefig(norm_plot_path, dpi=150)
    plt.close()
    print(f"Saved normalization effects plot to: {norm_plot_path}")
    
    # Summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    max_ratio = np.max(frame_power) / np.min(frame_power)
    if max_ratio > 5:
        print(f"\n⚠️  Source power varies by {max_ratio:.1f}x across frames")
        print("   This large variation can appear as 'flashes' in visualization.")
    else:
        print(f"\n✅ Source power varies by only {max_ratio:.1f}x across frames")
        print("   This is relatively stable.")
    
    if len(spike_frames) > 0:
        print(f"\n⚠️  Found {len(spike_frames)} spike-and-return patterns")
        print("   These are frames where power spikes briefly and returns to normal.")
        print("   This is the classic 'flash' artifact pattern.")
    else:
        print(f"\n✅ No spike-and-return patterns detected")
    
    eeg_ratio = np.max(frame_max_abs) / np.min(frame_max_abs)
    if eeg_ratio > 3 and abs(corr_maxabs_power) > 0.5:
        print(f"\n⚠️  EEG normalization factor varies by {eeg_ratio:.1f}x")
        print(f"   And is correlated ({corr_maxabs_power:.2f}) with source power.")
        print("   Per-window EEG normalization may be causing amplitude swings.")
    
    return {
        'n_windows': n_windows,
        'power_ratio': max_ratio,
        'n_spike_patterns': len(spike_frames),
        'spike_times': timestamps[spike_frames].tolist() if spike_frames else [],
        'large_jump_times': timestamps[large_jump_frames].tolist() if len(large_jump_frames) > 0 else [],
        'eeg_maxabs_ratio': eeg_ratio,
        'eeg_power_correlation': corr_maxabs_power,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path", type=str)
    parser.add_argument("--annotation_csv", type=str, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--overlap_fraction", type=float, default=0.5)
    
    args = parser.parse_args()
    
    analyze_frame_dynamics(
        edf_path=Path(args.edf_path),
        annotation_csv=Path(args.annotation_csv) if args.annotation_csv else None,
        max_windows=args.max_windows,
        overlap_fraction=args.overlap_fraction,
    )
