#!/usr/bin/env python3
"""
Compare flash detection between original per-window normalization and global normalization.

This script runs both approaches and produces a comparison report showing whether
global normalization reduces the "flash" artifacts.

It also checks if any high-activity frames correspond to neurologist annotations.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
    _find_default_model_checkpoint,
    _infer_model,
)
from diagnose_flashes import (
    load_annotations,
    match_flashes_to_annotations,
    FlashEvent,
    _load_model,
    _rename_channels_to_target,
    _build_full_montage_raw,
    _segment_signal,
)


def run_comparison(
    edf_path: Path,
    annotation_csv: Optional[Path] = None,
    simu_name: str = "mes_debug",
    subject: str = "fsaverage",
    source_space: str = "ico3",
    model_path: Optional[Path] = None,
    inter_layer: int = 4096,
    kernel_size: int = 5,
    train_loss: str = "cosine",
    overlap_fraction: float = 0.5,
    max_windows: Optional[int] = None,
    z_threshold: float = 2.0,
    output_dir: Optional[Path] = None,
):
    """Run comparison between per-window and global normalization."""
    
    repo_root = REPO_ROOT
    
    # Setup paths and config
    simu_root = repo_root / "simulation" / subject
    config_path = (
        simu_root / "constrained" / "standard_1020" / source_space
        / "simu" / simu_name / f"{simu_name}{source_space}_config.json"
    )
    
    with config_path.open() as f:
        general_config = json.load(f)
    
    general_config["simu_name"] = simu_name
    general_config["electrode_space"]["electrode_montage"] = "standard_1020"
    general_config.setdefault("eeg_snr", "infdb")
    
    # Load head model
    folders = FolderStructure(str(simu_root), general_config)
    electrode_space = HeadModel.ElectrodeSpace(folders, general_config)
    source_space_obj = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space, source_space_obj, folders, subject)
    
    fs = float(general_config["rec_info"]["fs"])
    window_samples = int(general_config["rec_info"]["n_times"])
    window_seconds = window_samples / fs
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_fraction))))
    
    # Load model
    if model_path:
        model_checkpoint = Path(model_path)
    else:
        token = f"{simu_name}{source_space}_"
        model_checkpoint = _find_default_model_checkpoint(token, inter_layer)
    
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
    
    # Compute global normalization factor (robust - 99th percentile)
    global_max_abs = float(np.percentile(np.abs(data), 99))
    
    print(f"\nProcessing with overlap={overlap_fraction}, max_windows={max_windows}")
    print(f"Global max_abs (99th percentile): {global_max_abs:.6e}")
    
    # Process windows with BOTH approaches
    predictions_perwindow = []
    predictions_global = []
    local_max_abs_values = []
    
    n_sources = head_model.source_space.n_sources
    
    for seg_idx, (start_sample, segment) in enumerate(
        _segment_signal(data, window_samples, step_samples, pad=True, max_windows=max_windows)
    ):
        local_max_abs = float(np.abs(segment).max())
        if local_max_abs <= 0:
            continue
        
        local_max_abs_values.append(local_max_abs)
        
        # Per-window normalization (original approach)
        eeg_perwindow = torch.from_numpy((segment / local_max_abs).astype(np.float32))
        pred_perwindow = _infer_model(
            model, eeg_perwindow, train_loss,
            max_src_val=1.0, max_eeg_val=local_max_abs, leadfield=leadfield
        )
        
        # Global normalization (new approach)
        eeg_global = torch.from_numpy((segment / global_max_abs).astype(np.float32))
        pred_global = _infer_model(
            model, eeg_global, train_loss,
            max_src_val=1.0, max_eeg_val=global_max_abs, leadfield=leadfield
        )
        
        predictions_perwindow.append({
            'start_time': start_sample / fs,
            'end_time': (start_sample + window_samples) / fs,
            'predictions': pred_perwindow.detach().cpu().numpy(),
        })
        
        predictions_global.append({
            'start_time': start_sample / fs,
            'end_time': (start_sample + window_samples) / fs,
            'predictions': pred_global.detach().cpu().numpy(),
        })
    
    print(f"Processed {len(predictions_perwindow)} windows")
    
    # Extract activity timelines
    n_windows = len(predictions_perwindow)
    activity_perwindow = np.zeros((n_sources, n_windows), dtype=np.float32)
    activity_global = np.zeros((n_sources, n_windows), dtype=np.float32)
    timestamps = np.zeros(n_windows, dtype=np.float32)
    
    for i in range(n_windows):
        # Per-window
        win_pred = predictions_perwindow[i]['predictions']
        total_activity = np.abs(win_pred).sum(axis=0)
        peak_idx = np.argmax(total_activity)
        activity_perwindow[:, i] = win_pred[:, peak_idx]
        
        # Global
        win_pred_g = predictions_global[i]['predictions']
        total_activity_g = np.abs(win_pred_g).sum(axis=0)
        peak_idx_g = np.argmax(total_activity_g)
        activity_global[:, i] = win_pred_g[:, peak_idx_g]
        
        timestamps[i] = (predictions_perwindow[i]['start_time'] + predictions_perwindow[i]['end_time']) / 2.0
    
    # Compute power and detect flashes
    power_perwindow = np.sum(activity_perwindow ** 2, axis=0)
    power_global = np.sum(activity_global ** 2, axis=0)
    
    def detect_flashes_from_power(power, timestamps, z_thresh):
        z = (power - power.mean()) / power.std()
        flash_idx = np.where(z > z_thresh)[0]
        flashes = [
            FlashEvent(
                frame_idx=int(idx),
                timestamp=float(timestamps[idx]),
                z_score=float(z[idx]),
                power=float(power[idx])
            )
            for idx in flash_idx
        ]
        return flashes, z
    
    flashes_perwindow, z_perwindow = detect_flashes_from_power(power_perwindow, timestamps, z_threshold)
    flashes_global, z_global = detect_flashes_from_power(power_global, timestamps, z_threshold)
    
    # Load annotations if available
    annotations = []
    n_matches_perwindow = 0
    n_matches_global = 0
    
    if annotation_csv and annotation_csv.exists():
        annotations, _ = load_annotations(annotation_csv)
        
        # Get unique annotations
        unique_times = set()
        unique_annotations = []
        for ann in annotations:
            mid = (ann.start_time + ann.end_time) / 2
            key = f"{mid:.2f}"
            if key not in unique_times:
                unique_times.add(key)
                unique_annotations.append(ann)
        
        flashes_perwindow, n_matches_perwindow = match_flashes_to_annotations(
            flashes_perwindow, unique_annotations, tolerance=0.5
        )
        flashes_global, n_matches_global = match_flashes_to_annotations(
            flashes_global, unique_annotations, tolerance=0.5
        )
    
    # Print results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    local_ratio = max(local_max_abs_values) / min(local_max_abs_values)
    print(f"\nEEG Normalization Factor Variation:")
    print(f"  Per-window: {local_ratio:.2f}x variation")
    print(f"  Global:     1.00x (no variation)")
    
    print(f"\nSource Power Ratio (max/min):")
    print(f"  Per-window: {power_perwindow.max()/power_perwindow.min():.2f}x")
    print(f"  Global:     {power_global.max()/power_global.min():.2f}x")
    
    print(f"\nFlashes Detected (z > {z_threshold}):")
    print(f"  Per-window: {len(flashes_perwindow)}")
    print(f"  Global:     {len(flashes_global)}")
    
    if len(flashes_global) < len(flashes_perwindow):
        reduction = (1 - len(flashes_global) / max(1, len(flashes_perwindow))) * 100
        print(f"  → Global normalization reduced flashes by {reduction:.1f}%")
    elif len(flashes_global) == len(flashes_perwindow):
        print(f"  → No change in number of flashes")
    else:
        increase = (len(flashes_global) / max(1, len(flashes_perwindow)) - 1) * 100
        print(f"  → Global normalization increased flashes by {increase:.1f}%")
    
    if annotations:
        print(f"\nAnnotation Matching (tolerance=0.5s):")
        print(f"  Per-window flashes matching: {n_matches_perwindow}/{len(flashes_perwindow)}")
        print(f"  Global flashes matching:     {n_matches_global}/{len(flashes_global)}")
    
    # Frame-to-frame stability
    jumps_perwindow = np.abs(np.diff(power_perwindow))
    jumps_global = np.abs(np.diff(power_global))
    
    print(f"\nFrame-to-Frame Stability:")
    print(f"  Per-window mean jump: {jumps_perwindow.mean():.6e}")
    print(f"  Global mean jump:     {jumps_global.mean():.6e}")
    print(f"  Improvement:          {(1 - jumps_global.mean()/jumps_perwindow.mean())*100:.1f}%")
    
    # Setup output
    if output_dir is None:
        output_dir = repo_root / "results" / "flash_comparison" / edf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 1. Per-window normalization factor
    ax = axes[0]
    ax.plot(timestamps, local_max_abs_values, 'b-', linewidth=0.8)
    ax.axhline(global_max_abs, color='r', linestyle='--', label=f'Global (99th %ile)')
    ax.set_ylabel('EEG max_abs')
    ax.set_title('Per-Window vs Global Normalization Factor')
    ax.legend()
    
    # 2. Source power comparison
    ax = axes[1]
    ax.plot(timestamps, power_perwindow / power_perwindow.mean(), 'b-', alpha=0.7, linewidth=0.8, label='Per-window')
    ax.plot(timestamps, power_global / power_global.mean(), 'r-', alpha=0.7, linewidth=0.8, label='Global')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Normalized Power')
    ax.set_title('Source Power (normalized to mean)')
    ax.legend()
    
    # 3. Z-scores comparison
    ax = axes[2]
    ax.plot(timestamps, z_perwindow, 'b-', alpha=0.7, linewidth=0.8, label='Per-window')
    ax.plot(timestamps, z_global, 'r-', alpha=0.7, linewidth=0.8, label='Global')
    ax.axhline(z_threshold, color='orange', linestyle='--', label=f'z={z_threshold} threshold')
    ax.set_ylabel('Z-score')
    ax.set_title('Power Z-scores (flash detection)')
    ax.legend()
    
    # 4. Frame-to-frame jumps
    ax = axes[3]
    ax.plot(timestamps[1:], jumps_perwindow / jumps_perwindow.mean(), 'b-', alpha=0.7, linewidth=0.8, label='Per-window')
    ax.plot(timestamps[1:], jumps_global / jumps_global.mean(), 'r-', alpha=0.7, linewidth=0.8, label='Global')
    ax.set_ylabel('Normalized Jump')
    ax.set_xlabel('Time (s)')
    ax.set_title('Frame-to-Frame Power Jumps (normalized to mean)')
    ax.legend()
    
    # Add annotation markers
    if annotations:
        for ax in axes:
            for ann in annotations[:50]:
                mid = (ann.start_time + ann.end_time) / 2
                if timestamps.min() <= mid <= timestamps.max():
                    ax.axvline(mid, color='green', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    plot_path = output_dir / 'comparison_plot.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot to: {plot_path}")
    
    # Save results
    results = {
        'n_windows': n_windows,
        'local_max_abs_ratio': float(local_ratio),
        'perwindow': {
            'power_ratio': float(power_perwindow.max() / power_perwindow.min()),
            'n_flashes': len(flashes_perwindow),
            'n_annotation_matches': n_matches_perwindow,
            'mean_jump': float(jumps_perwindow.mean()),
        },
        'global': {
            'power_ratio': float(power_global.max() / power_global.min()),
            'n_flashes': len(flashes_global),
            'n_annotation_matches': n_matches_global,
            'mean_jump': float(jumps_global.mean()),
        },
        'improvement': {
            'flash_reduction_pct': float((1 - len(flashes_global) / max(1, len(flashes_perwindow))) * 100) if flashes_perwindow else 0,
            'jump_reduction_pct': float((1 - jumps_global.mean() / jumps_perwindow.mean()) * 100),
        }
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to: {output_dir / 'comparison_results.json'}")
    
    # Final summary
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if results['improvement']['jump_reduction_pct'] > 10:
        print("\n✅ Global normalization significantly improved frame-to-frame stability.")
        print("   Consider updating run_edf_inference.py to use global normalization.")
    else:
        print("\n⚠️  Global normalization had minimal impact on stability.")
        print("   The variation may be from the inverse solver itself.")
    
    if len(flashes_global) > 0 and n_matches_global < len(flashes_global) * 0.3:
        print("\n⚠️  Most detected high-activity frames do NOT match neurologist annotations.")
        print("   Consider:")
        print("   1. Adding temporal regularization to the inverse solver")
        print("   2. Applying bandpass filtering to the source estimates")
        print("   3. Using a Kalman filter for temporal smoothing")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path", type=str)
    parser.add_argument("--annotation_csv", type=str, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--overlap_fraction", type=float, default=0.5)
    parser.add_argument("--z_threshold", type=float, default=2.0)
    
    args = parser.parse_args()
    
    run_comparison(
        edf_path=Path(args.edf_path),
        annotation_csv=Path(args.annotation_csv) if args.annotation_csv else None,
        max_windows=args.max_windows,
        overlap_fraction=args.overlap_fraction,
        z_threshold=args.z_threshold,
    )
