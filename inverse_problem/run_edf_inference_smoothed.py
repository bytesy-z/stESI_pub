#!/usr/bin/env python3
"""
EDF Inference with Temporal Smoothing Options.

This script implements three temporal smoothing approaches to reduce "flash" artifacts:
1. Exponential Moving Average (EMA) - simple, fast, adjustable smoothing
2. Bandpass Filtering - removes high-frequency noise from source estimates
3. Kalman-like Filter - state-space tracking with process noise control

Usage:
    python run_edf_inference_smoothed.py path/to/file.edf --smoothing_method ema --smoothing_alpha 0.3
    python run_edf_inference_smoothed.py path/to/file.edf --smoothing_method bandpass --lowcut 0.5 --highcut 4.0
    python run_edf_inference_smoothed.py path/to/file.edf --smoothing_method kalman --process_noise 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy.signal import windows as scipy_windows

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from models.cnn_1d import CNN1Dpl
from plot_1dcnn_inference_heatmap import (
    BrainGeometry,
    _ensure_within_repo,
    _find_default_model_checkpoint,
    _infer_model,
    _load_brain_geometry,
)


# =============================================================================
# TEMPORAL SMOOTHING METHODS
# =============================================================================

def apply_exponential_smoothing(
    activity: np.ndarray,
    alpha: float = 0.3,
    bidirectional: bool = True
) -> np.ndarray:
    """
    Apply Exponential Moving Average (EMA) smoothing to source activity.
    
    Args:
        activity: (n_sources, n_frames) array
        alpha: Smoothing factor (0-1). Lower = more smoothing.
               0.1 = heavy smoothing, 0.5 = moderate, 0.9 = light
        bidirectional: If True, apply forward-backward to avoid phase shift
        
    Returns:
        Smoothed activity array
    """
    n_sources, n_frames = activity.shape
    smoothed = np.zeros_like(activity)
    
    # Forward pass
    smoothed[:, 0] = activity[:, 0]
    for t in range(1, n_frames):
        smoothed[:, t] = alpha * activity[:, t] + (1 - alpha) * smoothed[:, t-1]
    
    if bidirectional:
        # Backward pass to remove phase shift
        backward = np.zeros_like(activity)
        backward[:, -1] = smoothed[:, -1]
        for t in range(n_frames - 2, -1, -1):
            backward[:, t] = alpha * smoothed[:, t] + (1 - alpha) * backward[:, t+1]
        smoothed = backward
    
    return smoothed


def apply_bandpass_filter(
    activity: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 4.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to source activity to remove high-frequency noise.
    
    Args:
        activity: (n_sources, n_frames) array
        fs: Sampling frequency of frames (frames per second)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered activity array
    """
    n_sources, n_frames = activity.shape
    
    # Need at least 3*order samples for filtering
    if n_frames < 3 * order + 1:
        print(f"Warning: Not enough frames ({n_frames}) for bandpass filter. Returning unfiltered.")
        return activity
    
    # Nyquist frequency
    nyq = fs / 2.0
    
    # Normalize frequencies
    low = lowcut / nyq
    high = highcut / nyq
    
    # Clamp to valid range
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    try:
        # Design Butterworth bandpass filter
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each source
        filtered = np.zeros_like(activity)
        for i in range(n_sources):
            # Use filtfilt for zero-phase filtering
            filtered[i, :] = scipy_signal.filtfilt(b, a, activity[i, :], padlen=min(3*order, n_frames-1))
        
        return filtered
    except Exception as e:
        print(f"Warning: Bandpass filter failed ({e}). Returning unfiltered.")
        return activity


def apply_kalman_filter(
    activity: np.ndarray,
    process_noise: float = 0.1,
    measurement_noise: float = 1.0
) -> np.ndarray:
    """
    Apply a simple Kalman-like filter for temporal smoothing.
    
    This implements a random walk state model where sources are expected to
    change slowly over time. The process_noise controls how much change is allowed.
    
    Args:
        activity: (n_sources, n_frames) array
        process_noise: Q - how much the state is expected to change per frame.
                       Lower = smoother output. Try 0.01-0.5
        measurement_noise: R - how much we trust the measurements.
                          Lower = trust measurements more. Usually set to 1.0
        
    Returns:
        Filtered activity array
    """
    n_sources, n_frames = activity.shape
    
    # State estimate and covariance for each source
    x_est = np.zeros((n_sources, n_frames))  # State estimates
    
    # Initialize with first observation
    x_est[:, 0] = activity[:, 0]
    P = np.ones(n_sources) * measurement_noise  # Initial covariance
    
    Q = process_noise  # Process noise
    R = measurement_noise  # Measurement noise
    
    for t in range(1, n_frames):
        # Prediction step (random walk model: x_t = x_{t-1} + noise)
        x_pred = x_est[:, t-1]
        P_pred = P + Q
        
        # Update step
        K = P_pred / (P_pred + R)  # Kalman gain
        x_est[:, t] = x_pred + K * (activity[:, t] - x_pred)
        P = (1 - K) * P_pred
    
    # Optional: backward smoothing pass (RTS smoother)
    x_smooth = np.zeros_like(x_est)
    x_smooth[:, -1] = x_est[:, -1]
    
    P_smooth = P.copy()
    P_pred_stored = np.ones(n_sources) * (P + Q)
    
    for t in range(n_frames - 2, -1, -1):
        # Backward smoothing
        P_pred_t = P + Q
        J = P / P_pred_t  # Smoother gain
        x_smooth[:, t] = x_est[:, t] + J * (x_smooth[:, t+1] - x_est[:, t])
    
    return x_smooth


def apply_median_filter(
    activity: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply median filter to remove spike artifacts.
    
    Args:
        activity: (n_sources, n_frames) array
        kernel_size: Size of median filter window (odd number)
        
    Returns:
        Filtered activity array
    """
    from scipy.ndimage import median_filter
    
    # Apply along time axis
    filtered = median_filter(activity, size=(1, kernel_size))
    return filtered


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _canonical_lookup(names):
    return {name.lower(): name for name in names}


def _rename_channels_to_target(raw, target_names):
    lookup = _canonical_lookup(target_names)
    rename_map = {}
    for ch in raw.ch_names:
        key = ch.lower()
        if key in lookup:
            rename_map[ch] = lookup[key]
        else:
            key = ch.replace(" ", "").lower()
            if key in lookup:
                rename_map[ch] = lookup[key]
    if rename_map:
        raw.rename_channels(rename_map)


def _build_full_montage_raw(source_raw, target_info, target_names):
    data = np.zeros((len(target_names), source_raw.n_times), dtype=np.float64)
    missing = []
    for idx, name in enumerate(target_names):
        if name in source_raw.ch_names:
            data[idx] = source_raw.get_data(picks=[name])[0]
        else:
            missing.append(name)
    full_info = target_info.copy()
    full_raw = mne.io.RawArray(data, full_info, verbose=False)
    if missing:
        full_raw.info["bads"] = missing.copy()
        full_raw.interpolate_bads(reset_bads=True, verbose=False, mode="accurate")
    return full_raw, missing


def _segment_signal(data, window_samples, step_samples, pad, max_windows):
    n_samples = data.shape[1]
    if n_samples == 0:
        return
    start = 0
    n_generated = 0
    while start < n_samples and (max_windows is None or n_generated < max_windows):
        end = start + window_samples
        segment = data[:, start:end]
        if segment.shape[1] < window_samples:
            if not pad:
                break
            pad_width = window_samples - segment.shape[1]
            segment = np.pad(segment, ((0, 0), (0, pad_width)), mode="edge")
        yield start, segment
        n_generated += 1
        start += step_samples
        if step_samples <= 0:
            break


def _load_model(checkpoint_path, n_electrodes, n_sources, inter_layer=4096, kernel_size=5):
    net_params = {
        "channels": [n_electrodes, inter_layer, n_sources],
        "kernel_size": kernel_size,
        "bias": False,
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,
    }
    model = CNN1Dpl(**net_params)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _prepare_output_dir(base_dir, repo_root, edf_path, smoothing_method):
    if base_dir:
        out_dir = Path(base_dir).expanduser()
    else:
        out_dir = repo_root / "results" / f"edf_inference_{smoothing_method}" / edf_path.stem
    out_dir = _ensure_within_repo(out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def detect_flashes(power: np.ndarray, z_threshold: float = 2.0) -> Tuple[np.ndarray, int]:
    """Detect frames with power above z_threshold standard deviations."""
    z = (power - power.mean()) / power.std()
    flash_idx = np.where(z > z_threshold)[0]
    return flash_idx, len(flash_idx)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run 1dCNN inference with temporal smoothing options."
    )
    parser.add_argument("edf_path", type=str, help="Path to EDF file")
    
    # Model/data parameters
    parser.add_argument("--simu_name", default="mes_debug")
    parser.add_argument("--subject", default="fsaverage")
    parser.add_argument("--source_space", default="ico3")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--inter_layer", type=int, default=4096)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--train_loss", default="cosine")
    
    # Windowing parameters
    parser.add_argument("--window_seconds", type=float, default=None)
    parser.add_argument("--overlap_fraction", type=float, default=0.5)
    parser.add_argument("--max_windows", type=int, default=None)
    
    # Smoothing parameters
    parser.add_argument("--smoothing_method", 
                        choices=["none", "ema", "bandpass", "kalman", "median", "all"],
                        default="all",
                        help="Smoothing method to apply")
    parser.add_argument("--ema_alpha", type=float, default=0.3,
                        help="EMA smoothing factor (0.1=heavy, 0.5=moderate, 0.9=light)")
    parser.add_argument("--bandpass_lowcut", type=float, default=0.5,
                        help="Bandpass low cutoff frequency (Hz)")
    parser.add_argument("--bandpass_highcut", type=float, default=4.0,
                        help="Bandpass high cutoff frequency (Hz)")
    parser.add_argument("--kalman_process_noise", type=float, default=0.1,
                        help="Kalman filter process noise (lower=smoother)")
    parser.add_argument("--median_kernel", type=int, default=3,
                        help="Median filter kernel size")
    
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--z_threshold", type=float, default=2.0,
                        help="Z-score threshold for flash detection")
    
    args = parser.parse_args()
    
    repo_root = REPO_ROOT
    edf_path = Path(args.edf_path).expanduser().resolve()
    
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    
    # Setup paths and load config
    simu_root = repo_root / "simulation" / args.subject
    config_path = (
        simu_root / "constrained" / "standard_1020" / args.source_space
        / "simu" / args.simu_name / f"{args.simu_name}{args.source_space}_config.json"
    )
    
    with config_path.open() as f:
        general_config = json.load(f)
    
    general_config["simu_name"] = args.simu_name
    general_config["electrode_space"]["electrode_montage"] = "standard_1020"
    general_config.setdefault("eeg_snr", "infdb")
    
    # Load head model
    folders = FolderStructure(str(simu_root), general_config)
    electrode_space = HeadModel.ElectrodeSpace(folders, general_config)
    source_space_obj = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space, source_space_obj, folders, args.subject)
    
    # Setup window parameters
    fs = float(general_config["rec_info"]["fs"])
    if args.window_seconds is None:
        window_samples = int(general_config["rec_info"]["n_times"])
        window_seconds = window_samples / fs
    else:
        window_seconds = float(args.window_seconds)
        window_samples = max(1, int(round(window_seconds * fs)))
    
    overlap_fraction = min(max(args.overlap_fraction, 0.0), 0.95)
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_fraction))))
    
    # Output directory
    output_dir = _prepare_output_dir(args.output_dir, repo_root, edf_path, args.smoothing_method)
    
    # Load model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        token = f"{args.simu_name}{args.source_space}_"
        model_path = _find_default_model_checkpoint(token, args.inter_layer)
        if model_path is None:
            raise FileNotFoundError("Cannot find model checkpoint")
    
    model = _load_model(
        model_path,
        head_model.electrode_space.n_electrodes,
        head_model.source_space.n_sources,
        args.inter_layer,
        args.kernel_size,
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
    
    full_raw, missing = _build_full_montage_raw(
        raw, head_model.electrode_space.info, head_model.electrode_space.info.ch_names
    )
    if missing:
        print(f"Interpolated {len(missing)} missing channels")
    
    full_raw.set_eeg_reference("average", projection=False, verbose=False)
    data = full_raw.get_data()
    
    # Use global normalization (99th percentile)
    global_max_abs = float(np.percentile(np.abs(data), 99))
    print(f"Global normalization factor: {global_max_abs:.6e}")
    
    leadfield = torch.from_numpy(head_model.fwd["sol"]["data"]).float()
    
    # Process windows
    all_predictions = []
    
    print(f"\nProcessing windows (overlap={overlap_fraction}, max={args.max_windows})...")
    for seg_idx, (start_sample, segment) in enumerate(
        _segment_signal(data, window_samples, step_samples, pad=True, max_windows=args.max_windows)
    ):
        local_max_abs = float(np.abs(segment).max())
        if local_max_abs <= 0:
            continue
        
        eeg_tensor = torch.from_numpy((segment / global_max_abs).astype(np.float32))
        pred = _infer_model(
            model, eeg_tensor, args.train_loss,
            max_src_val=1.0, max_eeg_val=global_max_abs, leadfield=leadfield
        )
        
        all_predictions.append({
            'start_time': start_sample / fs,
            'end_time': (start_sample + window_samples) / fs,
            'predictions': pred.detach().cpu().numpy(),
        })
    
    print(f"Processed {len(all_predictions)} windows")
    
    # Extract raw activity timeline
    n_windows = len(all_predictions)
    n_sources = all_predictions[0]['predictions'].shape[0]
    
    activity_raw = np.zeros((n_sources, n_windows), dtype=np.float32)
    timestamps = np.zeros(n_windows, dtype=np.float32)
    
    for i, window in enumerate(all_predictions):
        win_pred = window['predictions']
        total_activity = np.abs(win_pred).sum(axis=0)
        peak_idx = np.argmax(total_activity)
        activity_raw[:, i] = win_pred[:, peak_idx]
        timestamps[i] = (window['start_time'] + window['end_time']) / 2.0
    
    # Calculate frame rate
    if len(timestamps) > 1:
        frame_rate = 1.0 / np.mean(np.diff(timestamps))
    else:
        frame_rate = 1.0
    
    print(f"Frame rate: {frame_rate:.2f} FPS")
    
    # ==========================================================================
    # APPLY SMOOTHING METHODS
    # ==========================================================================
    
    results = {}
    
    # Raw (no smoothing)
    power_raw = np.sum(activity_raw ** 2, axis=0)
    flash_idx_raw, n_flashes_raw = detect_flashes(power_raw, args.z_threshold)
    results['none'] = {
        'activity': activity_raw,
        'power': power_raw,
        'n_flashes': n_flashes_raw,
        'power_ratio': power_raw.max() / power_raw.min(),
        'label': 'Raw (No Smoothing)'
    }
    
    methods_to_run = ['ema', 'bandpass', 'kalman', 'median'] if args.smoothing_method == 'all' else [args.smoothing_method]
    
    if 'ema' in methods_to_run:
        print(f"\nApplying Exponential Moving Average (alpha={args.ema_alpha})...")
        activity_ema = apply_exponential_smoothing(activity_raw, alpha=args.ema_alpha, bidirectional=True)
        power_ema = np.sum(activity_ema ** 2, axis=0)
        flash_idx_ema, n_flashes_ema = detect_flashes(power_ema, args.z_threshold)
        results['ema'] = {
            'activity': activity_ema,
            'power': power_ema,
            'n_flashes': n_flashes_ema,
            'power_ratio': power_ema.max() / power_ema.min(),
            'label': f'EMA (α={args.ema_alpha})'
        }
    
    if 'bandpass' in methods_to_run:
        print(f"\nApplying Bandpass Filter ({args.bandpass_lowcut}-{args.bandpass_highcut} Hz)...")
        activity_bp = apply_bandpass_filter(
            activity_raw, fs=frame_rate,
            lowcut=args.bandpass_lowcut, highcut=args.bandpass_highcut
        )
        power_bp = np.sum(activity_bp ** 2, axis=0)
        flash_idx_bp, n_flashes_bp = detect_flashes(power_bp, args.z_threshold)
        results['bandpass'] = {
            'activity': activity_bp,
            'power': power_bp,
            'n_flashes': n_flashes_bp,
            'power_ratio': power_bp.max() / power_bp.min(),
            'label': f'Bandpass ({args.bandpass_lowcut}-{args.bandpass_highcut} Hz)'
        }
    
    if 'kalman' in methods_to_run:
        print(f"\nApplying Kalman Filter (process_noise={args.kalman_process_noise})...")
        activity_kalman = apply_kalman_filter(activity_raw, process_noise=args.kalman_process_noise)
        power_kalman = np.sum(activity_kalman ** 2, axis=0)
        flash_idx_kalman, n_flashes_kalman = detect_flashes(power_kalman, args.z_threshold)
        results['kalman'] = {
            'activity': activity_kalman,
            'power': power_kalman,
            'n_flashes': n_flashes_kalman,
            'power_ratio': power_kalman.max() / power_kalman.min(),
            'label': f'Kalman (Q={args.kalman_process_noise})'
        }
    
    if 'median' in methods_to_run:
        print(f"\nApplying Median Filter (kernel={args.median_kernel})...")
        activity_median = apply_median_filter(activity_raw, kernel_size=args.median_kernel)
        power_median = np.sum(activity_median ** 2, axis=0)
        flash_idx_median, n_flashes_median = detect_flashes(power_median, args.z_threshold)
        results['median'] = {
            'activity': activity_median,
            'power': power_median,
            'n_flashes': n_flashes_median,
            'power_ratio': power_median.max() / power_median.min(),
            'label': f'Median (k={args.median_kernel})'
        }
    
    # ==========================================================================
    # PRINT COMPARISON
    # ==========================================================================
    
    print("\n" + "="*70)
    print("SMOOTHING COMPARISON RESULTS")
    print("="*70)
    print(f"\nFlash detection threshold: z > {args.z_threshold}")
    print(f"Number of frames: {n_windows}")
    print(f"Frame rate: {frame_rate:.2f} FPS")
    
    print(f"\n{'Method':<35} {'Flashes':<10} {'Power Ratio':<15} {'Reduction':<15}")
    print("-" * 75)
    
    baseline_flashes = results['none']['n_flashes']
    for method, data in results.items():
        reduction = (1 - data['n_flashes'] / max(1, baseline_flashes)) * 100 if baseline_flashes > 0 else 0
        print(f"{data['label']:<35} {data['n_flashes']:<10} {data['power_ratio']:<15.2f} {reduction:>+.1f}%")
    
    # Frame-to-frame stability
    print(f"\n{'Method':<35} {'Mean Jump':<15} {'Max Jump':<15} {'Jump Reduction':<15}")
    print("-" * 75)
    
    baseline_jumps = np.abs(np.diff(results['none']['power']))
    baseline_mean_jump = baseline_jumps.mean()
    
    for method, data in results.items():
        jumps = np.abs(np.diff(data['power']))
        mean_jump = jumps.mean()
        max_jump = jumps.max()
        reduction = (1 - mean_jump / baseline_mean_jump) * 100 if baseline_mean_jump > 0 else 0
        print(f"{data['label']:<35} {mean_jump:<15.2e} {max_jump:<15.2e} {reduction:>+.1f}%")
    
    # ==========================================================================
    # GENERATE COMPARISON PLOT
    # ==========================================================================
    
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 3 * n_methods))
    
    colors = {'none': 'gray', 'ema': 'blue', 'bandpass': 'green', 'kalman': 'red', 'median': 'purple'}
    
    for i, (method, data) in enumerate(results.items()):
        color = colors.get(method, 'black')
        
        # Power over time
        ax = axes[i, 0]
        norm_power = data['power'] / data['power'].mean()
        ax.plot(timestamps, norm_power, color=color, linewidth=0.8, alpha=0.8)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
        
        # Mark flashes
        flash_idx, _ = detect_flashes(data['power'], args.z_threshold)
        if len(flash_idx) > 0:
            ax.scatter(timestamps[flash_idx], norm_power[flash_idx], 
                      c='red', s=20, zorder=5, alpha=0.7)
        
        ax.set_ylabel('Normalized Power')
        ax.set_title(f"{data['label']} - {data['n_flashes']} flashes")
        if i == n_methods - 1:
            ax.set_xlabel('Time (s)')
        
        # Power histogram
        ax = axes[i, 1]
        ax.hist(norm_power, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Normalized Power')
        ax.set_ylabel('Count')
        ax.set_title(f"Distribution (ratio={data['power_ratio']:.1f}x)")
    
    plt.tight_layout()
    plot_path = output_dir / 'smoothing_comparison.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot to: {plot_path}")
    
    # ==========================================================================
    # SAVE BEST RESULT AS ANIMATION DATA
    # ==========================================================================
    
    # Find best method (lowest flash count, excluding 'none')
    smoothed_results = {k: v for k, v in results.items() if k != 'none'}
    if smoothed_results:
        best_method = min(smoothed_results.keys(), key=lambda k: smoothed_results[k]['n_flashes'])
        best_result = smoothed_results[best_method]
        
        print(f"\nBest smoothing method: {best_result['label']}")
        print(f"  Flashes reduced from {baseline_flashes} to {best_result['n_flashes']}")
        
        # Save animation data with best smoothing
        geom = _load_brain_geometry(head_model)
        
        triangles_list = []
        vertex_offset = 0
        for surf in geom.surfaces:
            adjusted_faces = surf.faces + vertex_offset
            triangles_list.append(adjusted_faces)
            vertex_offset += len(surf.vertices_mm)
        triangles = np.vstack(triangles_list).astype(np.int32)
        
        animation_data = {
            'activity': best_result['activity'].astype(np.float32),
            'timestamps': timestamps.astype(np.float32),
            'source_positions': np.ascontiguousarray(geom.positions_mm.astype(np.float32)),
            'triangles': triangles,
            'fps': np.array(int(frame_rate), dtype=np.int32),
            'smoothing_method': best_method,
        }
        
        animation_path = output_dir / 'animation_data_smoothed.npz'
        np.savez_compressed(str(animation_path), **animation_data)
        print(f"Saved smoothed animation data to: {animation_path}")
    
    # Save detailed results
    summary = {
        'n_windows': n_windows,
        'frame_rate': float(frame_rate),
        'z_threshold': args.z_threshold,
        'parameters': {
            'ema_alpha': args.ema_alpha,
            'bandpass': [args.bandpass_lowcut, args.bandpass_highcut],
            'kalman_process_noise': args.kalman_process_noise,
            'median_kernel': args.median_kernel,
        },
        'results': {
            method: {
                'n_flashes': int(data['n_flashes']),
                'power_ratio': float(data['power_ratio']),
                'flash_reduction_pct': float((1 - data['n_flashes'] / max(1, baseline_flashes)) * 100) if baseline_flashes > 0 else 0,
            }
            for method, data in results.items()
        }
    }
    
    with open(output_dir / 'smoothing_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved results to: {output_dir / 'smoothing_results.json'}")
    
    # ==========================================================================
    # RECOMMENDATIONS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Find most effective method
    if smoothed_results:
        best_flash_reduction = max(
            (1 - v['n_flashes'] / max(1, baseline_flashes)) * 100 
            for v in smoothed_results.values()
        )
        
        if best_flash_reduction > 50:
            print(f"\n✅ Smoothing significantly reduced flashes (up to {best_flash_reduction:.0f}% reduction)")
            print(f"   Recommended method: {best_result['label']}")
        elif best_flash_reduction > 20:
            print(f"\n⚠️  Moderate flash reduction achieved ({best_flash_reduction:.0f}%)")
            print("   Consider combining methods or adjusting parameters")
        else:
            print(f"\n❌ Limited flash reduction ({best_flash_reduction:.0f}%)")
            print("   The variability may be inherent to the inverse solver")
            print("   Consider:")
            print("   - Heavier smoothing (lower alpha, higher process noise)")
            print("   - Different inverse solver approach")
            print("   - Accepting the variability as characteristic of the method")


if __name__ == "__main__":
    main()
