#!/usr/bin/env python3
"""
Test the inverse solver on simulated data with known ground truth.

This verifies that:
1. The solver correctly localizes sources in simulation
2. Frame-to-frame variability in simulated data is comparable to real EDF
3. Smoothing methods preserve source localization accuracy

This helps determine if the "flashes" seen in real data are:
- Artifacts of the solver (would appear in simulation too)
- Characteristics of real EEG data (would NOT appear in simulation)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from loaders import EsiDatasetds_new
from models.cnn_1d import CNN1Dpl
from plot_1dcnn_inference_heatmap import (
    _find_default_model_checkpoint,
    _infer_model,
)
from run_edf_inference_smoothed import (
    apply_exponential_smoothing,
    apply_kalman_filter,
    detect_flashes,
)


def load_model(checkpoint_path, n_electrodes, n_sources, inter_layer=4096, kernel_size=5):
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


def compute_localization_error(
    pred: np.ndarray,
    truth: np.ndarray,
    positions: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute localization error between predicted and true source activity.
    
    Args:
        pred: (n_sources,) predicted activity
        truth: (n_sources,) ground truth activity
        positions: (n_sources, 3) source positions in mm
        
    Returns:
        Tuple of (center_of_mass_error_mm, peak_error_mm, correlation)
    """
    # Get absolute values
    pred_abs = np.abs(pred)
    truth_abs = np.abs(truth)
    
    # Normalize to get weights
    pred_norm = pred_abs / (pred_abs.sum() + 1e-10)
    truth_norm = truth_abs / (truth_abs.sum() + 1e-10)
    
    # Center of mass
    pred_com = np.sum(positions * pred_norm[:, np.newaxis], axis=0)
    truth_com = np.sum(positions * truth_norm[:, np.newaxis], axis=0)
    com_error = np.linalg.norm(pred_com - truth_com)
    
    # Peak location error
    pred_peak_idx = np.argmax(pred_abs)
    truth_peak_idx = np.argmax(truth_abs)
    peak_error = np.linalg.norm(positions[pred_peak_idx] - positions[truth_peak_idx])
    
    # Correlation
    correlation = np.corrcoef(pred, truth)[0, 1] if truth.std() > 0 else 0.0
    
    return com_error, peak_error, correlation


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--simu_name", default="mes_debug")
    parser.add_argument("--subject", default="fsaverage")
    parser.add_argument("--source_space", default="ico3")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of simulation samples to test")
    parser.add_argument("--inter_layer", type=int, default=4096)
    parser.add_argument("--eeg_snr", type=float, default=5.0)
    
    args = parser.parse_args()
    
    repo_root = REPO_ROOT
    simu_root = repo_root / "simulation" / args.subject
    
    # Load config
    config_path = (
        simu_root / "constrained" / "standard_1020" / args.source_space
        / "simu" / args.simu_name / f"{args.simu_name}{args.source_space}_config.json"
    )
    
    with config_path.open() as f:
        general_config = json.load(f)
    
    general_config["simu_name"] = args.simu_name
    general_config["electrode_space"]["electrode_montage"] = "standard_1020"
    general_config["eeg_snr"] = args.eeg_snr
    
    # Load head model
    folders = FolderStructure(str(simu_root), general_config)
    electrode_space = HeadModel.ElectrodeSpace(folders, general_config)
    source_space = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space, source_space, folders, args.subject)
    
    # Load dataset
    print(f"Loading simulation dataset (SNR={args.eeg_snr})...")
    
    config_path_str = str(config_path)
    
    dataset = EsiDatasetds_new(
        str(simu_root),
        config_path_str,
        args.simu_name,
        args.source_space,
        "standard_1020",  # electrode_montage
        args.n_samples,
        args.eeg_snr,
        noise_type={"white": 1.0, "pink": 0.0},
        norm="linear"
    )
    
    print(f"Loaded {len(dataset)} samples")
    
    # Load model
    token = f"{args.simu_name}{args.source_space}_"
    model_path = _find_default_model_checkpoint(token, args.inter_layer)
    if model_path is None:
        raise FileNotFoundError("Cannot find model checkpoint")
    
    model = load_model(
        model_path,
        head_model.electrode_space.n_electrodes,
        head_model.source_space.n_sources,
        args.inter_layer,
    )
    
    leadfield = torch.from_numpy(head_model.fwd["sol"]["data"]).float()
    positions_mm = head_model.source_space.positions * 1e3
    
    # Process each sample
    n_samples = min(args.n_samples, len(dataset))
    n_sources = head_model.source_space.n_sources
    
    # Store predictions and ground truth for analysis
    all_predictions = []  # (n_samples, n_sources, n_times)
    all_truths = []
    
    com_errors = []
    peak_errors = []
    correlations = []
    
    print(f"\nProcessing {n_samples} samples...")
    
    for idx in range(n_samples):
        eeg, src = dataset[idx]
        max_eeg = dataset.max_eeg[idx]
        max_src = dataset.max_src[idx]
        
        # Run inference
        pred = _infer_model(
            model, eeg, "cosine",
            max_src_val=float(max_src.item()),
            max_eeg_val=float(max_eeg.item()),
            leadfield=leadfield
        )
        
        pred_np = pred.detach().cpu().numpy()  # (n_sources, n_times)
        truth_np = (src * max_src).numpy()  # (n_sources, n_times)
        
        all_predictions.append(pred_np)
        all_truths.append(truth_np)
        
        # Compute error at peak timepoint
        total_activity = np.abs(truth_np).sum(axis=0)
        peak_t = np.argmax(total_activity)
        
        com_err, peak_err, corr = compute_localization_error(
            pred_np[:, peak_t], truth_np[:, peak_t], positions_mm
        )
        
        com_errors.append(com_err)
        peak_errors.append(peak_err)
        correlations.append(corr)
    
    # Stack all predictions
    all_pred_array = np.stack(all_predictions)  # (n_samples, n_sources, n_times)
    all_truth_array = np.stack(all_truths)
    
    print("\n" + "="*70)
    print("SIMULATION VALIDATION RESULTS")
    print("="*70)
    
    print(f"\nLocalization Accuracy (at peak timepoint):")
    print(f"  Center-of-Mass Error: {np.mean(com_errors):.1f} ± {np.std(com_errors):.1f} mm")
    print(f"  Peak Location Error:  {np.mean(peak_errors):.1f} ± {np.std(peak_errors):.1f} mm")
    print(f"  Correlation:          {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
    
    # Analyze temporal variability in simulated data
    # Treat each sample as a "window" in the temporal sequence
    print("\n" + "="*70)
    print("TEMPORAL VARIABILITY IN SIMULATION")
    print("="*70)
    
    # Use peak timepoint activity from each sample
    n_times_per_sample = all_pred_array.shape[2]
    
    # Option 1: Treat consecutive samples as temporal windows
    # Get peak activity per sample
    activity_timeline = np.zeros((n_sources, n_samples))
    for i in range(n_samples):
        pred = all_pred_array[i]  # (n_sources, n_times)
        total = np.abs(pred).sum(axis=0)
        peak_t = np.argmax(total)
        activity_timeline[:, i] = pred[:, peak_t]
    
    power_simu = np.sum(activity_timeline ** 2, axis=0)
    power_ratio_simu = power_simu.max() / power_simu.min()
    
    # Detect flashes in simulation
    flash_idx_simu, n_flashes_simu = detect_flashes(power_simu, z_threshold=2.0)
    
    print(f"\nSimulation Statistics (treating samples as temporal windows):")
    print(f"  Power ratio (max/min): {power_ratio_simu:.2f}x")
    print(f"  Flashes detected (z>2): {n_flashes_simu}")
    
    # Frame-to-frame jumps
    jumps_simu = np.abs(np.diff(power_simu))
    print(f"  Mean frame-to-frame jump: {jumps_simu.mean():.6e}")
    print(f"  Max frame-to-frame jump:  {jumps_simu.max():.6e}")
    
    # Apply smoothing and check if it degrades localization
    print("\n" + "="*70)
    print("SMOOTHING IMPACT ON LOCALIZATION")
    print("="*70)
    
    # EMA smoothing
    activity_ema = apply_exponential_smoothing(activity_timeline, alpha=0.15, bidirectional=True)
    power_ema = np.sum(activity_ema ** 2, axis=0)
    _, n_flashes_ema = detect_flashes(power_ema, z_threshold=2.0)
    
    # Kalman smoothing
    activity_kalman = apply_kalman_filter(activity_timeline, process_noise=0.01)
    power_kalman = np.sum(activity_kalman ** 2, axis=0)
    _, n_flashes_kalman = detect_flashes(power_kalman, z_threshold=2.0)
    
    # Compute localization error with smoothed predictions
    def compute_smoothed_errors(smoothed_activity, truth_activity, positions):
        errors = []
        for i in range(smoothed_activity.shape[1]):
            # Get corresponding ground truth
            truth = truth_activity[:, i]
            pred = smoothed_activity[:, i]
            com_err, _, corr = compute_localization_error(pred, truth, positions)
            errors.append((com_err, corr))
        return errors
    
    # Get ground truth activity timeline
    truth_timeline = np.zeros((n_sources, n_samples))
    for i in range(n_samples):
        truth = all_truth_array[i]
        total = np.abs(truth).sum(axis=0)
        peak_t = np.argmax(total)
        truth_timeline[:, i] = truth[:, peak_t]
    
    raw_errors = compute_smoothed_errors(activity_timeline, truth_timeline, positions_mm)
    ema_errors = compute_smoothed_errors(activity_ema, truth_timeline, positions_mm)
    kalman_errors = compute_smoothed_errors(activity_kalman, truth_timeline, positions_mm)
    
    print(f"\n{'Method':<25} {'COM Error (mm)':<20} {'Correlation':<15} {'Flashes':<10}")
    print("-" * 70)
    
    raw_com = np.mean([e[0] for e in raw_errors])
    raw_corr = np.mean([e[1] for e in raw_errors])
    print(f"{'Raw (no smoothing)':<25} {raw_com:<20.1f} {raw_corr:<15.3f} {n_flashes_simu:<10}")
    
    ema_com = np.mean([e[0] for e in ema_errors])
    ema_corr = np.mean([e[1] for e in ema_errors])
    print(f"{'EMA (α=0.15)':<25} {ema_com:<20.1f} {ema_corr:<15.3f} {n_flashes_ema:<10}")
    
    kalman_com = np.mean([e[0] for e in kalman_errors])
    kalman_corr = np.mean([e[1] for e in kalman_errors])
    print(f"{'Kalman (Q=0.01)':<25} {kalman_com:<20.1f} {kalman_corr:<15.3f} {n_flashes_kalman:<10}")
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON: SIMULATION vs REAL EDF")
    print("="*70)
    
    # These are the values from the real EDF analysis
    real_power_ratio = 79.82
    real_n_flashes = 16
    
    print(f"\n{'Metric':<30} {'Simulation':<20} {'Real EDF':<20}")
    print("-" * 70)
    print(f"{'Power ratio (max/min)':<30} {power_ratio_simu:<20.2f} {real_power_ratio:<20.2f}")
    print(f"{'Flashes detected (z>2)':<30} {n_flashes_simu:<20} {real_n_flashes:<20}")
    
    if power_ratio_simu < real_power_ratio * 0.5:
        print(f"\n✅ Simulation has {real_power_ratio/power_ratio_simu:.1f}x LESS variability than real EDF")
        print("   This suggests the 'flashes' in real data are from the EEG itself,")
        print("   not from the inverse solver.")
    elif power_ratio_simu > real_power_ratio * 2:
        print(f"\n⚠️  Simulation has {power_ratio_simu/real_power_ratio:.1f}x MORE variability than real EDF")
        print("   This suggests the solver may be contributing to artifacts.")
    else:
        print(f"\n⚠️  Simulation has comparable variability to real EDF")
        print("   The flashes may be a combination of solver behavior and real EEG variability.")
    
    # Check smoothing impact
    print("\n" + "="*70)
    print("SMOOTHING RECOMMENDATIONS")
    print("="*70)
    
    if ema_com < raw_com * 1.5 and ema_corr > raw_corr * 0.8:
        print("\n✅ EMA smoothing maintains localization accuracy")
        print(f"   COM error: {raw_com:.1f} → {ema_com:.1f} mm ({(ema_com/raw_com-1)*100:+.1f}%)")
        print(f"   Correlation: {raw_corr:.3f} → {ema_corr:.3f}")
    else:
        print("\n⚠️  EMA smoothing degrades localization accuracy")
        print("   Consider using lighter smoothing or different method")
    
    if kalman_com < raw_com * 1.5 and kalman_corr > raw_corr * 0.8:
        print("\n✅ Kalman smoothing maintains localization accuracy")
        print(f"   COM error: {raw_com:.1f} → {kalman_com:.1f} mm ({(kalman_com/raw_com-1)*100:+.1f}%)")
        print(f"   Correlation: {raw_corr:.3f} → {kalman_corr:.3f}")
    else:
        print("\n⚠️  Kalman smoothing degrades localization accuracy")
    
    # Save results
    output_dir = repo_root / "results" / "simulation_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'n_samples': n_samples,
        'eeg_snr': args.eeg_snr,
        'localization': {
            'com_error_mean': float(np.mean(com_errors)),
            'com_error_std': float(np.std(com_errors)),
            'peak_error_mean': float(np.mean(peak_errors)),
            'peak_error_std': float(np.std(peak_errors)),
            'correlation_mean': float(np.mean(correlations)),
            'correlation_std': float(np.std(correlations)),
        },
        'variability': {
            'power_ratio': float(power_ratio_simu),
            'n_flashes': int(n_flashes_simu),
        },
        'smoothing': {
            'ema': {
                'com_error': float(ema_com),
                'correlation': float(ema_corr),
                'n_flashes': int(n_flashes_ema),
            },
            'kalman': {
                'com_error': float(kalman_com),
                'correlation': float(kalman_corr),
                'n_flashes': int(n_flashes_kalman),
            }
        }
    }
    
    with open(output_dir / 'simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to: {output_dir / 'simulation_results.json'}")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Localization error histogram
    ax = axes[0, 0]
    ax.hist(com_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(com_errors), color='red', linestyle='--', label=f'Mean: {np.mean(com_errors):.1f} mm')
    ax.set_xlabel('Center-of-Mass Error (mm)')
    ax.set_ylabel('Count')
    ax.set_title('Localization Error Distribution')
    ax.legend()
    
    # 2. Power over "time" (samples)
    ax = axes[0, 1]
    sample_idx = np.arange(n_samples)
    ax.plot(sample_idx, power_simu / power_simu.mean(), 'b-', alpha=0.7, label='Raw')
    ax.plot(sample_idx, power_ema / power_ema.mean(), 'g-', alpha=0.7, label='EMA')
    ax.plot(sample_idx, power_kalman / power_kalman.mean(), 'r-', alpha=0.7, label='Kalman')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Normalized Power')
    ax.set_title('Source Power Over Samples')
    ax.legend()
    
    # 3. Correlation histogram
    ax = axes[1, 0]
    ax.hist(correlations, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(correlations), color='red', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
    ax.set_xlabel('Correlation (pred vs truth)')
    ax.set_ylabel('Count')
    ax.set_title('Source Activity Correlation Distribution')
    ax.legend()
    
    # 4. Smoothing comparison
    ax = axes[1, 1]
    methods = ['Raw', 'EMA', 'Kalman']
    com_vals = [raw_com, ema_com, kalman_com]
    corr_vals = [raw_corr, ema_corr, kalman_corr]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, com_vals, width, label='COM Error (mm)', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, corr_vals, width, label='Correlation', color='green', alpha=0.7)
    
    ax.set_ylabel('COM Error (mm)', color='blue')
    ax2.set_ylabel('Correlation', color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title('Smoothing Impact on Localization')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = output_dir / 'simulation_validation.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
