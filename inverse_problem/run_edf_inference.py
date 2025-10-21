#!/usr/bin/env python3
"""Convert long-form EDF recordings to the 1dCNN input format, run inference, and emit plots."""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import torch
from scipy.io import savemat

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
    _save_interactive_heatmap,
)

@dataclass
class SegmentSummary:
    index: int
    start_sample: int
    start_time_seconds: float
    eeg_max_abs: float
    mat_path: Path


@dataclass
class BestWindow:
    summary: SegmentSummary
    prediction: torch.Tensor
    peak_time_index: int
    peak_activity: np.ndarray
    score: float


def _canonical_lookup(names: Sequence[str]) -> Dict[str, str]:
    return {name.lower(): name for name in names}


def _rename_channels_to_target(raw: mne.io.BaseRaw, target_names: Sequence[str]) -> None:
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


def _build_full_montage_raw(
    source_raw: mne.io.BaseRaw,
    target_info: mne.Info,
    target_names: Sequence[str],
) -> Tuple[mne.io.RawArray, List[str]]:
    data = np.zeros((len(target_names), source_raw.n_times), dtype=np.float64)
    missing: List[str] = []
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


def _segment_signal(
    data: np.ndarray,
    window_samples: int,
    step_samples: int,
    pad: bool,
    max_windows: Optional[int],
) -> Iterable[Tuple[int, np.ndarray]]:
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


def _save_segment_to_mat(segment: np.ndarray, path: Path) -> None:
    savemat(str(path), {"eeg_data": {"EEG": segment.astype(np.float32)}})


def _load_model(
    checkpoint_path: Path,
    n_electrodes: int,
    n_sources: int,
    inter_layer: int,
    kernel_size: int,
) -> CNN1Dpl:
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


def _compute_window_score(pred: torch.Tensor) -> Tuple[float, int]:
    energy = pred.abs().sum(dim=0)
    peak_idx = int(torch.argmax(energy).item())
    return float(energy[peak_idx].item()), peak_idx


def _prepare_output_dir(base_dir: Optional[str], repo_root: Path, edf_path: Path) -> Path:
    if base_dir:
        out_dir = Path(base_dir).expanduser()
    else:
        out_dir = repo_root / "results" / "edf_inference" / edf_path.stem
    out_dir = _ensure_within_repo(out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "segments").mkdir(exist_ok=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1dCNN inference on an EDF recording.")
    parser.add_argument("edf_path", type=str, help="Path to the EDF file to process.")
    parser.add_argument(
        "--simu_name",
        default="mes_debug",
        help="Simulation name matching the training configuration (used to locate head model metadata).",
    )
    parser.add_argument(
        "--subject",
        default="fsaverage",
        help="Subject folder under simulation/ containing the head model assets.",
    )
    parser.add_argument(
        "--orientation",
        default="constrained",
        choices=["constrained", "unconstrained"],
        help="Source orientation of the head model.",
    )
    parser.add_argument(
        "--electrode_montage",
        default="standard_1020",
        help="Electrode montage name used during training.",
    )
    parser.add_argument(
        "--source_space",
        default="ico3",
        help="Source space name (e.g. ico3).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Explicit path to 1dCNN checkpoint; if omitted, the repository results folder is searched automatically.",
    )
    parser.add_argument(
        "--inter_layer",
        type=int,
        default=4096,
        help="Intermediate channel size used during training.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="1dCNN kernel size used during training.",
    )
    parser.add_argument(
        "--train_loss",
        default="cosine",
        help="Loss used during training (controls post-scaling).",
    )
    parser.add_argument(
        "--norm",
        default="linear",
        choices=["linear", "max-max"],
        help="Normalisation used during dataset generation.",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=None,
        help="Optional window duration in seconds. Defaults to the training n_times / fs.",
    )
    parser.add_argument(
        "--overlap_fraction",
        type=float,
        default=0.0,
        help="Fractional overlap between consecutive windows (0 = no overlap, 0.5 = 50%%).",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Optional cap on the number of windows to process (processed sequentially).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where segments, plots, and metadata will be stored (defaults under results/edf_inference/).",
    )
    parser.add_argument(
        "--open_plot",
        action="store_true",
        help="Open the resulting interactive plot in a browser after generation.",
    )
    parser.add_argument(
        "--no_pad_last",
        action="store_true",
        help="If set, drop the trailing window instead of padding to full length.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = REPO_ROOT
    edf_path = Path(args.edf_path).expanduser().resolve()
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    try:
        _ensure_within_repo(edf_path, repo_root)
    except ValueError:
        print(f"Warning: EDF file {edf_path} is outside the repository; proceeding regardless.")

    simu_root = repo_root / "simulation" / args.subject
    simu_root = _ensure_within_repo(simu_root, repo_root)

    config_path = (
        simu_root
        / args.orientation
        / args.electrode_montage
        / args.source_space
        / "simu"
        / args.simu_name
        / f"{args.simu_name}{args.source_space}_config.json"
    )
    config_path = _ensure_within_repo(config_path, repo_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not locate simulation config: {config_path}")

    with config_path.open() as f:
        general_config = json.load(f)

    general_config["simu_name"] = args.simu_name
    general_config["electrode_space"]["electrode_montage"] = args.electrode_montage
    general_config.setdefault("eeg_snr", "infdb")

    folders = FolderStructure(str(simu_root), general_config)
    electrode_space = HeadModel.ElectrodeSpace(folders, general_config)
    source_space = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space, source_space, folders, args.subject)

    if args.window_seconds is None:
        window_samples = int(general_config["rec_info"]["n_times"])
        fs = float(general_config["rec_info"]["fs"])
        window_seconds = window_samples / fs
    else:
        fs = float(general_config["rec_info"]["fs"])
        window_seconds = float(args.window_seconds)
        window_samples = max(1, int(round(window_seconds * fs)))

    overlap_fraction = min(max(args.overlap_fraction, 0.0), 0.95)
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_fraction))))

    output_dir = _prepare_output_dir(args.output_dir, repo_root, edf_path)
    segments_dir = output_dir / "segments"

    if args.model_path:
        model_path = _ensure_within_repo(Path(args.model_path), repo_root)
    else:
        token = f"{args.simu_name}{args.source_space}_"
        checkpoint = _find_default_model_checkpoint(token, args.inter_layer)
        if checkpoint is None:
            raise FileNotFoundError("Unable to locate 1dCNN checkpoint automatically. Specify --model_path explicitly.")
        model_path = _ensure_within_repo(checkpoint, repo_root)

    model = _load_model(
        model_path,
        head_model.electrode_space.n_electrodes,
        head_model.source_space.n_sources,
        args.inter_layer,
        args.kernel_size,
    )

    print(f"Loaded EDF: {edf_path}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick(picks="eeg")
    _rename_channels_to_target(raw, head_model.electrode_space.info.ch_names)
    try:
        raw.set_montage("standard_1020", match_case=False, verbose=False, on_missing="warn")
    except RuntimeError:
        pass

    raw.set_eeg_reference("average", projection=False, verbose=False)

    if not math.isclose(raw.info["sfreq"], fs, rel_tol=1e-6, abs_tol=1e-6):
        raw.resample(fs, npad="auto", verbose=False)

    full_raw, missing = _build_full_montage_raw(raw, head_model.electrode_space.info, head_model.electrode_space.info.ch_names)
    if missing:
        print(f"Interpolated {len(missing)} missing channels to match training montage.")
    full_raw.set_eeg_reference("average", projection=False, verbose=False)

    data = full_raw.get_data()
    leadfield = torch.from_numpy(head_model.fwd["sol"]["data"]).float()

    segment_summaries: List[SegmentSummary] = []
    best_window: Optional[BestWindow] = None
    any_segment = False

    for seg_idx, item in enumerate(
        _segment_signal(
            data,
            window_samples,
            step_samples,
            pad=not args.no_pad_last,
            max_windows=args.max_windows,
        )
    ):
        any_segment = True
        start_sample, segment = item
        max_abs = float(np.abs(segment).max())
        if max_abs <= 0:
            print(f"Skipping window {seg_idx} because it contains zero variance.")
            continue

        mat_path = segments_dir / f"segment_{seg_idx:04d}.mat"
        _save_segment_to_mat(segment, mat_path)

        summary = SegmentSummary(
            index=seg_idx,
            start_sample=start_sample,
            start_time_seconds=start_sample / fs,
            eeg_max_abs=max_abs,
            mat_path=mat_path,
        )
        segment_summaries.append(summary)

        eeg_tensor = torch.from_numpy((segment / max_abs).astype(np.float32))
        pred = _infer_model(
            model,
            eeg_tensor,
            args.train_loss,
            max_src_val=1.0,
            max_eeg_val=max_abs,
            leadfield=leadfield,
        )
        score, peak_idx = _compute_window_score(pred)
        peak_activity = pred[:, peak_idx].abs().cpu().numpy()

        if best_window is None or score > best_window.score:
            best_window = BestWindow(
                summary=summary,
                prediction=pred.detach().cpu(),
                peak_time_index=peak_idx,
                peak_activity=peak_activity,
                score=score,
            )

    if best_window is None or not any_segment:
        raise RuntimeError("All windows were skipped due to zero variance or preprocessing issues.")

    metadata_path = output_dir / "segments_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(
            [
                {
                    "index": s.index,
                    "start_sample": s.start_sample,
                    "start_time_seconds": s.start_time_seconds,
                    "segment_seconds": window_seconds,
                    "eeg_max_abs": s.eeg_max_abs,
                    "mat_path": str(s.mat_path.relative_to(output_dir)),
                }
                for s in segment_summaries
            ],
            f,
            indent=2,
        )

    geom: BrainGeometry = _load_brain_geometry(head_model)
    cmap_name = "inferno"

    interactive_path = output_dir / (
        f"{edf_path.stem}_window{best_window.summary.index:04d}_t{best_window.peak_time_index:03d}_interactive.html"
    )
    _save_interactive_heatmap(
        geom,
        best_window.peak_activity,
        title=(
            f"1dCNN Inference — window {best_window.summary.index} @ t={best_window.peak_time_index}"
        ),
        cmap_name=cmap_name,
        output_path=interactive_path,
    )

    print(f"Saved interactive plot to {interactive_path}")

    summary_path = output_dir / "best_window_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "window_index": best_window.summary.index,
                "start_time_seconds": best_window.summary.start_time_seconds,
                "window_duration_seconds": window_seconds,
                "peak_time_index": best_window.peak_time_index,
                "peak_time_seconds": best_window.peak_time_index / fs,
                "score": best_window.score,
                "segment_mat": str(best_window.summary.mat_path.relative_to(output_dir)),
                "interactive_plot": str(interactive_path.relative_to(output_dir)),
            },
            f,
            indent=2,
        )

    if args.open_plot:
        webbrowser.open(interactive_path.as_uri())


if __name__ == "__main__":
    main()
