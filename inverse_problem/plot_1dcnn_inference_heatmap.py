"""Run a local 1dCNN inference and visualise ground-truth vs predicted source activity on a 3D brain."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from loaders import EsiDatasetds_new
from models.cnn_1d import CNN1Dpl
from utils import utl


@dataclass
class BrainSurface:
    vertices_mm: np.ndarray
    faces: np.ndarray


@dataclass
class BrainGeometry:
    positions_mm: np.ndarray
    surfaces: Iterable[BrainSurface]


def _ensure_within_repo(path: Path, repo_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.relative_to(repo_root)
    return resolved


def _find_default_model_checkpoint(dataset_token: str, inter_layer: int) -> Optional[Path]:
    candidates = list(REPO_ROOT.glob("results/**/1dcnn_model.pt"))
    if not candidates:
        return None

    token = dataset_token.lower()
    inter = f"interlayer_{inter_layer}".lower()

    def score(path: Path) -> int:
        lower = path.as_posix().lower()
        s = 0
        if token in lower:
            s += 4
        if inter in lower:
            s += 3
        if "model_1dcnn" in lower:
            s += 2
        if "sereega" in lower:
            s += 1
        return s

    best = max(candidates, key=score)
    if score(best) == 0:
        return None
    return best


def _load_brain_geometry(head_model: HeadModel.HeadModel) -> BrainGeometry:
    surfaces = []
    for hemi in head_model.fwd["src"]:
        rr_mm = hemi["rr"] * 1e3
        tris = hemi["tris"].copy()
        surfaces.append(BrainSurface(vertices_mm=rr_mm, faces=tris))

    positions_mm = head_model.source_space.positions * 1e3
    return BrainGeometry(positions_mm=positions_mm, surfaces=surfaces)


def _set_axes_equal(ax, coords: np.ndarray) -> None:
    max_range = np.max(coords.max(axis=0) - coords.min(axis=0))
    mid = coords.mean(axis=0)
    half = max_range / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)


def _plot_heatmap(ax, geom: BrainGeometry, activity: np.ndarray, title: str, norm: Normalize, cmap: str):
    for surf in geom.surfaces:
        mesh = Poly3DCollection(
            surf.vertices_mm[surf.faces],
            alpha=0.05,
            facecolor=(0.8, 0.8, 0.8),
            edgecolor="none",
        )
        ax.add_collection3d(mesh)

    scatter = ax.scatter(
        geom.positions_mm[:, 0],
        geom.positions_mm[:, 1],
        geom.positions_mm[:, 2],
        c=activity,
        cmap=cmap,
        norm=norm,
        s=12,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    _set_axes_equal(ax, geom.positions_mm)
    ax.view_init(elev=20, azim=120)
    return scatter


def _infer_model(
    model: CNN1Dpl,
    eeg: torch.Tensor,
    train_loss: str,
    max_src_val: float,
    max_eeg_val: float,
    leadfield: torch.Tensor,
) -> torch.Tensor:
    eeg = eeg.float()
    with torch.no_grad():
        pred_norm = model.model(eeg.unsqueeze(0)).squeeze(0)

    if train_loss.lower() == "cosine":
        eeg_unscaled = eeg * max_eeg_val
        pred_scaled = utl.gfp_scaling(eeg_unscaled, pred_norm, leadfield)
    else:
        pred_scaled = pred_norm * max_src_val

    return pred_scaled


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise 1dCNN inference vs ground-truth on a 3D brain.")
    parser.add_argument("--simu_name", default="mes_debug", help="Name of the simulation inside simulation/fsaverage/.../simu/.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the sample to render.")
    parser.add_argument("--to_load", type=int, default=-1, help="How many samples to load (-1 = all available).")
    parser.add_argument("--eeg_snr", type=float, default=5.0, help="SNR used when loading the dataset.")
    parser.add_argument("--orientation", default="constrained", choices=["constrained", "unconstrained"], help="Source orientation.")
    parser.add_argument("--electrode_montage", default="standard_1020", help="Electrode montage name.")
    parser.add_argument("--source_space", default="ico3", help="Source space name (e.g. ico3).")
    parser.add_argument("--subject", default="fsaverage", help="Subject folder under simulation/.")
    parser.add_argument("--model_path", type=str, default=None, help="Optional explicit path to 1dCNN weights inside the repo.")
    parser.add_argument("--inter_layer", type=int, default=4096, help="Intermediate channel size used during training.")
    parser.add_argument("--kernel_size", type=int, default=5, help="1dCNN kernel size used during training.")
    parser.add_argument("--train_loss", default="cosine", help="Loss used during training (affects post-scaling).")
    parser.add_argument("--norm", default="linear", choices=["linear", "max-max"], help="Normalisation used during dataset generation.")
    parser.add_argument("--cmap", default="inferno", help="Colormap for activity heatmaps.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory inside the repository where the figure will be saved (defaults to results/<dataset>/eval/heatmaps).",
    )
    parser.add_argument("--show", action="store_true", help="Display the interactive figure in addition to saving it.")

    args = parser.parse_args()

    repo_root = REPO_ROOT
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
        raise FileNotFoundError(f"Could not find simulation config: {config_path}")

    with config_path.open() as f:
        general_config = json.load(f)

    general_config["eeg_snr"] = args.eeg_snr
    general_config["simu_name"] = args.simu_name
    general_config["electrode_space"]["electrode_montage"] = args.electrode_montage

    folders = FolderStructure(str(simu_root), general_config)
    electrode_space = HeadModel.ElectrodeSpace(folders, general_config)
    source_space = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space, source_space, folders, args.subject)

    dataset_len = len(general_config.get("ids", []))
    if dataset_len == 0:
        dataset_len = 1
    to_load = dataset_len if args.to_load < 0 else min(args.to_load, dataset_len)

    dataset = EsiDatasetds_new(
        str(simu_root),
        str(config_path),
        args.simu_name,
        args.source_space,
        args.electrode_montage,
        to_load,
        args.eeg_snr,
        noise_type={"white": 1.0, "pink": 0.0},
        norm=args.norm,
    )

    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(f"sample_idx {args.sample_idx} is out of bounds for dataset of length {len(dataset)}")

    dataset_token = f"{args.simu_name}{args.source_space}_"
    if args.model_path:
        model_path = _ensure_within_repo(Path(args.model_path), repo_root)
    else:
        checkpoint = _find_default_model_checkpoint(dataset_token, args.inter_layer)
        if checkpoint is None:
            raise FileNotFoundError(
                "Could not automatically locate a 1dCNN checkpoint inside the repository."
            )
        model_path = _ensure_within_repo(checkpoint, repo_root)

    net_kwargs = {
        "channels": [
            head_model.electrode_space.n_electrodes,
            args.inter_layer,
            head_model.source_space.n_sources,
        ],
        "kernel_size": args.kernel_size,
        "bias": False,
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,
    }

    model = CNN1Dpl(**net_kwargs)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    eeg_sample, src_sample = dataset[args.sample_idx]
    eeg_sample = eeg_sample.float()
    src_sample = src_sample.float()

    max_eeg_val = float(dataset.max_eeg[args.sample_idx].item())
    max_src_val = float(dataset.max_src[args.sample_idx].item())
    src_unscaled = src_sample * max_src_val

    leadfield = torch.from_numpy(head_model.fwd["sol"]["data"]).float()
    pred_unscaled = _infer_model(
        model,
        eeg_sample,
        args.train_loss,
        max_src_val,
        max_eeg_val,
        leadfield,
    )

    energy_over_time = src_unscaled.abs().sum(dim=0)
    peak_t = int(torch.argmax(energy_over_time).item())

    gt_activity = src_unscaled[:, peak_t].abs().cpu().numpy()
    pred_activity = pred_unscaled[:, peak_t].abs().cpu().numpy()

    geom = _load_brain_geometry(head_model)
    vmax = max(float(gt_activity.max()), float(pred_activity.max()), 1e-9)
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(14, 6))
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")

    sc1 = _plot_heatmap(ax_gt, geom, gt_activity, "Ground Truth Activity", norm, args.cmap)
    sc2 = _plot_heatmap(ax_pred, geom, pred_activity, "1dCNN Inferred Activity", norm, args.cmap)

    fig.subplots_adjust(wspace=0.05, hspace=0.0)
    cbar = fig.colorbar(sc2, ax=[ax_gt, ax_pred], shrink=0.75, pad=0.02)
    cbar.set_label("|Activity| (a.u.)")

    fig.suptitle(
        f"Simulation '{args.simu_name}' — sample {args.sample_idx} @ t={peak_t} (of {src_sample.shape[1]} time points)",
        fontsize=12,
    )
    default_output = (
        repo_root
        / "results"
        / dataset_token
        / "eval"
        / "heatmaps"
    )
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output
    output_dir = _ensure_within_repo(output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / f"{args.simu_name}_sample{args.sample_idx:03d}_t{peak_t}.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")

    print(f"Saved heatmap figure to {figure_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
