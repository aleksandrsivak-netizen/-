"""Matplotlib visualization helpers for demo artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dem import DEMData
from .simulator import Trajectory


def save_trajectory_overlay(
    dem: DEMData,
    truth_trajectory: Trajectory | None,
    estimated_trajectory: dict,
    output_path: str,
) -> str:
    """Save a DEM map with truth and estimated trajectories."""

    plt = _load_pyplot()
    output = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    extent = [
        dem.origin_x_m,
        dem.origin_x_m + dem.width_m,
        dem.origin_y_m,
        dem.origin_y_m + dem.height_m,
    ]
    image = ax.imshow(dem.elevation, origin="lower", extent=extent, cmap="terrain", aspect="equal")
    fig.colorbar(image, ax=ax, label="Elevation MSL, m")

    if truth_trajectory is not None:
        ax.plot(truth_trajectory.x_m, truth_trajectory.y_m, color="white", linewidth=2.5, label="Truth")
        ax.plot(truth_trajectory.x_m[0], truth_trajectory.y_m[0], "o", color="white", markersize=6)

    start = estimated_trajectory.get("start", {})
    end = estimated_trajectory.get("end", {})
    if start and end:
        ax.plot(
            [start.get("x_m"), end.get("x_m")],
            [start.get("y_m"), end.get("y_m")],
            color="crimson",
            linewidth=2.0,
            linestyle="--",
            label="Estimated",
        )
        ax.plot(start.get("x_m"), start.get("y_m"), "o", color="crimson", markersize=6)

    ax.set_title("Terrain Overlay With Flight Track")
    ax.set_xlabel("Local X, m")
    ax.set_ylabel("Local Y, m")
    ax.legend(loc="best")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output)


def save_correlation_heatmap(
    heatmap: np.ndarray,
    azimuth_values: np.ndarray,
    output_path: str,
) -> str:
    """Save a candidate score heatmap."""

    plt = _load_pyplot()
    output = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    image = ax.imshow(np.asarray(heatmap, dtype=float), origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="Combined score")
    ax.set_title("Correlation Search Heatmap")
    ax.set_xlabel("Candidate X index")
    ax.set_ylabel("Candidate Y index")
    if azimuth_values.size:
        ax.text(
            0.01,
            0.99,
            f"Azimuth bins: {azimuth_values.size}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=9,
        )
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output)


def save_profile_comparison(
    measured_profile: np.ndarray,
    reference_profile: np.ndarray,
    output_path: str,
) -> str:
    """Save a measured-vs-reference terrain profile plot."""

    plt = _load_pyplot()
    output = _prepare_output_path(output_path)
    measured = np.asarray(measured_profile, dtype=float)
    reference = np.asarray(reference_profile, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(measured, label="Measured terrain", linewidth=2.0)
    ax.plot(reference, label="Best DEM reference", linewidth=1.8, linestyle="--")
    ax.set_title("Measured Terrain Profile vs DEM Reference")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Elevation MSL, m")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output)


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save demo plots") from exc
    return plt


def _prepare_output_path(output_path: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
