#!/usr/bin/env python3
"""
1‑D variable‑density Cartesian k‑space trajectory (Python version)

Gustav Strijkers – May 2025
Amsterdam UMC
g.j.strijkers@amsterdamumc.nl
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────────
def _must_be_pos_even_int(x: int, name: str = "value") -> None:
    """Raise if *x* is not a positive, even integer."""
    if not (isinstance(x, int) and x > 0 and x % 2 == 0):
        raise ValueError(f"{name} must be a positive, even integer (got {x!r}).")


# ────────────────────────────────────────────────────────────────────────────────
# Core
# ────────────────────────────────────────────────────────────────────────────────
def generate_cartesian_1d_trajectory(
    target_kspace_size: int = 192,
    trajectory_length: int = 256,
    sigma: int = 8,
    density_shape: str = "gauss",
    n_samples: int = 240_000,
    plot: bool = True,
    export: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a zig‑zag 1‑D Cartesian trajectory, simulate random filling,
    and (optionally) plot & export the result.

    Returns
    -------
    traj : ndarray[int]
        The k‑line index of every acquired view (length = *trajectory_length*).
    filling : ndarray[float]
        Normalised histogram of how often each k‑line is visited.
    """
    # Sanity checks (mirror the MATLAB assertions)
    _must_be_pos_even_int(target_kspace_size, "target_kspace_size")
    _must_be_pos_even_int(trajectory_length, "trajectory_length")
    _must_be_pos_even_int(sigma, "sigma")

    # ------------------------------------------------------------------ k‑space target
    k_space_center = target_kspace_size // 2 + 1  # MATLAB is 1‑based
    extra_lines = trajectory_length - target_kspace_size
    k = np.arange(1, target_kspace_size + 1)

    if density_shape.lower() == "gauss":
        d = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((k - k_space_center) ** 2) / (2 * sigma**2))
        )
    else:
        raise ValueError(f"Unsupported density_shape: {density_shape!r}")

    # Discretise the density so ∑df == extra_lines
    incr = 0.9
    df = np.round(d * incr).astype(int)
    while df.sum() <= extra_lines:
        incr += 0.001
        df = np.round(d * incr).astype(int)

    pm = True  # plus/minus toggle – identical to MATLAB logic
    while df.sum() > extra_lines:
        loc = np.where(df == 1)[0]
        if loc.size == 0:
            break  # safety
        df[loc[0] if pm else loc[-1]] = 0
        pm = not pm

    d_int = 1 + df  # guaranteed length matching trajectory_length

    # ------------------------------------------------------------------ zig‑zag trajectory
    traj = np.empty(trajectory_length, dtype=int)
    cnt = 0
    for k_idx in range(1, target_kspace_size + 1, 2):  # odd indices ↑
        traj[cnt : cnt + d_int[k_idx - 1]] = k_idx
        cnt += d_int[k_idx - 1]
    for k_idx in range(target_kspace_size, 1, -2):  # even indices ↓
        traj[cnt : cnt + d_int[k_idx - 1]] = k_idx
        cnt += d_int[k_idx - 1]

    traj = traj - target_kspace_size / 2 - 1  # centre to ±k‑max

    # ------------------------------------------------------------------ simulate random filling
    samples = np.random.randint(0, trajectory_length, size=n_samples)
    filling = np.zeros(target_kspace_size, dtype=float)
    for s in samples:
        idx = int(traj[s] + target_kspace_size / 2)  # 0‑based now
        filling[idx] += 1
    filling /= filling.sum()

    # ------------------------------------------------------------------ visualisation
    if plot:
        FIG_W, FIG_H = 12, 5
        LW = 2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), tight_layout=True)

        ax1.plot(traj, linewidth=LW)
        ax1.set_title("Trajectory")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("K‑line")
        ax1.set_xlim(0, trajectory_length)
        ax1.set_ylim(traj.min() * 1.1, traj.max() * 1.1)
        ax1.grid(True)

        ax2.plot(filling, linewidth=LW)
        ax2.set_title("Estimated filling of k‑space")
        ax2.set_xlabel("K‑line")
        ax2.set_ylabel("Filling")
        ax2.set_xlim(0, target_kspace_size)
        ax2.set_ylim(0, filling.max() * 1.1)
        ax2.grid(True)

        plt.show()

    # ------------------------------------------------------------------ export
    if export:
        fname = Path(
            f"cartesian1D_{target_kspace_size}_{trajectory_length}_{sigma}.txt"
        )
        np.savetxt(fname, traj, fmt="%d", delimiter=",")
        print(f"Trajectory written to: {fname}")

    return traj, filling


# ────────────────────────────────────────────────────────────────────────────────
# Script entry point
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_cartesian_1d_trajectory()
