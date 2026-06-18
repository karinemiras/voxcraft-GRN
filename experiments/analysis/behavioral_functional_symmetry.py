import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np


STEP_RE = re.compile(r"<<<Step(\d+) Time:([0-9.eE+-]+)>>>(.*?)<<<>>>")


def parse_history(history_path):
    text = Path(history_path).read_text(encoding="utf-8", errors="ignore")
    frames = []
    times = []
    materials = None

    for match in STEP_RE.finditer(text):
        voxels = []
        frame_materials = []

        for record in match.group(3).split(";"):
            values = [value for value in record.strip().split(",") if value != ""]
            if len(values) < 14:
                continue

            # The history format stores each voxel as x,y,z,...,material,...
            voxels.append([float(values[0]), float(values[1]), float(values[2])])
            frame_materials.append(int(float(values[13])))

        if voxels:
            times.append(float(match.group(2)))
            frames.append(voxels)
            if materials is None:
                materials = frame_materials

    if not frames:
        raise ValueError(f"No history frames found in {history_path}")

    return np.asarray(frames, dtype=float), np.asarray(times, dtype=float), materials


def correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def best_lagged_correlation(a, b, max_lag):
    best_corr = np.nan
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        # Negative lag: b is shifted earlier. Positive lag: a is shifted earlier.
        if lag < 0:
            aa = a[:lag]
            bb = b[-lag:]
        elif lag > 0:
            aa = a[lag:]
            bb = b[:-lag]
        else:
            aa = a
            bb = b

        if len(aa) < 3:
            continue

        corr = correlation(aa, bb)
        if np.isnan(best_corr) or corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_corr, best_lag


def y_mirrored_pairs(initial_positions):
    pairs = []
    used = set()
    y_min = np.min(initial_positions[:, 1])
    y_max = np.max(initial_positions[:, 1])
    y_center_twice = y_min + y_max

    # Pair voxels using their initial x,z coordinates and mirrored y coordinate.
    lookup = {
        tuple(np.round(position, 6)): idx
        for idx, position in enumerate(initial_positions)
    }

    for idx, position in enumerate(initial_positions):
        if idx in used:
            continue

        mirrored = np.array([position[0], y_center_twice - position[1], position[2]])
        mirror_idx = lookup.get(tuple(np.round(mirrored, 6)))

        if mirror_idx is None or mirror_idx == idx or mirror_idx in used:
            continue

        pairs.append((idx, mirror_idx))
        used.add(idx)
        used.add(mirror_idx)

    return pairs


def local_x_velocity(positions):
    # Remove whole-body translation, then measure each voxel's internal x-motion.
    com_x = positions[:, :, 0].mean(axis=1)
    local_x = positions[:, :, 0] - com_x[:, None]
    return np.diff(local_x, axis=0)


def left_right_motion_metrics(positions, max_lag_fraction):
    pairs = y_mirrored_pairs(positions[0])
    x_velocity = local_x_velocity(positions)
    n_steps = x_velocity.shape[0]
    max_lag = max(1, int(round(n_steps * max_lag_fraction)))

    same_time_corrs = []
    best_lag_corrs = []
    best_lag_fractions = []

    for left_idx, right_idx in pairs:
        left_signal = x_velocity[:, left_idx]
        right_signal = x_velocity[:, right_idx]

        # Metric 1: high means mirrored sides move together along x at the same time.
        same_time_corrs.append(correlation(left_signal, right_signal))

        # Metric 2: high lag with high correlation means delayed/alternating coordination.
        best_corr, best_lag = best_lagged_correlation(left_signal, right_signal, max_lag)
        best_lag_corrs.append(best_corr)
        best_lag_fractions.append(abs(best_lag) / n_steps)

    return {
        "y_mirrored_pair_count": len(pairs),
        "lr_same_time_x_correlation": np.nanmean(same_time_corrs) if same_time_corrs else np.nan,
        "lr_best_lag_x_correlation": np.nanmean(best_lag_corrs) if best_lag_corrs else np.nan,
        "lr_best_lag_fraction": np.nanmean(best_lag_fractions) if best_lag_fractions else np.nan,
    }


def front_back_motion_metric(positions):
    initial_x = positions[0, :, 0]
    x_mid = np.median(initial_x)
    back_indices = np.where(initial_x <= x_mid)[0]
    front_indices = np.where(initial_x > x_mid)[0]

    # If the median split is degenerate, split sorted voxels into two halves.
    if len(back_indices) == 0 or len(front_indices) == 0:
        sorted_indices = np.argsort(initial_x)
        half = len(sorted_indices) // 2
        back_indices = sorted_indices[:half]
        front_indices = sorted_indices[half:]

    x_velocity = local_x_velocity(positions)

    # Metric 3: high means front and back have different internal x-motion patterns.
    front_signal = x_velocity[:, front_indices].mean(axis=1)
    back_signal = x_velocity[:, back_indices].mean(axis=1)
    raw_difference = np.mean(np.abs(front_signal - back_signal))
    scale = np.mean(np.abs(front_signal)) + np.mean(np.abs(back_signal))
    normalized_difference = raw_difference / scale if scale > 0 else np.nan

    return {
        "front_voxel_count": len(front_indices),
        "back_voxel_count": len(back_indices),
        "front_back_x_motion_difference": raw_difference,
        "front_back_x_motion_difference_norm": normalized_difference,
    }


def metrics_for_history(history_path, max_lag_fraction):
    positions, times, materials = parse_history(history_path)
    row = {
        "history_path": history_path,
        "history_file": os.path.basename(history_path),
        "num_frames": positions.shape[0],
        "num_voxels": positions.shape[1],
        "duration": times[-1] - times[0] if len(times) > 1 else 0.0,
    }
    row.update(left_right_motion_metrics(positions, max_lag_fraction))
    row.update(front_back_motion_metric(positions))
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Measure simple behavioral functional-symmetry metrics from Voxcraft .history files."
    )
    parser.add_argument("histories", nargs="+", help="one or more .history files")
    parser.add_argument("--output_csv", default="", help="optional CSV output path")
    parser.add_argument(
        "--max_lag_fraction",
        default=0.5,
        type=float,
        help="largest time shift to test for left-right lag, as a fraction of the history length",
    )
    args = parser.parse_args()

    rows = [
        metrics_for_history(history_path, args.max_lag_fraction)
        for history_path in args.histories
    ]

    fieldnames = list(rows[0].keys())
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.output_csv}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
