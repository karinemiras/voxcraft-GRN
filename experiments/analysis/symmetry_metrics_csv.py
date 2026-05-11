import csv
import importlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from utils.config import Config


def build_algorithm(args):
    module = importlib.import_module(f"algorithms.{args.algorithm}")
    cls = getattr(module, "EA")
    original_argv = sys.argv
    try:
        sys.argv = [
            "symmetry_metrics_csv.py",
            "--study_name", args.study_name,
            "--experiments", args.experiments,
            "--runs", args.runs,
            "--out_path", args.out_path,
            "--max_voxels", str(args.max_voxels),
            "--cube_face_size", str(args.cube_face_size),
            "--env_conditions", args.env_conditions,
            "--plastic", str(args.plastic),
            "--algorithm", args.algorithm,
            "--enforced_symmetry", str(args.enforced_symmetry),
        ]
        return cls(args)
    finally:
        sys.argv = original_argv


def axis_symmetry(phenotype, axis):
    occupied = phenotype != 0
    mirrored_occupied = np.flip(occupied, axis=axis)
    mirrored_modules = int((occupied & mirrored_occupied).sum())
    total_modules = int(occupied.sum())
    return round(mirrored_modules / total_modules, 4)


def axis_type_symmetry(phenotype, axis):
    occupied = phenotype != 0
    mirrored_phenotype = np.flip(phenotype, axis=axis)
    mirrored_same_type_modules = int((occupied & (phenotype == mirrored_phenotype)).sum())
    total_modules = int(occupied.sum())
    return round(mirrored_same_type_modules / total_modules, 4)


def symmetry_metrics(phenotype):
    return {
        "x_symmetry": axis_symmetry(phenotype, axis=0),
        "y_symmetry": axis_symmetry(phenotype, axis=1),
        "z_symmetry": axis_symmetry(phenotype, axis=2),
        "x_type_symmetry": axis_type_symmetry(phenotype, axis=0),
        "y_type_symmetry": axis_type_symmetry(phenotype, axis=1),
        "z_type_symmetry": axis_type_symmetry(phenotype, axis=2),
    }


def main():
    args = Config()._get_params()
    experiments = args.experiments.split(",")
    voxel_types_list = args.voxel_types.split(",")
    runs = [int(run) for run in args.runs.split(",")]
    ea = build_algorithm(args)

    output_csv = args.output_csv
    if not output_csv:
        output_csv = f"{args.out_path}/{args.study_name}/analysis/additional_metrics.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = [
        "study_name",
        "experiment_name",
        "run",
        "robot_id",
        "born_generation",
        "num_voxels",
        "x_symmetry",
        "y_symmetry",
        "z_symmetry",
        "x_type_symmetry",
        "y_type_symmetry",
        "z_type_symmetry",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp_idx, experiment_name in enumerate(experiments):
            voxel_types = voxel_types_list[exp_idx]
            print(f"Measuring symmetry for {experiment_name}...")

            for run in runs:
                db_path = f"{args.out_path}/{args.study_name}/{experiment_name}/run_{run}/run_{run}"
                if not os.path.exists(db_path):
                    print(f"Skipping missing DB: {db_path}")
                    continue

                print(f"  run {run}: {db_path}")
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        "SELECT robot_id, born_generation, genome FROM all_robots ORDER BY robot_id"
                    )

                    count = 0
                    for row in rows:
                        count += 1
                        genome = json.loads(row["genome"])
                        phenotype = ea.develop_phenotype(genome, voxel_types)
                        metrics = symmetry_metrics(phenotype)
                        writer.writerow({
                            "study_name": args.study_name,
                            "experiment_name": experiment_name,
                            "run": run,
                            "robot_id": row["robot_id"],
                            "born_generation": row["born_generation"],
                            "num_voxels": int((phenotype != 0).sum()),
                            **metrics,
                        })
                        if count % 500 == 0:
                            print(f"    measured {count} robots")
                    print(f"    measured {count} robots")
                finally:
                    conn.close()

    print(f"Saved symmetry metrics to {output_csv}")


if __name__ == "__main__":
    main()
