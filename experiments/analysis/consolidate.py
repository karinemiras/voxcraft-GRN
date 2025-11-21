import argparse
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, select

# make repo root importable 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from algorithms.EA_classes import Robot, GenerationSurvivor 
from utils.config import Config


class Analysis:
    def __init__(self, args):
        self.study_name = args.study_name
        self.experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
        self.runs = [int(r) for r in args.runs.split(",") if r.strip()]
        self.final_gen = int(args.final_gen)
        self.path = f"{args.out_path}/{self.study_name}"

        # which columns to summarize
        self.metrics = ["uniqueness",
                        "fitness",
                        "num_voxels"]

    def _resolve_db_path(self, base_path: str):
        """
        Accept either:
          - a direct sqlite file (…/run_X/experiment.sqlite3 or custom name), or
          - a run directory (…/run_X) containing experiment.sqlite3
        """
        if os.path.isdir(base_path):
            candidate = os.path.join(base_path, "experiment.sqlite3")
            return candidate if os.path.exists(candidate) else None
        return base_path if os.path.exists(base_path) else None

    def consolidate(self):
        print("consolidating...")

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(f"{self.path}/analysis", exist_ok=True)

        frames = []
        robots_frames = []

        for experiment in self.experiments:
            for run in self.runs:
                # user provided a directory path earlier; support both cases
                db_base = os.path.join(self.path, experiment, f"run_{run}")
                print(db_base)
                db_path = self._resolve_db_path(db_base)
                if db_path is None:
                    print(f"[warn] DB not found, skipping: {db_base}")
                    continue

                engine = create_engine(f"sqlite:///{db_path}", future=True)
                with engine.connect() as conn:
                    # survivors ⨝ robots
                    stmt = (
                        select(
                            GenerationSurvivor.generation,
                            GenerationSurvivor.robot_id,
                            GenerationSurvivor.fitness,
                            GenerationSurvivor.uniqueness,
                            Robot.born_generation,
                            Robot.num_voxels,
                        )
                        .join(Robot, Robot.robot_id == GenerationSurvivor.robot_id)
                    )
                    df = pd.read_sql(stmt, conn)

                    # full robots table
                    robots_stmt = select(
                        Robot.robot_id,
                        Robot.born_generation,
                        Robot.num_voxels,
                    )
                    df_robots = pd.read_sql(robots_stmt, conn)

                # tag with experiment/run
                df["experiment"] = experiment
                df["run"] = run
                frames.append(df)

                df_robots["experiment"] = experiment
                df_robots["run"] = run
                robots_frames.append(df_robots)

        if not frames:
            print("[warn] no data found; nothing to consolidate.")
            return

        # === all_df (filtered) =================================================
        all_df = pd.concat(frames, ignore_index=True)
        all_df = all_df[all_df["generation"] <= self.final_gen].reset_index(drop=True)
        all_df.to_csv(f"{self.path}/analysis/gens_robots.csv", index=False)

        # === consolidated robots ==============================================
        if robots_frames:
            robots_all = pd.concat(robots_frames, ignore_index=True)
            robots_all.to_csv(f"{self.path}/analysis/all_robots.csv", index=False)
        else:
            print("[warn] no robots rows found.")

        # === inner: within runs per generation (mean & max) ====================
        inner = (
            all_df.groupby(["experiment", "run", "generation"], as_index=False)
            .agg(
                fitness_mean=("fitness", "mean"),
                fitness_max=("fitness", "max"),
                uniqueness_mean=("uniqueness", "mean"),
                num_voxels_mean=("num_voxels", "mean"),
            )
        )
        inner.to_csv(f"{self.path}/analysis/gens_robots_inner.csv", index=False)

        # === outer: across runs per generation (median, q25, q75) ==============
        agg_spec = {}

        # summarize MEANS for all metrics
        for m in self.metrics:
            col = f"{m}_mean"
            agg_spec[f"{col}_median"] = (col, "median")
            agg_spec[f"{col}_q25"] = (col, lambda x: x.quantile(0.25))
            agg_spec[f"{col}_q75"] = (col, lambda x: x.quantile(0.75))

        # and ONLY FITNESS has MAX
        col = "fitness_max"
        agg_spec[f"{col}_median"] = (col, "median")
        agg_spec[f"{col}_q25"] = (col, lambda x: x.quantile(0.25))
        agg_spec[f"{col}_q75"] = (col, lambda x: x.quantile(0.75))

        outer = inner.groupby(["experiment", "generation"], as_index=False).agg(**agg_spec)
        outer.to_csv(f"{self.path}/analysis/gens_robots_outer.csv", index=False)

        print("consolidated!")


# --- CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    args = Config()._get_params()
    Analysis(args).consolidate()
