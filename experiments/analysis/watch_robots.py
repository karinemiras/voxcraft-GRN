import os
import sys
import math
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple, Optional

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from algorithms.EA_classes import Robot, GenerationSurvivor
from utils.draw import draw_phenotype
from utils.config import Config
from simulation.simulation_resources import prepare_robot_files, simulate_voxcraft_batch


# ── small DB helpers ──────────────────────────────────────────────────────────
def open_session(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def survivors_exist(session) -> bool:
    return session.query(func.count(GenerationSurvivor.generation.distinct())).scalar() > 0


def topN_for_generation(session, gen: int, limit: int) -> List[Tuple[Robot, Optional[GenerationSurvivor]]]:
    q = (
        session.query(Robot, GenerationSurvivor)
        .join(GenerationSurvivor, GenerationSurvivor.robot_id == Robot.robot_id)
        .filter(GenerationSurvivor.generation == gen)
        .order_by(GenerationSurvivor.fitness.desc().nullslast())
    )
    return q.limit(limit).all()


def robots_by_ids(session, ids: Iterable[int]) -> List[Robot]:
    if not ids:
        return []
    return session.query(Robot).filter(Robot.robot_id.in_(list(ids))).all()


def latest_survivor(session, robot_id: int) -> Optional[GenerationSurvivor]:
    return (
        session.query(GenerationSurvivor)
        .filter(GenerationSurvivor.robot_id == robot_id)
        .order_by(GenerationSurvivor.generation.desc())
        .first()
    )


# ── core pipeline pieces ──────────────────────────────────────────────────────
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def build_EA(args):
    module_name = f"algorithms.{args.algorithm}"
    EA_cls = getattr(importlib.import_module(module_name), "EA")
    print(args)
    return EA_cls(args)


def to_float(x) -> float:
    return float(x) if x is not None else float("nan")


def draw_develop_prepare(
    EA, args, tf_for_exp: str, out_dir: str, rank_idx: int, robot_row: Robot, fitness: float, gen_label: Optional[int],
) -> SimpleNamespace:
    phenotype = EA.develop_phenotype(robot_row.genome, tf_for_exp)
    draw_phenotype(
        phenotype,
        robot_row.robot_id,
        args.cube_face_size,
        rank_idx,
        round(fitness, 4) if math.isfinite(fitness) else fitness,
        out_dir,
    )
    ind = SimpleNamespace(
        robot_id=robot_row.robot_id,
        genome=robot_row.genome,
        phenotype=phenotype,
        generation=gen_label,
        fitness=fitness,
        rank=rank_idx,
        out_dir=out_dir,
    )
    prepare_robot_files(ind, args)
    return ind


def simulate_batch_if_any(pop: List[SimpleNamespace], args, label: str):
    if not pop:
        return
    print(f"    simulating {len(pop)} robot(s) [{label}]...")
    simulate_voxcraft_batch(pop, args)
    print("    simulation finished.")


# ── main orchestration ────────────────────────────────────────────────────────
def main():
    args = Config()._get_params()

    experiments = args.experiments.split(",")
    runs = [int(x) for x in args.runs.split(",")]
    generations = [int(x) for x in args.generations.split(",")]
    tfs = args.tfs.split(",")

    max_robots = getattr(args, "max_robots", 50)
    manual_ids = [1245] # if empty, takes best from gens
    rid_raw = getattr(args, "robot_ids", "")
    if isinstance(rid_raw, str) and rid_raw.strip():
        manual_ids = [int(x) for x in rid_raw.split(",") if x.strip().isdigit()]

    EA = build_EA(args)

    for exp_idx, experiment_name in enumerate(experiments):
        tf_for_exp = tfs[exp_idx]
        print(experiment_name)

        for run in runs:
            print(" run:", run)

            snapshot_root = ensure_dir(f"{args.out_path}/{args.study_name}/analysis/watch/{experiment_name}/run_{run}")
            db_path = f"{args.out_path}/{args.study_name}/{experiment_name}/run_{run}/run_{run}"

            with open_session(db_path) as session:
                if not survivors_exist(session):
                    print("  (no completed generations found in DB)")
                    continue

                # ── MODE A: manual IDs for this run ────────────────────────
                if manual_ids:
                    gen_dir = ensure_dir(f"{snapshot_root}/manual_selection")
                    robots = robots_by_ids(session, manual_ids)
                    found_ids = {r.robot_id for r in robots}
                    missing = [rid for rid in manual_ids if rid not in found_ids]
                    if missing:
                        print(f"    warning: IDs not found in this run's DB: {missing}")

                    pop = []
                    for i, r in enumerate(robots):
                        surv = latest_survivor(session, r.robot_id)
                        fitness = to_float(surv.fitness) if surv else float("nan")
                        gen_label = surv.generation if surv else None
                        ind = draw_develop_prepare(EA, args, tf_for_exp, gen_dir, i, r, fitness, gen_label)
                        pop.append(ind)
                    simulate_batch_if_any(pop, args, "manual")
                    continue  # skip per-gen mode

                # ── MODE B: top-N per generation ───────────────────────────
                for gen in generations:
                    print("  gen:", gen)
                    gen_dir = ensure_dir(f"{snapshot_root}/gen_{gen}")

                    rows = topN_for_generation(session, gen, max_robots)
                    if not rows:
                        print("    (no survivors for this generation)")
                        continue

                    pop = []
                    for i, (robot_row, surv_row) in enumerate(rows):
                        fitness = to_float(surv_row.fitness)
                        ind = draw_develop_prepare(EA, args, tf_for_exp, gen_dir, i, robot_row, fitness, gen)
                        pop.append(ind)

                    #simulate_batch_if_any(pop, args, f"gen {gen}")

    print("All done.")


if __name__ == "__main__":
    main()
