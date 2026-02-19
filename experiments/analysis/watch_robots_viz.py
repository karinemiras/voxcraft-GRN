# This script is localized in the server, and loads server data because a dir is mounted in ripper_data_mount,
# but it runs voxcraft-viz locally on MAC.
# mounted path: /Users/karinemiras/Documents/ripper_data_mount/voxcraft/experiments/analysis/watch_robots_viz.py

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root: .../ripper_data_mount/voxcraft
sys.path.append(str(ROOT))

from utils.config import Config

VIZPATH="/Users/karinemiras/projects/voxcraft-viz/build"
PARAMS_EXP = "locomotion.sh"
VIZ_SECONDS=10
NUMBESTS=1

def load_sh_params(path: Path) -> dict:
    params = {}
    text = path.read_text()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            params[k] = v

    return params


def parse_fit_and_id(filename: str):
    """
    Expected snapshot naming:
      <rank>_<fitness>_<id>.png
    Example:
      0_12.34_791.png
    """
    parts = filename.split("_")
    if len(parts) < 3:
        return None, None

    fit = parts[1]
    rid = Path(parts[2]).stem  # strips ".png"

    return fit, rid


def localize_remote_path(remote_path: str | Path, mount_root: Path) -> Path:
    """
    Map server-style absolute paths like /working_data/... to:
      <mount_root>/working_data/...
    """
    p = Path(str(remote_path))

    if p.exists():
        return p

    if p.is_absolute():
        return (mount_root / str(p).lstrip("/")).resolve()

    return (mount_root / p).resolve()


def history_path_for(out_path_local: Path, study: str, experiment: str, run: int, robot_id: str) -> Path:
    """
    Known server layout:
      working_data/<study>/<experiment>/run_<run>/simulations/<id>.history
    """
    return out_path_local / study / experiment / f"run_{run}" / "simulations" / f"{robot_id}.history"


def main():
    args = Config()._get_params()

    # --- load params from experiments/locomotion.sh -------------------------
    params_path = ROOT / "experiments" / PARAMS_EXP
    if not params_path.exists():
        raise FileNotFoundError(f"params file not found: {params_path}")

    params = load_sh_params(params_path)

    local_build_dir = Path(params.get("LOCAL_BUILD_DIR", VIZPATH))
    voxcraft_viz = Path(params.get("VOXCRAFT_VIZ", str(local_build_dir / "voxcraft-viz")))

    if not voxcraft_viz.exists():
        raise FileNotFoundError(
            f"voxcraft-viz not found at: {voxcraft_viz}\n"
            f"Set VOXCRAFT_VIZ=... in {params_path} or update LOCAL_BUILD_DIR."
        )

    # ----------------------------------------------------------------------
    # ORIGINAL LOGIC (DO NOT REMOVE)
    # study = args.study_name
    # experiments_name = [e for e in args.experiments.split(",") if e.strip()]
    # runs = [int(r) for r in args.runs.split(",") if r.strip()]
    # generations = [int(g) for g in args.generations.split(",") if g.strip()]
    # out_path = args.out_path
    # ----------------------------------------------------------------------

    # --- your CURRENT TEST VALUES (kept exactly as requested) ---------------
    study = "voxlocnew"
    experiments_name = ["crosspropxyv4s25"]
    runs = [1,2,3,4,5,6,7,8]
    generations = [50]
    out_path = args.out_path  # likely "/working_data"
    # -----------------------------------------------------------------------

    bests = int(getattr(args, "bests", NUMBESTS))

    mount_root = ROOT.parent
    out_path_local = localize_remote_path(out_path, mount_root)

    # snapshots folder (on mounted working_data)
    snapshots_root = out_path_local / study / "analysis" / "snapshots"

    print("out_path (config):", out_path)
    print("out_path (local): ", out_path_local)
    print("Snapshots root:   ", snapshots_root)
    print("Study:", study)
    print("Experiments:", experiments_name)
    print("Runs:", runs)
    print("Generations:", generations)
    print("Top bests:", bests)
    print("voxcraft-viz:", voxcraft_viz)

    wait = bool(int(params.get("VIZ_WAIT", "0")))

    for gen in generations:
        for experiment_name in experiments_name:
            print(f"\n=== {experiment_name} | gen {gen} ===")

            for run in runs:
                print(f"  run {run}")

                snap_dir = snapshots_root / experiment_name / f"run_{run}" / f"gen_{gen}"
                if not snap_dir.exists():
                    print(f"    [skip] missing snapshots: {snap_dir}")
                    continue

                files = sorted(
                    os.listdir(snap_dir),
                    key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 10**9,
                )[:bests]

                if not files:
                    print("    [skip] no snapshot files")
                    continue

                for fname in files:
                    fit, rid = parse_fit_and_id(fname)
                    if fit is None:
                        print(f"    [skip] bad filename: {fname}")
                        continue

                    hist = history_path_for(out_path_local, study, experiment_name, run, rid)
                    if not hist.exists():
                        print(f"    [missing] history for id={rid} fit={fit}: {hist}")
                        continue

                    print(f"    -> id={rid} fitness={fit}")
                    cmd = [str(voxcraft_viz), str(hist)]

                    p = subprocess.Popen(cmd)

                    try:
                        # If voxcraft-viz exits naturally (no loop), this just waits
                        p.wait(timeout=VIZ_SECONDS)
                    except subprocess.TimeoutExpired:
                        # Ask politely to close
                        p.terminate()
                        # Wait until it shuts down cleanly
                        p.wait()


if __name__ == "__main__":
    main()