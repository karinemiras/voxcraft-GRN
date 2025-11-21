import os, sys
import numpy as np
import time
import subprocess
from pathlib import Path
from math import inf


# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
# from VoxcraftVXD import VXD
# from VoxcraftVXA import VXA



ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_COLORS
from simulation.VoxcraftVXD import VXD
from simulation.VoxcraftVXA import VXA

def trim_phenotype_materials(phenotype):
    # Remove empty layers (prevents starting with a floating body)
    body = phenotype
    x_mask = np.any(body != 0, axis=(1, 2))
    body = body[x_mask]
    y_mask = np.any(body != 0, axis=(0, 2))
    body = body[:, y_mask]
    z_mask = np.any(body != 0, axis=(0, 1))
    body = body[:, :, z_mask]
    return body


def prepare_robot_files(individual, args):

    phenotype = individual.phenotype

    out_path = f"{args.out_path}/{args.study_name}/{args.experiment_name}/robots/robot{individual.id}"
    os.makedirs(out_path, exist_ok=True)

    body = trim_phenotype_materials(phenotype)

    # pass vxa tags in here
    vxa = VXA(EnableExpansion=1,
              VaryTempEnabled=1,
              TempEnabled=1,
              SimTime=5,
              TempAmplitude=1,
              TempPeriod=2)

    in_phase = 0
    off_phase = 0.5

    # Create materials with different properties
    # E is stiffness in Pascals
    # RHO is the density
    # CTE is the coefficient of thermal expansion (proportional to voxel size)
    # TempPhase 0-1 (in relation to period)

    mat1 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['bone'], E=1e8, RHO=1e4, TempPhase=in_phase)  # stiffer, passive
    mat2 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['fat'], E=1e6, RHO=1e4, CTE=0.5, TempPhase=in_phase)  # softer, active
    mat3 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['muscle'], E=1e6, RHO=1e4, CTE=0.5, TempPhase=off_phase)  # softer, active

    # Write out the vxa (robot) to data/ directory
    vxa.write(f"{out_path}/base.vxa")

    # material vs phase data for VXD
    MAT_PHASE = {
        mat1: in_phase,
        mat2: in_phase,
        mat3: off_phase,
    }

    # Phase array: same shape as body, phase comes from the material that occupies each voxel
    phase = np.zeros_like(body, dtype=float)
    for mat_id, phase_val in MAT_PHASE.items():
        phase[body == mat_id] = phase_val

    # Generate a VXD file
    vxd = VXD()
    # pass vxd tags in here to overwrite vxa tags
    vxd.set_tags(RecordVoxel=1)
    vxd.set_data(body, phase_offsets=phase)

    # Write out the vxd to data
    vxd.write(f"{out_path}/{individual.id}.vxd")

    # vxd file can have any name, but there must be only one per folder
    # vxa file must be called base.vxa

def simulate_voxcraft_batch(population, args):
    """
    Run voxcraft-sim for all individuals, with at most 2 simulations
    running in parallel at any time. Each simulation has a timeout.
    After each simulation, read fitness from the report file into ind.fitness.
    """
    sim_bin = Path(args.docker_path) / "voxcraft-sim" / "build" / "voxcraft-sim"
    worker_bin = Path(args.docker_path) / "voxcraft-sim" / "build" / "vx3_node_worker"

    if not sim_bin.exists():
        raise FileNotFoundError(f"Simulator binary not found at {sim_bin}")

    if not worker_bin.exists():
        raise FileNotFoundError(f"Worker binary not found at {worker_bin}")

    MAX_PARALLEL = 2          # max number of sims at once
    SIM_TIMEOUT = 60          # seconds per robot before we kill it

    def robot_dir_for(ind):
        return (
            Path(args.out_path)
            / args.study_name
            / args.experiment_name
            / "robots"
            / f"robot{ind.id}"
        )

    def parse_fitness_from_report(report_file: Path) -> float:
        """
        Parse fitness by string search, because the 'XML' is not valid
        (<5> tags are illegal XML).

        Looks for the first <fitness_score>...</fitness_score> and returns the float.
        """
        if not report_file.exists():
            raise FileNotFoundError(f"Report file not found: {report_file}")

        text = report_file.read_text(encoding="utf-8", errors="ignore")
        start_tag = "<fitness_score>"
        end_tag = "</fitness_score>"

        start = text.find(start_tag)
        if start == -1:
            raise ValueError(f"No {start_tag} in {report_file}")
        start += len(start_tag)

        end = text.find(end_tag, start)
        if end == -1:
            raise ValueError(f"No {end_tag} in {report_file}")

        val_str = text[start:end].strip()
        return float(val_str)

    # list of (ind, Popen, history_file_handle, history_path, report_path)
    procs = []

    # --- start processes, at most MAX_PARALLEL at a time ---
    for ind in population:
        robot_dir = robot_dir_for(ind)

        if not robot_dir.exists():
            raise FileNotFoundError(f"Robot directory missing for {ind.id}: {robot_dir}")

        base_vxa = robot_dir / "base.vxa"
        vxd_file = robot_dir / f"{ind.id}.vxd"

        if not base_vxa.exists():
            raise FileNotFoundError(f"base_vxa missing for {ind.id}: {base_vxa}")
        if not vxd_file.exists():
            raise FileNotFoundError(f"VXD file missing for {ind.id}: {vxd_file}")

        history_file = robot_dir / f"{ind.id}.history"
        report_file = robot_dir / f"{ind.id}_report.xml"

        # throttle: keep at most MAX_PARALLEL running
        while True:
            running = [p for _, p, _, _, _ in procs if p.poll() is None]
            if len(running) < MAX_PARALLEL:
                break
            time.sleep(0.1)

        cmd = [
            str(sim_bin),
            "-l",  # run locally
            "-w", str(worker_bin),
            "-i", str(robot_dir),
            "-o", str(report_file),
            "-f",
        ]

       # print(f"[SIM-START] {ind.id} -> {robot_dir}")
        out_f = open(history_file, "w")
        p = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=subprocess.PIPE,
            text=True,
        )
        procs.append((ind, p, out_f, history_file, report_file))

    # --- wait for all processes to finish, with timeout & fitness parsing ---
    errors = []
    for ind, p, out_f, history_file, report_file in procs:
        try:
            stderr = p.communicate(timeout=SIM_TIMEOUT)[1]
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {ind.id} exceeded {SIM_TIMEOUT}s, killing.")
            p.kill()
            try:
                stderr = p.communicate(timeout=5)[1]
            except subprocess.TimeoutExpired:
                stderr = "<no stderr after kill>"
            out_f.close()
            ind.fitness = -inf
            errors.append(f"[TIMEOUT] {ind.id}: {stderr}")
            continue

        out_f.close()

        if p.returncode != 0:
            print(f"[SIM-ERROR] {ind.id} exit code {p.returncode}")
            ind.fitness = -inf
            errors.append(f"[SIM-ERROR] {ind.id}: {stderr}")
            continue

        if not history_file.exists():
            msg = f"[SIM-WARN] {ind.id} finished but history file missing: {history_file}"
            ind.fitness = -inf
            errors.append(msg)
            continue

        # --- parse fitness ---
        try:
            fitness = parse_fitness_from_report(report_file)
            ind.fitness = fitness
            print(f"[FITNESS] {ind.id} = {fitness:.6g}")
        except Exception as e:
            msg = f"[SIM-REPORT-ERROR] {ind.id}: {e}"
            ind.fitness = -inf
            errors.append(msg)

      #  print(f"[SIM-DONE] {ind.id}")

    if errors:
        print("[SIM-SUMMARY] Some simulations had issues:")
        for e in errors:
            print("  " + e)
