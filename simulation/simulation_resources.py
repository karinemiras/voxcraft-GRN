import os, sys
import numpy as np
import time
import subprocess
from pathlib import Path

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

    out_path = f"{args.out_path}/{args.study_name}/{args.experiment_name}/run_{args.run}/robots/robot{individual.id}"
    os.makedirs(out_path, exist_ok=True)

    body = trim_phenotype_materials(phenotype)

    def amp_for_vol_gain(cte: float, vol_gain: float) -> float:
        """
        Given CTE (linear per °C) and desired peak volumetric gain (e.g., 0.5 for +50%),
        return the required temperature amplitude (ΔT).
        (1 + cte*ΔT)^3 - 1 = vol_gain  =>  ΔT = ( (1+vol_gain)**(1/3) - 1 ) / cte
        """
        return ((1.0 + vol_gain) ** (1.0 / 3.0) - 1.0) / cte

    def jitter_period(base_period: float, frac: float = 0.10) -> float:
        """Jitter period ±frac to discourage brittle timing hacks."""
        return base_period * (1.0 + np.random.uniform(-frac, frac))

    CTE = 0.5
    TARGET_VOL_GAIN = 0.50  # ~50%
    TEMP_AMP = amp_for_vol_gain(CTE, TARGET_VOL_GAIN)  # ~ 0.289
    FREQ = 5.0  # Hz
    BASE_PERIOD = 1.0 / FREQ  # 0.2 s
    JITTER_FRAC = 0.10  # ±10%
    SAFE_MAX_LINEAR_STR = 0.30  # disallow >30% linear strain at peak
    DT_FRAC = 0.4  # tighter than 0.95 for stability with soft + strong actuation

    # Safety guard: linear factor = 1 ± (CTE * TEMP_AMP) must stay reasonable
    # assert (CTE * TEMP_AMP) < SAFE_MAX_LINEAR_STR, (
    #     f"Actuation too large: CTE*AMP={CTE * TEMP_AMP:.3f} (limit {SAFE_MAX_LINEAR_STR})"
    # )

    # ---- build VXA with your conventions --------------------------------------
    vxa = VXA(
        SimTime=args.simulation_time,
        EnableExpansion=1,
        VaryTempEnabled=1,
        TempEnabled=1,
        # Stabilize integration for soft bodies under strong actuation:
        DtFrac=DT_FRAC,
        TempPeriod=BASE_PERIOD, # TODO: jitter_period(BASE_PERIOD, JITTER_FRAC),
        TempAmplitude=TEMP_AMP,  # *** ΔT since TempBase = 0 ***
        TempBase=0,  # your base
    )

    in_phase = 0
    off_phase = 0.5

    # Create materials with different properties
    # E is stiffness in Pascals
    # RHO is the density
    # CTE is the coefficient of thermal expansion (proportional to voxel size per degree)
    # TempPhase 0-1 (in relation to period)

    mat1 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['bone'], E=1e8, RHO=1e4)  # stiff, passive
    mat2 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['fat'], E=7e5, RHO=1.2e4)  # soft, passive, heavier
    mat3 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['muscle'], E=1e6, RHO=1e4, CTE=CTE, TempPhase=in_phase)  # medium-soft, active
    mat4 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['muscle_offp'], E=1e6, RHO=1e4, CTE=CTE, TempPhase=off_phase)  # medium-soft, active
    
    # Write out the vxa (robot) to data/ directory
    vxa.write(f"{out_path}/base.vxa")

    # material vs phase data for VXD
    MAT_PHASE = {
        mat1: in_phase,
        mat2: in_phase,
        mat3: in_phase,
        mat4: off_phase,
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
    After each simulation, read displacement from the report file into ind.
    voscraft-sim git version: dd8668f99ff74fb2fc2ee60deb91287a3b519d8f
    """
    sim_bin = Path(args.docker_path) / "voxcraft-sim" / "build" / "voxcraft-sim"
    worker_bin = Path(args.docker_path) / "voxcraft-sim" / "build" / "vx3_node_worker"

    if not sim_bin.exists():
        raise FileNotFoundError(f"Simulator binary not found at {sim_bin}")

    if not worker_bin.exists():
        raise FileNotFoundError(f"Worker binary not found at {worker_bin}")

    MAX_PARALLEL = 2          # max number of sims at once
    SIM_TIMEOUT = 60          # seconds per robot before we kill it

    out_path_hist = (
            Path(args.out_path)
            / args.study_name
            / args.experiment_name
            / f"run_{args.run}"
            / "simulations"
    )
    os.makedirs(out_path_hist, exist_ok=True)

    def robot_dir_for(ind):
        return (
            Path(args.out_path)
            / args.study_name
            / args.experiment_name
            / f"run_{args.run}"
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

        # does not evaluate invalid individuals
        if not ind.valid:
            continue

        robot_dir = robot_dir_for(ind)

        if not robot_dir.exists():
            raise FileNotFoundError(f"Robot directory missing for {ind.id}: {robot_dir}")

        base_vxa = robot_dir / "base.vxa"
        vxd_file = robot_dir / f"{ind.id}.vxd"

        if not base_vxa.exists():
            raise FileNotFoundError(f"base_vxa missing for {ind.id}: {base_vxa}")
        if not vxd_file.exists():
            raise FileNotFoundError(f"VXD file missing for {ind.id}: {vxd_file}")

        history_file = out_path_hist / f"{ind.id}.history"
        report_file = out_path_hist / f"{ind.id}_report.xml"

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
            errors.append(f"[TIMEOUT] {ind.id}: {stderr}")
            continue

        out_f.close()

        if p.returncode != 0:
            print(f"[SIM-ERROR] {ind.id} exit code {p.returncode}")
            errors.append(f"[SIM-ERROR] {ind.id}: {stderr}")
            continue

        if not history_file.exists():
            msg = f"[SIM-WARN] {ind.id} finished but history file missing: {history_file}"
            errors.append(msg)
            continue

        # --- parse behaviors ---
        try:
            # they call it fitness but here it is a behavior that may or not become the fitness
            displacement_xy = parse_fitness_from_report(report_file)
            ind.displacement_xy = displacement_xy
        except Exception as e:
            msg = f"[SIM-REPORT-ERROR] {ind.id}: {e}"
            errors.append(msg)

        #  print(f"[SIM-DONE] {ind.id}")

    # if errors:
    #     print("[SIM-SUMMARY] Some simulations had issues:")
    #     for e in errors:
    #         print("  " + e)
