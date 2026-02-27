import os, sys
import re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.voxel_types import VOXEL_TYPES_COLORS, VOXEL_TYPES_COLORS_NOBONE
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

    #CTE = 0.01
    #TARGET_VOL_GAIN = 0.20

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

    # E is stiffness in Pascals
    # RHO is the density
    # CTE is the coefficient of thermal expansion (proportional to voxel size per degree)
    # TempPhase 0-1 (in relation to period)

    # Create materials with different properties (order of materials matters to match voxel_types)
    if args.voxel_types == 'withbone':
        mat1 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['bone'],
                                E=1e8, RHO=1e4,
                                uStatic=args.ustatic, uDynamic=args.udynamic)  # stiff, passive

        mat2 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['fat'],
                                E=7e5, RHO=1.2e4,
                                uStatic=args.ustatic, uDynamic=args.udynamic)  # soft, passive

        mat3 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['phase_muscle'],
                                E=1e6, RHO=1e4, CTE=CTE, TempPhase=in_phase,
                                uStatic=args.ustatic, uDynamic=args.udynamic) # medium-soft, active

        mat4 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS['offphase_muscle'],
                                E=1e6, RHO=1e4, CTE=CTE, TempPhase=off_phase,
                                uStatic=args.ustatic, uDynamic=args.udynamic)  # medium-soft, active

    if args.voxel_types == 'nobone':
        mat1 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS_NOBONE['fat'],
                                E=7e5, RHO=1.2e4,
                                uStatic=args.ustatic, uDynamic=args.udynamic)  # soft, passive

        mat2 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS_NOBONE['fat2'],
                                E=7e5, RHO=1.2e4,
                                uStatic=args.ustatic, uDynamic=args.udynamic)  # soft, passive

        mat3 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS_NOBONE['phase_muscle'],
                                E=1e6, RHO=1e4, CTE=CTE, TempPhase=in_phase,
                                uStatic=args.ustatic, uDynamic=args.udynamic) # medium-soft, active

        mat4 = vxa.add_material(RGBA=VOXEL_TYPES_COLORS_NOBONE['offphase_muscle'],
                                E=1e6, RHO=1e4, CTE=CTE, TempPhase=off_phase,
                                uStatic=args.ustatic, uDynamic=args.udynamic)  # medium-soft, active

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

