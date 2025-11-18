import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from VoxcraftVXD import VXD
from VoxcraftVXA import VXA

import numpy as np


def trim_phenotype_materials(phenotype):
    # Remove empty layers (prevents starting with a floating body)
    trimmed_phenotype_materials = phenotype
    x_mask = np.any(trimmed_phenotype_materials != 0, axis=(1, 2))
    trimmed_phenotype_materials = trimmed_phenotype_materials[x_mask]
    y_mask = np.any(trimmed_phenotype_materials != 0, axis=(0, 2))
    trimmed_phenotype_materials = trimmed_phenotype_materials[:, y_mask]
    z_mask = np.any(trimmed_phenotype_materials != 0, axis=(0, 1))
    trimmed_phenotype_materials = trimmed_phenotype_materials[:, :, z_mask]
    return trimmed_phenotype_materials


def simulate(individual, args):

    out_path = f"{args.out_path}/{args.study_name}/{args.experiment_name}/simulations"
    os.makedirs(out_path, exist_ok=True)

    sim_path = f"{args.docker_path}/voxcraft-sim/build"
    print(out_path)
    print(sim_path)

    phenotype = individual.phenotype

    trimmed_phenotype_materials = trim_phenotype_materials(phenotype)

    # pass vxa tags in here
    vxa = VXA(EnableExpansion=1,
              VaryTempEnabled=1,
              TempEnabled=1,
              SimTime=5,
              TempAmplitude=1,
              TempPeriod=2,
              EnableCilia=1)

    # Create materials with different properties
    # E is stiffness in Pascals
    # RHO is the density
    # CTE is the coefficient of thermal expansion (proportional to voxel size)
    # TempPhase 0-1 (in relation to period)
    # returns the material ID
    
    mat1 = vxa.add_material(RGBA=(10, 10, 10), E=1e6, RHO=1e4)  # stiff, passive
    mat2 = vxa.add_material(RGBA=(200, 0, 0), E=1e4, RHO=1e4, CTE=0.5, TempPhase=0)  # soft (actuated)
    # mat3 = vxa.add_material(RGBA=(0, 0, 67), E=5e+006, RHO=1e4, CTE=0.5, TempPhase=0.5) # soft (muscle), actuated offphase
    mat3 = vxa.add_material(RGBA=(0, 0, 67), E=5e+006, RHO=1e4, CTE=0.5, hasCilia=1)  # soft (muscle), actuated, w cilia

    # Write out the vxa (robot) to data/ directory
    vxa.write(f"{out_path}/{individual.id}.vxa")

    # Generate a VXD file
    vxd = VXD()
    # pass vxd tags in here to overwrite vxa tags
    vxd.set_tags(RecordVoxel=1)
    vxd.set_data(trimmed_phenotype_materials)

    # Write out the vxd to data
    vxd.write(f"{out_path}/{individual.id}.vxd")

    # TODO: call sim
