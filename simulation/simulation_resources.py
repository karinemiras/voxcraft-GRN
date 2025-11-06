import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from VoxcraftVXD import VXD
from VoxcraftVXA import VXA

import numpy as np


def simulate(phenotype, sim_path, thread):

    # Remove empty layers (prevents starting with a floating body)
    trimmed_phenotype_materials = phenotype
    x_mask = np.any(trimmed_phenotype_materials != 0, axis=(1, 2))
    trimmed_phenotype_materials = trimmed_phenotype_materials[x_mask]
    y_mask = np.any(trimmed_phenotype_materials != 0, axis=(0, 2))
    trimmed_phenotype_materials = trimmed_phenotype_materials[:, y_mask]
    z_mask = np.any(trimmed_phenotype_materials != 0, axis=(0, 1))
    trimmed_phenotype_materials = trimmed_phenotype_materials[:, :, z_mask]

    # pass vxa tags in here
    vxa = VXA(EnableExpansion=1, SimTime=5)

    # Create two materials with different properties
    # returns the material ID
    mat1 = vxa.add_material(RGBA=(10, 10, 10), E=1e6, RHO=1e4)  # stiff, passive
    mat2 = vxa.add_material(RGBA=(200, 0, 0), E=1e4, RHO=1e4, CTE=0.5, TempPhase=0)  # soft (actuated)
    mat3 = vxa.add_material(RGBA=(100, 0, 67), E=1e3, RHO=1e4, CTE=0.5, TempPhase=0)  # softer (actuated)

    # Write out the vxa to data/ directory
    # TODO vxa.write(f"{sim_path}/thread_{thread}/base.vxa")
    #vxa.write(f"{sim_path}/base.vxa")

    # Generate a VXD file
    vxd = VXD()
    # pass vxd tags in here to overwrite vxa tags
    vxd.set_tags(RecordVoxel=1)
    vxd.set_data(trimmed_phenotype_materials)

    # Write out the vxd to data
    #vxd.write(f"{sim_path}/thread_{thread}/robot.vxd")
    #vxd.write(f"{sim_path}/robot.vxd")

    # TODO: call sim
