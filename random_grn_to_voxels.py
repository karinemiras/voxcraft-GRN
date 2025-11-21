import numpy as np
from math import pi
import random
from algorithms.GRN_3D import GRN, initialization, mutation_type1, unequal_crossover
from simulation.VoxcraftVXA import VXA
from simulation.VoxcraftVXD import VXD


# USER SHOULD CHANGE THIS !
USER_VOXCRAFT_FOLDER = 'voxcraft-sim/demos'

INI_GENOME_SIZE = 150
genome = initialization(random.Random(57), INI_GENOME_SIZE)

phenotype = GRN(
    max_voxels=10,
    cube_face_size=4,
    genotype=genome,
).develop()

# convert GRN-generated cells into a numpy that represents the voxels of the robot.
phenotype_materials = np.zeros(phenotype.shape, dtype=int)
for index, value in np.ndenumerate(phenotype):
    phenotype_materials[index] = value.voxel_type if value != 0 else 0

# Remove empty layers (prevents starting with a floating body)
body = phenotype_materials
x_mask = np.any(body != 0, axis=(1, 2))
body = body[x_mask]
y_mask = np.any(body != 0, axis=(0, 2))
body = body[:, y_mask]
z_mask = np.any(body != 0, axis=(0, 1))
body = body[:, :, z_mask]

# Generate a Base VXA file
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

mat1 = vxa.add_material(RGBA=(0, 0, 100), E=1e8, RHO=1e4, TempPhase=in_phase) # stiffer, passive
mat2 = vxa.add_material(RGBA=(100, 0, 0), E=1e6, RHO=1e4, CTE=0.5, TempPhase=in_phase) # softer, active
mat3 = vxa.add_material(RGBA=(0, 100, 0), E=1e6, RHO=1e4, CTE=0.5, TempPhase=off_phase) # softer, active

# Write out the vxa to data/ directory
vxa.write(f"{USER_VOXCRAFT_FOLDER}/base.vxa")

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

print('robot exported:\n', body)

# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1)
vxd.set_data(body, phase_offsets=phase)
# Write out the vxd to data/
vxd.write(f"{USER_VOXCRAFT_FOLDER}/robot.vxd")

# vxd file can have any name, but there must be only one per folder
# vxa file must be called base.vxa

# Now you can simulate robot.vxd in voxcraft-sim with this terminal command
# ./voxcraft-sim -i ../demos/ >  robot.history
# later, you can load the robot.history into voxcraft-viz to watch the simulation results
