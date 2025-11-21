import numpy as np

from simulation.VoxcraftVXA import VXA
from simulation.VoxcraftVXD import VXD

# USER SHOULD CHANGE THIS !
USER_VOXCRAFT_FOLDER = 'voxcraft-sim/demos'

np.random.seed(12)

# Generate a Base VXA file
# See here for list of vxa tags: https://gpuvoxels.readthedocs.io/en/docs/
vxa = VXA(EnableExpansion=1,
          VaryTempEnabled=1,
          SimTime=5,
          TempEnabled=1,
          TempAmplitude=1,
          TempPeriod=2)

in_phase = 0
off_phase = 0.5

# Create materials with different properties
# E is stiffness in Pascals
# RHO is the density
# CTE is the coefficient of thermal expansion (proportional to voxel size)
# TempPhase 0-1 (in relation to period)
# returns the material ID
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

# Create random body array between 0 and maximum material ID
body = np.random.randint(0, 4, size=(2, 2, 1))

# Phase array: same shape as body, phase comes from the material that occupies each voxel
phase = np.zeros_like(body, dtype=float)
for mat_id, phase_val in MAT_PHASE.items():
    phase[body == mat_id] = phase_val

print('robot exported:\n', body)

# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1)
vxd.set_data(body, phase_offsets=phase)
# Write out the vxd to data directory
vxd.write(f"{USER_VOXCRAFT_FOLDER}/robot.vxd")

# vxd file can have any name, but there must be only one per folder
# vxa file must be called base.vxa

# Now you can simulate robot.vxd in voxcraft-sim with this terminal command
# ./voxcraft-sim -i ../inputs/ >  robot.history
# later, you can load the robot.history into voxcraft-viz to watch the simulation results


