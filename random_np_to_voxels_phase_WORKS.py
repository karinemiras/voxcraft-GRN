import numpy as np

from simulation.VoxcraftVXA import VXA
from simulation.VoxcraftVXD import VXD

# USER SHOULD CHANGE THIS !
USER_VOXCRAFT_FOLDER = 'voxcraft-sim/demos'

np.random.seed(978)

# Generate a Base VXA file
# See here for list of vxa tags: https://gpuvoxels.readthedocs.io/en/docs/
vxa = VXA(EnableExpansion=1,
          VaryTempEnabled=1,
          SimTime=5,
          TempAmplitude=1,
          TempPeriod=2) # pass vxa tags in here

# Create two materials with different properties
# returns the material ID
mat2 = vxa.add_material(RGBA=(100, 0, 0), E=1e6, RHO=1e4, CTE=0.5) # softer, active
mat3 = vxa.add_material(RGBA=(0, 100, 0), E=1e6, RHO=1e4, CTE=0.5, TempPhase=0.5) # softer, active

# phase per material ID, matching how you defined them
MAT_PHASE = {
    mat2: 0.0,   # active, base phase
    mat3: 0.5,   # active, π out of phase
}

# Write out the vxa to data/ directory
vxa.write(f"{USER_VOXCRAFT_FOLDER}/base.vxa")

# Create random body array between 0 and maximum material ID
body = np.random.randint(0, 3, size=(2, 2, 1))
print('robot exported:\n', body)


# Phase array: same shape as body, phase comes from the material that occupies each voxel
phase = np.zeros_like(body, dtype=float)
for mat_id, phase_val in MAT_PHASE.items():
    phase[body == mat_id] = phase_val
print(phase)

# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwrite vxa tags
vxd.set_data(body, phase_offsets=phase)
# Write out the vxd to data directory
vxd.write(f"{USER_VOXCRAFT_FOLDER}/robot.vxd")

# Now you can simulate robot.vxd in voxcraft-sim with this terminal command
# ./voxcraft-sim -i ../inputs/ >  robot.history
# later, you can load the robot.history into voxcraft-viz to watch the simulation results


