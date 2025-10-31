import numpy as np

from simulation.VoxcraftVXA import VXA
from simulation.VoxcraftVXD import VXD

# USER SHOULD CHANGE THIS !
USER_VOXCRAFT_FOLDER = 'voxcraft-sim/inputs'

# Generate a Base VXA file
# See here for list of vxa tags: https://gpuvoxels.readthedocs.io/en/docs/
vxa = VXA(EnableExpansion=1, SimTime=1) # pass vxa tags in here

# Create two materials with different properties
mat1 = vxa.add_material(RGBA=(0, 140, 0), E=1e8, RHO=1e4) # returns the material ID
mat2 = vxa.add_material(RGBA=(55, 0, 0), E=1e7, RHO=1e4)
print('materials: ', mat1, mat2)

# Write out the vxa to data/ directory
vxa.write(f"{USER_VOXCRAFT_FOLDER}/base.vxa")

# Create random body array between 0 and maximum material ID
body = np.random.randint(0, 3, size=(3, 3, 3))

print('robot exported:\n', body)

# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
vxd.set_data(body)
# Write out the vxd to data directory
vxd.write(f"{USER_VOXCRAFT_FOLDER}/robot.vxd")

# Now you can simulate robot.vxd in voxcraft-sim with this terminal command
# ./voxcraft-sim -i ../inputs/ >  robot.history
# later, you can load the robot.history into voxcraft-viz to watch the simulation results
