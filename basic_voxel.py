import numpy as np

from simulation.VoxcraftVXA import VXA
from simulation.VoxcraftVXD import VXD

# Generate a Base VXA file
# See here for list of vxa tags: https://gpuvoxels.readthedocs.io/en/docs/
vxa = VXA(EnableExpansion=1, SimTime=5,
          VaryTempEnabled=1, TempEnabled=1, TempAmplitude=1, TempPeriod=2
          ) # pass vxa tags in here

# Create two materials with different properties
mat1 = vxa.add_material(RGBA=(255,0,255), E=5e4, RHO=1e4, CTE=0.5) # returns the material ID
mat2 = vxa.add_material(RGBA=(255,0,0), E=1e8, RHO=1e4, CTE=0.5, TempPhase=0.5)

# Write out the vxa to data/ directory
vxa.write("voxcraft-sim/inputs/base.vxa")

# Create random body array between 0 and maximum material ID
body = np.random.randint(0,mat2+1,size=(2,2,1))
print(body)


# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
vxd.set_data(body)
# Write out the vxd to data/
vxd.write("voxcraft-sim/inputs/robot.vxd")