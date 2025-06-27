import numpy as np

from VoxcraftVXA import VXA
from VoxcraftVXD import VXD

# Generate a Base VXA file
# See here for list of vxa tags: https://gpuvoxels.readthedocs.io/en/docs/
vxa = VXA(EnableExpansion=1, SimTime=1) # pass vxa tags in here

# Create two materials with different properties
mat1 = vxa.add_material(RGBA=(0,140,0), E=1e8, RHO=1e4) # returns the material ID
mat2 = vxa.add_material(RGBA=(55,120,0), E=1e8, RHO=1e4)
print(mat1, mat2)
# Write out the vxa to data/ directory
vxa.write("base.vxa")

# Create random body array between 0 and maximum material ID
body = np.random.randint(0,2,size=(3,3,3))
middle_pos = tuple(s // 2 for s in body.shape)
print(body)
print(middle_pos)
body[middle_pos] = 666
print(body)

# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
vxd.set_data(body)
# Write out the vxd to data/
vxd.write("robot1.vxd")