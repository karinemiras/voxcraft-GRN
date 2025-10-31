import numpy as np
import random
from algorithms.GRN_3D import GRN, initialization, mutation_type1, unequal_crossover
from simulation.VoxcraftVXA import VXA
from simulation.VoxcraftVXD import VXD


# USER SHOULD CHANGE THIS !
USER_VOXCRAFT_FOLDER = 'voxcraft-sim/inputs'


INI_GENOME_SIZE = 150
genome = initialization(random.Random(), INI_GENOME_SIZE)

phenotype = GRN(
    max_voxels=16,
    cube_face_size=4,
    genotype=genome,
).develop()

# convert GRNs-generated cells into a numpy that represent the voxels of the robot.
phenotype_materials = np.zeros(phenotype.shape, dtype=int)
for index, value in np.ndenumerate(phenotype):
    phenotype_materials[index] = value.voxel_type if value != 0 else 0

# Remove empty layers (prevents starting with a floating body)
trimmed_phenotype_materials = phenotype_materials
x_mask = np.any(trimmed_phenotype_materials != 0, axis=(1, 2))
trimmed_phenotype_materials = trimmed_phenotype_materials[x_mask]
y_mask = np.any(trimmed_phenotype_materials != 0, axis=(0, 2))
trimmed_phenotype_materials = trimmed_phenotype_materials[:, y_mask]
z_mask = np.any(trimmed_phenotype_materials != 0, axis=(0, 1))
trimmed_phenotype_materials = trimmed_phenotype_materials[:, :, z_mask]

print('robot: ')
print(trimmed_phenotype_materials)

# Generate a Base VXA file
vxa = VXA(EnableExpansion=1, SimTime=5) # pass vxa tags in here

# Create two materials with different properties
mat1 = vxa.add_material(RGBA=(10, 10, 10), E=1e6, RHO=1e4) # stiff, passive
mat2 = vxa.add_material(RGBA=(200, 0, 0), E=1e4, RHO=1e4, CTE=0.5, TempPhase=0) # soft (actuated)
mat3 = vxa.add_material(RGBA=(100, 0, 67), E=1e3, RHO=1e4, CTE=0.5, TempPhase=0.5) # softer (actuated)
print('materials: ', mat1, mat2, mat3)

# Write out the vxa to data/ directory
vxa.write(f"{USER_VOXCRAFT_FOLDER}/base.vxa")

# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
vxd.set_data(trimmed_phenotype_materials)
# Write out the vxd to data/, eg
vxd.write(f"{USER_VOXCRAFT_FOLDER}/robot.vxd")

# Now you can simulate robot.vxd in voxcraft-sim with this terminal command
# ./voxcraft-sim -i ../inputs/ >  robot.history
# later, you can load the robot.history into voxcraft-viz to watch the simulation results
