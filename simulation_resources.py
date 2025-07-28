from VoxcraftVXD import VXD
from VoxcraftVXA import VXA

def simulate(phenotype, sim_path, thread):

    vxa = VXA(EnableExpansion=1, SimTime=1) # pass vxa tags in here

    # Create two materials with different properties
    # returns the material ID
    mat1 = vxa.add_material(RGBA=(220, 220, 220), E=1e8, RHO=1e4)  # bone
    mat2 = vxa.add_material(RGBA=(255, 230, 128), E=1e8, RHO=1e4)  # fat
    mat3 = vxa.add_material(RGBA=(180, 30, 40), E=1e8, RHO=1e4)    # muscle

    # Write out the vxa to data/ directory
    # TODO : replace paths via param
    #vxa.write(f"{sim_path}/thread_{thread}/base.vxa")
    vxa.write(f"voxcraft-sim/inputs/base.vxa")

    # Generate a VXD file
    vxd = VXD()
    vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
    vxd.set_data(phenotype)
    # Write out the vxd to data

    #vxd.write(f"{sim_path}/thread_{thread}/robot.vxd")
    vxd.write(f"voxcraft-sim/inputs/robot.vxd")

    # TODO: call sim
