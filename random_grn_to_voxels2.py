
import numpy as np
from math import pi
import random
from algorithms.GRN_3D import GRN, initialization, mutation_type1, unequal_crossover

from simulation.VoxcraftVXD import VXD
from lxml import etree

# USER SHOULD CHANGE THIS !
USER_VOXCRAFT_FOLDER = 'voxcraft-sim/inputs'



from lxml import etree

class VXA:
    def __init__(self, EnableExpansion=1, VaryTempEnabled=1, TempAmplitude=8, TempPeriod=2,
                 SimTime=6, RecordStepSize=1, Lattice_Dim=0.01):
        root = etree.XML("<VXA/>"); root.set("Version","1.1"); self.tree=etree.ElementTree(root)
        sim=etree.SubElement(root,"Simulator")
        etree.SubElement(sim,"EnableCilia").text="0"
        etree.SubElement(sim,"EnableExpansion").text=str(EnableExpansion)
        integ=etree.SubElement(sim,"Integration"); etree.SubElement(integ,"DtFrac").text="0.95"
        damp=etree.SubElement(sim,"Damping")
        etree.SubElement(damp,"BondDampingZ").text="1"; etree.SubElement(damp,"ColDampingZ").text="0.8"; etree.SubElement(damp,"SlowDampingZ").text="0.01"
        stop=etree.SubElement(sim,"StopCondition"); f=etree.SubElement(stop,"StopConditionFormula"); sub=etree.SubElement(f,"mtSUB")
        etree.SubElement(sub,"mtVAR").text="t"; etree.SubElement(sub,"mtCONST").text=str(SimTime)
        hist=etree.SubElement(sim,"RecordHistory")
        etree.SubElement(hist,"RecordStepSize").text=str(RecordStepSize)
        etree.SubElement(hist,"RecordVoxel").text="1"; etree.SubElement(hist,"RecordLink").text="0"; etree.SubElement(hist,"RecordFixedVoxels").text="1"
        env=etree.SubElement(root,"Environment"); therm=etree.SubElement(env,"Thermal")
        etree.SubElement(therm,"VaryTempEnabled").text=str(VaryTempEnabled)
        etree.SubElement(therm,"TempPeriod").text=str(TempPeriod)
        etree.SubElement(therm,"TempAmplitude").text=str(TempAmplitude)
        etree.SubElement(therm,"TempBase").text="0"
        grav=etree.SubElement(env,"Gravity"); etree.SubElement(grav,"GravEnabled").text="1"
        etree.SubElement(grav,"GravAcc").text="-9.81"; etree.SubElement(grav,"FloorEnabled").text="1"
        vxc=etree.SubElement(root,"VXC"); vxc.set("Version","0.94")
        lat=etree.SubElement(vxc,"Lattice"); etree.SubElement(lat,"Lattice_Dim").text=str(Lattice_Dim)
        self.palette=etree.SubElement(vxc,"Palette")
        self.struct=etree.SubElement(vxc,"Structure"); self.struct.set("Compression","ASCII_READABLE")
        etree.SubElement(self.struct,"X_Voxels").text="3"
        etree.SubElement(self.struct,"Y_Voxels").text="1"
        etree.SubElement(self.struct,"Z_Voxels").text="1"
        self.data=etree.SubElement(self.struct,"Data")
        self.next_mat=1

    def add_material(self, *, E=1e6, RHO=1e4, P=0.35, CTE=0.5, isFixed=0, RGBA=(255,0,0,255)):
        mid=self.next_mat; self.next_mat+=1
        r,g,b,a=[c/255 for c in RGBA]
        m=etree.SubElement(self.palette,"Material"); etree.SubElement(m,"Name").text=str(mid)
        disp=etree.SubElement(m,"Display")
        for tag,val in zip(("Red","Green","Blue","Alpha"),(r,g,b,a)): etree.SubElement(disp,tag).text=str(val)
        mech=etree.SubElement(m,"Mechanical")
        etree.SubElement(mech,"isMeasured").text="1"; etree.SubElement(mech,"Fixed").text=str(isFixed)
        etree.SubElement(mech,"sticky").text="0"; etree.SubElement(mech,"Cilia").text="0"; etree.SubElement(mech,"MatModel").text="0"
        etree.SubElement(mech,"Elastic_Mod").text=str(E); etree.SubElement(mech,"Fail_Stress").text="0"
        etree.SubElement(mech,"Density").text=str(RHO); etree.SubElement(mech,"Poissons_Ratio").text=str(P)
        etree.SubElement(mech,"CTE").text=str(CTE); etree.SubElement(mech,"uStatic").text="1"; etree.SubElement(mech,"uDynamic").text="0.8"
        return mid

    def set_data_layers(self, layers_as_strings):
        for c in list(self.data): self.data.remove(c)
        for s in layers_as_strings: etree.SubElement(self.data,"Layer").text=etree.CDATA(s)

    def set_phase_offset(self, layers_xy_numbers):
        old=self.struct.find("PhaseOffset")
        if old is not None: self.struct.remove(old)
        po=etree.SubElement(self.struct,"PhaseOffset")
        for z_layer in layers_xy_numbers:
            rows=[" ".join(str(v) for v in row) for row in z_layer]
            etree.SubElement(po,"Layer").text=etree.CDATA(", ".join(rows))

    def write(self, fn):
        with open(fn,"w",encoding="utf-8") as f:
            f.write(etree.tostring(self.tree, encoding="unicode", pretty_print=True))

v = VXA()

# Both actuating materials have CTE>0 → both expand/contract
actA = v.add_material(E=5e6, RHO=1e4, CTE=0.5, RGBA=(255,0,0,255))
actB = v.add_material(E=5e6, RHO=1e4, CTE=0.5, RGBA=(255,128,0,255))
anchor = v.add_material(E=5e7, RHO=1e4, CTE=0.0, isFixed=1, RGBA=(0,255,0,255))  # just to pin one end

# Geometry: [anchor][actA][actB]
v.set_data_layers([f"{anchor}{actA}{actB}"])  # "312" etc., depending on IDs

# Per-voxel phase: actA=0.0, actB=0.5 (anchor can be 0.0)
v.set_phase_offset([ [ [0.0, 0.0, 0.5] ] ])

v.write("both_expand_out_of_phase.vxa")
print("Wrote both_expand_out_of_phase.vxa")




# Write out the vxa to data/ directory
v.write(f"{USER_VOXCRAFT_FOLDER}/base.vxa")

# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
robot=np.random.randint(0, 3, size=(2, 2, 1))
print(robot)
vxd.set_data(robot)
# Write out the vxd to data/
vxd.write(f"{USER_VOXCRAFT_FOLDER}/robot.vxd")

# Now you can simulate robot.vxd in voxcraft-sim with this terminal command
# ./voxcraft-sim -i ../inputs/ >  robot.history
# later, you can load the robot.history into voxcraft-viz to watch the simulation results
