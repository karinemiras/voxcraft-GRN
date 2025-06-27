import math
import numpy as np
import random
from VoxcraftVXA import VXA
from VoxcraftVXD import VXD

# TODO: save seeds later
def GRN_random():
    genome_size = 150+1
    genotype = [round(random.uniform(0, 1), 2) for _ in range(genome_size)]
    return genotype


class DS:
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    FRONT = 5
    BACK = 6


class GRN:

    # develops a Gene Regulatory network
    def __init__(self, max_voxels, tfs, genotype, querying_seed, env_condition, n_env_conditions, plastic_body):

        self.max_voxels = max_voxels
        self.genotype = genotype
        self.querying_seed = querying_seed
        self.env_condition = env_condition
        self.n_env_conditions = n_env_conditions
        self.plastic_body = plastic_body
        self.random = None
        self.cells = []
        self.phenotype = None
        self.genes = []
        self.quantity_voxels = 0

        self.regulatory_transcription_factor_idx = 0
        self.regulatory_v1_idx = 1
        self.regulatory_v2_idx = 2
        self.transcription_factor_idx = 3
        self.transcription_factor_amount_idx = 4
        self.diffusion_site_idx = 5
        self.types_nucleotides = 6
        self.diffusion_sites_qt = 6

        self.promoter_threshold = 0.8
        self.concentration_decay = 0.005
        self.structural_tfs = None

        three_voxels = {'bone': 1, 'fat': 2, 'muscle': 3}

        # if u increase number of reg tfs without increasing voxels tf or geno size,
        # too many only-head robots are sampled
        if tfs == 'reg2m3':  # balanced, number of regulatory tfs similar to number of voxels tfs
            self.regulatory_tfs = 2
            self.structural_tfs = three_voxels
        elif tfs == '':  # more regulatory, number of regulatory tfs greater than the number of voxels tfs
            pass

        # structural_tfs use initial indexes and regulatory tfs uses final indexes
        self.product_tfs = []
        for tf in range(1, len(self.structural_tfs)+1):
            self.product_tfs.append(f'TF{tf}')

        self.increase_scaling = 100
        self.intra_diffusion_rate = self.concentration_decay/2
        self.inter_diffusion_rate = self.intra_diffusion_rate/8
        self.dev_steps = 100
        self.concentration_threshold = self.genotype[0]
        self.genotype = self.genotype[1:]
        # TODO: evolve all params in the future?

    def develop(self):

        self.random = random.Random()#self.querying_seed)
        self.quantity_nodes = 0
        self.develop_body()
        self.phenotype.finalize()

        return self.phenotype

    def develop_body(self):
        self.gene_parser()
        self.regulate()

    def develop_knockout(self, knockouts):

        self.random = random.Random(self.querying_seed)
        self.quantity_nodes = 0
        self.gene_parser()

        if knockouts is not None:
            self.genes = self.genes[np.logical_not(np.isin(np.arange(self.genes.shape[0]), knockouts))]

        self.regulate()
        self.phenotype.finalize()

        return self.phenotype, self.genes

    # parses genotype to discover promoter sites and compose genes
    def gene_parser(self):
        nucleotide_idx = 0
        while nucleotide_idx < len(self.genotype):

            if self.genotype[nucleotide_idx] < self.promoter_threshold:
                # if there are nucleotides enough to compose a gene
                if (len(self.genotype)-1-nucleotide_idx) >= self.types_nucleotides:
                    regulatory_transcription_factor = self.genotype[nucleotide_idx+self.regulatory_transcription_factor_idx+1]
                    regulatory_v1 = self.genotype[nucleotide_idx+self.regulatory_v1_idx+1]
                    regulatory_v2 = self.genotype[nucleotide_idx+self.regulatory_v2_idx+1]
                    transcription_factor = self.genotype[nucleotide_idx+self.transcription_factor_idx+1] # gene product
                    transcription_factor_amount = self.genotype[nucleotide_idx+self.transcription_factor_amount_idx+1]
                    diffusion_site = self.genotype[nucleotide_idx+self.diffusion_site_idx+1]

                    # begin: converts tfs values into labels #
                    total = len(self.structural_tfs) + self.regulatory_tfs
                    range_size = 1 / total
                    limits = np.linspace(0, 1 - range_size, total)
                    limits = [round(limit, 2) for limit in limits]

                    for idx in range(0, len(limits)-1):

                        if regulatory_transcription_factor >= limits[idx] and regulatory_transcription_factor < limits[idx+1]:
                            regulatory_transcription_factor_label = 'TF'+str(idx+1)
                        elif regulatory_transcription_factor >= limits[idx+1]:
                            regulatory_transcription_factor_label = 'TF' + str(len(limits))

                        if transcription_factor >= limits[idx] and transcription_factor < limits[idx+1]:
                            transcription_factor_label = 'TF'+str(idx+1)
                        elif transcription_factor >= limits[idx+1]:
                            transcription_factor_label = 'TF'+str(len(limits))
                    # ends: converts tfs values into labels #

                    # begin: converts diffusion sites values into labels #
                    range_size = 1 / self.diffusion_sites_qt
                    limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]
                    for idx in range(0, len(limits) - 1):
                        if limits[idx+1] > diffusion_site >= limits[idx]:
                            diffusion_site_label = idx
                        elif diffusion_site >= limits[idx+1]:
                            diffusion_site_label = len(limits)-1
                    # ends: converts diffusion sites values into labels #

                    gene = [regulatory_transcription_factor_label, regulatory_v1, regulatory_v2,
                            transcription_factor_label, transcription_factor_amount, diffusion_site_label]

                    self.genes.append(gene)

                    nucleotide_idx += self.types_nucleotides
            nucleotide_idx += 1
        self.genes = np.array(self.genes)

    def net_parser(self):

        connections = []
        numbers_regulators = []
        self.gene_parser()
        for id_regulated, gene_regulated in enumerate(self.genes):
            number_regulators = 0
            for id_regulator, gene_regulator in enumerate(self.genes):
                if gene_regulated[self.regulatory_transcription_factor_idx] == gene_regulator[
                    self.transcription_factor_idx]:
                    connections.append((id_regulator, id_regulated))
                    number_regulators += 1
            numbers_regulators.append(number_regulators)
        return connections, numbers_regulators

    def regulate(self):
        #TODO: provide param for voxel dimensions and voxel types
        # 0 means no voxel
        self.phenotype = np.random.randint(0, 2, size=(3, 3, 3)) # x, y, z
        self.maternal_injection()
        self.growth()

    # develop embryo from single cell
    def growth(self):

        maximum_reached = False
        for t in range(0, self.dev_steps):

            # develops cells in order of age
            for idxc in range(0, len(self.cells)):

                cell = self.cells[idxc]
                self.increase(cell)
                # for tf in cell.transcription_factors:
                #     self.intra_diffusion(tf, cell)
                #     self.inter_diffusion(tf, cell)

                # try to grow new cell
                self.place_voxel(cell)

                if self.quantity_voxels == self.max_voxels - 1:
                    maximum_reached = True
                    break

                # do decay only after possible growth,
                # so that small increases have more chance to have an effect before decaying too much
                # this means that first injection/parsing decays a bit before expression,
                # but that is neglectable in comparison to injection size
                for tf in cell.transcription_factors:
                    self.decay(tf, cell)

            if maximum_reached:
                break

    # increase of originally expressed genes (meaning that gene products resulting from diffusion/split do not increase)
    def increase(self, cell):

        # for all genes in the dna #TODO: easier to loop original_genes instead
        for idg, gene in enumerate(self.genes):

            # if that gene was originally expressed (during dna parse at cell split)
            if idg in cell.original_genes:

                # increases a genes tf if there is enough of its regulatory tf
                if cell.transcription_factors.get(gene[self.regulatory_transcription_factor_idx]):

                    tf_in_all_sites = sum(cell.transcription_factors[gene[self.regulatory_transcription_factor_idx]])

                    regulatory_min_val = min(float(gene[self.regulatory_v1_idx]),
                                             float(gene[self.regulatory_v2_idx]))
                    regulatory_max_val = max(float(gene[self.regulatory_v1_idx]),
                                             float(gene[self.regulatory_v2_idx]))

                    if tf_in_all_sites >= regulatory_min_val and tf_in_all_sites <= regulatory_max_val:
                        cell.transcription_factors[gene[self.transcription_factor_idx]][int(gene[self.diffusion_site_idx])] += \
                            float(gene[self.transcription_factor_amount_idx]) \
                            / float(self.increase_scaling)

    def inter_diffusion(self, tf, cell):

        for ds in range(0, self.diffusion_sites_qt):

            # back slot of all voxels but core share with parent
            if ds == Core.BACK and \
                    (type(cell.developed_voxel) == ActiveHinge or type(cell.developed_voxel) == Brick):
                if cell.transcription_factors[tf][Core.BACK] >= self.inter_diffusion_rate:

                    cell.transcription_factors[tf][Core.BACK] -= self.inter_diffusion_rate

                    # updates or includes
                    if cell.developed_voxel._parent.cell.transcription_factors.get(tf):
                        cell.developed_voxel._parent.cell.transcription_factors[tf][cell.developed_voxel.direction_from_parent] += self.inter_diffusion_rate
                    else:
                        cell.developed_voxel._parent.cell.transcription_factors[tf] = [0] * self.diffusion_sites_qt
                        cell.developed_voxel._parent.cell.transcription_factors[tf][cell.developed_voxel.direction_from_parent] += self.inter_diffusion_rate

            # concentrations of sites without slot are also shared with single child in the case of joint
            elif ds in [Core.LEFT, Core.FRONT, Core.RIGHT] and type(cell.developed_voxel) == ActiveHinge:

                if cell.developed_voxel.children[Core.FRONT] is not None \
                        and cell.transcription_factors[tf][ds] >= self.inter_diffusion_rate:
                    cell.transcription_factors[tf][ds] -= self.inter_diffusion_rate

                    # updates or includes
                    if cell.developed_voxel.children[Core.FRONT].cell.transcription_factors.get(tf):
                        cell.developed_voxel.children[Core.FRONT].cell.transcription_factors[tf][Core.BACK] += self.inter_diffusion_rate
                    else:
                        cell.developed_voxel.children[Core.FRONT].cell.transcription_factors[tf] = [0] * self.diffusion_sites_qt
                        cell.developed_voxel.children[Core.FRONT].cell.transcription_factors[tf][Core.BACK] += self.inter_diffusion_rate
            else:

                # everyone shares with children
                #TODO: this does not allow for children of active joint to receive diffusion: fix it
                if cell.developed_voxel.children[ds] is not None \
                    and cell.transcription_factors[tf][ds] >= self.inter_diffusion_rate:
                    cell.transcription_factors[tf][ds] -= self.inter_diffusion_rate

                    # updates or includes
                    if cell.developed_voxel.children[ds].cell.transcription_factors.get(tf):
                        cell.developed_voxel.children[ds].cell.transcription_factors[tf][Core.BACK] += self.inter_diffusion_rate
                    else:
                        cell.developed_voxel.children[ds].cell.transcription_factors[tf] = [0] * self.diffusion_sites_qt
                        cell.developed_voxel.children[ds].cell.transcription_factors[tf][Core.BACK] += self.inter_diffusion_rate

    def intra_diffusion(self, tf, cell):

        # for each site in original slots order
        for ds in range(0, self.diffusion_sites_qt):

            # finds sites at right and left (cyclically)
            ds_left = ds - 1 if ds - 1 >= 0 else self.diffusion_sites_qt - 1
            ds_right = ds + 1 if ds + 1 <= self.diffusion_sites_qt - 1 else 0

            # first right
            if cell.transcription_factors[tf][ds] >= self.intra_diffusion_rate:
                cell.transcription_factors[tf][ds] -= self.intra_diffusion_rate
                cell.transcription_factors[tf][ds_right] += self.intra_diffusion_rate
            #  then left
            if cell.transcription_factors[tf][ds] >= self.intra_diffusion_rate:
                cell.transcription_factors[tf][ds] -= self.intra_diffusion_rate
                cell.transcription_factors[tf][ds_left] += self.intra_diffusion_rate

    def decay(self, tf, cell):
        # decay in all sites
        for ds in range(0, self.diffusion_sites_qt):
            cell.transcription_factors[tf][ds] = \
                max(0, cell.transcription_factors[tf][ds] - self.concentration_decay)

    def place_voxel(self, parent_cell):

        product_concentrations = []
        for idm in range(0, len(self.structural_tfs)-1):
            # sum concentration of all diffusion sites
            concentration = sum(parent_cell.transcription_factors[self.product_tfs[idm]]) \
                if parent_cell.transcription_factors.get(self.product_tfs[idm]) else 0
            product_concentrations.append(concentration)

        # chooses tf with the highest concentration
        idx_max = product_concentrations.index(max(product_concentrations))

        # rotation is at the end of the list
        # concentration_rotation = sum(cell.transcription_factors[self.product_tfs[-1]]) \
        #     if cell.transcription_factors.get(self.product_tfs[-1]) else 0

        # if tf concentration above a threshold
        if product_concentrations[idx_max] > self.concentration_threshold:

            # grows in the free diffusion site with the highest concentration
            freeslots = np.array([c is None for c in parent_cell.children])

            if any(freeslots):

                true_indices = np.where(freeslots)[0]
                values_at_true_indices = np.array(parent_cell.transcription_factors[self.product_tfs[idx_max]])[true_indices]
                max_value_index = np.argmax(values_at_true_indices)
                position_of_max_value = true_indices[max_value_index]
                slot = position_of_max_value

                potential_child_coord, child_slot = self.find_child_slot(parent_cell.xyz_coordinates, slot)
                if self.phenotype[potential_child_coord] == 0:
                    voxel_type = self.structural_tfs[idx_max]

                    self.quantity_voxels += 1
                    self.new_cell(voxel_type, parent_cell, slot, child_slot, potential_child_coord)

    def new_cell(self, voxel_type, parent_cell, parent_slot, child_slot, xyz_coordinates):

        new_cell = Cell()
        new_cell.voxel_type = voxel_type
        new_cell.parent_cell = parent_cell
        new_cell.xyz_coordinates = xyz_coordinates

        # share concentrations in diffusion site of parent with child
        for tf in parent_cell.transcription_factors:

            if parent_cell.transcription_factors[tf][parent_slot] > 0:
                half_concentration = parent_cell.transcription_factors[tf][parent_slot] / 2
                parent_cell.transcription_factors[tf][parent_slot] = half_concentration
                new_cell.transcription_factors[tf] = [0, 0, 0, 0]
                new_cell.transcription_factors[tf][child_slot] = half_concentration

        self.express_genes(new_cell)
        self.cells.append(new_cell)

    def find_child_slot(self, xyz_coordinates_parent, parent_slot):

        x = 0
        y = 1
        z = 2

        if parent_slot == DS.LEFT:
            child_slot = DS.RIGHT
            xyz_coordinates_child = xyz_coordinates_parent.copy()
            xyz_coordinates_child[x] -= 1

        if parent_slot == DS.RIGHT:
            child_slot = DS.LEFT
            xyz_coordinates_child = xyz_coordinates_parent.copy()
            xyz_coordinates_child[x] += 1

        if parent_slot == DS.FRONT:
            child_slot = DS.BACK
            xyz_coordinates_child = xyz_coordinates_parent.copy()
            xyz_coordinates_child[y] += 1

        if parent_slot == DS.BACK:
            child_slot = DS.FRONT
            xyz_coordinates_child = xyz_coordinates_parent.copy()
            xyz_coordinates_child[y] -= 1

        if parent_slot == DS.UP:
            child_slot = DS.DOWN
            xyz_coordinates_child = xyz_coordinates_parent.copy()
            xyz_coordinates_child[z] += 1

        if parent_slot == DS.DOWN:
            child_slot = DS.UP
            xyz_coordinates_child = xyz_coordinates_parent.copy()
            xyz_coordinates_child[z] -= 1

        return xyz_coordinates_child, child_slot

    # karines original injection
    def maternal_injection(self):

        # injects maternal tf into zygot and starts development of the first cell
        # the tf injected is regulatory tf of the first gene in the genetic string
        # the amount injected is the minimum for the regulatory tf to regulate its regulated product
        first_gene_idx = 0
        tf_label_idx = 0
        min_value_idx = 1
        # TODO: do not inject nor grow if there are no genes (unlikely)
        mother_tf_label = self.genes[first_gene_idx][tf_label_idx]
        mother_tf_injection = float(self.genes[first_gene_idx][min_value_idx])

        first_cell = Cell()
        # distributes injection among diffusion sites
        first_cell.transcription_factors[mother_tf_label] = \
            [mother_tf_injection/self.diffusion_sites_qt] * self.diffusion_sites_qt

        self.express_genes(first_cell)
        self.cells.append(first_cell)
        middle_pos = [s // 2 for s in self.phenotype.shape]
        first_cell.xyz_coordinates = middle_pos
        #TODO: define type of first voxel based on expression
        self.phenotype[middle_pos] = self.structural_tfs['muscle']

    def express_genes(self, new_cell):

        for idg, gene in enumerate(self.genes):

            regulatory_min_val = min(float(gene[self.regulatory_v1_idx]),
                                     float(gene[self.regulatory_v2_idx]))
            regulatory_max_val = max(float(gene[self.regulatory_v1_idx]),
                                     float(gene[self.regulatory_v2_idx]))

            if new_cell.transcription_factors.get(gene[self.regulatory_transcription_factor_idx]):
                # expresses a gene if its regulatory tf is present and within a range
                tf_in_all_sites = sum(new_cell.transcription_factors[gene[self.regulatory_transcription_factor_idx]])
                if tf_in_all_sites >= regulatory_min_val and tf_in_all_sites <= regulatory_max_val:

                    # update or add
                    if new_cell.transcription_factors.get(gene[self.transcription_factor_idx]):
                        new_cell.transcription_factors[gene[self.transcription_factor_idx]] \
                            [int(gene[self.diffusion_site_idx])] += float(gene[self.transcription_factor_amount_idx])
                    else:
                        new_cell.transcription_factors[gene[self.transcription_factor_idx]] = [0] * self.diffusion_sites_qt
                        new_cell.transcription_factors[gene[self.transcription_factor_idx]] \
                        [int(gene[self.diffusion_site_idx])] = float(gene[self.transcription_factor_amount_idx])

                    new_cell.original_genes.append(idg)


class Cell:

    def __init__(self, voxel_type, parent_cell, xyz_coordinates):# -> None:
        self.voxel_type = voxel_type
        self.transcription_factors = {}
        self.original_genes = []
        self.xyz_coordinates = xyz_coordinates
        self.parent_cell = parent_cell
        self.children = []


genotype = GRN_random()

body = GRN(max_voxels=8, tfs='reg2m2', genotype=genotype, querying_seed=666,
                              env_condition="", n_env_conditions=1, plastic_body=0).develop()

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