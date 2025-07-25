from GRN_3D import GRN
from GRN_3D import GRN_random, unequal_crossover, mutation_type1
import numpy as np
import copy
from random import Random, random

from VoxcraftVXA import VXA
from VoxcraftVXD import VXD

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


rng = Random()
seed = random()
print('seed', seed)
rng.seed(seed)

genotype = GRN_random(rng)

cube_face_size = 4


def develop(id_individual, individual):
    genome = individual.genome
    body_voxels = GRN(max_voxels=8, cube_face_size=cube_face_size, tfs='reg2m3',
                      genotype=genome, rng=rng,
                      env_condition="", n_env_conditions=1, plastic_body=0).develop()

    #vxa = VXA(EnableExpansion=1, SimTime=1) # pass vxa tags in here

    # Create two materials with different properties
    # returns the material ID
    # mat1 = vxa.add_material(RGBA=(220, 220, 220), E=1e8, RHO=1e4) # bone
    # mat2 = vxa.add_material(RGBA=(255, 230, 128), E=1e4, RHO=1e4) # fat
    # mat3 = vxa.add_material(RGBA=(180, 30, 40), E=1e3, RHO=1e4) #muscle
    #
    #
    # # Write out the vxa to data/ directory
    # vxa.write("voxcraft-sim/inputs/base.vxa")


    body = np.zeros(body_voxels.shape, dtype=int)

    for index, value in np.ndenumerate(body_voxels):
        body[index] = value.voxel_type if value != 0 else 0

    # Remove empty y-layers
    x_mask = np.any(body != 0, axis=(1, 2))
    body = body[x_mask]
    y_mask = np.any(body != 0, axis=(0, 2))
    body = body[:, y_mask]
    z_mask = np.any(body != 0, axis=(0, 1))
    body = body[:, :, z_mask]

    # print('2body')
    # print(body)
    #
    # # Generate a VXD file
    # vxd = VXD()
    # vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
    # vxd.set_data(body)
    # # Write out the vxd to data
    # vxd.write("voxcraft-sim/inputs/robot1.vxd")


    # Define color map for values in body
    color_map = {
        1: (220/255, 220/255, 220/255, 0.5),  # Bone
        2: (255/255, 230/255, 128/255, 0.5),  # Fat
        3: (180/255, 30/255, 40/255, 0.5)     # Muscle
    }


    # Function to draw a single 1x1x1 cube
    def draw_cube(ax, position, color):
        x, y, z = position
        vertices = np.array([
            [x, y, z],
            [x + 1, y, z],
            [x + 1, y + 1, z],
            [x, y + 1, z],
            [x, y, z + 1],
            [x + 1, y, z + 1],
            [x + 1, y + 1, z + 1],
            [x, y + 1, z + 1]
        ])
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=0.5))

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through body and draw cubes
    x_dim, y_dim, z_dim = body.shape
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                val = body[x, y, z]
                if val > 0 and val in color_map:
                    draw_cube(ax, (y, x, z), color_map[val]) # invert y and x to match voxcraft-viz

    # Set limits to match array shape exactly
    ax.set_xticks(np.arange(0, cube_face_size + 1, 1))
    ax.invert_yaxis()  # to match voxcraft-viz
    ax.set_yticks(np.arange(0, cube_face_size+ 1, 1))
    ax.set_zticks(np.arange(0, cube_face_size+ 1, 1))

    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Visualization (1x1x1 Cubes, Correct Colors)')
    plt.tight_layout()

    # Save the image
    plt.savefig(f"../working_data/voxcraft-GRN/{id_individual}.png", dpi=300)



# Constants
POPULATION_SIZE = 100  # μ
OFFSPRING_COUNT = 100  # λ
TOURNAMENT_SIZE = 4
GENERATIONS = 3#100
RUNS = 1

# Global ID counter
next_id = 0
def get_next_id():
    global next_id
    id_ = next_id
    next_id += 1
    return id_

class Individual:
    def __init__(self, genome):
        self.id = get_next_id()
        self.genome = genome
        self.fitness = 0.0  # Leave zero for now

    def copy(self):
        new_ind = copy.deepcopy(self)
        new_ind.id = get_next_id()  # Ensure new unique ID for the copy
        return new_ind

def initialize_population(size):
    return [Individual(GRN_random(rng)) for _ in range(size)]

def mutate(individual, rng):
    print(individual.genome)
    individual.genome = mutation_type1(individual.genome, rng)

def crossover(parent1, parent2, rng):
    child_genome = unequal_crossover(parent1.copy(), parent2.copy(), rng)
    child = Individual(child_genome)
    return child

def tournament_selection(population, k):
    return max(rng.sample(population, k), key=lambda ind: ind.fitness)

def evolutionary_algorithm():
    for run in range(RUNS):
        # Initialization
        population = initialize_population(POPULATION_SIZE)

        for generation in range(GENERATIONS):
            # Generate offspring
            offspring = []
            for _ in range(OFFSPRING_COUNT // 2):
                parent1 = tournament_selection(population, TOURNAMENT_SIZE)
                parent2 = tournament_selection(population, TOURNAMENT_SIZE)

                child1 = crossover(parent1, parent2, rng)
                child2 = child1 # TODO: fix crossover later
                mutate(child1, rng)
                mutate(child2, rng)
                offspring.append(child1)
                offspring.append(child2)
                develop(child1.id, child1)
                develop(child2.id, child2)

            # Combine parents and offspring
            combined = population + offspring

            # Select next generation
            new_population = []
            for _ in range(POPULATION_SIZE):
                winner = tournament_selection(combined, TOURNAMENT_SIZE)
                new_population.append(winner.copy())

            population = new_population

        print(f"Run {run + 1} completed.")

# Run the algorithm
if __name__ == "__main__":
    evolutionary_algorithm()






