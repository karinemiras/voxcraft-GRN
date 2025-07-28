from GRN_3D import GRN, initialization, mutation_type1, unequal_crossover
from simulation_resources import simulate
from utils import draw_phenotype

import copy
import random
import numpy as np

rng = random.Random()
seed = random.randint(0, 2 ** 32 - 1)
print('seed', seed)
rng.seed(seed)

PROMOTOR_THRESHOLD = 0.8
TYPES_NUCLEOTIDES = 6
CUBE_FACE_SIZE = 4
MAX_GENOME_SIZE = 1000
INI_GENOME_SIZE = 150

POPULATION_SIZE = 10  # μ
OFFSPRING_COUNT = 10  # λ
TOURNAMENT_SIZE = 4
GENERATIONS = 2
RUNS = 1

id_counter = 1


def develop_phenotype(individual):
    genome = individual.genome
    # TODO: take prams form config
    phenotype = GRN(promoter_threshold=PROMOTOR_THRESHOLD, types_nucleotides=TYPES_NUCLEOTIDES,
                    max_voxels=8, cube_face_size=CUBE_FACE_SIZE, tfs='reg2m3',
                    genotype=genome, env_condition="", n_env_conditions=1, plastic_body=0).develop()

    phenotype_materials = np.zeros(phenotype.shape, dtype=int)
    for index, value in np.ndenumerate(phenotype):
        phenotype_materials[index] = value.voxel_type if value != 0 else 0

    # Remove empty layers
    trimmed_phenotype_materials = phenotype_materials
    x_mask = np.any(trimmed_phenotype_materials != 0, axis=(1, 2))
    trimmed_phenotype_materials = trimmed_phenotype_materials[x_mask]
    y_mask = np.any(trimmed_phenotype_materials != 0, axis=(0, 2))
    trimmed_phenotype_materials = trimmed_phenotype_materials[:, y_mask]
    z_mask = np.any(trimmed_phenotype_materials != 0, axis=(0, 1))
    trimmed_phenotype_materials = trimmed_phenotype_materials[:, :, z_mask]

    return trimmed_phenotype_materials


class Individual:
    def __init__(self, genome):
        self.id = id_counter
        self.genome = genome
        self.phenotype = None
        self.fitness = 0.0

        id_counter += 1


def initialize_population(size):
    ind = [Individual(initialization(rng, INI_GENOME_SIZE)) for _ in range(size)]
    return ind


def mutate(individual):
    individual.genome = mutation_type1(rng, individual.genome)


def crossover(parent1, parent2):
    child_genome = unequal_crossover(rng, PROMOTOR_THRESHOLD, TYPES_NUCLEOTIDES, MAX_GENOME_SIZE,
                                     parent1.copy(), parent2.copy(), )
    child = Individual(child_genome)
    return child


def tournament_selection(population, k):
    return max(rng.sample(population, k), key=lambda ind: ind.fitness)


def evolutionary_algorithm():
    for run in range(RUNS):
        # Initialization
        population = initialize_population(POPULATION_SIZE)

        avg_fitness_per_gen = []  # For plotting

        for generation in range(GENERATIONS):

            # Generate offspring
            offspring = []
            for _ in range(OFFSPRING_COUNT // 2):
                parent1 = tournament_selection(population, TOURNAMENT_SIZE)
                parent2 = tournament_selection(population, TOURNAMENT_SIZE)

                child1 = crossover(parent1, parent2)
                child2 = crossover(parent1, parent2)
                mutate(child1)
                mutate(child2)
                offspring.append(child1)
                offspring.append(child2)

                child1.phenotype = develop_phenotype(child1)
                child2.phenotype = develop_phenotype(child2)

                draw_phenotype(child1.phenotype, child1.id, CUBE_FACE_SIZE)
                draw_phenotype(child2.phenotype, child2.id, CUBE_FACE_SIZE)

                simulate(child1.phenotype, child1.id)
                simulate(child2.phenotype, child2.id)

            # Combine parents and offspring
            combined = population + offspring
            compute_novelty_fitness(combined)

            # Select next generation
            new_population = []
            for _ in range(POPULATION_SIZE):
                winner = tournament_selection(combined, TOURNAMENT_SIZE)
                new_population.append(winner.copy())

            population = new_population
            compute_novelty_fitness(population)

            # Store average fitness for plotting
            avg_fitness = np.mean([ind.fitness for ind in population])
            avg_fitness_per_gen.append(avg_fitness)
            print(generation, avg_fitness)

        print(f"Run {run + 1} completed.")
        # fig = plt.figure()
        # plt.plot(range(len(avg_fitness_per_gen)), avg_fitness_per_gen, label=f'Run {run + 1}')
        # plt.xlabel('Generation')
        # plt.ylabel('Average Novelty (Fitness)')
        # plt.title('Fitness Progression Across Generations')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # fig.savefig("fitness_plot.png")  # Save the figure to file


def tree_edit_distance(g1, g2):
    """
    Placeholder for tree edit distance.
    Replace this with your actual implementation.
    """
    # Simplified: use numpy and compare as flat arrays for now.
    # You should replace this with your true tree edit distance.
    g1_flat = np.array(g1).flatten()
    g2_flat = np.array(g2).flatten()
    return np.sum(g1_flat != g2_flat)  # Hamming distance as placeholder


def compute_novelty_fitness(population, k=10):
    """
    Assigns fitness to each individual based on novelty,
    using average distance to k nearest neighbors.
    """
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = tree_edit_distance(ind.phenotype, other.phenotype)
                distances.append(d)
        nearest_distances = sorted(distances)[:k]
        ind.fitness = np.mean(nearest_distances)


# Run the algorithm
if __name__ == "__main__":
    evolutionary_algorithm()






