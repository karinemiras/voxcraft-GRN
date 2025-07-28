from GRN_3D import GRN, initialization, mutation_type1, unequal_crossover
from simulation_resources import simulate
from utils import draw_phenotype
from metrics import *
from config import Config

import random
import numpy as np


class Individual:
    def __init__(self, genome, id_counter):
        self.id = id_counter
        self.genome = genome
        self.phenotype = None
        self.fitness = 0.0


class EA:
    def __init__(self):
        args = Config()._get_params()
       # self.mainpath = args.mainpath #TODO: use script later
        self.mainpath ='../working_data/voxcraft-GRN'
        self.sim_path = 'voxcraft-sim/inputs'


        self.rng = random.Random()
        seed = random.randint(0, 2**32 - 1)
        print('seed', seed)
        self.rng.seed(seed)

        self.CUBE_FACE_SIZE = args.cube_face_size
        self.max_voxels = args.max_voxels

        self.tfs = args.tfs
        self.MAX_GENOME_SIZE = 1000
        self.INI_GENOME_SIZE = 150
        self.PROMOTOR_THRESHOLD = 0.8
        self.TYPES_NUCLEOTIDES = 6
        self.plastic = args.plastic
        self.env_conditions = args.env_conditions

        self.POPULATION_SIZE = args.population_size
        self.offspring_size = args.offspring_size
        self.tournament_k = args.tournament_k
        self.num_generations = args.num_generations
        self.id_counter = 0

        #TOFO: try to recover if exidst otherwise create folder

    def develop_phenotype(self, individual):
        genome = individual.genome
        # TODO: take prams form config
        phenotype = GRN(promoter_threshold=self.PROMOTOR_THRESHOLD, types_nucleotides=self.TYPES_NUCLEOTIDES,
                          max_voxels=self.max_voxels, cube_face_size=self.CUBE_FACE_SIZE, tfs=self.tfs,
                          genotype=genome, env_conditions=self.env_conditions, plastic=self.plastic).develop()

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

    def initialize_population(self, size):
        ind = []
        for _ in range(size):
            self.id_counter += 1
            ind.append(Individual(initialization(self.rng, self.INI_GENOME_SIZE), self.id_counter))
        return ind

    def mutate(self, individual):
        individual.genome = mutation_type1(self.rng, individual.genome)

    def crossover(self, parent1, parent2):
        child_genome = unequal_crossover(self.rng, self.PROMOTOR_THRESHOLD, self.TYPES_NUCLEOTIDES, self.MAX_GENOME_SIZE,
                                         list(parent1.genome), list(parent2.genome))
        self.id_counter += 1
        child = Individual(child_genome, self.id_counter)
        return child

    def tournament_selection(self, population, k):
        return max(self.rng.sample(population, k), key=lambda ind: ind.fitness)

    def run(self):

        # Initialization
        population = self.initialize_population(self.POPULATION_SIZE)

        avg_fitness_per_gen = []  # For plotting

        for generation in range(self.num_generations):

            # Generate offspring
            offspring = []
            for _ in range(self.offspring_size // 2):
                parent1 = self.tournament_selection(population, self.tournament_k)
                parent2 = self.tournament_selection(population, self.tournament_k)

                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                offspring.append(child1)
                offspring.append(child2)

                child1.phenotype = self.develop_phenotype(child1)
                child2.phenotype = self.develop_phenotype(child2)

                draw_phenotype(child1.phenotype, child1.id, self.CUBE_FACE_SIZE)
                draw_phenotype(child2.phenotype, child2.id, self.CUBE_FACE_SIZE)

                simulate(child1.phenotype, self.sim_path, 1) #TODFO: make thread
                simulate(child2.phenotype, self.sim_path, 1)

            # Combine parents and offspring
            combined = population + offspring
            uniqueness(combined)

            # Select next generation
            new_population = []
            for _ in range(self.POPULATION_SIZE):
                winner = self.tournament_selection(combined, self.tournament_k)
                new_population.append(winner)

            population = new_population
            uniqueness(population)

            # Store average fitness for plotting
            avg_fitness = np.mean([ind.fitness for ind in population])
            avg_fitness_per_gen.append(avg_fitness)
            print(generation, avg_fitness)

        print(f"Run completed.")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(range(len(avg_fitness_per_gen)), avg_fitness_per_gen, label=f'Run ')
        plt.xlabel('Generation')
        plt.ylabel('Average Novelty (Fitness)')
        plt.title('Fitness Progression Across Generations')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig("fitness_plot.png")  # Save the figure to file


EA().run()




