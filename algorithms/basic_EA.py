import os
import sys
import numpy as np

from experiment import Experiment
from EA_classes import Individual
from GRN_3D import GRN, initialization, mutation_type1, unequal_crossover

# TODO: use more elegant path import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulation.simulation_resources import simulate
from utils import draw_phenotype
from metrics import phenotype_abs_metrics, behavior_abs_metrics, relative_metrics
from config import Config


class EA(Experiment):
    def __init__(self):
        args = Config()._get_params()
        super().__init__(args)  # sets out_path, DB, session, rng, id_counter

        # experiment-level params used by EA logic
        self.MAX_GENOME_SIZE = 1000
        self.INI_GENOME_SIZE = 150
        self.PROMOTOR_THRESHOLD = 0.8
        self.TYPES_NUCLEOTIDES = 6

        self.sim_path = args.sim_path
        self.CUBE_FACE_SIZE = args.cube_face_size
        self.max_voxels = args.max_voxels
        self.tfs = args.tfs
        self.plastic = args.plastic
        self.env_conditions = args.env_conditions
        self.POPULATION_SIZE = args.population_size
        self.offspring_size = args.offspring_size
        self.crossover_prob = args.crossover_prob
        self.mutation_prob = args.mutation_prob
        self.tournament_k = args.tournament_k
        self.num_generations = args.num_generations
        self.fitness_metric = args.fitness_metric

    # ---------- EA-specific utilities ----------

    def develop_phenotype(self, genome):
        phenotype = GRN(
            promoter_threshold=self.PROMOTOR_THRESHOLD,
            types_nucleotides=self.TYPES_NUCLEOTIDES,
            max_voxels=self.max_voxels,
            cube_face_size=self.CUBE_FACE_SIZE,
            tfs=self.tfs,
            genotype=genome,
            env_conditions=self.env_conditions,
            plastic=self.plastic,
        ).develop()

        phenotype_materials = np.zeros(phenotype.shape, dtype=int)
        for index, value in np.ndenumerate(phenotype):
            phenotype_materials[index] = value.voxel_type if value != 0 else 0

        return phenotype_materials

    def initialize_population(self, size):
        individuals = []
        for _ in range(size):
            self.id_counter += 1
            individuals.append(Individual(initialization(self.rng, self.INI_GENOME_SIZE), self.id_counter))
        return individuals

    def mutate(self, individual):
        if self.rng.uniform(0, 1) <= self.mutation_prob:
            individual.genome = mutation_type1(self.rng, individual.genome)

    def crossover(self, parent1, parent2):
        if self.rng.uniform(0, 1) <= self.crossover_prob:
            child_genome = unequal_crossover(
                self.rng,
                self.PROMOTOR_THRESHOLD,
                self.TYPES_NUCLEOTIDES,
                self.MAX_GENOME_SIZE,
                list(parent1.genome),
                list(parent2.genome),
            )
        else:
            chosen = self.rng.choice((parent1, parent2))
            child_genome = list(chosen.genome)

        self.id_counter += 1
        child = Individual(child_genome, self.id_counter)
        return child

    def tournament_selection(self, population, k):
        # tie-breaks handled by max() on fitness
        return max(self.rng.sample(population, k), key=lambda ind: ind.fitness)

    # ---------- Main run ----------

    def run(self):
        last_gen, recovered_population = self._recover_state()

        if recovered_population is None:
            # Fresh start
            generation = 0
            population = self.initialize_population(self.POPULATION_SIZE)
            for ind in population:
                ind.phenotype = self.develop_phenotype(ind.genome)
                phenotype_abs_metrics(ind)
                simulate(ind.phenotype, self.sim_path, 1)
                behavior_abs_metrics(ind)
            relative_metrics(population, self.fitness_metric)

            # persist parents as both robots and survivors for gen 0
            self._persist_generation_atomic(generation, population, population)
            start_gen = 1
            print(f"Finished generation {generation}.")
        else:
            # Continue from the next generation after the last completed one
            population = recovered_population
            start_gen = last_gen + 1
            print(
                f"Recovered last completed generation = {last_gen}, "
                f"population size = {len(population)}, next id = {self.id_counter + 1}"
            )

        for generation in range(start_gen, self.num_generations):
            # Generate offspring
            offspring = []
            for _ in range(self.offspring_size):
                parent1 = self.tournament_selection(population, self.tournament_k)
                parent2 = self.tournament_selection(population, self.tournament_k)

                child = self.crossover(parent1, parent2)
                self.mutate(child)
                offspring.append(child)

                child.phenotype = self.develop_phenotype(child.genome)
                # draw_phenotype(child.phenotype, child.id, self.CUBE_FACE_SIZE, self.out_path)
                phenotype_abs_metrics(child)
                simulate(child.phenotype, self.sim_path, 1)
                behavior_abs_metrics(child)

            # Combine parents and offspring into a pool
            pool = population + offspring
            relative_metrics(pool, self.fitness_metric)

            # Select next generation (unique winners)
            new_population = []
            pool = pool.copy()
            for _ in range(self.POPULATION_SIZE):
                k = min(self.tournament_k, len(pool))
                contestants = self.rng.sample(pool, k)
                winner = max(contestants, key=lambda ind: ind.fitness)
                new_population.append(winner)
                pool.remove(winner)  # ensures uniqueness

            population = new_population
            relative_metrics(population, self.fitness_metric)

            # Persist this generation atomically
            self._persist_generation_atomic(generation, offspring, population)
            print(f"Finished generation {generation}.")

        try:
            self.session.close()
        except Exception:
            pass

        print("Finished optimizing.")


if __name__ == "__main__":
    EA().run()
