from GRN_3D import GRN, initialization, mutation_type1, unequal_crossover

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.simulation_resources import simulate
from utils import draw_phenotype
from metrics import *
from config import Config

import random
import numpy as np
import json
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, JSON, ForeignKey, UniqueConstraint,
    PrimaryKeyConstraint
)
from sqlalchemy import func  # <<< add this
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError
import sqlite3
from sqlalchemy import event
from sqlalchemy.engine import Engine


# Enable FK enforcement in SQLite (otherwise FK errors won't trip the transaction)
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


Base = declarative_base()


class ExperimentInfo(Base):
    __tablename__ = "experiment_info"
    id = Column(Integer, primary_key=True, autoincrement=True)
    seed = Column(Integer, nullable=False)


class Robot(Base):
    __tablename__ = "all_robots"
    # store your own evolutionary ID, not the DB PK
    robot_id = Column(Integer, primary_key=True)          # matches Individual.id
    born_generation = Column(Integer, nullable=False)
    fitness = Column(Float, default=0.0)
    genome = Column(JSON, nullable=False)                 # list or dict; SQLAlchemy will JSON-encode for SQLite


class GenerationSurvivor(Base):
    __tablename__ = "generation_survivors"
    generation = Column(Integer, nullable=False)
    robot_id = Column(Integer, ForeignKey("all_robots.robot_id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint("generation", "robot_id", name="pk_generation_robot"),
    )


class Individual:
    def __init__(self, genome, id_counter):
        self.id = id_counter
        self.genome = genome
        self.phenotype = None
        self.fitness = 0.0


class EA:
    def __init__(self):
        args = Config()._get_params()
        self.out_path = f'{args.out_path}/{args.study_name}/{args.experiment_name}/{args.run}'
        self.sim_path = args.sim_path

        os.makedirs(self.out_path, exist_ok=True)
        self.db_path = os.path.join(self.out_path, "experiment.sqlite3")
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.session = self.Session()

        # manages experiment seed for reproducibility
        self.rng = random.Random()
        info = self.session.query(ExperimentInfo).first()
        if info is None:
            seed = random.randint(0, 2 ** 32 - 1)
            print('seed (new)', seed)
            self.rng.seed(seed)
            self.session.add(ExperimentInfo(seed=seed))
            self.session.commit()
        else:
            print('seed (reused)', info.seed)
            self.rng.seed(info.seed)

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
        self.crossover_prob = args.crossover_prob
        self.mutation_prob = args.mutation_prob
        self.tournament_k = args.tournament_k
        self.num_generations = args.num_generations
        self.id_counter = 0

    def _individual_from_robot(self, r: Robot) -> Individual:
        ind = Individual(genome=r.genome, id_counter=r.robot_id)
        ind.fitness = float(r.fitness) if r.fitness is not None else 0.0
        return ind

    def _recover_state(self):
        """
        Returns (last_completed_generation, recovered_population or None).

        If there is no completed generation, last_completed_generation is None
        and population is None.
        """
        with self.Session() as s:
            last_gen = s.query(func.max(GenerationSurvivor.generation)).scalar()
            if last_gen is None:
                # Assert the invariant: no robots should exist either
                if s.query(Robot).count() != 0:
                    raise RuntimeError("DB inconsistent: robots exist but no survivors. Clean or migrate.")
                self.id_counter = 0
                return None, None

            # Rebuild population = survivors from last completed generation
            rows = (
                s.query(Robot)
                .join(GenerationSurvivor, GenerationSurvivor.robot_id == Robot.robot_id)
                .filter(GenerationSurvivor.generation == last_gen)
                .all()
            )
            population = [self._individual_from_robot(r) for r in rows]

            # Set next ID
            max_id = s.query(func.max(Robot.robot_id)).scalar()
            self.id_counter = int(max_id) if max_id is not None else 0

            return int(last_gen), population

    def develop_phenotype(self, individual):
        genome = individual.genome
        phenotype = GRN(promoter_threshold=self.PROMOTOR_THRESHOLD, types_nucleotides=self.TYPES_NUCLEOTIDES,
                          max_voxels=self.max_voxels, cube_face_size=self.CUBE_FACE_SIZE, tfs=self.tfs,
                          genotype=genome, env_conditions=self.env_conditions, plastic=self.plastic).develop()

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

        return trimmed_phenotype_materials

    def initialize_population(self, size):
        ind = []
        for _ in range(size):
            self.id_counter += 1
            ind.append(Individual(initialization(self.rng, self.INI_GENOME_SIZE), self.id_counter))
        return ind

    def mutate(self, individual):
        if self.rng.uniform(0, 1) <= self.mutation_prob:
            individual.genome = mutation_type1(self.rng, individual.genome)

    def crossover(self, parent1, parent2):
        if self.rng.uniform(0, 1) <= self.crossover_prob:
            child_genome = unequal_crossover(self.rng, self.PROMOTOR_THRESHOLD, self.TYPES_NUCLEOTIDES, self.MAX_GENOME_SIZE,
                                             list(parent1.genome), list(parent2.genome))
        else:
            chosen = self.rng.choice((parent1, parent2))
            child_genome = list(chosen.genome)

        self.id_counter += 1
        child = Individual(child_genome, self.id_counter)
        return child

    def tournament_selection(self, population, k):
        return max(self.rng.sample(population, k), key=lambda ind: ind.fitness)

    #  one atomic save per generation
    def _persist_generation_atomic(self, generation, robots_this_gen, survivors_this_gen):
        # Use a fresh session per generation to keep transactions clean
        with self.Session() as s, s.begin():  # s.begin() = single atomic transaction
            # Stage robot rows first (so FK to robot_id exists when survivors insert)
            for ind in robots_this_gen:
                self._stage_robot(s, ind, born_generation=generation)
            s.flush()  # optional: surfaces issues before adding survivors

            # Stage survivors
            self._stage_generation_survivors(s, generation, survivors_this_gen)
            # exiting the with-block commits; any exception rolls back everything

    def _stage_robot(self, s, individual, born_generation):
        row = s.get(Robot, individual.id)
        if row is None:
            s.add(Robot(
                robot_id=individual.id,
                born_generation=int(born_generation),
                fitness=float(individual.fitness) if individual.fitness is not None else 0.0,
                genome=individual.genome
            ))
        else:
            row.fitness = float(individual.fitness) if individual.fitness is not None else row.fitness

    def _stage_generation_survivors(self, s, generation, survivors):
        for ind in survivors:
            s.merge(GenerationSurvivor(generation=int(generation), robot_id=int(ind.id)))

    def run(self):

        last_gen, recovered_population = self._recover_state()

        if recovered_population is None:
            # Fresh start
            population = self.initialize_population(self.POPULATION_SIZE)
            generation = 0
            self._persist_generation_atomic(generation, population, population)
            start_gen = 1
            print(f"Finished generation {generation}.")
        else:
            # Continue from the next generation after the last completed one
            population = recovered_population
            start_gen = last_gen + 1
            print(f"Recovered last completed generation = {last_gen}, "
                  f"population size = {len(population)}, next id = {self.id_counter + 1}")

       # avg_fitness_per_gen = []  # For plotting

        for generation in range(start_gen, self.num_generations):

            # Generate offspring
            offspring = []
            for _ in range(self.offspring_size): # // 2): TODO: two complementary children
                parent1 = self.tournament_selection(population, self.tournament_k)
                parent2 = self.tournament_selection(population, self.tournament_k)

                child1 = self.crossover(parent1, parent2)
                # child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                # self.mutate(child2)
                offspring.append(child1)
                # offspring.append(child2)

                child1.phenotype = self.develop_phenotype(child1)
                # child2.phenotype = self.develop_phenotype(child2)

                # draw_phenotype(child1.phenotype, child1.id, self.CUBE_FACE_SIZE, self.out_path)
                # draw_phenotype(child2.phenotype, child2.id, self.CUBE_FACE_SIZE, self.out_path)

                simulate(child1.phenotype, self.sim_path, 1)
                # simulate(child2.phenotype, self.sim_path, 1)

            # Combine parents and offspring into a pool
            pool = population + offspring
            uniqueness(pool)

            # Select next generation
            new_population = []
            pool = pool.copy()
            for _ in range(self.POPULATION_SIZE):
                k = min(self.tournament_k, len(pool))
                contestants = self.rng.sample(pool, k)
                winner = max(contestants, key=lambda ind: ind.fitness)
                new_population.append(winner)
                pool.remove(winner)  # ensures uniqueness

            population = new_population
            uniqueness(population)

            self._persist_generation_atomic(generation, offspring, population)
            print(f"Finished generation {generation}.")

            # Store average fitness for plotting
            # avg_fitness = np.mean([ind.fitness for ind in population])
            # avg_fitness_per_gen.append(avg_fitness)
            # print(generation, avg_fitness)

        try:
            self.session.close()
        except Exception:
            pass

        print(f"Finished optimizing.")

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # plt.plot(range(len(avg_fitness_per_gen)), avg_fitness_per_gen, label=f'Run ')
        # plt.xlabel('Generation')
        # plt.ylabel('Average Novelty (Fitness)')
        # plt.title('Fitness Progression Across Generations')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # fig.savefig("fitness_plot.png")  # Save the figure to file


EA().run()




