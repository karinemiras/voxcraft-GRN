# experiment.py
import os
import random
import sqlite3
from sqlalchemy import create_engine, func, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from EA_classes import Base, Robot, GenerationSurvivor, Individual, ExperimentInfo


# Enable FK enforcement in SQLite (otherwise FK errors won't trip the transaction)
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


class Experiment:
    """
    Handles experiment bookkeeping:
      - output/db paths
      - SQLAlchemy engine/session
      - RNG seed management
      - state recovery from DB
      - atomic persistence per generation
    """

    def __init__(self, args):
        # paths
        self.out_path = f"{args.out_path}/{args.study_name}/{args.experiment_name}"
        os.makedirs(self.out_path, exist_ok=True)
        self.db_path = os.path.join(self.out_path, f'run_{args.run}')

        # DB
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.session = self.Session()

        # RNG (seed is persisted in the DB for reproducibility)
        self.rng = random.Random()
        info = self.session.query(ExperimentInfo).first()
        if info is None:
            seed = random.randint(0, 2**32 - 1)
            print("seed (new)", seed)
            self.rng.seed(seed)
            self.session.add(ExperimentInfo(seed=seed))
            self.session.commit()
        else:
            print("seed (reused)", info.seed)
            self.rng.seed(info.seed)

        # running ID counter for Individuals/Robots
        self.id_counter = 0

    # ---------- Recovery ----------

    def _individual_from_robot(self, r: Robot) -> Individual:
        ind = Individual(genome=r.genome, id_counter=r.robot_id)
        ind.num_voxels = float(r.num_voxels) if r.num_voxels is not None else 0.0
        return ind

    def _recover_state(self):
        """
        Returns (last_completed_generation, recovered_population or None).

        If there is no completed generation, returns (None, None).
        Requires subclass to implement `develop_phenotype(genome)`.
        """
        with self.Session() as s:
            last_gen = s.query(func.max(GenerationSurvivor.generation)).scalar()
            if last_gen is None:
                # Assert the invariant: no robots should exist either
                if s.query(Robot).count() != 0:
                    raise RuntimeError(
                        "DB inconsistent: robots exist but no survivors. Clean or migrate."
                    )
                self.id_counter = 0
                return None, None

            # Rebuild population = survivors from last completed generation
            rows = (
                s.query(Robot, GenerationSurvivor)
                .join(GenerationSurvivor, GenerationSurvivor.robot_id == Robot.robot_id)
                .filter(GenerationSurvivor.generation == last_gen)
                .all()
            )

            population = []
            for r, gs in rows:
                ind = self._individual_from_robot(r)
                # Hook into subclass to rebuild phenotype
                ind.phenotype = self.develop_phenotype(ind.genome)
                ind.fitness = float(gs.fitness or 0.0)
                ind.uniqueness = float(gs.uniqueness or 0.0)
                population.append(ind)

            # Set next ID
            max_id = s.query(func.max(Robot.robot_id)).scalar()
            self.id_counter = int(max_id) if max_id is not None else 0

            return int(last_gen), population

    # ---------- Persistence (one atomic save per generation) ----------

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
            s.add(
                Robot(
                    robot_id=individual.id,
                    born_generation=int(born_generation),
                    genome=individual.genome,
                    num_voxels=float(individual.num_voxels) if individual.num_voxels is not None else 0.0,
                )
            )
        else:
            row.num_voxels = (
                float(individual.num_voxels) if individual.num_voxels is not None else row.num_voxels
            )

    def _stage_generation_survivors(self, s, generation, survivors):
        for ind in survivors:
            s.merge(
                GenerationSurvivor(
                    generation=int(generation),
                    robot_id=int(ind.id),
                    fitness=float(ind.fitness or 0.0),
                    uniqueness=float(ind.uniqueness or 0.0),
                )
            )
