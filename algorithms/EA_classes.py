from sqlalchemy.orm import declarative_base
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, JSON, ForeignKey, UniqueConstraint,
    PrimaryKeyConstraint
)

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
    genome = Column(JSON, nullable=False)                 # list or dict; SQLAlchemy will JSON-encode for SQLite

    num_voxels = Column(Float, default=0.0)


class GenerationSurvivor(Base):
    __tablename__ = "generation_survivors"
    generation = Column(Integer, nullable=False)
    robot_id = Column(Integer, ForeignKey("all_robots.robot_id"), nullable=False)

    fitness = Column(Float, default=0.0)
    uniqueness = Column(Float, default=0.0)

    __table_args__ = (
        PrimaryKeyConstraint("generation", "robot_id", name="pk_generation_robot"),
    )


class Individual:
    def __init__(self, genome, id_counter):
        self.id = id_counter
        self.genome = genome
        self.phenotype = None

        # absolute
        self.num_voxels = 0.0

        # relative
        self.fitness = 0.0
        self.uniqueness = 0.0