from sqlalchemy.orm import declarative_base
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, JSON, ForeignKey, UniqueConstraint,
    PrimaryKeyConstraint
)
from math import inf

Base = declarative_base()

# DB CLASSES

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
    genome_size = Column(Float, default=0.0)
    valid = Column(Float, default=0.0)
    displacement_xy = Column(Float, default=0.0)
    num_voxels = Column(Float, default=0.0)
    bone_count = Column(Float, default=0.0)
    bone_prop  = Column(Float, default=0.0)
    fat_count = Column(Float, default=0.0)
    fat_prop  = Column(Float, default=0.0)
    muscle_count = Column(Float, default=0.0)
    muscle_prop  = Column(Float, default=0.0)
    muscle_offp_count = Column(Float, default=0.0)
    muscle_offp_prop  = Column(Float, default=0.0)


class GenerationSurvivor(Base):
    __tablename__ = "generation_survivors"
    generation = Column(Integer, nullable=False)
    robot_id = Column(Integer, ForeignKey("all_robots.robot_id"), nullable=False)

    fitness = Column(Float, default=0.0)
    uniqueness = Column(Float, default=0.0)

    __table_args__ = (
        PrimaryKeyConstraint("generation", "robot_id", name="pk_generation_robot"),
    )

# EA CLASSES


class Individual:
    def __init__(self, genome, id_counter):
        self.id = id_counter
        self.genome = genome
        self.genome_size = 0.0
        self.phenotype = None
        self.valid = 0  # False: assumes it is invalid until proved otherwise

        # behavior
        self.displacement_xy = float('-inf')  # invalid is not evaluated and receives worst value

        # absolute morphology
        self.num_voxels = 0.0
        self.bone_count = 0.0
        self.bone_prop = 0.0
        self.fat_count = 0.0
        self.fat_prop = 0.0
        self.muscle_count = 0.0
        self.muscle_prop = 0.0
        self.muscle_offp_count = 0.0
        self.muscle_offp_prop = 0.0

        # relative
        self.fitness = 0.0
        self.uniqueness = 0.0