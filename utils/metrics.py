import numpy as np
import sys
from pathlib import Path
from math import inf

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.voxel_types import VOXEL_TYPES

METRICS_ABS = [
    # genotypic
    "genome_size",

    # behavioral
    "displacement",

    # phenotypic
    "num_voxels",
    "bone_count",
    "bone_prop",
    "fat_count",
    "fat_prop",
    "muscle_count",
    "muscle_prop",
    "muscle_offp_count",
    "muscle_offp_prop",
]

METRICS_REL = [
                "uniqueness",
                "fitness",
                "local_novelty",
               ]


def relative_metrics(population, args):
    uniqueness(population)
    local_novelty(population)
    set_fitness(population, args.fitness_metric)


def genopheno_abs_metrics(individual):

    # genome
    genome_size(individual)

    # phenotype
    num_voxels(individual)
    update_material_metrics(individual)
    test_validity(individual)


def behavior_abs_metrics(population):
    # displacement_xy is calculated by voxcraft itself and collected in simulation_resources.py
    # as center-of-mass displacement in meters: x^2 + y^2

    # TODO: implement others and treat for -inf
    pass


def genome_size(individual):
    individual.genome_size = len(individual.genome)


def num_voxels(individual):  # size / mass proxy
    individual.num_voxels = int((individual.phenotype != 0).sum())


def update_material_metrics(individual):
    grid = np.asarray(individual.phenotype, dtype=int)

    filled_total = int((grid != 0).sum())
    individual.filled_total = filled_total

    for name, mid in VOXEL_TYPES.items():
        count = int((grid == mid).sum())
        prop = (count / filled_total) if filled_total > 0 else 0.0

        setattr(individual, f"{name}_count", count)
        setattr(individual, f"{name}_prop", round(prop,2))


def set_fitness(population, fitness_metric):
    for ind in population:
        ind.fitness = float(getattr(ind, fitness_metric, None))


def test_validity(individual):
    has_muscle = individual.muscle_count >= 1
    has_offp   = individual.muscle_offp_count >= 1
    individual.valid = has_muscle and has_offp


def tree_edit_distance(g1, g2):
    a = np.asarray(g1)
    b = np.asarray(g2)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    one_zero = (a == 0) ^ (b == 0)  # 0 vs non-zero → 1.0 (different shape)
    both_nonzero_diff = (a != 0) & (b != 0) & (a != b)  # non-zero vs different non-zero → 0.5 (different material)

    return float(one_zero.sum() + 0.5 * both_nonzero_diff.sum())


def uniqueness(population):
    # average distance to all pop using edit tree distance
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = tree_edit_distance(ind.phenotype, other.phenotype)
                distances.append(d)
        ind.uniqueness = np.mean(distances)


def local_novelty(population):
    k = 5
    # average distance to k nearest neighbors using edit tree distance
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = tree_edit_distance(ind.phenotype, other.phenotype)
                distances.append(d / max(ind.num_voxels, other.num_voxels))
        nearest_distances = sorted(distances)[:k]
        ind.local_novelty = np.mean(nearest_distances)


