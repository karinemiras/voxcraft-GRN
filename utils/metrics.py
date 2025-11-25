import numpy as np
import sys
from pathlib import Path
from math import inf

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.voxel_types import VOXEL_TYPES


def relative_metrics(population, fitness_metric):
    uniqueness(population)
    set_fitness(population, fitness_metric)

def phenotype_abs_metrics(individual):
    num_voxels(individual)
    update_material_metrics(individual)
    test_validity(individual)

def behavior_abs_metrics(population):
    pass

def num_voxels(individual):               # size / mass proxy
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
        if ind.valid:
            ind.fitness = float(getattr(ind, fitness_metric, 0.0))
        else:
            ind.fitness = float('-inf')
            
def test_validity(individual):
    if individual.muscle_count < 1 or individual.muscle_offp_count < 1:
        individual.valid = False

def tree_edit_distance(g1, g2):
    a = np.asarray(g1)
    b = np.asarray(g2)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    one_zero = (a == 0) ^ (b == 0)  # 0 vs non-zero → 1.0 (different shape)
    both_nonzero_diff = (a != 0) & (b != 0) & (a != b)  # non-zero vs different non-zero → 0.5 (different material)

    return float(one_zero.sum() + 0.5 * both_nonzero_diff.sum())

def uniqueness(population, k=10):
    """
    Assigns fitness to each individual based on local 'novelty' = uniqueness,
    using average distance to k nearest neighbors.
    """
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = tree_edit_distance(ind.phenotype, other.phenotype)
                distances.append(d)
        nearest_distances = sorted(distances)[:k]
        ind.uniqueness = np.mean(nearest_distances)



