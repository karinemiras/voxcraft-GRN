import numpy as np
import sys
from pathlib import Path
import math

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
    "phase_muscle_count",
    "phase_muscle_prop",
    "offphase_muscle_count",
    "offphase_muscle_prop",
]

METRICS_REL = [
                "uniqueness",
                "fitness",
                "age",
                "dominated_disp_age",
                "novelty"
               ]

# metrics relative to other individuals or factors like time
def relative_metrics(population, args, generation):
    uniqueness(population)
    novelty(population)
    age(population, generation)
    pareto_dominance_count( population,
                            objectives=(("age", "min"), ("displacement", "max")), out_attr="dominated_disp_age")
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

def age(population, generation):
    for ind in population:
        age = generation - ind.born_generation + 1
        ind.age = age

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
    has_phase_muscle = individual.phase_muscle_count >= 1
    has_offphase_muscle   = individual.offphase_muscle_count >= 1
    individual.valid = has_phase_muscle and has_offphase_muscle


def tree_edit_distance(g1, g2):
    a = np.asarray(g1)
    b = np.asarray(g2)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    one_zero = (a == 0) ^ (b == 0)  # 0 vs non-zero → 1.0 (different shape)
    both_nonzero_diff = (a != 0) & (b != 0) & (a != b)  # non-zero vs different non-zero → 0.5 (different material)

    return float(one_zero.sum() + 0.5 * both_nonzero_diff.sum())


def uniqueness(population):
    # average morphological distance to all current pop using edit tree distance
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = tree_edit_distance(ind.phenotype, other.phenotype)
                distances.append(d)
        ind.uniqueness = np.mean(distances)


def novelty(population):
    # TODO: replace by regular novelty ie with archive
    k = 5
    # average morphological distance to k nearest neighbors using edit tree distance
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = tree_edit_distance(ind.phenotype, other.phenotype)
                distances.append(d / max(ind.num_voxels, other.num_voxels))
        nearest_distances = sorted(distances)[:k]
        ind.novelty = np.mean(nearest_distances)

def pareto_dominance_count(
    population,
    objectives=(("age", "min"), ("displacement", "max")),
    out_attr="dominates_count",
):
    """
    For each individual, count how many others it Pareto-dominates
    Dominance rule:
      A dominates B iff
        - A is no worse than B in all objectives, AND
        - A is strictly better in at least one objective.
    """
    # Normalize directions and validate
    obj_specs = []
    for attr, direction in objectives:
        d = direction.strip().lower()
        obj_specs.append((attr, d))

    def dominates(a, b) -> bool:
        no_worse_all = True
        strictly_better_any = False

        for attr, d in obj_specs:
            av = getattr(a, attr)
            bv = getattr(b, attr)

            if d == "min":
                if av > bv:
                    no_worse_all = False
                    break
                if av < bv:
                    strictly_better_any = True
            else:  # "max"
                if av < bv:
                    no_worse_all = False
                    break
                if av > bv:
                    strictly_better_any = True

        return no_worse_all and strictly_better_any

    # Init output
    for ind in population:
        setattr(ind, out_attr, 0)

    # O(n^2) dominance counting
    n = len(population)
    for i in range(n):
        a = population[i]
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            if dominates(a, population[j]):
                cnt += 1
        setattr(a, out_attr, cnt)

