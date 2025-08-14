import numpy as np


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


def relative_metrics(population, fitness_metric):
    uniqueness(population)
    set_fitness(population, fitness_metric)


def set_fitness(population, fitness_metric):
    for ind in population:
        ind.fitness = float(getattr(ind, fitness_metric, 0.0))


def phenotype_abs_metrics(population):
    pass


def behavior_abs_metrics(population):
    pass




