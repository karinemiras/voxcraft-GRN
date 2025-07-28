import numpy as np

def tree_edit_distance(g1, g2):
    # TODO: increment
    g1_flat = np.array(g1).flatten()
    g2_flat = np.array(g2).flatten()
    return np.sum(g1_flat != g2_flat)  # Hamming distance as placeholder


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
        ind.fitness = np.mean(nearest_distances)