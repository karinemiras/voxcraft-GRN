# import numpy as np
# import sys
# from pathlib import Path
# import math
# from sklearn.neighbors import KDTree
#
# ROOT = Path(__file__).resolve().parent.parent
# sys.path.append(str(ROOT))
# from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_NOBONE
#
# METRICS_ABS = [
#     # genotypic
#     "genome_size",
#
#     # behavioral
#     "displacement",
#
#     # phenotypic
#     "num_voxels",
#     "bone_count",
#     "bone_prop",
#     "fat_count",
#     "fat_prop",
#     "fat2_count",
#     "fat2_prop",
#     "phase_muscle_count",
#     "phase_muscle_prop",
#     "offphase_muscle_count",
#     "offphase_muscle_prop",
# ]
#
# METRICS_REL = [
#                 "uniqueness",
#                 "fitness",
#                 "age",
#                 # "dominated_disp_age",
#                 "dominated_disp_nov",
#                 "novelty",
#                 "novelty_weighted"
#                ]
#
# # metrics relative to other individuals or factors like time
# def relative_metrics(population, args, generation, novelty_archive=None):
#     uniqueness(population)
#     novelty(population, novelty_archive)
#     novelty_weighted(population)
#     age(population, generation)
#     # pareto_dominance_count( population,
#     #                         objectives=(("age", "min"), ("displacement", "max")), out_attr="dominated_disp_age")
#     pareto_dominance_count(population,
#                            objectives=(("novelty", "max"), ("displacement", "max")), out_attr="dominated_disp_nov")
#     set_fitness(population, args.fitness_metric)
#
#
# def genopheno_abs_metrics(individual, args):
#
#     # genome
#     genome_size(individual)
#
#     # phenotype
#     num_voxels(individual)
#     update_material_metrics(individual, args)
#     test_validity(individual)
#
#
# def behavior_abs_metrics(population):
#     # displacement_xy is calculated by voxcraft itself and collected in simulation_resources.py
#     # as center-of-mass displacement in meters: x^2 + y^2
#
#     # TODO: implement others and treat for -inf
#     pass
#
# def update_material_metrics(individual, args):
#     if args.voxel_types == 'withbone':
#         voxel_types = VOXEL_TYPES
#     if args.voxel_types == 'nobone':
#         voxel_types = VOXEL_TYPES_NOBONE
#
#     grid = np.asarray(individual.phenotype, dtype=int)
#     filled_total = int((grid != 0).sum())
#     individual.filled_total = filled_total
#
#     for name, mid in voxel_types.items():
#
#         count = int((grid == mid).sum())
#         prop = (count / filled_total) if filled_total > 0 else 0.0
#
#         setattr(individual, f"{name}_count", count)
#         setattr(individual, f"{name}_prop", round(prop,2))
#
# def set_fitness(population, fitness_metric):
#     for ind in population:
#         ind.fitness = float(getattr(ind, fitness_metric, None))
#
# def test_validity(individual):
#     has_phase_muscle = individual.phase_muscle_count >= 1
#     has_offphase_muscle   = individual.offphase_muscle_count >= 1
#     individual.valid = has_phase_muscle and has_offphase_muscle
#
# def age(population, generation):
#     for ind in population:
#         age = generation - ind.born_generation + 1
#         ind.age = age
#
# def genome_size(individual):
#     individual.genome_size = len(individual.genome)
#
#
# def num_voxels(individual):  # size / mass proxy
#     individual.num_voxels = int((individual.phenotype != 0).sum())
#
# def distance(g1, g2):
#     a = np.asarray(g1)
#     b = np.asarray(g2)
#
#     if a.shape != b.shape:
#         raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
#
#     one_zero = (a == 0) ^ (b == 0)  # 0 vs non-zero → 1.0 (different shape)
#     both_nonzero_diff = (a != 0) & (b != 0) & (a != b)  # non-zero vs different non-zero → 0.5 (different material)
#
#     return float(one_zero.sum() + 0.5 * both_nonzero_diff.sum())
#
#
# def uniqueness(population):
#     # average morphological distance to all current pop using Hamming distance
#     for i, ind in enumerate(population):
#         distances = []
#         for j, other in enumerate(population):
#             if i != j:
#                 d = distance(ind.phenotype, other.phenotype)
#                 distances.append(d / max(ind.num_voxels, other.num_voxels))
#         ind.uniqueness = np.mean(distances)
#
# # def novelty_weighted(population):
# #     for ind in population:
# #         novelty_weighted = ind.displacement * ind.novelty
# #         ind.novelty_weighted = novelty_weighted
#
# def novelty_weighted(population):
#     beta = 0.8
#     for ind in population:
#         novelty_weighted = ind.displacement * ind.novelty + beta * ind.displacement
#         ind.novelty_weighted = novelty_weighted
#
# def novelty(population, novelty_archive, k=5, M=50, embed_fn=None):
#     pool = list(population) + list(novelty_archive or [])
#
#     if embed_fn is None:
#         # minimal embedding: 1D vector
#         embed_fn = lambda ind: np.array([ind.num_voxels], dtype=np.float32)
#
#     X = np.vstack([embed_fn(ind) for ind in pool]).astype(np.float32)
#     tree = KDTree(X)
#
#     for ind in population:
#         qi = embed_fn(ind).reshape(1, -1)
#         _, idxs = tree.query(qi, k=min(M + 1, len(pool)))
#         idxs = idxs[0]
#
#         dists = []
#         for j in idxs:
#             other = pool[j]
#             if other is ind:
#                 continue
#             d = distance(ind.phenotype, other.phenotype)
#             dists.append(d / max(ind.num_voxels, other.num_voxels))
#
#         kk = min(k, len(dists))
#         ind.novelty = float(np.partition(np.asarray(dists, dtype=np.float32), kk - 1)[:kk].mean()) if kk else 0.0
#
#
# def pareto_dominance_count(
#     population,
#     objectives=(("age", "min"), ("displacement", "max")),
#     out_attr="dominates_count",
# ):
#     """
#     For each individual, count how many others it Pareto-dominates
#     Dominance rule:
#       A dominates B iff
#         - A is no worse than B in all objectives, AND
#         - A is strictly better in at least one objective.
#     """
#     # Normalize directions and validate
#     obj_specs = []
#     for attr, direction in objectives:
#         d = direction.strip().lower()
#         obj_specs.append((attr, d))
#
#     def dominates(a, b) -> bool:
#         no_worse_all = True
#         strictly_better_any = False
#
#         for attr, d in obj_specs:
#             av = getattr(a, attr)
#             bv = getattr(b, attr)
#
#             if d == "min":
#                 if av > bv:
#                     no_worse_all = False
#                     break
#                 if av < bv:
#                     strictly_better_any = True
#             else:  # "max"
#                 if av < bv:
#                     no_worse_all = False
#                     break
#                 if av > bv:
#                     strictly_better_any = True
#
#         return no_worse_all and strictly_better_any
#
#     # Init output
#     for ind in population:
#         setattr(ind, out_attr, 0)
#
#     # O(n^2) dominance counting
#     n = len(population)
#     for i in range(n):
#         a = population[i]
#         cnt = 0
#         for j in range(n):
#             if i == j:
#                 continue
#             if dominates(a, population[j]):
#                 cnt += 1
#         setattr(a, out_attr, cnt)
#



import numpy as np
import sys
from pathlib import Path
import math
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_NOBONE


# -----------------------------------------------------------------------------
# Absolute metrics: per-individual, independent of other individuals/time
# -----------------------------------------------------------------------------
METRICS_ABS = [
    # genotypic
    "genome_size",

    # behavioral
    "displacement",

    # phenotypic (counts/props)
    "num_voxels",
    "bone_count",
    "bone_prop",
    "fat_count",
    "fat_prop",
    "fat2_count",
    "fat2_prop",
    "phase_muscle_count",
    "phase_muscle_prop",
    "offphase_muscle_count",
    "offphase_muscle_prop",

    # NEW: geometry-aware morpho metrics (layout matters, not only proportions)
    # Contact / ground interaction proxies
    "contact_area",
    "contact_prop",
    "contact_muscle_prop",
    "contact_passive_prop",
    "contact_bone_prop",   # meaningful only in withbone

    # Vertical stratification / layering
    "mean_z_bone",
    "mean_z_fat",
    "mean_z_fat2",
    "mean_z_phase_muscle",
    "mean_z_offphase_muscle",
    "mean_z_passive",      # combined passive mean height
    "mean_z_muscle",       # combined muscle mean height
    "layering_muscle_minus_passive",

    # Shape / compactness proxies
    "surface_voxels",
    "surface_ratio",
    "extent_x",
    "extent_y",
    "extent_z",
    "bbox_fill",
    "elongation_xy",
    "flatness",

    # Symmetry / directional bias proxies (useful for directed locomotion)
    "mirror_x_diff",
    "mirror_y_diff",

    # Muscle clustering / organization
    "muscle_components",
    "largest_muscle_component_prop",
]


# -----------------------------------------------------------------------------
# Relative metrics: depend on population, archive, or time
# -----------------------------------------------------------------------------
METRICS_REL = [
    "uniqueness",
    "fitness",
    "age",
    "dominated_disp_nov",
    "novelty",
    "novelty_weighted",
]


# metrics relative to other individuals or factors like time
def relative_metrics(population, args, generation, novelty_archive=None):
    uniqueness(population)
    novelty(population, novelty_archive)
    novelty_weighted(population)
    age(population, generation)

    pareto_dominance_count(
        population,
        objectives=(("novelty", "max"), ("displacement", "max")),
        out_attr="dominated_disp_nov"
    )
    set_fitness(population, args.fitness_metric)


def genopheno_abs_metrics(individual, args):
    # genome
    genome_size(individual)

    # phenotype
    num_voxels(individual)
    update_material_metrics(individual, args)
    test_validity(individual)

    # NEW: geometry-aware metrics that capture *arrangement*, not just proportions
    update_morpho_geometry_metrics(individual, args)


def behavior_abs_metrics(population):
    # displacement is calculated by voxcraft itself and collected elsewhere
    pass


# -----------------------------------------------------------------------------
# Material composition metrics
# -----------------------------------------------------------------------------
def update_material_metrics(individual, args):
    if args.voxel_types == 'withbone':
        voxel_types = VOXEL_TYPES
    if args.voxel_types == 'nobone':
        voxel_types = VOXEL_TYPES_NOBONE

    grid = np.asarray(individual.phenotype, dtype=int)
    filled_total = int((grid != 0).sum())
    individual.filled_total = filled_total

    for name, mid in voxel_types.items():
        count = int((grid == mid).sum())
        prop = (count / filled_total) if filled_total > 0 else 0.0

        setattr(individual, f"{name}_count", count)
        setattr(individual, f"{name}_prop", round(prop, 2))


def set_fitness(population, fitness_metric):
    for ind in population:
        ind.fitness = float(getattr(ind, fitness_metric, None))


def test_validity(individual):
    has_phase_muscle = individual.phase_muscle_count >= 1
    has_offphase_muscle = individual.offphase_muscle_count >= 1
    individual.valid = has_phase_muscle and has_offphase_muscle


def age(population, generation):
    for ind in population:
        ind.age = generation - ind.born_generation + 1


def genome_size(individual):
    individual.genome_size = len(individual.genome)


def num_voxels(individual):
    individual.num_voxels = int((individual.phenotype != 0).sum())


# -----------------------------------------------------------------------------
# NEW: Geometry-aware morpho metrics
# -----------------------------------------------------------------------------
def update_morpho_geometry_metrics(individual, args):
    """
    Computes morphology metrics that are sensitive to *spatial arrangement*.
    These are especially useful for your 2x2 experiments:
      - withbone vs nobone
      - high friction vs low friction

    Why these help:
      - Proportions alone can't distinguish "bone core" vs "bone feet" vs "bone stripes".
      - Friction changes often shift "contact strategy" (what touches the ground, how much).
      - Removing bone changes whether evolution relies on rigid scaffolds vs distributed compliance.
    """
    if args.voxel_types == 'withbone':
        voxel_types = VOXEL_TYPES
    else:
        voxel_types = VOXEL_TYPES_NOBONE

    grid = np.asarray(individual.phenotype, dtype=int)

    # Identify material IDs (robust to different maps)
    bone_id = voxel_types.get("bone", None)
    fat_id = voxel_types.get("fat", None)
    fat2_id = voxel_types.get("fat2", None)
    phase_id = voxel_types.get("phase_muscle", None)
    off_id = voxel_types.get("offphase_muscle", None)

    muscle_ids = [mid for mid in [phase_id, off_id] if mid is not None]
    passive_ids = [mid for mid in [bone_id, fat_id, fat2_id] if mid is not None]

    # 1) Ground contact metrics
    contact_metrics(individual, grid, muscle_ids=muscle_ids, passive_ids=passive_ids, bone_id=bone_id)

    # 2) Vertical stratification (layering)
    layering_metrics(individual, grid, bone_id=bone_id, fat_id=fat_id, fat2_id=fat2_id,
                     phase_id=phase_id, off_id=off_id, passive_ids=passive_ids, muscle_ids=muscle_ids)

    # 3) Shape / compactness proxies
    bbox_metrics(individual, grid)
    surface_ratio(individual, grid)

    # 4) Symmetry / asymmetry proxies (useful with directed locomotion fitness = x)
    mirror_asymmetry(individual, grid)

    # 5) Muscle clustering / organization (banding vs speckled)
    muscle_clustering_metrics(individual, grid, muscle_ids=muscle_ids)


# -------------------------
# (1) Contact / ground layer
# -------------------------
def contact_metrics(ind, grid, muscle_ids, passive_ids, bone_id=None, eps=1e-12):
    """
    CONTACT METRICS (proxy for traction strategy):
      - contact_area: how many voxels touch ground (larger = "sole" / slider)
      - contact_prop: contact_area / num_voxels
      - contact_muscle_prop: fraction of contacting voxels that are muscle
      - contact_passive_prop: fraction passive on ground
      - contact_bone_prop: fraction bone on ground (only meaningful withbone)

    Interpretation:
      - High friction often allows smaller contact patches (feet/struts).
      - Low friction often favors broader contact or specialized contacting material.
      - With bone: bone at contact tends to imply "rigid foot" / lever strategies.
      - No bone: contact tends to be soft/passive with muscle patterns behind it.
    """
    # Assumes Z axis is the third axis and ground is at z=0
    bottom = grid[:, :, 0]
    contact_mask = (bottom != 0)

    contact_area = int(contact_mask.sum())
    ind.contact_area = contact_area
    ind.contact_prop = float(contact_area / (ind.num_voxels + eps))

    if contact_area == 0:
        ind.contact_muscle_prop = 0.0
        ind.contact_passive_prop = 0.0
        ind.contact_bone_prop = 0.0
        return

    bottom_vals = bottom[contact_mask]

    is_muscle = np.isin(bottom_vals, muscle_ids) if muscle_ids else np.zeros_like(bottom_vals, dtype=bool)
    is_passive = np.isin(bottom_vals, passive_ids) if passive_ids else np.zeros_like(bottom_vals, dtype=bool)

    ind.contact_muscle_prop = float(is_muscle.mean())
    ind.contact_passive_prop = float(is_passive.mean())

    if bone_id is not None:
        ind.contact_bone_prop = float((bottom_vals == bone_id).mean())
    else:
        ind.contact_bone_prop = 0.0


# -------------------------
# (2) Layering / stratification
# -------------------------
def _mean_z_for_mask(coords, mask):
    # coords: Nx3 indices of filled voxels; mask: boolean mask over those coords
    if mask.sum() == 0:
        return 0.0
    return float(coords[mask, 2].mean())


def layering_metrics(ind, grid, bone_id, fat_id, fat2_id, phase_id, off_id, passive_ids, muscle_ids):
    """
    LAYERING METRICS (who sits low/high):
      - mean_z_* per material: average height of that material
      - mean_z_passive / mean_z_muscle: combined grouping
      - layering_muscle_minus_passive: >0 means muscles are higher than passive on average

    Interpretation:
      - With bone: bone often becomes a "core" (higher mean_z) or "feet" (low mean_z).
      - Low friction often drives solutions where contact layer changes (so mean_z for passive/muscle shifts).
      - Useful for detecting "core + soft skirt" vs "muscles on the ground" strategies.
    """
    filled_coords = np.argwhere(grid != 0)  # Nx3
    if len(filled_coords) == 0:
        # set defaults
        ind.mean_z_bone = 0.0
        ind.mean_z_fat = 0.0
        ind.mean_z_fat2 = 0.0
        ind.mean_z_phase_muscle = 0.0
        ind.mean_z_offphase_muscle = 0.0
        ind.mean_z_passive = 0.0
        ind.mean_z_muscle = 0.0
        ind.layering_muscle_minus_passive = 0.0
        return

    vals = grid[filled_coords[:, 0], filled_coords[:, 1], filled_coords[:, 2]]

    def mean_z(mid):
        if mid is None:
            return 0.0
        return _mean_z_for_mask(filled_coords, vals == mid)

    ind.mean_z_bone = mean_z(bone_id)
    ind.mean_z_fat = mean_z(fat_id)
    ind.mean_z_fat2 = mean_z(fat2_id)
    ind.mean_z_phase_muscle = mean_z(phase_id)
    ind.mean_z_offphase_muscle = mean_z(off_id)

    passive_mask = np.isin(vals, passive_ids) if passive_ids else np.zeros_like(vals, dtype=bool)
    muscle_mask = np.isin(vals, muscle_ids) if muscle_ids else np.zeros_like(vals, dtype=bool)

    ind.mean_z_passive = _mean_z_for_mask(filled_coords, passive_mask)
    ind.mean_z_muscle = _mean_z_for_mask(filled_coords, muscle_mask)

    ind.layering_muscle_minus_passive = float(ind.mean_z_muscle - ind.mean_z_passive)


# -------------------------
# (3) Bounding box + fill + elongation
# -------------------------
def bbox_metrics(ind, grid, eps=1e-12):
    """
    SHAPE / COMPACTNESS METRICS:
      - extent_x/y/z: bounding box size of the filled morphology
      - bbox_fill: num_voxels / bbox_volume  (1.0 = fully filled block; low = sparse/branched)
      - elongation_xy: how stretched in x vs y (directionality / keel / sled shapes)
      - flatness: extent_z / max(extent_x, extent_y) (flat pancake vs tall body)

    Interpretation:
      - With bone + high friction may create protrusions/legs (lower bbox_fill).
      - No bone often yields compact bodies (higher bbox_fill).
      - Low friction may produce elongated "sled-like" shapes (higher elongation_xy).
    """
    coords = np.argwhere(grid != 0)
    if len(coords) == 0:
        ind.extent_x = ind.extent_y = ind.extent_z = 0
        ind.bbox_fill = 0.0
        ind.elongation_xy = 0.0
        ind.flatness = 0.0
        return

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ext = (maxs - mins + 1).astype(int)

    ind.extent_x, ind.extent_y, ind.extent_z = map(int, ext)

    bbox_vol = float(ext[0] * ext[1] * ext[2])
    ind.bbox_fill = float(ind.num_voxels / (bbox_vol + eps))

    ind.elongation_xy = float(max(ext[0], ext[1]) / (min(ext[0], ext[1]) + eps))
    ind.flatness = float(ext[2] / (max(ext[0], ext[1]) + eps))


# -------------------------
# (3b) Surface ratio (boundary voxels)
# -------------------------
def surface_ratio(ind, grid, eps=1e-12):
    """
    SURFACE METRIC:
      - surface_voxels: number of voxels that have at least one empty 6-neighbor
      - surface_ratio: surface_voxels / num_voxels

    Interpretation:
      - High surface_ratio often means "branched / leggy / spiky" morphologies.
      - Low surface_ratio means more compact, blob-like shapes.
      - With bone present, you often see higher surface_ratio (structural appendages).
      - No bone often shifts toward compact forms (lower surface_ratio).
    """
    filled = (grid != 0)
    if ind.num_voxels == 0:
        ind.surface_voxels = 0
        ind.surface_ratio = 0.0
        return

    # pad with empty space so neighbor checks don't need bounds checks
    p = np.pad(filled, 1, mode="constant", constant_values=False)

    center = p[1:-1, 1:-1, 1:-1]
    neigh_empty = (
        ~p[2:, 1:-1, 1:-1] | ~p[:-2, 1:-1, 1:-1] |
        ~p[1:-1, 2:, 1:-1] | ~p[1:-1, :-2, 1:-1] |
        ~p[1:-1, 1:-1, 2:] | ~p[1:-1, 1:-1, :-2]
    )

    surf = center & neigh_empty
    surface_voxels = int(surf.sum())
    ind.surface_voxels = surface_voxels
    ind.surface_ratio = float(surface_voxels / (ind.num_voxels + eps))


# -------------------------
# (4) Symmetry / asymmetry
# -------------------------
def mirror_asymmetry(ind, grid):
    """
    MIRROR ASYMMETRY METRICS (occupancy-based):
      - mirror_x_diff: difference between occupancy and x-mirrored occupancy
      - mirror_y_diff: same for y

    Interpretation:
      - Higher values = more left-right / front-back asymmetry.
      - With directed locomotion fitness = x, successful gaits often break symmetry.
      - Low friction can encourage keels / sleds / biased shapes -> higher asymmetry.
    """
    occ = (grid != 0).astype(np.uint8)

    mx = occ[::-1, :, :]  # mirror along x
    my = occ[:, ::-1, :]  # mirror along y

    ind.mirror_x_diff = float((occ != mx).mean())
    ind.mirror_y_diff = float((occ != my).mean())


# -------------------------
# (5) Muscle clustering / banding
# -------------------------
def muscle_clustering_metrics(ind, grid, muscle_ids):
    """
    MUSCLE CLUSTERING METRICS:
      - muscle_components: number of connected components among muscle voxels (6-neighborhood)
      - largest_muscle_component_prop: largest_component_size / total_muscle_voxels

    Interpretation:
      - Few components + large largest_component_prop: organized bands/blocks (often wave-like gaits).
      - Many components: speckled muscle distribution (often noisy exploration).
      - Expect differences across friction regimes:
          * low friction often favors organized banding / wave structures.
    """
    if not muscle_ids:
        ind.muscle_components = 0
        ind.largest_muscle_component_prop = 0.0
        return

    mask = np.isin(grid, muscle_ids)
    total = int(mask.sum())
    if total == 0:
        ind.muscle_components = 0
        ind.largest_muscle_component_prop = 0.0
        return

    comps, largest = count_components_6n(mask)
    ind.muscle_components = comps
    ind.largest_muscle_component_prop = float(largest / total)


def count_components_6n(mask3d):
    """
    Counts 6-neighborhood connected components in a boolean 3D mask.
    Returns: (num_components, largest_component_size)

    Kept simple and dependency-free (no scipy).
    """
    visited = np.zeros(mask3d.shape, dtype=bool)
    coords = np.argwhere(mask3d)
    if len(coords) == 0:
        return 0, 0

    # To speed membership checks a bit, we still use mask3d directly.
    X, Y, Z = mask3d.shape
    neigh = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    num = 0
    largest = 0

    for (x, y, z) in coords:
        if visited[x, y, z]:
            continue
        # new component
        num += 1
        stack = [(x, y, z)]
        visited[x, y, z] = True
        size = 0

        while stack:
            cx, cy, cz = stack.pop()
            size += 1
            for dx, dy, dz in neigh:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                    if mask3d[nx, ny, nz] and not visited[nx, ny, nz]:
                        visited[nx, ny, nz] = True
                        stack.append((nx, ny, nz))

        if size > largest:
            largest = size

    return num, largest


# -----------------------------------------------------------------------------
# Your existing distance/novelty machinery (unchanged)
# -----------------------------------------------------------------------------
def distance(g1, g2):
    a = np.asarray(g1)
    b = np.asarray(g2)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    one_zero = (a == 0) ^ (b == 0)  # 0 vs non-zero -> 1.0 (different shape)
    both_nonzero_diff = (a != 0) & (b != 0) & (a != b)  # non-zero vs different non-zero -> 0.5 (different material)

    return float(one_zero.sum() + 0.5 * both_nonzero_diff.sum())


def uniqueness(population):
    # average morphological distance to all current pop using custom Hamming-ish distance
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = distance(ind.phenotype, other.phenotype)
                distances.append(d / max(ind.num_voxels, other.num_voxels))
        ind.uniqueness = np.mean(distances)


def novelty_weighted(population):
    # Simple, interpretable fitness: d*n + beta*d = d*(n+beta)
    # - preserves "novelty matters among movers"
    # - increases average performance via the beta*d term
    beta = 0.8
    for ind in population:
        ind.novelty_weighted = ind.displacement * ind.novelty + beta * ind.displacement


def novelty(population, novelty_archive, k=5, M=50, embed_fn=None):
    pool = list(population) + list(novelty_archive or [])

    if embed_fn is None:
        # minimal embedding: 1D vector
        embed_fn = lambda ind: np.array([ind.num_voxels], dtype=np.float32)

    X = np.vstack([embed_fn(ind) for ind in pool]).astype(np.float32)
    tree = KDTree(X)

    for ind in population:
        qi = embed_fn(ind).reshape(1, -1)
        _, idxs = tree.query(qi, k=min(M + 1, len(pool)))
        idxs = idxs[0]

        dists = []
        for j in idxs:
            other = pool[j]
            if other is ind:
                continue
            d = distance(ind.phenotype, other.phenotype)
            dists.append(d / max(ind.num_voxels, other.num_voxels))

        kk = min(k, len(dists))
        ind.novelty = float(np.partition(np.asarray(dists, dtype=np.float32), kk - 1)[:kk].mean()) if kk else 0.0


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

    for ind in population:
        setattr(ind, out_attr, 0)

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