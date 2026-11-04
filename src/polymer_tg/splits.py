"""How we split data into training and test parts.

A *random* split tests "can the model predict polymers similar to the ones it
was trained on?" Because many polymers in the data are close analogues of each
other (e.g. the same backbone with a different side chain), a random split is
optimistic.

A *cluster* split groups chemically similar polymers together (Butina
clustering on Morgan fingerprints) and keeps each group entirely in either
the training or the test part. The model is then tested on chemistry it has
not seen, which is closer to the real use case of screening new polymers.
"""

import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupKFold, KFold

from . import SEED

# Butina distance cutoff = 1 - Tanimoto similarity. 0.7 means polymers are
# put in the same cluster as a centroid if their similarity is >= 0.3.
# Chosen so that clusters are chemically meaningful but no single cluster
# dominates the data (largest cluster ~15% of polymers).
BUTINA_CUTOFF = 0.7


def max_similarity_to_reference(query_fps, reference_fps):
    """For each query polymer, the Tanimoto similarity to its nearest
    neighbour in the reference set (1 = identical fingerprint, 0 = nothing in
    common). Used as a simple "how out-of-distribution is this polymer" score.
    """
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps))
                     for fp in query_fps])


def test_fold_similarities(fps, splits):
    """For a cross-validation scheme, the nearest-neighbour similarity of every
    polymer to the training part of the fold in which it was a *test* polymer.

    This tells us how "new" the test polymers looked to the model, so we can
    check that the cluster split really is harder than the random split.
    """
    sims = np.full(len(fps), np.nan)
    for train_idx, test_idx in splits:
        train_fps = [fps[i] for i in train_idx]
        sims[test_idx] = max_similarity_to_reference([fps[i] for i in test_idx], train_fps)
    return sims


def butina_clusters(fps, cutoff=BUTINA_CUTOFF):
    """Cluster fingerprints with the Butina algorithm.

    Butina picks the polymer with the most neighbours (within `cutoff`) as a
    cluster centre, assigns those neighbours to it, removes them, and repeats.
    It is deterministic (no random seed needed) and widely used in
    cheminformatics for exactly this purpose.

    Returns an integer cluster label for every polymer.
    """
    n = len(fps)
    # Butina wants the lower triangle of the distance matrix as a flat list.
    distances = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1 - s for s in sims)

    clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
    labels = np.empty(n, dtype=int)
    for cluster_id, members in enumerate(clusters):
        labels[list(members)] = cluster_id
    return labels


def random_cv_splits(n_samples, n_splits=5, seed=SEED):
    """Ordinary shuffled K-fold: list of (train_idx, test_idx) pairs."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.arange(n_samples)))


def cluster_cv_splits(groups, n_splits=5, seed=SEED):
    """Grouped K-fold: every cluster ends up in exactly one test fold, and a
    cluster is never split between training and test.

    GroupKFold with shuffle=True randomises which clusters go to which fold
    (with a fixed seed), while still balancing fold sizes.
    """
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(groups))
    return list(cv.split(idx, groups=groups))
