"""Turning polymer SMILES into numbers a model can use.

Three feature sets are compared in the notebook:
1. DESCRIPTORS          : 5 hand-picked, physically motivated RDKit descriptors
                          (the original baseline choice).
2. All RDKit descriptors: ~200 descriptors; more information, more redundancy.
3. Morgan fingerprints  : bit vector saying which small circular substructures
                          are present. Describes *chemistry* rather than bulk
                          properties, and is also what we use to measure how
                          similar two polymers are.

Polymer SMILES contain `*` for the two connection points of the repeat unit.
RDKit reads `*` as a "dummy" atom with zero mass, so it is ignored by
descriptors like molecular weight but still visible to fingerprints as
"this atom is a chain end".
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# The five baseline descriptors. Each one has a physical reason to relate to Tg:
# - rotatable bonds  : flexible backbones/side chains lower Tg
# - aromatic rings   : stiff rings raise Tg
# - fraction sp3 C   : more sp3 carbon usually means a more flexible chain
# - repeat-unit MW   : bulkier repeat units tend to restrict motion
# - TPSA             : polar surface area, a proxy for polar interactions/H-bonding
DESCRIPTORS = {
    "rotatable_bonds": Descriptors.NumRotatableBonds,
    "aromatic_rings": Descriptors.NumAromaticRings,
    "fraction_csp3": Descriptors.FractionCSP3,
    "repeat_unit_MW": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
}

# Every descriptor RDKit offers (name -> function).
ALL_DESCRIPTORS = dict(Descriptors.descList)

# Morgan fingerprint settings:
# radius 2  -> substructures up to 2 bonds from each atom (the common "ECFP4")
# 2048 bits -> few hash collisions for ~500 small repeat units
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def featurize(smiles, descriptors=DESCRIPTORS):
    """Compute RDKit descriptors for a Series of SMILES (baseline function)."""
    mols = smiles.apply(Chem.MolFromSmiles)
    return pd.DataFrame({name: mols.apply(func) for name, func in descriptors.items()})


def clean_descriptors(X_train, X_other):
    """Make a large descriptor table safe for modelling.

    Decisions are made using the *training* set only, then applied to the other
    set, so no information from the test polymers leaks into the features:
    - drop descriptors that are NaN/inf for any training polymer
      (some RDKit descriptors fail on the `*` dummy atoms);
    - drop descriptors that are constant in training (they carry no signal);
    - for the other set, fill any remaining NaN/inf with the training median.
    """
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    X_train = X_train.loc[:, X_train.std() > 0]

    X_other = X_other[X_train.columns].replace([np.inf, -np.inf], np.nan)
    X_other = X_other.fillna(X_train.median())

    # A few descriptors (e.g. Ipc) can reach ~1e30. Random forests do not care,
    # but such values wreck a linear model. A signed log keeps the ordering
    # and compresses the range; we apply it only to columns that need it.
    huge = X_train.columns[X_train.abs().max() > 1e6]
    for col in huge:
        X_train[col] = np.sign(X_train[col]) * np.log1p(X_train[col].abs())
        X_other[col] = np.sign(X_other[col]) * np.log1p(X_other[col].abs())
    return X_train, X_other


def _morgan_generator(radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_fingerprint_objects(smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """RDKit fingerprint objects. Used for fast Tanimoto similarity searches."""
    gen = _morgan_generator(radius, n_bits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def morgan_features(smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """Morgan fingerprints as a 0/1 DataFrame (one column per bit) for models."""
    gen = _morgan_generator(radius, n_bits)
    bits = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
                    dtype=np.uint8)
    columns = [f"morgan_{i}" for i in range(n_bits)]
    return pd.DataFrame(bits, columns=columns, index=smiles.index)
