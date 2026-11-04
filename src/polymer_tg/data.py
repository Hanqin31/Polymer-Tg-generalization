"""Loading and cleaning the Tg data.

Two files are used:
- data/train.csv                         : Kaggle training data; only rows with a
                                           measured Tg are kept (~511 rows).
- data/train_supplement/dataset3.csv     : 46 extra polymers with Tg. Kept aside as
                                           an external test set (an original design choice).
"""

import pandas as pd
from rdkit import Chem


def canonicalize(smiles):
    """Return RDKit's canonical SMILES so that the same polymer written in two
    different ways becomes the same string (needed to find duplicates).

    Returns None if RDKit cannot parse the SMILES, so bad rows can be dropped
    explicitly instead of crashing later.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_tg_data(data_dir="../data"):
    """Load the training rows that have a Tg value and the external set.

    Returns two DataFrames with columns: SMILES, Tg, canonical_smiles
    (the training frame also keeps the Kaggle `id`).
    """
    train = pd.read_csv(f"{data_dir}/train.csv")
    ext = pd.read_csv(f"{data_dir}/train_supplement/dataset3.csv")

    # train.csv has five target columns; most rows only have one of them.
    # We only study Tg, so keep rows where Tg was measured.
    train_tg = train[train["Tg"].notna()][["id", "SMILES", "Tg"]].reset_index(drop=True)
    ext = ext[ext["Tg"].notna()][["SMILES", "Tg"]].reset_index(drop=True)

    for df in (train_tg, ext):
        df["canonical_smiles"] = df["SMILES"].apply(canonicalize)
    return train_tg, ext


def deduplicate(df, name="data"):
    """Remove invalid SMILES and merge duplicate polymers.

    Why: if the same polymer appears twice, one copy can land in the training
    fold and the other in the test fold. The model then "predicts" a polymer
    it has already seen, which inflates test scores.

    Duplicates (same canonical SMILES) are merged into one row whose Tg is the
    mean of the reported values. The mean is a reasonable summary when the
    copies come from slightly different measurements.
    """
    n_start = len(df)
    df = df[df["canonical_smiles"].notna()]
    n_invalid = n_start - len(df)

    # groupby + mean merges duplicates; `first` keeps one representative
    # original SMILES / id for reference.
    agg = {col: "first" for col in df.columns if col not in ("canonical_smiles", "Tg")}
    agg["Tg"] = "mean"
    df = df.groupby("canonical_smiles", as_index=False, sort=False).agg(agg)

    print(f"{name}: {n_start} rows -> {len(df)} rows "
          f"({n_invalid} invalid SMILES, {n_start - n_invalid - len(df)} duplicates merged)")
    return df.reset_index(drop=True)


def remove_overlap(train_df, ext_df):
    """Drop external polymers that also appear in the training set.

    Why: the external set is supposed to measure performance on polymers the
    model has never seen. An overlapping polymer would be a free answer.
    """
    in_train = ext_df["canonical_smiles"].isin(train_df["canonical_smiles"])
    print(f"External polymers also in training (removed from external): {in_train.sum()}")
    return ext_df[~in_train].reset_index(drop=True)
