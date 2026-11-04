"""Reusable helpers for the polymer Tg generalization project.

The notebook in `notebooks/` imports from these modules so that the notebook
itself stays focused on the analysis story, while the code that does the work
lives in one place and can be read (and tested) on its own.

Modules
-------
data         : loading, SMILES canonicalization, de-duplication
features     : RDKit descriptors and Morgan fingerprints
splits       : fingerprint similarity, Butina clustering, CV splits
evaluation   : models, metrics, cross-validation over several test settings
uncertainty  : random-forest tree spread + conformal calibration
plotting     : small helpers for saving figures consistently
"""

# One seed for the whole project. Every function that involves randomness
# takes a `seed` argument whose default is this value.
SEED = 42
