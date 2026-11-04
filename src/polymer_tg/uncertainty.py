"""Uncertainty estimates for random-forest Tg predictions.

Step 1 - raw uncertainty (sigma)
    A random forest is an average of many decision trees, each trained on a
    different bootstrap sample of the data. Where the trees disagree, the
    prediction is less certain. We use the standard deviation of the 300 tree
    predictions as sigma.

Step 2 - calibration (conformal prediction)
    Raw sigma has the right *shape* (bigger for harder polymers, hopefully)
    but not the right *size*: there is no guarantee that y_pred +/- 1.64*sigma
    contains 90% of true values. Normalised conformal prediction fixes the size:
      a) on data the model has NOT seen (calibration set) compute the score
             s = |y_true - y_pred| / sigma
         i.e. "how many sigmas off was the prediction";
      b) take q = the 90% quantile of those scores;
      c) the 90% interval for a new polymer is y_pred +/- q * sigma.
    If calibration polymers and test polymers are "exchangeable" (drawn from
    the same distribution), this interval covers ~90% of test polymers.
    The research question is what happens when they are NOT - e.g. when the
    test polymers come from unseen chemical clusters.

    With only ~400 training polymers per fold we cannot afford a separate
    calibration hold-out, so calibration scores come from an *inner*
    5-fold cross-validation on the training fold (each training polymer is
    predicted once by a forest that did not see it). The inner folds can be
    random or cluster-based; comparing the two is part of the analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone

from . import SEED
from .splits import cluster_cv_splits, random_cv_splits

# Avoid dividing by zero when all trees agree exactly (rare).
MIN_SIGMA = 1e-3


def rf_mean_and_std(forest, X):
    """Prediction (mean over trees) and sigma (std over trees)."""
    X = np.asarray(X)
    tree_preds = np.stack([tree.predict(X) for tree in forest.estimators_])
    return tree_preds.mean(axis=0), np.maximum(tree_preds.std(axis=0), MIN_SIGMA)


def calibration_scores(forest, X, y, inner_splits):
    """Nonconformity scores |y - y_pred| / sigma from inner cross-validation."""
    X, y = np.asarray(X), np.asarray(y)
    scores = np.empty(len(y))
    for tr, te in inner_splits:
        f = clone(forest).fit(X[tr], y[tr])
        mu, sigma = rf_mean_and_std(f, X[te])
        scores[te] = np.abs(y[te] - mu) / sigma
    return scores


def conformal_quantile(scores, level):
    """Quantile of the calibration scores with the standard finite-sample
    correction: use the ceil((n+1)*level)/n empirical quantile, which makes the
    coverage guarantee hold exactly for exchangeable data.
    """
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * level) / n)
    return np.quantile(scores, q_level, method="higher")


def run_uncertainty_experiment(forest, X_train, y_train, outer_splits,
                               X_test=None, y_test=None, groups=None,
                               n_inner=5, seed=SEED):
    """Fit, predict and collect calibration scores for one test setting.

    Two ways of using the function:
    - cross-validation setting: pass `outer_splits` (list of (train, test)).
    - external setting: pass outer_splits=None plus X_test / y_test; the forest
      is trained on all of X_train and evaluated on X_test.

    `groups` (cluster labels for X_train) enables cluster-based inner
    calibration in addition to random inner calibration.

    Returns
    -------
    preds  : DataFrame, one row per test polymer: y_true, y_pred, sigma, fold
    scores : dict {(fold, "random"|"cluster"): calibration score array}
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    if outer_splits is None:
        # External setting: a single "fold" = all training data vs X_test.
        folds = [(np.arange(len(y_train)), None)]
    else:
        folds = outer_splits

    rows, scores = [], {}
    for k, (tr, te) in enumerate(folds):
        X_te = np.asarray(X_test) if te is None else X_train[te]
        y_te = np.asarray(y_test) if te is None else y_train[te]

        # Inner splits are made on the outer-training polymers only, so the
        # test polymers never influence the calibration.
        inner = {"random": random_cv_splits(len(tr), n_inner, seed)}
        if groups is not None:
            inner["cluster"] = cluster_cv_splits(np.asarray(groups)[tr], n_inner, seed)
        for name, splits in inner.items():
            scores[(k, name)] = calibration_scores(forest, X_train[tr], y_train[tr], splits)

        f = clone(forest).fit(X_train[tr], y_train[tr])
        mu, sigma = rf_mean_and_std(f, X_te)
        rows.append(pd.DataFrame({"y_true": y_te, "y_pred": mu, "sigma": sigma, "fold": k,
                                  "test_index": np.arange(len(y_te)) if te is None else te}))
    return pd.concat(rows, ignore_index=True), scores


def interval_halfwidth(preds, scores, level, method):
    """Half-width of the `level` prediction interval for every test polymer.

    method = "gaussian" : sigma * z  (treat tree spread as a normal std; no calibration)
             "random" / "cluster" : sigma * conformal quantile from that inner CV
    """
    if method == "gaussian":
        return preds["sigma"].to_numpy() * norm.ppf(0.5 + level / 2)
    q = preds["fold"].map(lambda k: conformal_quantile(scores[(k, method)], level))
    return preds["sigma"].to_numpy() * q.to_numpy()


def coverage_curve(preds, scores, method, levels=np.linspace(0.1, 0.95, 18)):
    """Observed coverage (fraction of true Tg inside the interval) for each
    nominal level. A perfectly calibrated model lies on the diagonal."""
    abs_err = np.abs(preds["y_true"] - preds["y_pred"]).to_numpy()
    return pd.DataFrame({
        "nominal": levels,
        "observed": [np.mean(abs_err <= interval_halfwidth(preds, scores, lv, method))
                     for lv in levels],
    })


def no_skill_interval(y_train, y_test, level=0.9):
    """Reference point for interval *width*: ignore the polymer entirely and
    predict "training mean +/- q", where q is the conformal quantile of
    |y - mean| in training. A useful model must give narrower intervals than
    this at the same coverage.

    Returns (coverage, interval width in degC).
    """
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    q = conformal_quantile(np.abs(y_train - y_train.mean()), level)
    coverage = np.mean(np.abs(y_test - y_train.mean()) <= q)
    return coverage, 2 * q


def coverage_table(results, level=0.9, methods=("gaussian", "random", "cluster")):
    """Coverage and average interval width at one nominal level.

    results : dict setting -> (preds, scores) from run_uncertainty_experiment
    """
    rows = []
    for setting, (preds, scores) in results.items():
        abs_err = np.abs(preds["y_true"] - preds["y_pred"]).to_numpy()
        for method in methods:
            hw = interval_halfwidth(preds, scores, level, method)
            rows.append({"setting": setting, "interval method": method,
                         f"coverage @ {level:.0%}": np.mean(abs_err <= hw),
                         "mean interval width (degC)": np.mean(2 * hw)})
    return pd.DataFrame(rows)
