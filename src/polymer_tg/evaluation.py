"""Models, metrics and the three test settings.

Test settings
-------------
1. "Random CV"   : 5-fold cross-validation with random folds.
2. "Cluster CV"  : 5-fold cross-validation where whole chemical clusters are
                   held out (see splits.py).
3. "External"    : train on all training polymers, predict dataset3.

Why cross-validation instead of one 80/20 split: with ~511 polymers, a single
test set has only ~100 polymers and the score depends a lot on which polymers
happen to land in it. Averaging over 5 folds (and reporting the spread) gives
a more stable estimate and an honest error bar.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import SEED


def make_models(seed=SEED):
    """The two models compared throughout the project.

    - Ridge regression: a linear model with a penalty on large coefficients.
      Plain linear regression breaks down when there are more features than
      polymers (2048 fingerprint bits vs ~400 training polymers), Ridge does
      not. Features are standardised first so the penalty treats them equally,
      and the penalty strength (alpha) is chosen by internal cross-validation
      on the training fold only.
    - Random forest: the original baseline's non-linear model (300 trees,
      scikit-learn defaults otherwise). Also gives an uncertainty estimate
      for free from the spread of its trees (see uncertainty.py).
    """
    return {
        "Ridge regression": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
    }


def regression_metrics(y_true, y_pred):
    """MAE, RMSE and R2 in one dictionary.

    - MAE  : average absolute error in degC; easy to interpret.
    - RMSE : like MAE but punishes large errors more; RMSE >> MAE means a few
             polymers are predicted very badly.
    - R2   : fraction of the Tg variance explained. It depends on how spread
             out Tg is in the test set, so compare it with care across sets.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def bootstrap_metric_std(y_true, y_pred, n_boot=1000, seed=SEED):
    """Standard deviation of each metric over bootstrap resamples of a single
    test set. Gives the External set (only 46 polymers) an error bar that is
    comparable to the fold-to-fold spread reported for cross-validation.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    boot = [regression_metrics(y_true[i], y_pred[i])
            for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return pd.DataFrame(boot).std().to_dict()


def cross_val_predict_splits(model, X, y, splits):
    """Fit a fresh copy of `model` on each training fold and predict the test
    fold. Returns out-of-fold predictions (one per polymer) and the fold id.
    """
    X, y = np.asarray(X), np.asarray(y)
    pred = np.full(len(y), np.nan)
    fold = np.full(len(y), -1)
    for k, (tr, te) in enumerate(splits):
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        fold[te] = k
    return pred, fold


def evaluate_all(feature_sets, y_train, y_ext, cv_settings, models=None):
    """Run every (feature set x model x test setting) combination.

    Parameters
    ----------
    feature_sets : dict name -> (X_train, X_ext)
    cv_settings  : dict setting name -> list of (train_idx, test_idx)
    Returns
    -------
    summary     : one row per combination with mean and std of each metric
    predictions : one row per predicted polymer (for parity plots)
    """
    models = models or make_models()
    y_train, y_ext = np.asarray(y_train), np.asarray(y_ext)
    summary, predictions = [], []

    for feat_name, (X_tr, X_ext) in feature_sets.items():
        for model_name, model in models.items():
            # --- cross-validation settings -------------------------------
            for setting, splits in cv_settings.items():
                pred, fold = cross_val_predict_splits(model, X_tr, y_train, splits)
                per_fold = pd.DataFrame([regression_metrics(y_train[fold == k], pred[fold == k])
                                         for k in range(len(splits))])
                summary.append(_summary_row(feat_name, model_name, setting,
                                            per_fold.mean(), per_fold.std()))
                predictions.append(pd.DataFrame({
                    "features": feat_name, "model": model_name, "setting": setting,
                    "y_true": y_train, "y_pred": pred}))

            # --- external set: train on everything, predict dataset3 -----
            m = clone(model).fit(np.asarray(X_tr), y_train)
            pred = m.predict(np.asarray(X_ext))
            summary.append(_summary_row(feat_name, model_name, "External",
                                        pd.Series(regression_metrics(y_ext, pred)),
                                        pd.Series(bootstrap_metric_std(y_ext, pred))))
            predictions.append(pd.DataFrame({
                "features": feat_name, "model": model_name, "setting": "External",
                "y_true": y_ext, "y_pred": pred}))

    return pd.DataFrame(summary), pd.concat(predictions, ignore_index=True)


def _summary_row(feat_name, model_name, setting, mean, std):
    row = {"features": feat_name, "model": model_name, "setting": setting}
    for metric in ("MAE", "RMSE", "R2"):
        row[metric] = mean[metric]
        row[f"{metric}_std"] = std[metric]
    return row


def format_summary(summary):
    """Pretty table: 'mean ± std' strings, one row per feature set x model,
    one column group per test setting."""
    out = summary.copy()
    for metric, digits in (("MAE", 1), ("RMSE", 1), ("R2", 2)):
        out[metric] = [f"{m:.{digits}f} ± {s:.{digits}f}"
                       for m, s in zip(summary[metric], summary[f"{metric}_std"])]
    order = ["Random CV", "Cluster CV", "External"]
    out["setting"] = pd.Categorical(out["setting"], categories=order, ordered=True)
    table = out.pivot_table(index=["features", "model"], columns="setting",
                            values=["MAE", "RMSE", "R2"], aggfunc="first", observed=True)
    table = table.swaplevel(axis=1).reindex(columns=order, level=0)
    return table.reindex(columns=["MAE", "RMSE", "R2"], level=1)
