"""The SHAP-guided COMBIO discovery procedure.

Pass source training data to ``fit``. Reserve any independent test cohort
outside this class; its labels are never an input to formula discovery.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def _frame(X, names=None):
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        out.columns = [str(name) for name in out.columns]
    else:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric table")
        columns = names or [f"lab_{j}" for j in range(values.shape[1])]
        out = pd.DataFrame(values, columns=columns)
    if out.columns.has_duplicates:
        raise ValueError("Laboratory column names must be unique")
    if names is not None:
        missing = set(names) - set(out.columns)
        if missing:
            raise ValueError(f"Missing laboratory columns: {sorted(missing)}")
        out = out.loc[:, names]
    try:
        out = out.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every laboratory column must be numeric") from exc
    if np.isinf(out.to_numpy()).any():
        raise ValueError("X contains infinity; use missing values for unavailable labs")
    return out


def _log_labs(values):
    return np.log(np.clip(np.asarray(values, dtype=float), 1e-3, None))


def _direction(values, attributions):
    """Conservative global trend in one feature's validation SHAP contribution."""
    mask = np.isfinite(values) & np.isfinite(attributions)
    x, phi = np.asarray(values[mask], float), np.asarray(attributions[mask], float)
    if len(x) < 20 or len(np.unique(x)) < 5:
        return "ambiguous"
    rho = float(spearmanr(x, phi).statistic)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    contrast = float(phi[x >= q3].mean() - phi[x <= q1].mean())
    bins = np.minimum(np.floor(rankdata(x, method="average") / (len(x) + 1) * 5).astype(int), 4)
    means = np.array([phi[bins == j].mean() if np.any(bins == j) else np.nan
                      for j in range(5)])
    changes = np.diff(means)
    if not np.all(np.isfinite(means)) or not np.isfinite(rho):
        return "ambiguous"
    reverses = np.any(changes > 1e-12) and np.any(changes < -1e-12)
    if not reverses and np.sign(rho) == np.sign(contrast) and rho != 0:
        return "positive" if rho > 0 else "negative"
    return "ambiguous"


def _signs_allowed(final_exponents, directions):
    return all(power == 0 or direction == "ambiguous" or
               np.sign(power) == (1 if direction == "positive" else -1)
               for power, direction in zip(final_exponents, directions))


class COMBIO:
    """Select a compact multiplicative laboratory formula with Tree SHAP.

    A gradient-boosted reference model is fitted on inner training data. Mean
    absolute Tree SHAP on inner validation ranks five labs; clear value–SHAP
    trends constrain the signs of the final risk-oriented exponents. Integer
    exponents in ``{-2,-1,0,1,2}`` with two to four nonzero terms are searched
    by validation AUROC. Orientation is fixed from inner training labels.

    The caller must reserve a test set before ``fit``. ``score`` returns raw
    ``log C``, with higher scores oriented toward higher outcome risk. It
    returns NaN where any of the five selected labs is unavailable. It does
    not convert the score into a calibrated probability.

    ``groups`` keeps repeated observations from the same person together in
    the inner split. A temporal or other external test partition remains the
    caller's responsibility.
    """

    def __init__(self, k_top=5, max_terms=4, validation_size=0.30, random_state=42):
        self.k_top = k_top
        self.max_terms = max_terms
        self.validation_size = validation_size
        self.random_state = random_state

    def fit(self, X, y, *, groups=None):
        frame = _frame(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(frame) or not np.isin(labels, [0, 1]).all():
            raise ValueError("y must be a binary 0/1 vector matching X")
        labels = labels.astype(int)
        if len(np.unique(labels)) != 2:
            raise ValueError("Both outcome classes are required")
        if not (2 <= self.k_top <= frame.shape[1]):
            raise ValueError("k_top must be between 2 and the number of labs")
        if not (2 <= self.max_terms <= self.k_top):
            raise ValueError("max_terms must be between 2 and k_top")
        if not (0 < self.validation_size < 1):
            raise ValueError("validation_size must lie between 0 and 1")

        idx = np.arange(len(frame))
        if groups is None:
            train, validation = train_test_split(
                idx, test_size=self.validation_size, stratify=labels,
                random_state=self.random_state)
        else:
            groups = np.asarray(groups)
            if groups.ndim != 1 or len(groups) != len(frame) or pd.isna(groups).any():
                raise ValueError("groups must have one nonmissing value per row")
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.validation_size,
                                         random_state=self.random_state)
            train, validation = next(splitter.split(idx, labels, groups))
            if set(groups[train]) & set(groups[validation]):
                raise AssertionError("Patient groups overlap between training and validation")
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[validation])) != 2:
            raise ValueError("Training and validation must each contain both outcomes")

        values = frame.to_numpy()
        reference = HistGradientBoostingClassifier(
            random_state=self.random_state, max_iter=250, learning_rate=0.05,
            max_leaf_nodes=31, l2_regularization=1.0).fit(values[train], labels[train])
        explainer = shap.TreeExplainer(reference, model_output="raw",
                                      feature_perturbation="tree_path_dependent")
        phi = np.asarray(explainer.shap_values(values[validation]), float)
        if phi.shape != (len(validation), frame.shape[1]):
            raise RuntimeError("Unexpected Tree SHAP output shape")
        raw = reference.decision_function(values[validation])
        additive_error = np.max(np.abs(float(np.asarray(explainer.expected_value).item()) +
                                       phi.sum(axis=1) - raw))
        if additive_error > 1e-5:
            raise RuntimeError("Tree SHAP does not reconstruct reference log odds")

        names = list(frame.columns)
        mean_abs = np.mean(np.abs(phi), axis=0)
        order = sorted(range(len(names)), key=lambda j: (-mean_abs[j], names[j]))[:self.k_top]
        top = [names[j] for j in order]
        directions = [_direction(values[validation, j], phi[:, j]) for j in order]
        logs = _log_labs(values[:, order])
        complete = np.isfinite(logs).all(axis=1)
        orientation_rows = (train if len(train) <= 25_000 else
                            np.random.default_rng(self.random_state).choice(
                                train, 25_000, replace=False))
        oriented_train = orientation_rows[complete[orientation_rows]]
        complete_validation = validation[complete[validation]]
        if (len(np.unique(labels[oriented_train])) != 2 or
                len(np.unique(labels[complete_validation])) != 2):
            raise ValueError("Selected labs need complete cases with both classes in both inner subsets")

        best = None
        for candidate in product((-2, -1, 0, 1, 2), repeat=self.k_top):
            terms = sum(power != 0 for power in candidate)
            if not 2 <= terms <= self.max_terms:
                continue
            raw_train_score = logs[oriented_train] @ np.asarray(candidate)
            orientation = 1 if roc_auc_score(labels[oriented_train], raw_train_score) >= 0.5 else -1
            final = tuple(orientation * power for power in candidate)
            if not _signs_allowed(final, directions):
                continue
            auc = roc_auc_score(labels[complete_validation],
                                orientation * (logs[complete_validation] @ np.asarray(candidate)))
            key = (round(float(auc), 4), -terms)
            if best is None or key > best[0]:
                best = key, final, float(auc)
        if best is None:
            raise RuntimeError("No formula satisfies the SHAP direction constraints")

        self.feature_names_ = names
        self.top_features_ = top
        self.shap_importance_ = {names[j]: float(mean_abs[j]) for j in order}
        self.shap_directions_ = dict(zip(top, directions))
        self.all_exponents_ = dict(zip(top, best[1]))
        self.exponents_ = {name: power for name, power in self.all_exponents_.items() if power}
        self.validation_auroc_ = best[2]
        self.shap_additivity_error_ = float(additive_error)
        self.training_indices_ = train.copy()
        self.validation_indices_ = validation.copy()
        return self

    def score(self, X):
        """Return raw, risk-oriented log-composite values, or NaN if inputs are missing."""
        if not hasattr(self, "all_exponents_"):
            raise RuntimeError("Fit COMBIO before scoring")
        frame = _frame(X, self.top_features_)
        logs = _log_labs(frame.to_numpy())
        return logs @ np.asarray(list(self.all_exponents_.values()), float)

    @property
    def formula_(self):
        if not hasattr(self, "exponents_"):
            raise RuntimeError("Fit COMBIO before reading the formula")
        numerator, denominator = [], []
        for name, power in self.exponents_.items():
            term = name if abs(power) == 1 else f"{name}^{abs(power)}"
            (numerator if power > 0 else denominator).append(term)
        top = " × ".join(numerator) if numerator else "1"
        return top if not denominator else f"{top} / ({' × '.join(denominator)})"
