import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from combio import COMBIO
from combio.model import _direction


def _cohort():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.lognormal(size=(420, 6)), columns=list("abcdef"))
    risk = np.log(X.a) - np.log(X.b) + 0.5 * np.log(X.c)
    y = (risk + rng.normal(0, 0.7, len(X)) > np.median(risk)).astype(int)
    return X, y


def test_direction_rule_does_not_force_nonmonotone_trends():
    values = np.arange(100, dtype=float)
    assert _direction(values, values) == "positive"
    assert _direction(values, -values) == "negative"
    assert _direction(values, (values - 50) ** 2) == "ambiguous"


def test_fit_reserves_inner_validation_and_scores_external_rows():
    X, y = _cohort()
    source, test, y_source, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    model = COMBIO().fit(source, y_source)
    assert set(model.training_indices_).isdisjoint(model.validation_indices_)
    assert len(model.top_features_) == 5
    assert 2 <= len(model.exponents_) <= 4
    assert all(model.shap_directions_[name] == "ambiguous" or
               np.sign(power) == (1 if model.shap_directions_[name] == "positive" else -1)
               for name, power in model.exponents_.items())
    score = model.score(test)
    assert np.isfinite(score).all()
    assert roc_auc_score(y_test, score) > 0.65
    unavailable = test.copy()
    unavailable.iloc[0, unavailable.columns.get_loc(model.top_features_[0])] = np.nan
    assert np.isnan(model.score(unavailable)[0])


def test_patient_groups_do_not_cross_inner_split():
    X, y = _cohort()
    groups = np.repeat(np.arange(len(X) // 2), 2)
    model = COMBIO().fit(X, y, groups=groups)
    assert set(groups[model.training_indices_]).isdisjoint(groups[model.validation_indices_])
