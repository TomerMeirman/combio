"""No-download example using simulated positive laboratory measurements."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from combio import COMBIO


rng = np.random.default_rng(42)
X = pd.DataFrame(rng.lognormal(mean=1.0, sigma=0.5, size=(700, 6)),
                 columns=["lab_a", "lab_b", "lab_c", "lab_d", "lab_e", "lab_f"])
signal = np.log(X.lab_a) - 1.5 * np.log(X.lab_b) + np.log(X.lab_c)
y = (signal + rng.normal(0, 0.7, len(X)) > np.median(signal)).astype(int)
X_source, X_test, y_source, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)

model = COMBIO().fit(X_source, y_source)
score = model.score(X_test)
observed = np.isfinite(score)
print("Selected labs:", model.top_features_)
print("SHAP directions:", model.shap_directions_)
print("Formula:", model.formula_)
print("Held-out AUROC:", round(roc_auc_score(y_test[observed], score[observed]), 3))
