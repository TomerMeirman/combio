# COMBIO

COMBIO discovers a small multiplicative formula from numerical laboratory
measurements. It fits a gradient-boosted reference model, uses Tree SHAP on an
inner validation set to rank five labs and identify clear directional trends,
then selects integer exponents by validation AUROC. The selected formula is a
single score: `log C = Σ exponent × log(lab)`.

## Install and try it

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/quickstart.py
```

## Use with your own data

Reserve a test cohort **before** fitting. `fit` splits only the supplied
source training data into inner training and validation subsets. Its reference
model uses inner training; SHAP ranks labs and the formula search uses inner
validation. Orientation is determined in inner training. The test cohort is
used only after the formula is fixed.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from combio import COMBIO

# X is a numeric pandas DataFrame; y contains 0/1 outcome labels.
X_source, X_test, y_source, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
model = COMBIO().fit(X_source, y_source)
print(model.formula_)
print(model.top_features_)
print(model.shap_directions_)

test_score = model.score(X_test)
observed = np.isfinite(test_score)
print(roc_auc_score(y_test[observed], test_score[observed]))
```

For repeated observations from one person, make the outer split by person and
pass source patient identifiers as `groups` to `fit`. For temporal or external
evaluation, reserve the later cohort outside `fit`. When comparing COMBIO with
other methods, use the same reserved test partition and report the eligible
complete-case subset for each comparison.

`score` returns risk-oriented **log C**, not a probability. It returns NaN when
any of the five selected labs is missing, including a lab assigned exponent
zero. Inputs are clipped below at 0.001 before logarithms, so preserve the
same laboratory units when scoring a new cohort. The `shap_directions_` values
describe trends in the fitted reference model, not clinical or causal effects.
If a trend is ambiguous, both exponent signs remain eligible.

## Worked notebook

[NHANES iron-overload example](examples/nhanes_iron.ipynb) constructs a public
cohort, reserves its test partition, runs the same SHAP-guided selection, and
scores the frozen formula. The notebook obtains seven public CDC XPT files only
when its explicit data-download cell is run. No dataset or participant record
is stored in this repository.

## Method details

The reference is a histogram gradient-boosted classifier (250 iterations,
learning rate 0.05, at most 31 leaves, L2 regularization 1.0). It is fitted
on the inner training subset. Positive-class raw-log-odds Tree SHAP values are
computed on inner validation. Mean absolute attribution ranks the five labs.
A sign constraint is used only when the Spearman value–SHAP trend agrees with
the upper-versus-lower-quartile mean attribution contrast and five ranked bins
show no trend reversal. Otherwise both signs are searched. The integer grid
is `{-2,-1,0,1,2}` with two to four nonzero terms. Each candidate is oriented
using inner training labels; validation AUROC chooses the formula, with ties
at four decimal places resolved toward fewer terms. These steps never receive
test labels.

The caller chooses the laboratory panel and excludes outcome-defining
measurements before fitting. The notebook shows that choice for one task.
This package implements formula discovery and raw scoring; calibration,
clinical use, and external validation require separate study-specific work.

## License

MIT. See [LICENSE](LICENSE).
