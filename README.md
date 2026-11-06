# COMBIO

**COMBIO: Automated Discovery of a Compact Composite Biomarker from Routine
Laboratory Panels**

COMBIO discovers a multiplicative biomarker with two to four laboratory
measurements: `C = product_i x_i^a_i`. It fits a gradient-boosted reference model,
uses Tree SHAP on inner validation to rank five candidate measurements and assess
their directional trends, then selects small integer exponents by validation
AUROC. The implementation returns the equivalent risk-oriented log-score,
`log C = sum_i a_i log(x_i)`.

## Install and try it

Python 3.12 or newer is required.

```bash
git clone https://github.com/TomerMeirman/combio.git
cd combio
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

The default settings implement the manuscript's discovery procedure:

1. **Reserve test data before discovery.** The manuscript reserves 30% of the
   source cohort for testing. Pass only the remaining source data to `fit`, which
   splits it 70:30 into inner training and validation. For repeated admissions,
   keep patients together in the outer split and supply `groups` for the inner
   split. External cohorts remain outside discovery.
2. **Fit the reference and rank measurements.** The reference is a
   `HistGradientBoostingClassifier` fitted on inner training, with 250 iterations,
   learning rate 0.05, at most 31 leaves and L2 regularization 1.0. Tree SHAP
   explains positive-class raw log odds on inner validation. Mean absolute SHAP
   ranks five candidate measurements; the code checks attribution additivity.
3. **Determine eligible exponent signs.** A positive or negative direction is
   assigned when Spearman correlation between a measurement and its SHAP values
   agrees in sign with the upper-minus-lower-quartile mean SHAP contrast, and
   five ranked groups show no trend reversal. Ambiguous or insufficient evidence
   leaves both signs eligible. Constraints apply to the final risk-oriented
   exponents.
4. **Select the formula on validation.** Search the integer grid
   `{-2,-1,0,1,2}` with two to four nonzero terms. Inner-training AUROC fixes each
   candidate's risk orientation; for large training subsets, the implementation
   uses a reproducible sample of at most 25,000 training rows for this step.
   The highest validation AUROC rounded to four decimal places wins, followed
   by fewer nonzero terms. Remaining exact ties retain the first candidate in
   the grid's iteration order.
5. **Score the reserved data after selection.** The formula and orientation are
   fixed before `score` is applied to test or external data. `score` returns
   `log C`; probability-based analyses require a separately fitted risk mapping.

The caller chooses the laboratory panel and excludes outcome-defining
measurements before fitting. The notebook shows that choice for one task.
This package implements formula discovery and raw scoring; calibration,
clinical use, and external validation require separate study-specific work.

The manuscript's full XGBoost model is a separate comparator. The discovery
reference used by this package is the histogram gradient-boosted model above.

## Funding

This research was supported by the Israel Science Foundation (grant No. 2672/24).

## License

MIT. See [LICENSE](LICENSE).
