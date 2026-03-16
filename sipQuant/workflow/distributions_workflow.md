# sipQuant — Distributions Workflow
SIP Global (Systematic Index Partners)

## Overview

The `distributions` module provides distribution fitting and evaluation for
commodity price data, returns series, and risk factor modelling. All routines
are pure NumPy; no scipy or external dependencies are required.

## Supported Distributions

| Distribution | Use Case                        |
|--------------|---------------------------------|
| Normal       | Return residuals, log returns   |
| Log-normal   | Spot price levels               |
| Student-t    | Heavy-tailed returns            |
| Gamma        | Positive skew, storage costs    |
| Copula (Gaussian / t) | Joint tail dependence  |

## Typical Workflow

### 1. Prepare Data

```python
import sipQuant as sq
import numpy as np

returns = np.diff(np.log(prices))       # log-returns
returns = returns[np.isfinite(returns)] # drop NaN / Inf
```

### 2. Fit a Distribution

```python
# Fit normal
fitNorm = sq.distributions.fitNormal(returns)
# fitNorm: {'mu': ..., 'sigma': ..., 'logLik': ..., 'aic': ..., 'bic': ...}

# Fit Student-t (MLE, heavier tails)
fitT = sq.distributions.fitStudentT(returns)
# fitT: {'mu': ..., 'sigma': ..., 'nu': ..., 'logLik': ..., 'aic': ..., 'bic': ...}
```

### 3. Compare Models via Information Criteria

```python
models = {'normal': fitNorm, 'studentT': fitT}
bestModel = min(models, key=lambda k: models[k]['aic'])
print(f"Best fit by AIC: {bestModel}")
```

### 4. Evaluate Tail Risk

```python
var95 = sq.distributions.quantile(fitT, alpha=0.05)
cvar95 = sq.distributions.conditionalTailExpectation(fitT, alpha=0.05)
```

### 5. Copula Fitting (Multi-Asset)

```python
# Joint tail dependence between two commodity return series
copulaResult = sq.distributions.gaussianCopula(returns_A, returns_B)
# copulaResult: {'rho': ..., 'logLik': ..., 'tailDependence': ...}
```

## Key Design Principles

- **MLE estimation** is used throughout; moment matching is available as a fallback.
- **AIC / BIC** are computed automatically for model selection.
- All fitted objects are plain `dict`s; no custom classes.
- Functions are camelCase and return consistent dict schemas.

## Notes

- For fewer than 30 observations, Student-t MLE may not converge; use normal.
- Copula fitting assumes marginals have been transformed to uniform via the
  empirical CDF before passing to the copula function.
- Log-normal fit expects strictly positive input values.
