# btQuant

btQuant is the Python companion package for Systematic Index Partners (SIP Global) quantitative infrastructure. It provides tools for quantitative finance, econometrics, simulation, portfolio optimization, and risk analytics. This package is intended for professional developers and researchers.

[![PyPI](https://img.shields.io/badge/PyPI-btQuant-blue)](https://pypi.org/project/btQuant/) [![Downloads](https://pepy.tech/badge/btQuant)](https://pepy.tech/project/btQuant)

## About SIP

Systematic Index Partners (SIP Global) is a quantitative investment and market infrastructure firm focused on alternative and underdeveloped markets. We design and deploy systematic trading strategies, indices, and market infrastructure to enable market making, price discovery, and brokerage in low-liquidity or previously unstructured markets.  

Our quantitative systems support structured pricing, standardized contracts, and repeatable trading processes. These systems are deployed alongside physical and digital infrastructure, including trading hubs, delivery points, data pipelines, execution tools, and operational platforms to ensure markets function efficiently.  

SIP participates actively in the markets we develop, providing liquidity, facilitating transactions, and supporting both physical and financial trading activity. By combining quantitative strategy, infrastructure, and market participation, SIP enables liquidity, price transparency, and sustainable market development.

## Installation

Install btQuant via PyPI:

```bash
pip install btQuant
````

## Design Philosophy

* Pure NumPy implementation for all mathematics
* Optimized vectorized operations for performance
* Consistent API with camelCase function names
* Lightweight outputs using arrays and dictionaries
* Production-ready code suitable for professional environments

## Modules

### options.py

Option pricing and Greeks calculation

* `blackScholes()` Black-Scholes European options with cost of carry
* `binomial()` Binomial tree (European/American)
* `trinomial()` Trinomial tree (European/American)
* `asian()` Asian options (geometric averaging)
* `binary()` Binary/digital options
* `spread()` Two-asset spread options
* `barrier()` Barrier options with rebate
* `simulate()` Monte Carlo simulation wrapper
* `impliedVol()` Implied volatility calculation
* `buildForwardCurve()` Forward curve construction
* `bootstrapCurve()` Bootstrap convenience yields from futures

### econometrics.py

Econometric analysis and hypothesis testing

* `ols()` Ordinary least squares with robust standard errors
* `whiteTest()` White's heteroskedasticity test
* `breuschPaganTest()` Breusch-Pagan test
* `durbinWatson()` Durbin-Watson autocorrelation statistic
* `ljungBox()` Ljung-Box autocorrelation test
* `adfTest()` Augmented Dickey-Fuller unit root test
* `kpssTest()` KPSS stationarity test
* `grangerCausality()` Granger causality testing

### ml.py

Machine learning algorithms

* `regressionTree()` / `decisionTree()` Tree-based models
* `predictTree()` Tree prediction
* `isolationForest()` / `anomalyScore()` Anomaly detection
* `kmeans()` K-means clustering
* `knn()` K-nearest neighbors
* `naiveBayes()` Gaussian naive Bayes
* `randomForest()` Random forest regressor
* `gradientBoosting()` Gradient boosting regressor
* `pca()` / `lda()` Dimensionality reduction
* `logisticRegression()` Logistic regression classifier

### portfolio.py

Portfolio optimization and allocation

* `blackLitterman()` Black-Litterman model
* `meanVariance()` Mean-variance optimization
* `minVariance()` Minimum variance portfolio
* `riskParity()` Risk parity allocation
* `equalWeight()` Equal weight portfolio
* `maxDiversification()` Maximum diversification
* `tangency()` / `maxSharpe()` Tangency/maximum Sharpe portfolio
* `efficientFrontier()` Efficient frontier computation
* `hierarchicalRiskParity()` HRP allocation
* `minCvar()` Minimum CVaR portfolio

### risk.py

Risk metrics and measures

* `parametricVar()` / `historicalVar()` Value at Risk
* `parametricCvar()` / `historicalCvar()` Conditional VaR
* `expectedShortfall()` Expected shortfall
* `drawdown()` / `maxDrawdownDuration()` Drawdown analysis
* `calmarRatio()` / `sharpeRatio()` / `sortinoRatio()` Performance ratios
* `omegaRatio()` / `treynorRatio()` / `informationRatio()` Additional ratios
* `modifiedVar()` / `hillTailIndex()` Advanced risk measures
* `beta()` Systematic risk
* `downsideDeviation()` / `ulcerIndex()` / `painIndex()` Downside risk
* `tailRatio()` / `capturRatio()` / `stabilityRatio()` Additional metrics

### sim.py

Time series simulation

* `gbm()` Geometric Brownian Motion
* `ou()` Ornstein-Uhlenbeck process
* `levyOu()` Lévy OU (OU with jumps)
* `ar1()` / `arma()` ARMA processes
* `markovSwitching()` Regime switching models
* `arch()` / `garch()` Volatility models
* `heston()` Heston stochastic volatility
* `cir()` / `vasicek()` Interest rate models
* `poisson()` / `compoundPoisson()` Jump processes
* `simulate()` General dispatcher function

### bootstrap.py

Curve and surface bootstrapping

* `curve()` 1D curve interpolation (linear/cubic/pchip)
* `surface()` 2D surface interpolation
* `zeroRateCurve()` Zero rate curve from bond prices
* `forwardCurve()` Forward rate derivation
* `discountCurve()` Discount factor curve
* `yieldCurve()` Yield curve bootstrapping
* `volSurface()` Implied volatility surface
* `creditCurve()` CDS curve bootstrapping
* `fxForwardCurve()` FX forward curve
* `inflationCurve()` Inflation curve from breakevens

### dimension.py

Dimensionality reduction techniques

* `pca()` Principal Component Analysis
* `lda()` Linear Discriminant Analysis
* `tsne()` t-SNE
* `ica()` Independent Component Analysis
* `nmf()` Non-negative Matrix Factorization
* `kernelPca()` Kernel PCA (RBF/poly/linear)
* `mds()` Multidimensional Scaling
* `isomap()` Isomap

### distributions.py

Distribution fitting and testing

* `fitNormal()` / `fitLognormal()` / `fitExponential()` Parametric fitting
* `fitGamma()` / `fitBeta()` / `fitT()` Additional distributions
* `fitMixture()` Gaussian mixture models (EM algorithm)
* `moments()` Calculate distribution moments
* `ksTest()` Kolmogorov-Smirnov test
* `adTest()` Anderson-Darling test
* `klDivergence()` / `jsDivergence()` Distribution divergence
* `quantile()` Quantile calculation
* `qqPlot()` Q-Q plot data generation

### factor.py

Factor models and analysis

* `famaFrench3()` Fama-French 3-factor model
* `carhart4()` Carhart 4-factor model
* `apt()` Arbitrage Pricing Theory
* `capm()` Capital Asset Pricing Model
* `estimateBeta()` Beta estimation
* `estimateFactorLoading()` Factor loading estimation
* `rollingBeta()` Rolling beta calculation
* `pcaFactors()` PCA factor extraction
* `factorMimicking()` Factor-mimicking portfolios (SMB/HML)
* `jensenAlpha()` Jensen's alpha
* `treynorMazuy()` Market timing model
* `multifactor()` Multi-factor regression

### fit.py

Time series model fitting

* `fitGbm()` Fit GBM to prices
* `fitOu()` Fit OU process
* `fitLevyOU()` Fit Levy OU process (OU with jumps)
* `fitAr1()` / `fitArma()` ARMA model fitting
* `fitGarch()` GARCH model fitting
* `fitHeston()` Heston model fitting
* `fitCir()` / `fitVasicek()` Interest rate model fitting
* `fitJumpDiffusion()` Jump-diffusion model fitting
* `fitCopula()` Gaussian copula fitting
* `fitDistributions()` Multi-distribution fitting with AIC ranking
* `aic()` / `bic()` Information criteria

## Usage Examples

```python
import numpy as np
from options import blackScholes
from risk import sharpeRatio
from sim import gbm
from portfolio import riskParity

# Option pricing
result = blackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, optType='call')
print(f"Call price: {result['price']:.2f}, Delta: {result['delta']:.3f}")

# Risk calculation
returns = np.random.normal(0.001, 0.02, 252)
sharpe = sharpeRatio(returns, riskFreeRate=0.0001)
print(f"Sharpe ratio: {sharpe:.3f}")

# Simulation
paths = gbm(mu=0.1, sigma=0.2, nSteps=252, nSims=1000, s0=100)
print(f"Simulated paths shape: {paths.shape}")

# Portfolio optimization
covMatrix = np.array([[0.04, 0.01], [0.01, 0.09]])
weights = riskParity(covMatrix)
print(f"Risk parity weights: {weights}")
```

## Documentation

Full documentation will be available at [btQuant Documentation](https://btquant.readthedocs.io).

## Converting to Pandas or Polars

All functions return NumPy arrays or dicts. Convert to DataFrames as needed:

```python
import pandas as pd
import polars as pl

paths = gbm(mu=0.1, sigma=0.2, nSteps=252, nSims=100)
df_pandas = pd.DataFrame(paths.T, columns=[f'sim_{i}' for i in range(100)])
df_polars = pl.DataFrame(paths.T)
```

## License

This project is licensed under GPL-3.0.

## Contact

For inquiries regarding btQuant, contact:

* Brayden Boyko ([brayden@sipglobally.com](mailto:brayden@sipglobally.com))