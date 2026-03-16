"""
sipQuant — Quantitative library for physically-settled commodity markets.
SIP Global (Systematic Index Partners)

Pure NumPy. No pandas, scipy, or external dependencies.

Modules
-------
schema      : Data contract layer — validated input containers.
options     : Option pricing: Black-Scholes, binomial, trinomial, barrier, spread, Monte Carlo.
econometrics: OLS, robust regression, stationarity tests, structural break tests.
sim         : Price process simulation: GBM, OU, Lévy-OU, GARCH, Heston, ARMA.
fit         : Model calibration: OU, Lévy-OU, GARCH, Heston, copula, distribution fitting.
bootstrap   : Curve and surface construction: forward curves, discount curves, vol surfaces.
portfolio   : Portfolio optimisation: mean-variance, HRP, risk parity, Black-Litterman.
risk        : Risk metrics: VaR, CVaR, drawdown, Sortino, Calmar, Hill estimator.
distributions: Distribution fitting: normal, lognormal, t, gamma, copulas.
dimension   : Dimensionality reduction: PCA, kernel PCA, t-SNE, ICA.
factor      : Factor models: CAPM, rolling beta, PCA factors.
ml          : Machine learning: regression trees, random forest, gradient boosting, k-means.
commodity   : Physical commodity pricing: seasonality, convenience yield, basis, grade adjustment.
otc         : OTC instrument pricing: swaps, Asian swaps, collars, physical forwards, swaptions.
book        : Dealer book: net Greeks, hedge ratios, P&L attribution, scenario shocks.
index       : Index construction, calculation, audit trail, IOSCO-aligned methodology.
liquidity   : Thin-market risk: liquidity-adjusted VaR, market impact, optimal execution.

Usage
-----
import sipQuant as sq

# Namespaced access
result = sq.options.blackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
swap   = sq.otc.commoditySwap(fixedPrice=142.0, indexCurve=curve, notional=500, schedule=sched, r=0.045)
ps     = sq.schema.PriceSeries(dates, values, source='broker_a', market='alberta_hay')
"""

from . import schema
from . import options
from . import econometrics
from . import sim
from . import bootstrap
from . import portfolio
from . import risk
from . import distributions
from . import dimension
from . import factor
from . import commodity
from . import otc
from . import book
from . import index
from . import liquidity

from . import ml
from . import fit

__version__ = '1.0.0'
__author__ = 'SIP Global — Systematic Index Partners'
__license__ = 'MIT'
