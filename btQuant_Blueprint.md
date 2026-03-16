# btQuant — Package Restructure Blueprint
### SIP Global — Systematic Index Partners
*Physically-settled commodity markets · OTC derivatives · Published indices*

**Re-brand as sipQuant in new repo - buildout in this repo and than we will transfer**

| | |
|---|---|
| **Scope** | Illiquid & underdeveloped commodity markets globally |
| **Constraint** | Pure NumPy — no pandas, scipy, or external dependencies |
| **Architecture** | Physical + OTC derivatives + published index infrastructure |
| **Version** | 2.0 (proposed) |

---

## 1. What to Remove — Bloat Assessment

The following functions exist in btQuant today but have no practical application to SIP's workflow. They target liquid market assumptions that do not hold in your markets, or solve academic problems that produce meaningless outputs on sparse commodity data.

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `famaFrench3()` | Fama-French 3-factor model | Requires tradeable SMB/HML factor portfolios. No liquid factor exposure in Alberta hay. | **REMOVE** |
| `carhart4()` | Carhart 4-factor with momentum | Momentum factor requires continuous liquid price history. | **REMOVE** |
| `apt()` | Arbitrage Pricing Theory | Requires observable tradeable factor premia. Not constructible in thin markets. | **REMOVE** |
| `factorMimicking()` | Construct SMB/HML mimicking portfolios | Requires cross-section of liquid assets. | **REMOVE** |
| `treynorMazuy()` | Market timing model | Measures mutual fund market-timing ability. | **REMOVE** |
| `lda() dimension.py` | Linear Discriminant Analysis | Classification-focused. No application here. | **REMOVE** |
| `nmf()` | Non-negative Matrix Factorization | Text/image decomposition. No commodity application. | **REMOVE** |
| `mds()` | Multidimensional Scaling | Visualisation only. | **REMOVE** |
| `isomap()` | Isomap manifold learning | High-dimensional visualisation. No pricing application. | **REMOVE** |
| `naiveBayes()` | Gaussian Naive Bayes classifier | Classification. Your problems are regression/estimation. | **REMOVE** |
| `logisticRegression()` | Logistic regression classifier | Classification. You need continuous regression outputs. | **REMOVE** |
| `vasicek()` | Vasicek interest rate model | Short-rate model for bond markets. | **REMOVE** |
| `cir()` | Cox-Ingersoll-Ross rate model | Interest rate model. No commodity application. | **REMOVE** |
| `fitVasicek()` | Fit Vasicek to rate data | Remove with vasicek(). | **REMOVE** |
| `fitCir()` | Fit CIR to rate data | Remove with cir(). | **REMOVE** |
| `creditCurve()` | Bootstrap CDS curve | CDS spreads require a credit market. | **REMOVE** |
| `inflationCurve()` | Inflation curve from breakevens | Requires inflation-linked bond market. | **REMOVE** |
| `asian() geometric` | Geometric Asian option only | Geometric averaging is an approximation. Keep arithmetic only. | **REMOVE** |
| `binary()` | Binary/digital option pricing | No current or planned product in your OTC book. | **REMOVE** |
| `jensenAlpha()` | Jensen's alpha vs benchmark | Your markets have no benchmark. | **REMOVE** |

Removing these reduces the package by ~20–25% of current function count without losing any capability relevant to SIP.

---

## 2. What to Keep — Core Retained Modules

### options.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `blackScholes()` | Black-Scholes with cost of carry | Price options on commodity forwards. Cost of carry captures storage + financing. | **KEEP** |
| `binomial()` | Binomial tree, American/European | American-exercise options where early physical delivery is possible. | **KEEP** |
| `trinomial()` | Trinomial tree, American/European | Higher accuracy for longer-dated OTC contracts. | **KEEP** |
| `barrier()` | Barrier options with rebate | Knock-in/knock-out clauses common in physical delivery OTC structures. | **KEEP** |
| `spread()` | Two-asset spread option | Price basis between two delivery points. | **KEEP** |
| `simulate()` | Monte Carlo wrapper | Path-dependent pricing for Asian average swaps. | **KEEP** |
| `impliedVol()` | Implied vol from market price | Back out vol from any observed OTC quote to build your vol surface. | **KEEP** |
| `buildForwardCurve()` | Forward curve from futures | Build the forward curve from observable proxy futures. | **KEEP** |
| `bootstrapCurve()` | Bootstrap convenience yields | Extract convenience yield from forward curve. | **KEEP** |

### econometrics.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `ols()` | OLS with robust standard errors | Proxy regression pricing — regress thin asset on liquid proxies. | **KEEP** |
| `whiteTest()` | White heteroskedasticity test | Test whether proxy regression residuals are heteroskedastic. | **KEEP** |
| `breuschPaganTest()` | Breusch-Pagan test | Companion heteroskedasticity test. | **KEEP** |
| `durbinWatson()` | Autocorrelation statistic | Detect serial correlation in pricing model residuals. | **KEEP** |
| `ljungBox()` | Ljung-Box test | More powerful autocorrelation test. | **KEEP** |
| `adfTest()` | Augmented Dickey-Fuller | Test stationarity before fitting any time series model. | **KEEP** |
| `kpssTest()` | KPSS stationarity test | Complement to ADF. | **KEEP** |
| `grangerCausality()` | Granger causality | Test whether a liquid proxy Granger-causes your thin asset. | **KEEP** |

### sim.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `gbm()` | Geometric Brownian Motion | Baseline price simulation for option pricing and scenario analysis. | **KEEP** |
| `ou()` | Ornstein-Uhlenbeck process | Mean-reversion to seasonal average — correct model for commodity spot prices. | **KEEP** |
| `levyOu()` | OU with jump component | Captures price shocks from weather events, crop failures, transport disruptions. | **KEEP** |
| `markovSwitching()` | Regime switching simulation | Simulate across different market regimes. | **KEEP** |
| `garch()` | GARCH volatility model | Model volatility clustering around seasonal peaks. | **KEEP** |
| `heston()` | Heston stochastic vol | Option pricing when vol itself is uncertain. | **KEEP** |
| `compoundPoisson()` | Compound Poisson jump process | Model lumpy physical trade arrivals in thin markets. | **KEEP** |
| `arma()` | ARMA time series | Short-term price forecasting and residual modelling. | **KEEP** |

### portfolio.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `blackLitterman()` | Black-Litterman allocation | Incorporate SIP's market views into position sizing. | **KEEP** |
| `meanVariance()` | Mean-variance optimisation | Baseline portfolio optimisation across commodity positions. | **KEEP** |
| `riskParity()` | Risk parity allocation | Equal-risk allocation — appropriate when return estimates are unreliable. | **KEEP** |
| `hierarchicalRiskParity()` | HRP allocation | Clustering-based — robust to covariance estimation error in sparse data. | **KEEP** |
| `minCvar()` | Minimum CVaR portfolio | Tail-risk-aware allocation for fat-tailed commodity exposures. | **KEEP** |
| `efficientFrontier()` | Efficient frontier | Visualise risk/return tradeoff across commodity positions. | **KEEP** |
| `maxSharpe()` | Maximum Sharpe portfolio | Optimal risk-adjusted allocation. | **KEEP** |

### risk.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `historicalVar()` | Historical simulation VaR | Non-parametric VaR — preferred for non-normal commodity distributions. | **KEEP** |
| `historicalCvar()` | Historical CVaR | Expected loss in tail — essential for book-level risk reporting. | **KEEP** |
| `parametricVar()` | Parametric VaR | Fast VaR estimate for position sizing decisions. | **KEEP** |
| `expectedShortfall()` | Expected shortfall | Regulatory-standard tail risk for OTC counterparty exposure. | **KEEP** |
| `drawdown()` | Drawdown series | Track capital drawdown on physical positions over time. | **KEEP** |
| `modifiedVar()` | Cornish-Fisher adjusted VaR | Accounts for skew and kurtosis. | **KEEP** |
| `hillTailIndex()` | Hill estimator for tail index | Measure fat-tail severity in thin markets. | **KEEP** |
| `beta()` | Systematic risk vs proxy | Measure proxy beta — direct input to hedge ratio calculation. | **KEEP** |
| `downsideDeviation()` | Downside deviation | Measure only negative volatility. | **KEEP** |
| `sortinoRatio()` | Sortino ratio | Risk-adjusted performance using downside deviation only. | **KEEP** |
| `calmarRatio()` | Calmar ratio | Return vs max drawdown. | **KEEP** |

### fit.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `fitOu()` | Fit OU process to prices | Calibrate mean-reversion speed and long-run mean. | **KEEP** |
| `fitLevyOU()` | Fit OU with jumps | Calibrate jump intensity and size from historical trade data. | **KEEP** |
| `fitGarch()` | Fit GARCH to returns | Calibrate volatility dynamics for option pricing and risk. | **KEEP** |
| `fitHeston()` | Fit Heston model | Calibrate stochastic vol model for longer-dated OTC contracts. | **KEEP** |
| `fitCopula()` | Fit Gaussian copula | Model joint distribution across correlated commodity positions. | **KEEP** |
| `fitDistributions()` | Multi-distribution fit with AIC | Find best-fit distribution for price returns. | **KEEP** |
| `fitArma()` | Fit ARMA model | Calibrate short-term price dynamics for forecasting. | **KEEP** |
| `aic()` | Akaike Information Criterion | Model selection. | **KEEP** |
| `bic()` | Bayesian Information Criterion | Companion to AIC. | **KEEP** |

### bootstrap.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `curve()` | 1D interpolation (linear/cubic/pchip) | Interpolate forward prices between observable contract dates. | **KEEP** |
| `surface()` | 2D surface interpolation | Build vol surface from sparse OTC quotes across strikes and tenors. | **KEEP** |
| `forwardCurve()` | Forward rate derivation | Build full commodity forward curve from observable futures. | **KEEP** |
| `discountCurve()` | Discount factor curve | Discount OTC cash flows for mark-to-model valuation. | **KEEP** |
| `volSurface()` | Implied vol surface | Build the vol surface for options at any strike/tenor. | **KEEP** |
| `fxForwardCurve()` | FX forward curve | Required for cross-border physical contracts. | **KEEP** |

---

## 3. What to Add — New Modules

### NEW: commodity.py

Physical commodity pricing primitives absent from standard quant libraries. Seasonality, convenience yield, basis, and grade adjustment are the core pricing inputs for ag markets.

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `stlDecompose(prices, period)` | STL decomposition into trend, seasonal, and residual | Decompose hay/stover prices into harvest seasonality vs trend vs noise. | **ADD** |
| `seasonalIndex(prices, period)` | Seasonal index by period — multipliers showing over/under relative to trend | Quantify how much hay prices typically rise in winter vs fall. | **ADD** |
| `fourierSeason(prices, n_harmonics)` | Fit Fourier series to seasonal pattern | Capture complex multi-harvest-cycle seasonality. | **ADD** |
| `convenienceYield(spot, forward, r, T)` | Back out convenience yield from observed spot and forward price | Core input to all commodity option pricing. | **ADD** |
| `storageCost(quantity, rate, T)` | Physical storage cost over time | Required for pricing deferred physical delivery contracts. | **ADD** |
| `basisCalc(local_price, benchmark_price)` | Calculate and track basis between local delivery point and benchmark | Alberta hay vs CME nearby corn — the basis is what SIP is trading. | **ADD** |
| `gradeAdj(price, base_spec, actual_spec, adj_matrix)` | Adjust price for quality/grade differentials vs contract specification | Hay moisture, protein, bale weight all create grade premiums/discounts. | **ADD** |
| `transportBasis(origin, destination, rate_per_unit)` | Transportation differential between delivery points | Price the cost of moving physical grain/hay between delivery points. | **ADD** |
| `localForwardCurve(spot, basis_curve, futures_curve)` | Build local delivery-point forward curve | SIP's core pricing output — the forward curve at the actual delivery point. | **ADD** |
| `physicalPremium(market_price, cost_of_production, margin)` | Premium above cost-of-production floor | Sets a floor on any mark-to-model price. | **ADD** |

### NEW: otc.py

Prices the actual OTC instruments SIP quotes on both sides of the book. All functions return a full Greeks dictionary for book.py aggregation.

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `commoditySwap(fixed_price, index_curve, notional, schedule, r)` | Price a fixed-float commodity swap | SIP's primary OTC product. A buyer locks in a fixed price; SIP manages float risk. | **ADD** |
| `asianSwap(index_curve, notional, schedule, r, avg_method)` | Price an Asian-average swap — settlement based on arithmetic average of index | Common for ag markets — settlement against monthly average prevents manipulation. | **ADD** |
| `collar(S, K_put, K_call, T, r, sigma, cost_of_carry)` | Price a collar (long put + short call). Returns combined Greeks. | Producers want price floors. SIP quotes collars for physical hedgers. | **ADD** |
| `physicalForward(S, basis, storage, r, T, quality_adj)` | Price a physically-delivered forward with basis, storage, and quality adjustment | The most basic physical commodity contract. | **ADD** |
| `basisSwap(basis_fixed, basis_curve, notional, schedule)` | Price a swap on the basis between two delivery points | SIP can hedge or monetise basis risk between delivery points. | **ADD** |
| `flexibleForward(S, K, T, r, sigma, flex_windows, cost_of_carry)` | Forward with delivery flexibility | Physical buyers need delivery flexibility. Standard models underprice this optionality. | **ADD** |
| `swaptionPrice(swap_params, T_expiry, sigma, r)` | Option to enter a commodity swap at a fixed rate at a future date | Allows counterparties to lock in the right to hedge without committing immediately. | **ADD** |
| `otcGreeks(instrument, params)` | Unified Greeks for any OTC instrument — delta, gamma, vega, theta, rho | Standardises output format for book.py aggregation. | **ADD** |
| `markToModel(instrument, params, market_context)` | Mark any OTC position to model value | Daily marking of the OTC book for margin calls, credit exposure, and settlement. | **ADD** |

### NEW: book.py

Aggregates individual OTC positions into a dealer book. Net Greeks, hedge requirements, and P&L attribution.

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `Book() class` | Container for a collection of OTC positions with Greeks stored per position | SIP's active OTC book tracking every outstanding swap, collar, and forward. | **ADD** |
| `addPosition(book, instrument, params, direction)` | Add a new OTC position to the book | Called every time SIP transacts a new deal. | **ADD** |
| `netGreeks(book)` | Aggregate all position Greeks into net book-level delta, gamma, vega, theta | Shows SIP's aggregate exposure. Drives hedging decisions. | **ADD** |
| `hedgeRatio(book, proxy_betas)` | Required proxy hedge position given net delta and proxy beta | Computes how many CME corn/canola contracts approximate the hay delta hedge. | **ADD** |
| `residualBasisRisk(book, proxy_betas)` | Unhedgeable basis risk remaining after proxy hedging | The naked basis exposure that cannot be eliminated. Sets reserve requirements. | **ADD** |
| `pnlAttribution(book, prices_t0, prices_t1)` | Decompose P&L into delta, gamma, vega, theta, and residual | Daily P&L explanation separating price moves from vol changes from time decay. | **ADD** |
| `scenarioShock(book, shocks_dict)` | Apply simultaneous price, vol, and basis shocks and return P&L impact | Stress test: what happens if hay drops 20% and basis widens 10% simultaneously? | **ADD** |
| `counterpartyExposure(book, discount_curve)` | Mark-to-market exposure per counterparty across all outstanding trades | How much does each counterparty owe SIP if all trades were unwound today? | **ADD** |
| `bookVaR(book, returns_history, confidence)` | Full book VaR using historical simulation across all correlated price factors | Aggregate risk measure for the entire OTC book. | **ADD** |
| `bookReport(book)` | Structured dict summary: positions, net Greeks, VaR, top risks | Daily book snapshot. Output feeds into index.py audit trail and counterparty reporting. | **ADD** |

### NEW: index.py

Constructs, calculates, and publishes SIP's commodity price indices. IOSCO-aligned with full audit trails.

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `defineIndex(name, constituents, weights_method, roll_rule, version)` | Define an index with immutable spec. Returns versioned index spec dict. | Every SIP index must have a defined, versioned spec pinned to each calculation date. | **ADD** |
| `weightConstituents(prices, method, volume, open_interest)` | Calculate constituent weights | Determines how much each trade or price observation contributes to the index. | **ADD** |
| `rollIndex(index_spec, current_date, prices)` | Execute the roll logic from expiring to next contract | Roll methodology is disclosed to counterparties. | **ADD** |
| `calculateIndex(index_spec, prices, date, context)` | Calculate index value for a given date. Logs all inputs. | Core daily calculation. Every input, weight, and output is stored in the audit log. | **ADD** |
| `auditLog(index_name, date, inputs, output, version)` | Append an immutable audit record for a given index calculation | IOSCO compliance. Every index calculation must be reproducible from the log. | **ADD** |
| `restatement(index_name, date, corrected_inputs, reason)` | Record a restatement of a past calculation | Documents corrections. Counterparties must be notified per methodology. | **ADD** |
| `methodologyVersion(index_spec)` | Return current methodology version and changelog | Counterparties need to know which methodology version their swap settled against. | **ADD** |
| `backfillIndex(index_spec, historical_prices)` | Calculate historical index time series from archived price data | Build the historical series for backtesting OTC structures and counterparty diligence. | **ADD** |
| `indexVol(index_values, window)` | Rolling realised volatility of the index | Implied vol for index-referenced options must be calibrated to index vol. | **ADD** |
| `constituentsReport(index_spec, date, prices)` | Structured transparency report of constituents, weights, and contributions | Published to counterparties. Shows exactly what prices drove the index on settlement dates. | **ADD** |

### Extensions: econometrics.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `huberRegression(X, y, epsilon)` | Huber M-estimator regression. Downweights outliers automatically. | Proxy regression when price history contains aberrant trades. | **ADD** |
| `theilSen(X, y)` | Theil-Sen non-parametric regression. Median of all pairwise slopes. | Most robust regression for very sparse datasets (< 50 observations). | **ADD** |
| `madVolatility(returns)` | Median Absolute Deviation volatility estimate | Single aberrant trade in thin market inflates std dev. MAD gives stable vol. | **ADD** |
| `robustOls(X, y, method)` | Drop-in replacement for ols() with estimator flag: 'ols', 'huber', or 'theilsen' | Use wherever ols() is currently called. Switch estimators without changing downstream code. | **ADD** |
| `chowTest(y, X, breakpoint)` | Chow test for structural break at a known candidate date | Test whether your pricing regression changed after a specific event. | **ADD** |
| `baiPerron(y, X, max_breaks)` | Bai-Perron test for multiple unknown structural breakpoints | Find structural breaks when you do not know in advance when they occurred. | **ADD** |
| `cusum(residuals)` | CUSUM test for parameter stability over time | Monitor whether your OU model or proxy regression is drifting. Triggers recalibration. | **ADD** |

### NEW: liquidity.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `liquidityAdjustedVar(returns, position_size, adv, confidence, holding_period)` | VaR adjusted for cost and time to unwind in a thin market | Your VaR on a 1000t hay position is not the same as a 10t position. | **ADD** |
| `marketImpact(order_size, adv, volatility, model)` | Estimate price impact of a trade as fraction of daily volume | A 500t trade in a 200t/day market has significant impact. | **ADD** |
| `liquidationCost(position_size, adv, volatility, urgency)` | Total cost to unwind a position over a given horizon | Required for accurate P&L on physical positions being unwound. | **ADD** |
| `effectiveBidAsk(trades, window)` | Estimate effective bid-ask spread from transaction data | Back out the effective spread from observed trade prices. | **ADD** |
| `amihudIlliquidity(returns, volume)` | Amihud illiquidity ratio — price impact per unit of volume | Measure how illiquid each commodity market is. Used in portfolio allocation. | **ADD** |
| `optimalExecution(position_size, adv, volatility, T, urgency)` | Almgren-Chriss optimal execution schedule across time | When SIP needs to build or unwind a large physical position, gives the optimal schedule. | **ADD** |

### Extensions: distributions.py

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `fitStudentT(X, Y)` | Fit bivariate Student-t copula. Symmetric tail dependence. | Drought hits hay and corn stover simultaneously. Gaussian copula underestimates this. | **ADD** |
| `fitClaytonCopula(X, Y)` | Fit Clayton copula. Lower tail dependence. | Price collapses across related ag commodities are more correlated than price spikes. | **ADD** |
| `fitGumbelCopula(X, Y)` | Fit Gumbel copula. Upper tail dependence. | Use when supply shortages drive simultaneous price spikes across your book. | **ADD** |
| `tailDependence(X, Y, copula_type)` | Upper and lower tail dependence coefficients from fitted copula | Quantify how correlated your commodity positions are in stress scenarios. | **ADD** |

---

## 4. Data Contract Layer — schema.py

No proprietary data. No ingestion logic. Defines the typed input structures that every btQuant function expects.

| Function | What it does | SIP application | Status |
|---|---|---|---|
| `PriceSeries(dates, values, source, market, grade)` | Validated container for a price time series | Every pricing and risk function takes a PriceSeries. Catches data quality issues at the boundary. | **ADD** |
| `SparsePriceSeries(dates, values, source, market)` | Extension for irregular observation intervals with gap metadata | Most SIP markets have weekly or irregular observations. | **ADD** |
| `TradeRecord(date, price, volume, grade, origin, destination, counterparty_id)` | Single physical trade observation. Validated for completeness. | Physical trades are the primary data source. | **ADD** |
| `QuoteSheet(date, bid, ask, mid, source, market, grade, tenor)` | OTC quote observation with full metadata | Standardises how broker quotes enter the pricing stack. | **ADD** |
| `ForwardCurve(tenors, prices, base_date, market, methodology)` | Validated forward curve container | Forward curves are the primary input to OTC pricing. | **ADD** |
| `OTCPosition(instrument_type, direction, notional, strike_or_fixed, expiry, counterparty_id, greeks)` | Validated OTC position record for book.py | Standardised position structure ensures book.py aggregation works regardless of instrument type. | **ADD** |
| `IndexSpec(name, version, constituents, weights_method, roll_rule, effective_date)` | Immutable index methodology specification | The governing document for OTC settlement. Immutability ensures no retroactive changes. | **ADD** |
| `validate(obj)` | Universal validation dispatcher. Returns errors list. | Fail fast with a clear error message at the input boundary. | **ADD** |

---

## 5. Proposed Package Structure — v2.0

| Module / Item | Notes |
|---|---|
| **options.py** | Retained. Option pricing suite for commodity derivatives. Foundation for otc.py. |
| **econometrics.py** | Retained + extended. Adds Huber, Theil-Sen, MAD vol, Chow, Bai-Perron, CUSUM. |
| **sim.py** | Retained. Simulation library. Vasicek and CIR removed. |
| **portfolio.py** | Retained. Portfolio optimisation. Factor model functions removed. |
| **risk.py** | Retained. Position-level risk metrics. Used by book.py. |
| **fit.py** | Retained. Model calibration suite. fitVasicek and fitCir removed. |
| **bootstrap.py** | Retained. Curve and surface construction. creditCurve and inflationCurve removed. |
| **distributions.py** | Retained + extended. Adds t-copula, Clayton, Gumbel, tail dependence. |
| **dimension.py** | Trimmed. Keep PCA, tSNE, kernelPCA. Remove LDA, NMF, MDS, Isomap. |
| **factor.py** | Trimmed. Keep CAPM, estimateBeta, rollingBeta, pcaFactors. Remove FF3, Carhart, APT. |
| **ml.py** | Trimmed. Keep tree models, forests, isolationForest, kmeans, knn. Remove naiveBayes, logisticRegression. |
| **commodity.py** | **NEW.** Physical commodity pricing: seasonality, convenience yield, basis, grade adjustment, local forward curve. |
| **otc.py** | **NEW.** OTC pricing: commodity swaps, Asian swaps, collars, physical forwards, basis swaps, flex forwards, swaptions. |
| **book.py** | **NEW.** Dealer book: net Greeks, proxy hedge ratios, residual basis risk, P&L attribution, scenario shocks, counterparty exposure. |
| **index.py** | **NEW.** Index construction, calculation, audit trail, methodology versioning. IOSCO-aligned. |
| **liquidity.py** | **NEW.** Thin market risk: liquidity-adjusted VaR, market impact, liquidation cost, effective bid-ask, optimal execution. |
| **schema.py** | **NEW.** Data contract layer: typed input schemas with validation. No proprietary data. |

**Total:** 10 retained/trimmed modules + 6 new modules = **16 modules**. ~20% smaller by function count (bloat removed), ~60% larger in capability relevant to SIP's actual workflow.

### Design constraints maintained throughout

| Constraint | Detail |
|---|---|
| **Pure NumPy** | All mathematical operations in NumPy only. No pandas, scipy, sklearn, or statsmodels. |
| **No data layer** | schema.py defines input shapes and validates them. No data ingestion or proprietary feeds. |
| **Consistent output** | All pricing functions return dicts. All risk functions return dicts. book.py can aggregate any otc.py output. |
| **Physical first** | Every OTC function handles physical delivery as the default. Cash settlement is an option flag. |
| **Audit by design** | index.py writes immutable audit records by construction. No way to calculate an index without logging inputs. |
| **Proxy-hedge aware** | book.py natively understands that delta hedges are proxy hedges with basis risk. |

---

## 6. Build Sequence

The modules have hard dependencies. Build in this order.

| Phase | Task |
|---|---|
| **Phase 1** | `schema.py` — everything depends on validated inputs. Build this first. |
| **Phase 2** | `commodity.py` — basis, seasonality, convenience yield feed into all OTC pricing. |
| **Phase 3** | `otc.py` — depends on `options.py` (existing) and `commodity.py` (phase 2). |
| **Phase 4** | `book.py` — depends on `otc.py` providing consistent Greeks dicts. |
| **Phase 5** | `index.py` — depends on `commodity.py` for price inputs and `schema.py` for validation. |
| **Phase 6** | `liquidity.py` — standalone. Can be built in parallel with phases 3–5. |
| **Phase 7** | `econometrics.py` extensions — add Huber, Theil-Sen, structural break tests. |
| **Phase 8** | `distributions.py` extensions — add tail copulas. |
| **Parallel** | Remove bloat functions from existing modules throughout phases 1–8. Each removal is independent. |

---

*btQuant v2.0 — SIP Global*
