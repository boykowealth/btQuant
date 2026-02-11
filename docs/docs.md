# btQuant Official Mathematical Documentation

## options.py

## options.py

### blackScholes(S, K, T, r, sigma, q=0.0, optType='call')

**Purpose**: Price European options using Black-Scholes-Merton model with cost of carry.

**Inputs**:
- $S$: Current asset price
- $K$: Strike price
- $T$: Time to maturity (years)
- $r$: Risk-free rate
- $\sigma$: Volatility
- $q$: Dividend yield or cost of carry adjustment
- optType: 'call' or 'put'

**Model**: 

$$
d_1 = \frac{\ln(S/K) + (r - q + 0.5\sigma^2)T}{\sigma\sqrt{T}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

Call price:
$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

Put price:
$$
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)
$$

**Applications**: Stock options, index options, currency options, futures options.

**Returns**: Dict with price, delta, gamma, vega, rho, theta.

---

### binomial(S, K, T, r, sigma, q=0.0, N=100, optType='call', american=False)

**Purpose**: Price options using a binomial tree (European or American).

**Inputs**:
- $N$: Number of time steps
- american: True for American-style exercise

**Model**:

$$
u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}
$$

$$
p = \frac{e^{(r-q)\Delta t} - d}{u - d}
$$

Option value recursion:
$$
V_i = e^{-r\Delta t}[p V_{i,u} + (1-p) V_{i,d}]
$$

For American options: $V_i = \max(V_i, \text{intrinsic value})$

**Applications**: American options, early exercise, discrete dividends.

**Returns**: Dict with price, delta, gamma, theta.

---

### trinomial(S, K, T, r, sigma, q=0.0, N=50, optType='call', american=False)

**Purpose**: Price options using trinomial tree (more stable than binomial).

**Model**:

$$
\Delta x = \sigma \sqrt{3 \Delta t}
$$

$$
p_u = 0.5\left(\frac{\sigma^2 \Delta t + \nu^2 \Delta t^2}{\Delta x^2} + \frac{\nu \Delta t}{\Delta x}\right), 
\quad
p_d = 0.5\left(\frac{\sigma^2 \Delta t + \nu^2 \Delta t^2}{\Delta x^2} - \frac{\nu \Delta t}{\Delta x}\right), 
\quad
p_m = 1 - p_u - p_d
$$

**Applications**: American options, barrier options, improved accuracy over binomial.

**Returns**: Dict with price, delta, gamma, theta.

---

### asian(S, K, T, r, sigma, q=0.0, nSteps=100, optType='call', avgType='geometric')

**Purpose**: Price Asian options (geometric averaging).

**Model**:

$$
\sigma_A = \sigma \sqrt{\frac{(n+1)(2n+1)}{6 n^2}}
$$

Adjusted drift:
$$
b_A = 0.5 (r - q - 0.5 \sigma^2) + 0.5 \sigma_A^2
$$

Use Black-Scholes formula with $\sigma_A$ and $b_A$.

**Applications**: Commodity averaging contracts, volatility smoothing, manipulation reduction.

**Returns**: Dict with price, delta, gamma, vega, rho, theta.

---

### binary(S, K, T, r, sigma, q=0.0, optType='call')

**Purpose**: Price binary (digital) options paying fixed amount at expiry.

**Model**:

Call payoff: $\mathbb{1}_{S_T > K}$

Price:
$$
C = e^{-rT} N(d_2)
$$

**Applications**: Binary bets, structured products, digital barrier options.

**Returns**: Dict with price, delta, gamma, vega, rho, theta.

---

### spread(S1, S2, K, T, r, sigma1, sigma2, rho, q1=0.0, q2=0.0, optType='call')

**Purpose**: Price spread options (difference between two underlyings).

**Model**: Margrabe-type approximation with correlation adjustment.

**Applications**: Commodity spreads, index spreads, multi-asset derivatives.

**Returns**: Dict with price, delta1, delta2, gamma1, gamma2, vega1, vega2.

---

### barrier(S, K, T, r, sigma, barrierLevel, q=0.0, optType='call', barrierType='down-and-out', rebate=0.0)

**Purpose**: Price barrier options (knock-in / knock-out).

**Applications**: Exotic options, risk management, structured products.

**Returns**: Dict with price, delta, gamma, vega.

---

### simulate(pricingModel, paths, r, T, **modelParams)

**Purpose**: Monte Carlo simulation of option prices using custom pricing function.

**Applications**: Validate analytic models, path-dependent simulations, stress testing.

**Returns**: Dict with price, stderr.

---

### impliedVol(price, S, K, T, r, optType='call', q=0.0, tol=1e-6, maxIter=100)

**Purpose**: Calculate implied volatility from market price.

**Model**: Newton-Raphson iteration on Black-Scholes vega.

**Applications**: Volatility surfaces, relative value trading, market calibration.

**Returns**: Implied volatility $\sigma$.

---

### buildForwardCurve(spotPrice, tenors, rates, storageCosts=None, convenienceYields=None)

**Purpose**: Build forward curve with storage costs and convenience yields.

**Applications**: Commodity forward pricing, curve modeling.

**Returns**: Numpy array of forward prices.

---

### bootstrapCurve(spotPrice, futuresPrices, tenors, assumedRate=0.05)

**Purpose**: Bootstrap convenience yields and storage costs from futures prices.

**Applications**: Commodity curve calibration, curve construction.

**Returns**: Dict with convenience_yields, storage_costs.

---

### generateRange(modelFunc, paramRanges, fixedParams, optType='call')

**Purpose**: Generate option prices across parameter ranges for sensitivity analysis.

**Applications**: Greeks surfaces, scenario analysis, parameter studies.

**Returns**: List of dicts with parameters and pricing results.

---

## econometrics.py

### ols(y, X, addConst=True, robust=False, covType='HC3')

**Purpose**: Ordinary least squares regression with optional robust standard errors.

**Model**:

$$
\hat{\beta} = (X^TX)^{-1}X^Ty
$$

Robust covariance (HC3):
$$
\text{Var}(\hat{\beta}) = (X^TX)^{-1}X^T\Omega X(X^TX)^{-1}
$$

where $\Omega_{ii} = \frac{e_i^2}{(1-h_i)^2}$ and $h_i$ is leverage.

**Applications**: Linear regression, factor models, econometric analysis.

**Returns**: Dict with coefficients, stdErrors, tStats, pValues, rSquared, adjRSquared, residuals, fittedValues, covMatrix.

---

### whiteTest(y, X, addConst=True)

**Purpose**: Test for heteroskedasticity in regression residuals.

**Model**:

Regress $e^2$ on $X$, $X^2$, and cross-products.

Test statistic:
$$
W = nR^2 \sim \chi^2_p
$$

**Applications**: Validate OLS assumptions, determine need for robust errors.

**Returns**: Dict with testStatistic, pValue, df, conclusion.

---

### breuschPaganTest(y, X, addConst=True)

**Purpose**: Test for heteroskedasticity (simpler than White test).

**Model**:

$$
\text{BP} = \frac{1}{2}\text{ESS} \sim \chi^2_k
$$

**Applications**: Heteroskedasticity testing, regression diagnostics.

**Returns**: Dict with testStatistic, pValue, df, conclusion.

---

### durbinWatson(residuals)

**Purpose**: Test for first-order autocorrelation in residuals.

**Model**:

$$
DW = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2}
$$

$DW \approx 2$: no autocorrelation; $DW < 2$: positive; $DW > 2$: negative.

**Applications**: Time series regression diagnostics, validate independence assumption.

**Returns**: Durbin-Watson statistic.

---

### ljungBox(residuals, lags=None)

**Purpose**: Test for autocorrelation at multiple lags.

**Model**:

$$
Q = n(n+2)\sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2_h
$$

**Applications**: Time series model diagnostics, ARMA residual testing.

**Returns**: Dict with lags, autocorrelations, qStats, pValues, criticalValues.

---

### adfTest(series, lags=None, regression='c', autolag='AIC')

**Purpose**: Augmented Dickey-Fuller test for unit root (non-stationarity).

**Model**:

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t
$$

Test $H_0: \gamma = 0$ (unit root).

**Applications**: Test stationarity, determine differencing order, cointegration analysis.

**Returns**: Dict with testStatistic, pValue, criticalValues, conclusion, optimalLag.

---

### kpssTest(series, lags=None, regression='c')

**Purpose**: KPSS test for stationarity (opposite null from ADF).

**Model**:

$$
\text{KPSS} = \frac{1}{n^2\hat{\sigma}^2}\sum_{t=1}^n S_t^2
$$

where $S_t = \sum_{i=1}^t e_i$.

Test $H_0$: series is stationary.

**Applications**: Confirm stationarity, complement to ADF test.

**Returns**: Dict with testStatistic, pValue, criticalValues, conclusion, lags.

---

### grangerCausality(y, x, maxLag=4)

**Purpose**: Test whether $x$ Granger-causes $y$ (predictive power).

**Model**:

Compare:
$$
y_t = \alpha + \sum_{i=1}^p \beta_i y_{t-i} + \epsilon_t
$$

vs:
$$
y_t = \alpha + \sum_{i=1}^p \beta_i y_{t-i} + \sum_{i=1}^p \gamma_i x_{t-i} + \epsilon_t
$$

**Applications**: Causality testing, lead-lag relationships, variable selection.

**Returns**: Dict with lags, fStats, pValues, conclusion.

---

## ml.py

### regressionTree(X, y, maxDepth=5)

**Purpose**: Build regression tree using MSE criterion.

**Model**:

At each node, find split $(j, s)$ that minimizes:
$$
\text{MSE} = \frac{1}{n}\left(\sum_{x_i \in L} (y_i - \bar{y}_L)^2 + \sum_{x_i \in R} (y_i - \bar{y}_R)^2\right)
$$

**Applications**: Non-linear regression, feature interactions, interpretable models.

**Returns**: Tree structure (nested dicts).

---

### predictTree(tree, X)

**Purpose**: Make predictions using fitted tree.

**Model**:

For each observation $x$, traverse tree:
$$
\hat{y}(x) = \begin{cases}
\text{left subtree} & \text{if } x_j \leq s \\
\text{right subtree} & \text{if } x_j > s
\end{cases}
$$

where $j$ is split feature and $s$ is split value.

**Applications**: Scoring new data from decision/regression trees.

**Returns**: Array of predictions.

---

### isolationForest(X, nTrees=100, maxSamples=None, maxDepth=10)

**Purpose**: Anomaly detection via isolation forest.

**Model**: 

Build ensemble of random trees. Anomalies have shorter path lengths:

$$
h(x) = e + c(n)
$$

where $e$ is external node depth and $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ is average path length.

Anomaly score:
$$
s(x) = 2^{-\frac{E[h(x)]}{c(n)}}
$$

Higher scores indicate anomalies.

**Applications**: Fraud detection, outlier identification, quality control.

**Returns**: List of isolation trees.

---

### anomalyScore(trees, X)

**Purpose**: Compute anomaly scores from isolation forest.

**Model**:

Average path length normalization:
$$
c(n) = 2H(n-1) - \frac{2(n-1)}{n}
$$

where $H(i)$ is harmonic number $\approx \ln(i) + 0.5772$.

Anomaly score:
$$
s(x,n) = 2^{-\frac{E[h(x)]}{c(n)}}
$$

Values close to 1 indicate anomalies, close to 0 indicate normal points.

**Returns**: Array of scores (higher = more anomalous).

---

### kmeans(X, k=3, maxIters=100, tol=1e-4)

**Purpose**: K-means clustering.

**Model**:

Objective (minimize within-cluster sum of squares):
$$
\min_{C_1,\ldots,C_k} \sum_{i=1}^k \sum_{x \in C_i} ||x - \mu_i||^2
$$

Algorithm:
1. Initialize centroids $\mu_1, \ldots, \mu_k$
2. Assign each point to nearest centroid:
$$
C_i = \{x : ||x - \mu_i|| \leq ||x - \mu_j|| \text{ for all } j\}
$$
3. Update centroids:
$$
\mu_i = \frac{1}{|C_i|}\sum_{x \in C_i} x
$$
4. Repeat until convergence

**Applications**: Customer segmentation, pattern recognition, data compression.

**Returns**: Tuple (centroids, labels).

---

### knn(XTrain, yTrain, XTest, k=3)

**Purpose**: K-nearest neighbors classification.

**Model**: 

Distance metric (Euclidean):
$$
d(x, x') = \sqrt{\sum_{j=1}^p (x_j - x'_j)^2}
$$

Classification:
$$
\hat{y}(x) = \text{mode}\{y_i : x_i \in N_k(x)\}
$$

where $N_k(x)$ are the $k$ nearest neighbors of $x$.

**Applications**: Classification, pattern matching, recommendation systems.

**Returns**: Array of predicted labels.

---

### naiveBayes(XTrain, yTrain, XTest)

**Purpose**: Gaussian naive Bayes classifier.

**Model**:

$$
P(y|x) \propto P(y)\prod_{j=1}^p P(x_j|y)
$$

Assume $P(x_j|y) \sim \mathcal{N}(\mu_{jy}, \sigma_{jy}^2)$.

**Applications**: Text classification, spam filtering, fast baseline classifier.

**Returns**: Array of predicted labels.

---

### decisionTree(X, y, maxDepth=5)

**Purpose**: Build classification tree using Gini impurity.

**Model**:

Gini impurity:
$$
G = 1 - \sum_{i=1}^C p_i^2
$$

**Applications**: Classification, feature selection, rule extraction.

**Returns**: Tree structure.

---

### randomForest(X, y, nEstimators=10, maxDepth=5, sampleRatio=0.8)

**Purpose**: Random forest regression ensemble.

**Model**: 

Build $B$ trees on bootstrapped samples, average predictions:
$$
\hat{f}(x) = \frac{1}{B}\sum_{b=1}^B T_b(x)
$$

Each tree $T_b$ is trained on bootstrap sample of size $n \times \text{sampleRatio}$.

Variance reduction:
$$
\text{Var}[\hat{f}(x)] = \frac{\sigma^2}{B}
$$

**Applications**: Regression with robustness, feature importance, non-linear modeling.

**Returns**: Prediction function.

---

### gradientBoosting(X, y, nEstimators=100, learningRate=0.1, maxDepth=3)

**Purpose**: Gradient boosting for regression.

**Model**:

Sequential additive model:
$$
f_m(x) = f_{m-1}(x) + \nu \cdot h_m(x)
$$

where $h_m$ is fitted to pseudo-residuals:
$$
r_{im} = y_i - f_{m-1}(x_i)
$$

Final prediction:
$$
\hat{y}(x) = f_0(x) + \nu\sum_{m=1}^M h_m(x)
$$

Learning rate $\nu$ controls shrinkage.

**Applications**: Regression, ranking, high accuracy prediction.

**Returns**: Prediction function.

---

### pca(X, nComponents=None)

**Purpose**: Principal component analysis for dimensionality reduction.

**Model**:

Center data:
$$
\tilde{X} = X - \bar{X}
$$

Covariance matrix:
$$
\Sigma = \frac{1}{n}\tilde{X}^T\tilde{X}
$$

Eigenvalue decomposition:
$$
\Sigma = W\Lambda W^T
$$

Project onto top $k$ eigenvectors:
$$
Z = \tilde{X}W_k
$$

Explained variance:
$$
\frac{\lambda_i}{\sum_j \lambda_j}
$$

**Applications**: Dimensionality reduction, visualization, noise reduction.

**Returns**: Dict with transformed, eigenvalues, eigenvectors, explainedVariance, mean.

---

### lda(X, y, nComponents=None)

**Purpose**: Linear discriminant analysis for supervised dimensionality reduction.

**Model**:

Between-class scatter:
$$
S_B = \sum_{i=1}^c n_i(\mu_i - \mu)(\mu_i - \mu)^T
$$

Within-class scatter:
$$
S_W = \sum_{i=1}^c \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T
$$

Maximize Fisher criterion:
$$
W = \arg\max_W \frac{|W^TS_BW|}{|W^TS_WW|}
$$

Solution: eigenvectors of $S_W^{-1}S_B$.

**Applications**: Classification, feature extraction, class separation.

**Returns**: Dict with transformed, eigenvalues, eigenvectors, mean.

---

### logisticRegression(X, y, learningRate=0.01, nIters=1000)

**Purpose**: Binary logistic regression classifier.

**Model**:

Logistic function:
$$
P(y=1|x) = \sigma(\beta^Tx + \beta_0) = \frac{1}{1 + e^{-(\beta^Tx + \beta_0)}}
$$

Log-likelihood:
$$
\ell(\beta) = \sum_{i=1}^n [y_i \log p_i + (1-y_i)\log(1-p_i)]
$$

Gradient ascent:
$$
\beta \leftarrow \beta + \eta \nabla\ell(\beta)
$$

where $\nabla\ell(\beta) = X^T(y - p)$.

**Applications**: Binary classification, probability estimation, baseline classifier.

**Returns**: Tuple (weights, bias, predict function).

---

## portfolio.py

### blackLitterman(covMatrix, pi, P, Q, tau=0.05)

**Purpose**: Combine market equilibrium with investor views.

**Model**:

Posterior returns:
$$
\mu_{BL} = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\pi + P^T\Omega^{-1}Q]
$$

**Applications**: Portfolio construction with views, tactical allocation.

**Returns**: Adjusted expected returns.

---

### meanVariance(expectedReturns, covMatrix, riskAversion=0.5)

**Purpose**: Mean-variance optimization.

**Model**:

Maximize:
$$
\mu^Tw - \frac{\lambda}{2}w^T\Sigma w
$$

subject to $\sum w_i = 1, w_i \geq 0$.

**Applications**: Portfolio optimization, risk-return tradeoff.

**Returns**: Optimal weights.

---

### minVariance(covMatrix)

**Purpose**: Minimum variance portfolio.

**Model**:

Minimize $w^T\Sigma w$ subject to $\sum w_i = 1, w_i \geq 0$.

**Applications**: Conservative allocation, risk minimization.

**Returns**: Minimum variance weights.

---

### riskParity(covMatrix)

**Purpose**: Risk parity (equal risk contribution) portfolio.

**Model**:

Each asset contributes equally to portfolio risk:
$$
w_i \frac{\partial \sigma_p}{\partial w_i} = \frac{1}{n}\sigma_p
$$

**Applications**: Diversified allocation, all-weather portfolios.

**Returns**: Risk parity weights.

---

### equalWeight(nAssets)

**Purpose**: Equal weight (1/N) portfolio.

**Model**: 

$$
w_i = \frac{1}{n} \quad \forall i
$$

Portfolio return:
$$
R_p = \frac{1}{n}\sum_{i=1}^n R_i
$$

**Applications**: Simple baseline, naive diversification.

**Returns**: Equal weights array.

---

### maxDiversification(covMatrix)

**Purpose**: Maximum diversification ratio portfolio.

**Model**:

Maximize:
$$
DR = \frac{w^T\sigma}{w^T\Sigma w}
$$

**Applications**: Maximize diversification benefit, reduce concentration.

**Returns**: Maximum diversification weights.

---

### tangency(expectedReturns, covMatrix, riskFreeRate=0)

**Purpose**: Tangency portfolio (maximum Sharpe ratio).

**Model**:

Maximize:
$$
SR = \frac{\mu^Tw - r_f}{\sqrt{w^T\Sigma w}}
$$

**Applications**: Optimal risky portfolio, CAPM, efficient frontier.

**Returns**: Tangency weights.

---

### efficientFrontier(expectedReturns, covMatrix, nPoints=50)

**Purpose**: Compute efficient frontier points.

**Applications**: Risk-return visualization, portfolio selection.

**Returns**: List of dicts with return, volatility, weights.

---

### hierarchicalRiskParity(covMatrix, returns)

**Purpose**: HRP allocation using hierarchical clustering.

**Model**:

1. Compute distance matrix from correlation:
$$
d_{ij} = \sqrt{\frac{1 - \rho_{ij}}{2}}
$$

2. Build dendrogram via hierarchical clustering

3. Recursive bisection with inverse-variance weighting:
$$
w_{\text{left}} = \frac{V_{\text{right}}}{V_{\text{left}} + V_{\text{right}}}, \quad w_{\text{right}} = \frac{V_{\text{left}}}{V_{\text{left}} + V_{\text{right}}}
$$

where $V$ is cluster variance.

**Applications**: Diversification, machine learning portfolios, alternative to MVO.

**Returns**: HRP weights.

---

### minCvar(expectedReturns, returns, alpha=0.95)

**Purpose**: Minimize conditional value at risk.

**Model**:

CVaR optimization:
$$
\min_w \text{CVaR}_\alpha(w^Tr) = \min_w E[-(w^Tr) | w^Tr \leq -\text{VaR}_\alpha]
$$

subject to:
$$
\sum_{i=1}^n w_i = 1, \quad w_i \geq 0
$$

CVaR is expected loss beyond VaR threshold (coherent risk measure).

**Applications**: Downside risk minimization, conservative allocation.

**Returns**: Minimum CVaR weights.

---

## risk.py

### parametricVar(returns, confidence=0.95)

**Purpose**: Value at Risk assuming normal distribution.

**Model**:

$$
\text{VaR}_\alpha = -(\mu + z_\alpha \sigma)
$$

**Applications**: Risk reporting, regulatory capital, position limits.

**Returns**: VaR estimate.

---

### historicalVar(returns, confidence=0.95)

**Purpose**: Non-parametric VaR from historical distribution.

**Model**: $(1-\alpha)$ quantile of return distribution.

**Applications**: Non-normal distributions, empirical risk.

**Returns**: Historical VaR.

---

### parametricCvar(returns, confidence=0.95)

**Purpose**: Conditional VaR (expected shortfall) assuming normal distribution.

**Model**:

$$
\text{CVaR}_\alpha = -\left(\mu + \sigma\frac{\phi(z_\alpha)}{1-\alpha}\right)
$$

**Applications**: Tail risk, coherent risk measure, optimization.

**Returns**: CVaR estimate.

---

### historicalCvar(returns, confidence=0.95)

**Purpose**: Non-parametric CVaR from historical distribution.

**Model**: Mean of returns below VaR threshold.

**Applications**: Tail risk estimation, portfolio optimization.

**Returns**: Historical CVaR.

---

### drawdown(returns)

**Purpose**: Maximum drawdown (peak-to-trough decline).

**Model**:

$$
DD_t = \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}
$$

**Applications**: Risk assessment, manager evaluation, drawdown constraints.

**Returns**: Maximum drawdown (negative value).

---

### calmarRatio(returns, riskFreeRate=0)

**Purpose**: Calmar ratio (return over max drawdown).

**Model**:

$$
\text{Calmar} = \frac{\text{Annualized Return}}{|\text{Max Drawdown}|}
$$

**Applications**: Performance measurement, fund comparison.

**Returns**: Calmar ratio.

---

### sharpeRatio(returns, riskFreeRate=0)

**Purpose**: Sharpe ratio (excess return per unit volatility).

**Model**:

$$
SR = \frac{E[R - R_f]}{\sigma(R - R_f)}
$$

**Applications**: Performance measurement, portfolio selection, fund ranking.

**Returns**: Sharpe ratio.

---

### sortinoRatio(returns, riskFreeRate=0, target=0)

**Purpose**: Sortino ratio (excess return per unit downside deviation).

**Model**:

$$
\text{Sortino} = \frac{E[R - R_f]}{\sigma_d}
$$

where $\sigma_d = \sqrt{E[\min(R - \text{target}, 0)^2]}$.

**Applications**: Downside risk-adjusted returns, asymmetric risk.

**Returns**: Sortino ratio.

---

### omegaRatio(returns, threshold=0.0)

**Purpose**: Omega ratio (probability-weighted gains to losses).

**Model**:

$$
\Omega = \frac{\int_\tau^\infty [1-F(x)]dx}{\int_{-\infty}^\tau F(x)dx}
$$

**Applications**: Full distribution risk measure, alternative to Sharpe.

**Returns**: Omega ratio.

---

### modifiedVar(returns, confidence=0.95)

**Purpose**: VaR adjusted for skewness and kurtosis via Cornish-Fisher expansion.

**Model**:

$$
z_{CF} = z + \frac{1}{6}(z^2-1)s + \frac{1}{24}(z^3-3z)(k-3) - \frac{1}{36}(2z^3-5z)s^2
$$

**Applications**: Non-normal distributions, fat tails, asymmetric risk.

**Returns**: Modified VaR.

---

### hillTailIndex(returns, k=50)

**Purpose**: Estimate tail index (measure of tail heaviness).

**Model**:

$$
\hat{\xi} = \frac{1}{k}\sum_{i=1}^k \ln\frac{X_{(i)}}{X_{(k+1)}}
$$

**Applications**: Extreme value theory, tail risk assessment.

**Returns**: Tail index (lower = heavier tail).

---

### beta(assetReturns, marketReturns)

**Purpose**: Systematic risk (market beta).

**Model**:

$$
\beta = \frac{\text{Cov}(R_a, R_m)}{\text{Var}(R_m)}
$$

**Applications**: CAPM, risk decomposition, hedging.

**Returns**: Beta coefficient.

---

### treynorRatio(returns, marketReturns, riskFreeRate=0)

**Purpose**: Treynor ratio (excess return per unit systematic risk).

**Model**:

$$
T = \frac{E[R - R_f]}{\beta}
$$

**Applications**: Performance measurement for diversified portfolios.

**Returns**: Treynor ratio.

---

### informationRatio(returns, benchmarkReturns)

**Purpose**: Information ratio (active return per unit tracking error).

**Model**:

$$
IR = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)}
$$

**Applications**: Active management evaluation, manager skill.

**Returns**: Information ratio.

---

### trackingError(returns, benchmarkReturns)

**Purpose**: Standard deviation of active returns.

**Model**:

$$
TE = \sigma(R_p - R_b)
$$

**Applications**: Active risk measurement, index tracking quality.

**Returns**: Tracking error.

---

## sim.py

### gbm(mu, sigma, nSteps, nSims, s0=1.0, dt=1/252)

**Purpose**: Simulate geometric Brownian motion.

**Model**:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

Solution:
$$
S_t = S_0 e^{(\mu - 0.5\sigma^2)t + \sigma W_t}
$$

**Applications**: Stock price simulation, option pricing, scenario analysis.

**Returns**: Array (nSims x nSteps).

---

### ou(theta, mu, sigma, nSteps, nSims, x0=0.0, dt=1/252)

**Purpose**: Simulate Ornstein-Uhlenbeck (mean-reverting) process.

**Model**:

$$
dX_t = \theta(\mu - X_t)dt + \sigma dW_t
$$

**Applications**: Interest rates, spreads, commodity prices, pairs trading.

**Returns**: Array (nSims x nSteps).

---

### levyOu(theta, mu, sigma, jumpLambda, jumpMu, jumpSigma, nSteps, nSims, x0=0.0, dt=1/252)

**Purpose**: OU process with jumps (Lévy process).

**Model**:

$$
dX_t = \theta(\mu - X_t)dt + \sigma dW_t + J_tdN_t
$$

where $N_t$ is Poisson process with intensity $\lambda$.

**Applications**: Commodity prices with shocks, electricity, regime shifts.

**Returns**: Array (nSims x nSteps).

---

### ar1(phi, intercept, sigma, nSteps, nSims, x0=0.0)

**Purpose**: Simulate AR(1) process.

**Model**:

$$
X_t = c + \phi X_{t-1} + \epsilon_t
$$

**Applications**: Discrete-time mean reversion, economic time series.

**Returns**: Array (nSims x nSteps).

---

### arma(arCoefs, maCoefs, sigma, nSteps, nSims, x0=0.0)

**Purpose**: Simulate ARMA process.

**Model**:

$$
X_t = \sum_{i=1}^p \phi_i X_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t
$$

**Applications**: Time series forecasting, economic modeling.

**Returns**: Array (nSims x nSteps).

---

### markovSwitching(mu1, sigma1, mu2, sigma2, p11, p22, nSteps, nSims, x0=0.0)

**Purpose**: Simulate regime-switching model.

**Model**: State alternates between two regimes with transition probabilities $p_{ij}$.

**Applications**: Business cycles, market regimes, structural breaks.

**Returns**: Array (nSims x nSteps).

---

### arch(alpha0, alpha1, nSteps, nSims)

**Purpose**: Simulate ARCH(1) process.

**Model**:

$$
X_t = \epsilon_t\sqrt{\sigma_t^2}, \quad \sigma_t^2 = \alpha_0 + \alpha_1 X_{t-1}^2
$$

**Applications**: Volatility clustering, heteroskedasticity.

**Returns**: Array (nSims x nSteps).

---

### garch(omega, alpha1, beta1, nSteps, nSims)

**Purpose**: Simulate GARCH(1,1) process.

**Model**:

$$
X_t = \epsilon_t\sqrt{\sigma_t^2}, \quad \sigma_t^2 = \omega + \alpha_1 X_{t-1}^2 + \beta_1 \sigma_{t-1}^2
$$

**Applications**: Volatility modeling, option pricing, risk management.

**Returns**: Array (nSims x nSteps).

---

### heston(mu, kappa, theta, sigma, rho, s0=100, v0=0.04, nSteps=252, nSims=1000, dt=1/252)

**Purpose**: Simulate Heston stochastic volatility model.

**Model**:

$$
dS_t = \mu S_t dt + \sqrt{V_t}S_t dW_1
$$

$$
dV_t = \kappa(\theta - V_t)dt + \sigma\sqrt{V_t}dW_2
$$

where $\text{Corr}(W_1, W_2) = \rho$.

**Applications**: Option pricing, volatility surface modeling.

**Returns**: Dict with prices and variances arrays.

---

### cir(kappa, theta, sigma, nSteps, nSims, r0=0.05, dt=1/252)

**Purpose**: Simulate Cox-Ingersoll-Ross interest rate model.

**Model**:

$$
dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t}dW_t
$$

**Applications**: Interest rate modeling, bond pricing, positive rates.

**Returns**: Array (nSims x nSteps).

---

### vasicek(kappa, theta, sigma, nSteps, nSims, r0=0.05, dt=1/252)

**Purpose**: Simulate Vasicek interest rate model.

**Model**:

$$
dr_t = \kappa(\theta - r_t)dt + \sigma dW_t
$$

**Applications**: Interest rate modeling, analytical tractability.

**Returns**: Array (nSims x nSteps).

---

### poisson(lambdaRate, nSteps, nSims)

**Purpose**: Simulate Poisson process (jump counts).

**Model**: $N_t \sim \text{Poisson}(\lambda t)$.

**Applications**: Event counting, jump arrivals, insurance claims.

**Returns**: Array (nSims x nSteps) of jump counts.

---

### compoundPoisson(lambdaRate, jumpMu, jumpSigma, nSteps, nSims, s0=100, dt=1/252)

**Purpose**: Simulate compound Poisson process (jumps with sizes).

**Model**: Random jump sizes at Poisson arrival times.

**Applications**: Insurance, credit events, operational risk.

**Returns**: Array (nSims x nSteps).

---

## bootstrap.py

### curve(x, y, z=None, method='cubic', nPoints=100)

**Purpose**: Bootstrap 1D curves using interpolation.

**Methods**:
- Linear: piecewise linear
- Cubic: cubic spline (smooth, continuous second derivative)
- PCHIP: monotonic cubic (preserves shape)

**Applications**: Yield curves, forward curves, term structures.

**Returns**: Dict with x and y arrays.

---

### surface(x, y, z, method='linear', gridSize=(100, 100))

**Purpose**: Bootstrap 2D surfaces.

**Applications**: Volatility surfaces, correlation surfaces, multidimensional pricing.

**Returns**: Dict with X, Y, Z grids.

---

### zeroRateCurve(maturities, prices, method='linear', nPoints=100)

**Purpose**: Bootstrap zero-coupon rate curve from bond prices.

**Model**:

$$
P(T) = e^{-r(T)T}
$$

**Applications**: Discount curve construction, bond valuation.

**Returns**: Dict with maturities and zeroRates.

---

### forwardCurve(maturities, zeroRates, method='linear', nPoints=100)

**Purpose**: Derive forward rates from zero rates.

**Model**:

$$
f(t_1, t_2) = \frac{r(t_2)t_2 - r(t_1)t_1}{t_2 - t_1}
$$

**Applications**: Forward rate agreements, interest rate derivatives.

**Returns**: Dict with maturities and forwardRates.

---

### discountCurve(maturities, zeroRates, method='linear', nPoints=100)

**Purpose**: Convert zero rates to discount factors.

**Model**:

$$
DF(T) = e^{-r(T)T}
$$

**Applications**: Present value calculations, bond pricing.

**Returns**: Dict with maturities and discountFactors.

---

### yieldCurve(maturities, prices, coupons=None, method='linear', nPoints=100)

**Purpose**: Bootstrap yield curve from bond data.

**Applications**: Government curves, corporate bonds, fixed income valuation.

**Returns**: Dict with maturities and yields.

---

### volSurface(maturities, strikes, impliedVols, method='linear', gridSize=(50, 50))

**Purpose**: Bootstrap implied volatility surface.

**Applications**: Option pricing, arbitrage-free interpolation, risk management.

**Returns**: Dict with maturities, strikes, vols grids.

---

### creditCurve(maturities, cdsSpreads, recoveryRate=0.4, method='linear', nPoints=100)

**Purpose**: Bootstrap credit default swap curve.

**Model**:

$$
\text{Hazard Rate} = \frac{\text{CDS Spread}}{1 - \text{Recovery Rate}}
$$

**Applications**: Credit risk, default probability estimation.

**Returns**: Dict with maturities and defaultProbabilities.

---

### fxForwardCurve(spot, domesticRates, foreignRates, maturities, method='linear', nPoints=100)

**Purpose**: Bootstrap FX forward curve.

**Model**:

$$
F(T) = S_0 e^{(r_d - r_f)T}
$$

**Applications**: Currency hedging, FX derivatives.

**Returns**: Dict with maturities and forwardRates.

---

### inflationCurve(maturities, breakevens, realRates, method='linear', nPoints=100)

**Purpose**: Derive inflation curve from breakeven rates.

**Model**:

$$
\text{Inflation} = \text{Breakeven} - \text{Real Rate}
$$

**Applications**: TIPS, inflation hedging, real vs nominal bonds.

**Returns**: Dict with maturities and inflationRates.

---

## dimension.py

### pca(X, nComponents=None)

**Purpose**: Principal component analysis.

**Model**: Eigenvalue decomposition of covariance matrix.

$$
\Sigma = W\Lambda W^T
$$

**Applications**: Dimensionality reduction, feature extraction, visualization, noise reduction.

**Returns**: Dict with transformed, eigenvalues, eigenvectors, explainedVariance, mean.

---

### lda(X, y, nComponents=None)

**Purpose**: Linear discriminant analysis.

**Model**: Maximize between-class to within-class scatter ratio.

$$
W = \arg\max \frac{|W^TS_BW|}{|W^TS_WW|}
$$

**Applications**: Supervised dimensionality reduction, classification preprocessing.

**Returns**: Dict with transformed, eigenvalues, eigenvectors, mean.

---

### tsne(X, nComponents=2, perplexity=30.0, maxIter=1000, learningRate=200.0)

**Purpose**: t-SNE for non-linear dimensionality reduction.

**Model**: Minimize KL divergence between high-dimensional and low-dimensional probability distributions.

**Applications**: Visualization, clustering analysis, pattern discovery.

**Returns**: Dict with transformed data.

---

### ica(X, nComponents=None, maxIter=200, tol=1e-4)

**Purpose**: Independent component analysis.

**Model**: Find independent sources in mixed signals via non-Gaussianity maximization.

**Applications**: Signal separation, feature extraction, blind source separation.

**Returns**: Dict with transformed, mixingMatrix, unmixingMatrix, mean.

---

### nmf(X, nComponents, maxIter=200, tol=1e-4)

**Purpose**: Non-negative matrix factorization.

**Model**:

$$
X \approx WH
$$

where $W, H \geq 0$.

**Applications**: Topic modeling, image processing, parts-based representation.

**Returns**: Dict with W (basis), H (coefficients).

---

### kernelPca(X, nComponents=None, kernel='rbf', gamma=None)

**Purpose**: Kernel PCA for non-linear dimensionality reduction.

**Model**: Apply PCA in kernel-induced feature space.

RBF kernel: $K(x, y) = e^{-\gamma||x-y||^2}$

**Applications**: Non-linear patterns, manifold learning.

**Returns**: Dict with transformed, eigenvalues, eigenvectors.

---

### mds(X, nComponents=2, metric=True, maxIter=300)

**Purpose**: Multidimensional scaling.

**Model**: Find low-dimensional embedding preserving pairwise distances.

**Applications**: Distance-based visualization, perceptual mapping.

**Returns**: Dict with transformed data.

---

### isomap(X, nComponents=2, nNeighbors=5)

**Purpose**: Isometric feature mapping (non-linear dimensionality reduction).

**Model**: MDS on geodesic distances in neighborhood graph.

**Applications**: Manifold learning, non-linear dimensionality reduction.

**Returns**: Dict with transformed data.

---

## distributions.py

### fitNormal(data)

**Purpose**: Fit normal distribution via MLE.

**Model**:

$$
\hat{\mu} = \frac{1}{n}\sum x_i, \quad \hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \hat{\mu})^2
$$

**Applications**: Return modeling, hypothesis testing, parametric VaR.

**Returns**: Dict with mu, sigma, logLikelihood.

---

### fitLognormal(data)

**Purpose**: Fit lognormal distribution.

**Model**: $\ln(X) \sim \mathcal{N}(\mu, \sigma^2)$.

**Applications**: Asset prices, positive quantities, skewed distributions.

**Returns**: Dict with mu, sigma (of log), logLikelihood.

---

### fitExponential(data)

**Purpose**: Fit exponential distribution.

**Model**:

$$
f(x) = \lambda e^{-\lambda x}, \quad \hat{\lambda} = \frac{1}{\bar{x}}
$$

**Applications**: Waiting times, durations, survival analysis.

**Returns**: Dict with lambda, logLikelihood.

---

### fitGamma(data, maxIter=100)

**Purpose**: Fit gamma distribution via method of moments.

**Model**:

$$
\hat{\alpha} = \frac{\bar{x}^2}{s^2}, \quad \hat{\beta} = \frac{\bar{x}}{s^2}
$$

**Applications**: Positive skewed data, loss distributions.

**Returns**: Dict with alpha (shape), beta (rate), logLikelihood.

---

### fitBeta(data, maxIter=100)

**Purpose**: Fit beta distribution.

**Model**: Distribution on $(0, 1)$.

**Applications**: Proportions, percentages, recovery rates.

**Returns**: Dict with alpha, beta, logLikelihood.

---

### fitT(data, maxIter=50)

**Purpose**: Fit Student's t distribution.

**Applications**: Heavy tails, robust statistics, financial returns.

**Returns**: Dict with df (degrees of freedom), mu, sigma, logLikelihood.

---

### fitMixture(data, nComponents=2, maxIter=100)

**Purpose**: Fit Gaussian mixture model via EM algorithm.

**Model**:

$$
f(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \sigma_k^2)
$$

**Applications**: Regime detection, clustering, multi-modal distributions.

**Returns**: Dict with means, sigmas, weights.

---

### moments(data)

**Purpose**: Calculate distribution moments.

**Model**:

First moment (mean):
$$
\mu = E[X] = \frac{1}{n}\sum_{i=1}^n x_i
$$

Second central moment (variance):
$$
\sigma^2 = E[(X-\mu)^2] = \frac{1}{n-1}\sum_{i=1}^n (x_i - \mu)^2
$$

Third standardized moment (skewness):
$$
\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3}
$$

Fourth standardized moment (kurtosis):
$$
\gamma_2 = \frac{E[(X-\mu)^4]}{\sigma^4} - 3
$$

**Returns**: Dict with mean, variance, skewness, kurtosis.

---

### ksTest(data, distName='normal', params=None)

**Purpose**: Kolmogorov-Smirnov goodness-of-fit test.

**Model**:

$$
D = \sup_x |F_n(x) - F(x)|
$$

**Applications**: Distribution testing, model validation.

**Returns**: Dict with statistic, pValue.

---

### adTest(data, distName='normal')

**Purpose**: Anderson-Darling test for normality.

**Model**: Weighted KS test emphasizing tails.

**Applications**: Normality testing, distribution validation.

**Returns**: Dict with statistic, criticalValues.

---

### klDivergence(p, q)

**Purpose**: Kullback-Leibler divergence.

**Model**:

$$
D_{KL}(P||Q) = \sum p_i \ln\frac{p_i}{q_i}
$$

**Applications**: Information theory, model comparison, distribution distance.

**Returns**: KL divergence value.

---

### jsDivergence(p, q)

**Purpose**: Jensen-Shannon divergence (symmetric).

**Model**:

$$
JSD(P||Q) = \frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)
$$

where $M = \frac{1}{2}(P + Q)$.

**Applications**: Symmetric distribution comparison, clustering.

**Returns**: JS divergence value.

---

### quantile(data, q)

**Purpose**: Calculate quantile.

**Model**:

Order statistics $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$.

Quantile:
$$
Q(p) = X_{(\lceil np \rceil)}
$$

or via linear interpolation:
$$
Q(p) = (1-\gamma)X_{(j)} + \gamma X_{(j+1)}
$$

where $np = j + \gamma$, $0 \leq \gamma < 1$.

**Applications**: Percentiles, VaR calculation, distribution summaries.

**Returns**: Quantile value.

---

### qqPlot(data, distName='normal')

**Purpose**: Generate Q-Q plot data for distribution assessment.

**Model**:

Empirical quantiles (sorted data):
$$
q_i^{\text{emp}} = X_{(i)}
$$

Theoretical quantiles:
$$
q_i^{\text{theo}} = F^{-1}\left(\frac{i - 0.5}{n}\right)
$$

where $F^{-1}$ is inverse CDF of reference distribution.

Points should lie on 45-degree line if distributions match.

**Applications**: Visual distribution testing, tail analysis.

**Returns**: Dict with theoretical and empirical quantiles.

---

## factor.py

### famaFrench3(marketReturns, smb, hml, betaM, betaSmb, betaHml, riskFree=0.02)

**Purpose**: Fama-French 3-factor model expected return.

**Model**:

$$
E[R_i] = R_f + \beta_M E[R_M - R_f] + \beta_{SMB}E[SMB] + \beta_{HML}E[HML]
$$

**Applications**: Equity expected returns, factor investing, performance attribution.

**Returns**: Expected return.

---

### carhart4(marketReturns, smb, hml, momentum, betaM, betaSmb, betaHml, betaMom, riskFree=0.02)

**Purpose**: Carhart 4-factor model (adds momentum to FF3).

**Model**:

$$
E[R_i] = R_f + \beta_M E[R_M] + \beta_{SMB}E[SMB] + \beta_{HML}E[HML] + \beta_{MOM}E[MOM]
$$

**Applications**: Momentum strategies, enhanced factor models.

**Returns**: Expected return.

---

### apt(riskFactors, factorBetas, riskFree=0.02)

**Purpose**: Arbitrage pricing theory expected return.

**Model**:

$$
E[R_i] = R_f + \sum_{j=1}^k \beta_j E[F_j]
$$

**Applications**: Multi-factor models, general equilibrium pricing.

**Returns**: Expected return.

---

### capm(marketReturn, beta, riskFree=0.02)

**Purpose**: Capital asset pricing model.

**Model**:

$$
E[R_i] = R_f + \beta(E[R_M] - R_f)
$$

**Applications**: Cost of equity, expected returns, baseline pricing model.

**Returns**: Expected return.

---

### estimateBeta(assetReturns, marketReturns)

**Purpose**: Estimate beta coefficient and alpha.

**Model**:

$$
R_i = \alpha + \beta R_M + \epsilon
$$

**Applications**: CAPM estimation, risk measurement, portfolio construction.

**Returns**: Dict with beta, alpha, rSquared.

---

### estimateFactorLoading(assetReturns, factorReturns)

**Purpose**: Estimate factor loadings via OLS.

**Model**:

Multi-factor regression:
$$
R_{i,t} = \alpha_i + \sum_{j=1}^k \beta_{ij} F_{j,t} + \epsilon_{i,t}
$$

OLS estimator:
$$
\hat{\beta} = (F^TF)^{-1}F^TR
$$

where $F$ is matrix of factor returns, $R$ is asset returns.

**Applications**: Multi-factor models, factor exposure analysis.

**Returns**: Dict with loadings, intercept, rSquared.

---

### rollingBeta(assetReturns, marketReturns, window=60)

**Purpose**: Calculate time-varying beta.

**Model**:

For each window $[t-w+1, t]$:
$$
\beta_t = \frac{\text{Cov}(R_{a,t-w+1:t}, R_{m,t-w+1:t})}{\text{Var}(R_{m,t-w+1:t})}
$$

Provides time series of betas:
$$
\{\beta_{w}, \beta_{w+1}, \ldots, \beta_T\}
$$

**Applications**: Dynamic risk, regime changes, conditional CAPM.

**Returns**: Array of rolling betas.

---

### pcaFactors(returns, nFactors=3)

**Purpose**: Extract principal component factors from returns.

**Model**:

Covariance matrix of returns:
$$
\Sigma = \frac{1}{T}\sum_{t=1}^T (r_t - \bar{r})(r_t - \bar{r})^T
$$

Extract top $k$ eigenvectors $W = [w_1, \ldots, w_k]$:
$$
\Sigma w_i = \lambda_i w_i
$$

Factor time series:
$$
F_t = W^T r_t
$$

**Applications**: Statistical factor models, dimension reduction, factor construction.

**Returns**: Dict with factors, loadings, explainedVariance.

---

### factorMimicking(assetReturns, characteristicData, nPortfolios=5)

**Purpose**: Create factor-mimicking portfolios (e.g., SMB, HML).

**Model**:

Sort assets by characteristic into $n$ portfolios. Long-short factor:
$$
F_t = \frac{1}{|H|}\sum_{i \in H} R_{i,t} - \frac{1}{|L|}\sum_{i \in L} R_{i,t}
$$

where $H$ is high characteristic portfolio, $L$ is low characteristic portfolio.

For SMB (size): long small cap, short large cap.
For HML (value): long high B/M, short low B/M.

**Applications**: Factor construction, custom factors, anomaly testing.

**Returns**: Dict with factorReturns.

---

### jensenAlpha(assetReturns, marketReturns, riskFree=0.0)

**Purpose**: Calculate Jensen's alpha (risk-adjusted excess return).

**Model**:

$$
\alpha = \bar{R}_i - [R_f + \beta(\bar{R}_M - R_f)]
$$

**Applications**: Performance evaluation, manager skill assessment.

**Returns**: Jensen's alpha.

---

### treynorMazuy(assetReturns, marketReturns, riskFree=0.0)

**Purpose**: Market timing model with quadratic term.

**Model**:

$$
R_i - R_f = \alpha + \beta(R_M - R_f) + \gamma(R_M - R_f)^2 + \epsilon
$$

**Applications**: Market timing evaluation, convexity in returns.

**Returns**: Dict with alpha, beta, gamma.

---

### informationRatio(assetReturns, benchmarkReturns)

**Purpose**: Information ratio (active return / tracking error).

**Model**:

$$
IR = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)}
$$

**Applications**: Active management, manager evaluation.

**Returns**: Information ratio.

---

### multifactor(assetReturns, factorReturns, riskFree=0.0)

**Purpose**: Multi-factor regression model.

**Applications**: Factor decomposition, attribution analysis, risk models.

**Returns**: Dict with alpha, betas, rSquared, residuals.

---

## fit.py

### fitGbm(prices, dt=1/252)

**Purpose**: Fit GBM parameters to price data.

**Model**: Maximum likelihood estimation for $\mu$ and $\sigma$.

**Applications**: Stock price modeling, Monte Carlo inputs.

**Returns**: Dict with mu (drift), sigma (volatility).

---

### fitOu(spread, dt=1/252)

**Purpose**: Fit OU process to mean-reverting data.

**Model**: Least squares estimation of OU parameters.

**Applications**: Pairs trading, interest rates, commodities.

**Returns**: Dict with theta, mu, sigma, halfLife.

---

### fitLevyOu(spread, jumpDetectionThreshold=0.4, dt=1/252)

**Purpose**: Fit OU process with jumps.

**Model**: Separate jump detection and OU parameter estimation.

**Applications**: Commodities with shocks, electricity prices, regime shifts.

**Returns**: Dict with theta, mu, sigma, halfLife, jumpLambda, jumpMu, jumpSigma.

---

### fitAr1(series)

**Purpose**: Fit AR(1) model.

**Model**:

$$
X_t = c + \phi X_{t-1} + \epsilon_t
$$

**Applications**: Time series modeling, forecasting, discrete mean reversion.

**Returns**: Dict with phi, intercept, sigma2.

---

### fitArma(series, p=1, q=1, maxIter=100)

**Purpose**: Fit ARMA model.

**Applications**: Time series forecasting, residual modeling.

**Returns**: Dict with arCoefs, maCoefs, sigma2.

---

### fitGarch(series, p=1, q=1, maxIter=50)

**Purpose**: Fit GARCH model to volatility clustering.

**Model**:

$$
\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2
$$

**Applications**: Volatility forecasting, option pricing, risk management.

**Returns**: Dict with params (omega, alphas, betas), aic, bic.

---

### fitHeston(prices, dt=1/252)

**Purpose**: Calibrate Heston stochastic volatility model.

**Applications**: Option pricing, volatility surface modeling.

**Returns**: Dict with mu, kappa, theta, sigmaV, rho, v0.

---

### fitCir(rates, dt=1/252)

**Purpose**: Fit CIR interest rate model.

**Applications**: Interest rate modeling, bond pricing.

**Returns**: Dict with kappa, theta, sigma.

---

### fitVasicek(rates, dt=1/252)

**Purpose**: Fit Vasicek interest rate model.

**Applications**: Interest rate modeling, analytical solutions.

**Returns**: Dict with kappa, theta, sigma.

---

### fitJumpDiffusion(prices, dt=1/252, threshold=3.0)

**Purpose**: Fit jump-diffusion model by separating jumps from diffusion.

**Applications**: Crash risk, discontinuous price movements.

**Returns**: Dict with mu, sigma, jumpLambda, jumpMu, jumpSigma.

---

### fitCopula(data1, data2, copulaType='gaussian')

**Purpose**: Fit Gaussian copula to bivariate data.

**Model**: Transform to uniform margins, estimate correlation.

**Applications**: Dependency modeling, portfolio risk, credit risk.

**Returns**: Dict with rho (correlation parameter).

---

### fitDistributions(data, distributions=None)

**Purpose**: Fit multiple distributions and rank by AIC.

**Applications**: Distribution selection, model comparison.

**Returns**: List of (distName, params, aic) sorted by AIC.

---

### aic(logLikelihood, nParams)

**Purpose**: Calculate Akaike Information Criterion.

**Model**:

$$
AIC = 2k - 2\ln(L)
$$

**Applications**: Model selection, penalized likelihood.

**Returns**: AIC value.

---

### bic(logLikelihood, nParams, nObs)

**Purpose**: Calculate Bayesian Information Criterion.

**Model**:

$$
BIC = k\ln(n) - 2\ln(L)
$$

**Applications**: Model selection, stronger penalty than AIC.

**Returns**: BIC value.