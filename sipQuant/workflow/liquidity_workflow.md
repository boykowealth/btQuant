# SIP Global — Liquidity Risk Management Workflow (Thin Commodity Markets)

## Overview

Agricultural commodity markets served by SIP Global are characterised by
episodic trading, wide bid-ask spreads, and high participant concentration.
This document describes how the risk desk uses `sipQuant.liquidity` to monitor
and manage liquidity risk across the dealer book and index eligibility pipeline.

---

## 1. Why Liquidity Risk Matters for SIP Markets

Thin markets create three distinct risks for SIP Global:

1. **Execution risk** — large positions cannot be unwound quickly without
   moving the market.
2. **Index reliability risk** — insufficient trade activity degrades VWAP
   quality, forcing reliance on proxy estimates.
3. **Margin risk** — illiquidity can cause sudden spread widening, inflating
   mark-to-market losses.

---

## 2. Liquidity-Adjusted VaR (LVAR)

Daily VaR is computed with a liquidity adjustment per Bangia et al.:

```python
result = sq.liquidity.liquidityAdjustedVar(returns, volumes, alpha=0.05)
```

The spread cost is either provided explicitly (when broker quotes are available)
or estimated from volume relative to the rolling mean. The liquidity cost
(0.5 × spreadCost) is added to the standard historical VaR.

LVAR is reported alongside standard VaR in the daily risk pack to make the
liquidity premium transparent to senior management.

---

## 3. Market Impact Assessment

Before executing any trade exceeding 10% of estimated ADV, the desk runs a
market impact pre-trade check:

```python
impact = sq.liquidity.marketImpact(tradeSize, adv, model='almgren-chriss')
```

The `impactBps` field is compared against the position's expected alpha. Trades
where market impact exceeds 50% of expected alpha are deferred or split.

---

## 4. Optimal Execution for Large Positions

When a position must be unwound over multiple days (e.g. following a counterparty
default or fund redemption), the desk computes the Almgren-Chriss optimal
execution schedule:

```python
exec_plan = sq.liquidity.optimalExecution(
    totalShares=position_size,
    T=liquidation_days,
    adv=market_adv,
    sigma=daily_vol,
    riskAversion=1e-6,
)
```

The `schedule` output (shares per period) is used as the trading instruction.
The `expectedCost` and `expectedVariance` outputs feed into the pre-trade
cost/benefit analysis submitted to the trading desk head.

---

## 5. Thin-Market Scoring for Index Eligibility

Each constituent market is scored monthly using:

```python
score = sq.liquidity.thinMarketScore(tradeRecords, window=30)
```

Scoring criteria for index inclusion:
- Score >= 0.20: eligible constituent
- Score 0.10–0.19: eligible but flagged; proxy supplement required
- Score < 0.10: suspended from index; proxy-only for that period

The score, nTrades, avgVolume, and priceCV are logged in the monthly
liquidity report reviewed by the Index Oversight Committee.

---

## 6. Position Concentration Risk

The HHI-based concentration check is run weekly across the dealer book:

```python
conc = sq.liquidity.concentrationRisk(position_sizes, market_adv_array)
```

Alert thresholds:
- HHI > 0.25: elevated concentration — review required
- participationRate > 0.20 in any single market: potential market impact
  concern — escalate to compliance

---

## 7. Liquidation Cost Estimation

For stress-testing and capital planning, the TWAP/VWAP liquidation cost
estimate is used:

```python
liq = sq.liquidity.optimalLiquidation(position, adv, sigma, timeHorizon)
```

`estimatedSlippage` feeds into the stressed liquidity coverage ratio
calculation reported to the Risk Committee quarterly.
