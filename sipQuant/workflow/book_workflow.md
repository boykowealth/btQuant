# SIP Global — OTC Dealer Book Workflow

## Overview

This document describes how SIP Global's trading desk manages its OTC dealer book
for physically-settled agricultural commodity positions using the `sipQuant.book`
module. The workflow covers daily net Greeks monitoring, pre-trade scenario analysis,
end-of-day P&L attribution, and margin reporting.

---

## 1. Book Composition

The SIP dealer book holds OTC positions across Alberta hay, Saskatchewan straw,
Manitoba oat, and related soft-commodity markets. Instruments include:

- Commodity swaps (fixed-for-floating on weekly index prices)
- Physical forwards (outright price commitments)
- Collars (structured buy-side protection)

Each position is stored as an `OTCPosition` dict from `sipQuant.schema`.

---

## 2. Daily Net Greeks Monitoring

Every morning at open, the risk desk calls `book.netGreeks(positions)` across the
full live position set. Output delta, gamma, vega, theta, and rho are logged to the
risk dashboard.

Alert thresholds (indicative):
- Net delta > ±500 tonnes equivalent → escalate to head trader
- Net vega > ±$50,000 → review vol exposure

Greeks are signed using the convention: buy/long/receive_fixed = +1,
sell/short/pay_fixed = −1.

---

## 3. Delta Hedging

When net delta breaches threshold, the desk uses `book.hedgeRatios(positions, hedgeInstrumentDelta)`
to calculate the number of futures contracts (or physical forward offsets) required
to flatten the delta. The `residualDelta` output confirms post-hedge exposure.

---

## 4. Scenario Shocks for Agricultural Positions

Before each monthly WASDE release (USDA World Agricultural Supply and Demand
Estimates), the risk team runs `book.scenarioShock(positions, scenarios)` with
standard shocks calibrated to historical WASDE surprise magnitudes:

- Bearish: priceShock = −$15/t, volShock = +0.03
- Neutral: priceShock = $0, volShock = 0
- Bullish: priceShock = +$15/t, volShock = −0.02

Delta and vega contributions are reviewed against book limits before the release.

---

## 5. End-of-Day P&L Attribution

`book.pnlAttribution(positions, priceMoves, volMoves, timeDecay)` decomposes
daily P&L into delta, gamma, vega, and theta components. The output feeds
directly into the daily risk report emailed to senior management.

Key field: `totalByPosition` — allows identification of concentrated P&L drivers.

---

## 6. Margin Estimation

`book.marginEstimate(positions)` provides a rapid intraday margin estimate
using notional, delta, and theta. This is an internal working estimate;
formal margin calls are governed by ISDA Credit Support Annexes.

---

## 7. Book Summary for Reporting

`book.bookSummary(positions)` is used in weekly risk committee decks to show:
- Total notional by instrument type
- Net Greeks snapshot
- Concentration risk (largest single position as fraction of book)
