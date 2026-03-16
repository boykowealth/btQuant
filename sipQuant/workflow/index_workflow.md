# SIP Global — IOSCO-Aligned Commodity Index Calculation Workflow

## Overview

This document describes the end-to-end pipeline SIP Global follows to calculate,
audit, and maintain its suite of physically-settled commodity price indices.
All calculations use `sipQuant.index` and adhere to the IOSCO Principles for
Financial Benchmarks (2013, revised 2021).

---

## 1. SIP Index Products

SIP Global currently publishes the following indices:

| Index Code   | Description                                    | Weighting  |
|------------- |------------------------------------------------|------------|
| SIP-AHI-001  | Alberta Hay Index — premium bale, 14% moisture | Equal      |
| SIP-AHI-002  | Alberta Hay Index — feed grade                 | Volume     |
| SIP-SKS-001  | Saskatchewan Straw Index — wheat straw         | Equal      |
| SIP-MBO-001  | Manitoba Oat Index — milling grade             | Volume     |
| SIP-BCF-001  | BC Forage Composite Index (multi-grade)        | Liquidity  |

Each index has a versioned `IndexSpec` dict stored in the methodology registry.

---

## 2. Trade Data Ingestion

Physical trade reports are received daily from registered reporting entities.
Each submission is validated against `sipQuant.schema.TradeRecord` before
entering the calculation pipeline:

- Price must be positive
- Volume must be positive
- Grade, origin, destination, and counterpartyId are mandatory

Rejected submissions are flagged for analyst review and excluded from that
day's calculation.

---

## 3. Index Calculation

For each index on each calculation date, the desk runs:

```python
result = sq.index.calculateIndex(tradeRecords, indexSpec, calculationDate)
```

Steps executed internally:

1. Filter trades to those on or before `calculationDate` whose `grade` matches
   a constituent in `indexSpec['constituents']`.
2. Compute VWAP per constituent across filtered trades.
3. Apply constituent weighting per `indexSpec['weightsMethod']`.
4. Return weighted sum as the index value.

Minimum trade count: if fewer than 3 trades exist for a constituent, a proxy
value is estimated using `index.proxyRegression` against a correlated series
before the weighted sum is computed.

---

## 4. Audit Trail Generation

Immediately after calculation, an IOSCO-aligned audit record is created:

```python
audit = sq.index.auditTrail(result, indexSpec)
```

The audit record captures:
- Timestamp, index name, methodology version
- Constituent-level VWAP and weight detail
- Data sources used
- Checksum for integrity verification

All audit records are archived in the methodology vault with a minimum
retention period of 7 years, per IOSCO Principle 15.

---

## 5. Proxy Construction for Sparse Markets

When a constituent market has insufficient trade activity, `index.proxyRegression`
constructs a proxy estimate using OLS or Huber regression against a correlated
liquid series (e.g. CME Corn futures for grain-based indices):

```python
proxy = sq.index.proxyRegression(targetSeries, proxySeries, method='huber')
```

The proxy is clearly flagged in the audit record as estimated, not observed.

---

## 6. Restatement Workflow

If an error is identified post-publication (e.g. a trade is found to be
non-arm's-length and must be excluded), the restatement procedure is:

1. Analyst documents the reason and corrected value.
2. `index.restatement(originalRecord, correctedValue, reason, analystId)` is
   called and the result is appended to the restatement log.
3. The correction is disclosed to subscribers within 1 business day.
4. The restatement log is reviewed quarterly by the Index Oversight Committee.

---

## 7. Roll Schedule Management

Index roll dates are generated using `index.rollSchedule(indexSpec, startDate,
endDate, step='monthly')`. Monthly rolls are the default for all current SIP
indices. The roll schedule is published to subscribers at the start of each
calendar year.

---

## 8. Backtesting and Methodology Review

`index.backtestIndex(tradeRecords, indexSpec, dates)` is run annually during
the methodology review cycle to assess:
- Index volatility and return characteristics
- Maximum drawdown in stress periods
- Stability of proxy contributions

Results are presented to the Index Oversight Committee for sign-off.
