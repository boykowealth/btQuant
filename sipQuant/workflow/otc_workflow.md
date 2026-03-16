# sipQuant otc — SIP Global Workflow Guide

## Purpose

The `otc` module prices physically-settled OTC commodity instruments. All functions
return price + Greeks dicts and integrate directly with `book.py` for portfolio
aggregation and `schema.OTCPosition` for position tracking.

---

## Function Reference

| Function | Instrument |
|---|---|
| `commoditySwap` | Fixed-float physical swap (most common SIP deal) |
| `asianSwap` | Asian (average-price) swap |
| `collar` | Long cap + short floor (zero-cost collar) |
| `physicalForward` | Outright physical forward contract |
| `swaption` | Option on a commodity swap |
| `asianOption` | Monte Carlo Asian option (arithmetic average) |

---

## Step-by-Step: Full OTC Deal Workflow

### 1. Build inputs from schema objects

```python
import sipQuant as sq
import numpy as np

# Forward curve from commodity module
fwd = sq.commodity.localForwardCurve(
    spotPrice=187.50, tenor=np.array([0.25, 0.5, 0.75, 1.0]),
    r=0.046, convYield=0.028, storageCost=0.02
)
schedule    = fwd['tenors']        # payment dates in years
indexCurve  = fwd['forwards']      # float index prices
r           = 0.046
notional    = 500.0                # tonnes
fixedPrice  = 190.0
```

### 2. Price a commodity swap

```python
swap = sq.otc.commoditySwap(
    fixedPrice = fixedPrice,
    indexCurve = indexCurve,
    notional   = notional,
    schedule   = schedule,
    r          = r
)

print(f"NPV:            {swap['npv']:+.2f}")
print(f"Fixed leg PV:   {swap['fixedLegPV']:.2f}")
print(f"Float leg PV:   {swap['floatLegPV']:.2f}")
print(f"DV01:           {swap['greeks']['dv01']:.4f}")
```

### 3. Price an Asian swap (monthly average settlement)

```python
# Monthly average of 22 price observations
monthly_prices = np.array([187.5, 188.2, 189.1, 190.0, 191.2,
                             190.8, 189.5, 188.7, 187.9, 188.4,
                             189.0, 190.5, 191.1, 192.0, 191.5,
                             190.2, 189.7, 188.9, 189.3, 190.1,
                             190.8, 191.3])

asian_swap = sq.otc.asianSwap(
    fixedPrice  = 190.0,
    indexPrices = monthly_prices,
    notional    = 500.0,
    r           = 0.046,
    T           = 1/12    # 1 month
)
print(f"Asian swap NPV: {asian_swap['npv']:+.2f}")
print(f"Average index:  {asian_swap['averageIndex']:.2f}")
```

### 4. Price a collar (zero-cost or near-zero-cost)

```python
# Hedge a long physical position: buy cap, sell floor
collar = sq.otc.collar(
    S          = 187.50,
    capStrike  = 200.0,    # buy call at 200
    floorStrike= 175.0,    # sell put at 175
    T          = 0.5,
    r          = 0.046,
    sigma      = 0.18,
    notional   = 500.0
)

print(f"Net collar premium: {collar['price']:+.4f}")
print(f"Cap cost:           {collar['capPrice']:.4f}")
print(f"Floor credit:       {collar['floorPrice']:.4f}")
print(f"Net delta:          {collar['greeks']['delta']:.4f}")
```

### 5. Record position in book

```python
pos = sq.schema.OTCPosition(
    instrumentType = 'commodity_swap',
    direction      = 'pay_fixed',
    notional       = notional,
    strikeOrFixed  = fixedPrice,
    expiry         = '2026-12-31',
    counterpartyId = 'CP_ANON_004',
    greeks = {
        'delta': swap['greeks']['delta'],
        'gamma': 0.0,
        'vega' : 0.0,
        'theta': 0.0,
        'rho'  : swap['greeks']['dv01'],
    }
)
errors = sq.schema.validate(pos)
assert errors == [], errors
```

### 6. Greeks and hedge sizing

```python
# Net book Greeks after adding the swap
book_greeks = sq.book.netGreeks([pos])
hedge_units = sq.book.hedgeRatios([pos], hedgeInstrumentDelta=1.0)
print(f"Net delta:   {book_greeks['delta']:.2f}")
print(f"Hedge units: {hedge_units['hedgeUnits']:.2f}")
```

### 7. Swaption for embedded optionality

```python
swpn = sq.otc.swaption(
    fixedPrice = 190.0,
    indexCurve = indexCurve,
    notional   = notional,
    schedule   = schedule,
    r          = 0.046,
    sigma      = 0.18,
    T          = 0.25,     # option expiry = 3 months
    optType    = 'call'
)
print(f"Swaption price:          {swpn['price']:.4f}")
print(f"Forward swap rate:       {swpn['forwardSwapRate']:.4f}")
```

---

## Adapting to Any Cluster

Replace `indexCurve` (from `commodity.localForwardCurve`), `fixedPrice`, and
`notional` with the target cluster's values. All OTC pricing functions are
commodity-agnostic — the same code prices Alberta Hay swaps, Western Canadian
Wheat collars, or Prairie Canola Asian options identically.
