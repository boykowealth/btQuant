# sipData — Package Blueprint
### SIP Global — Systematic Index Partners
*API client layer for proprietary commodity market data*

| | |
|---|---|
| **Role in stack** | Fetch → validate → return sipQuant-ready schema objects |
| **Depends on** | sipQuant (schema.py types only) |
| **Consumed by** | sipQuant, internal analyst workflows |
| **Auth** | API key — `SIP_DATA_KEY` environment variable |
| **Output** | sipQuant schema objects — never raw dicts, never dataframes |
| **Version** | 1.0 (proposed) |

---

## Design Philosophy

sipData has one job: get SIP's proprietary data into the hands of sipQuant functions with zero friction. It does not do mathematics. It does not do pricing. It does not transform outputs. Every function returns a sipQuant schema object directly, so the consumer passes the result straight into sipQuant without any intermediate handling.

Three principles govern every design decision:

**API keys never appear in code.** The key is loaded from the environment or a credential file. It is never passed as a function argument, never hardcoded, never logged. A notebook that calls `sd.prices.spot("alberta_hay")` contains no credentials anywhere.

**The boundary is the schema.** `sipData.prices.spot()` returns a `PriceSeries`. `sipData.trades.recent()` returns a list of `TradeRecord`. These are the exact types sipQuant functions accept. No conversion, no guessing field names, no mismatched column orders.

**Fail loudly at the boundary.** If the API returns data that fails sipQuant schema validation, sipData raises a clear error before the bad data reaches any pricing function. The error message names the field, the value, and the validation rule that failed.

---

## Package Structure

```
sipData/
  __init__.py         # configure(), version, top-level imports
  auth.py             # Key management, token storage, permission tiers
  client.py           # Base HTTP client — retry, rate limit, circuit breaker
  cache.py            # Local TTL cache for unchanged series
  stream.py           # WebSocket client for real-time quote streaming
  endpoints/
    __init__.py
    prices.py         # Spot, historical, forward curves
    trades.py         # Physical trade records
    quotes.py         # Broker quote sheets
    indices.py        # Published SIP index values and methodology
    markets.py        # Available markets, grades, delivery points
    reference.py      # Grade specs, adjustment matrices, transport rates
  exceptions.py       # Typed exception hierarchy
  utils.py            # Date helpers, pagination, response parsing
```

---

## Module Reference

### `__init__.py` — top-level configuration

| Function | What it does | Notes |
|---|---|---|
| `configure(key=None, env_var="SIP_DATA_KEY", cache_dir=None, timeout=30)` | Initialise the sipData session. Loads API key from environment if `key` not provided. Sets cache directory and request timeout. | Call once at the top of any script or notebook. Raises `AuthError` if no key is found. |
| `whoami()` | Returns dict of authenticated user info: key tier, permissions, rate limits, expiry. | Use to verify the key is valid and check which endpoints are accessible. |
| `markets()` | Returns list of all markets accessible under the current API key. | Each market includes available grades, delivery points, and data history depth. |
| `status()` | Returns API health status — uptime, latency, any degraded endpoints. | Call before a large data pull to check service availability. |

---

### `auth.py` — authentication and key management

| Function | What it does | SIP application |
|---|---|---|
| `load_key(env_var, path)` | Load API key from environment variable or credential file. Returns key string. Never logs or prints the key. | Internal — called by `configure()`. Supports both `SIP_DATA_KEY` env var and `~/.sip/credentials` file. |
| `validate_key(key)` | Send a lightweight validation request to confirm the key is active and not expired. Returns permission tier dict. | Called automatically on `configure()`. Raises `AuthError` with reason if key is invalid. |
| `permission_tier(key)` | Return the permission tier for the key: `read_data`, `read_indices`, `execute_trade`. | sipData requires `read_data` minimum. Some index endpoints require `read_indices`. |
| `rotate_key(old_key, new_key)` | Swap to a new API key, invalidate the old one, and persist the new key to credential file. | Key rotation without downtime. The old key remains valid for 60 seconds after rotation. |
| `KeyStore` class | Secure in-memory key store. Holds the active key for the session. Never serialises to disk except via `rotate_key()`. | All `client.py` requests pull the key from `KeyStore`, not from user-facing variables. |

---

### `client.py` — base HTTP client

All endpoint functions call through `client.py`. Users never interact with this module directly.

| Function / Class | What it does | Notes |
|---|---|---|
| `SIPClient(base_url, key_store, timeout, max_retries)` | Base HTTP client. Attaches auth header, handles serialisation, manages connection pooling. | One client instance per session, created by `configure()`. |
| `get(endpoint, params)` | Authenticated GET request. Returns parsed response dict. | Raises typed exceptions on 4xx/5xx. |
| `post(endpoint, body)` | Authenticated POST request. Used for bulk data requests. | Payload size is validated before sending. |
| `_retry(fn, max_retries, backoff)` | Exponential backoff retry wrapper. Retries on 429 (rate limit) and 5xx (server error). Never retries 4xx auth errors. | Retry budget: 3 attempts, 1s/2s/4s backoff. |
| `_rate_limit(tier)` | Token bucket rate limiter. Enforces per-key request rate based on permission tier. | Blocks (not drops) excess requests. Prevents API key suspension from accidental tight loops. |
| `CircuitBreaker` class | Tracks consecutive failures. Opens the circuit after 5 failures, blocks requests for 30s, then half-opens to test recovery. | Prevents hammering a degraded API endpoint. Raises `ServiceUnavailableError` when open. |

---

### `cache.py` — local TTL cache

| Function | What it does | SIP application |
|---|---|---|
| `Cache(directory, default_ttl)` | Initialise a local file-based cache. Stores serialised schema objects keyed by endpoint + params hash. | Default TTL: 3600s for historical series, 300s for spot prices, 60s for index values. |
| `get(key)` | Return cached schema object if present and not expired. Returns `None` on miss. | Transparent to endpoint callers — cache is checked automatically before any HTTP request. |
| `set(key, value, ttl)` | Store a schema object in cache with explicit TTL. | Called automatically after every successful API response. |
| `invalidate(pattern)` | Invalidate all cache entries matching a pattern (e.g. all entries for `alberta_hay`). | Call when you know data has changed and want fresh pulls. |
| `clear()` | Wipe the entire cache. | Use when switching API keys or environments. |
| `offline_mode(enabled)` | When enabled, raise `CacheOnlyError` instead of making HTTP requests. Returns cached data only. | Useful for analysts working on planes or with rate-limited keys. Requires prior data pull to have populated the cache. |

---

### `stream.py` — real-time streaming

| Function | What it does | SIP application |
|---|---|---|
| `subscribe(markets, grades, callback)` | Open a WebSocket connection and subscribe to real-time quote updates for the specified markets and grades. Calls `callback(QuoteSheet)` on each update. | Real-time price feed for live OTC quoting. The callback receives a sipQuant `QuoteSheet` schema object directly. |
| `unsubscribe(markets)` | Remove subscriptions for specified markets. | Reduces bandwidth when certain markets are no longer needed. |
| `close()` | Close the WebSocket connection cleanly. | Call on session teardown. Automatically called on program exit. |
| `reconnect(backoff)` | Re-establish dropped WebSocket connection with exponential backoff. | Called automatically on connection drop. Replays any missed updates if the server supports it. |

---

### `endpoints/prices.py` — price data

| Function | Returns | SIP application |
|---|---|---|
| `spot(market, grade, date=None)` | `PriceSeries` (single observation or latest) | Current spot price for a specific market and grade. The most common single call in any sipQuant workflow. |
| `historical(market, grade, start, end, frequency="daily")` | `PriceSeries` | Full historical price series for model calibration. `frequency` supports `"daily"`, `"weekly"`, `"monthly"`. Sparse markets will have gaps — returned as `SparsePriceSeries`. |
| `forward_curve(market, grade, base_date=None)` | `ForwardCurve` | Full forward curve at current date or a specified historical date. Ready to pass directly into `sipQuant.otc.commoditySwap()` or `sipQuant.commodity.convenienceYield()`. |
| `basis(local_market, benchmark, start, end)` | `SparsePriceSeries` | Historical basis between a local delivery point and a liquid benchmark. Basis values, not raw prices. |
| `proxy_series(market, proxies, start, end)` | `dict[str, PriceSeries]` | Fetch the target market price series plus all specified proxy series in a single call. Ready for `sipQuant.econometrics.ols()` proxy regression. |
| `vol_surface(market, grade, base_date=None)` | `dict` containing tenor/strike grid and implied vols | Implied vol surface for options pricing. Ready to pass into `sipQuant.bootstrap.volSurface()`. |
| `convenience_yield_history(market, grade, start, end)` | `SparsePriceSeries` | Historical convenience yield series backed out from SIP's forward curve archive. |

---

### `endpoints/trades.py` — physical trade records

| Function | Returns | SIP application |
|---|---|---|
| `recent(market, grade, days=30)` | `list[TradeRecord]` | Most recent physical trade observations. Primary input to index calculation and proxy regression. |
| `historical(market, grade, start, end)` | `list[TradeRecord]` | Full trade history for a market. Used to build price series where no continuous quote exists. |
| `by_delivery_point(origin, destination, start, end)` | `list[TradeRecord]` | Trade records filtered by specific origin/destination pair. Isolates basis for a transport corridor. |
| `by_grade(market, grade_spec, start, end)` | `list[TradeRecord]` | Trades filtered to a specific quality specification. Used to calibrate grade adjustment matrices. |
| `volume_profile(market, start, end, frequency="monthly")` | `dict` | Aggregated volume by period. Input to `sipQuant.index.weightConstituents()` for volume-weighted index construction. |

---

### `endpoints/quotes.py` — broker quote sheets

| Function | Returns | SIP application |
|---|---|---|
| `latest(market, grade)` | `QuoteSheet` | Most recent broker quote for a market/grade. Bid, ask, mid, and source. |
| `historical(market, grade, start, end)` | `list[QuoteSheet]` | Historical broker quote archive. Used to build vol surfaces and calibrate spread models. |
| `consensus(market, grade, date=None)` | `QuoteSheet` | Aggregated consensus quote across all broker sources, weighted by recency and source reliability. Mid is the primary SIP mark-to-model input when no trade has occurred. |
| `by_tenor(market, grade, tenors)` | `list[QuoteSheet]` | Quotes across multiple tenors for a single market/grade. Required for forward curve construction. |

---

### `endpoints/indices.py` — SIP published indices

| Function | Returns | SIP application |
|---|---|---|
| `current(index_id)` | `dict` with value, date, methodology version | Latest published index value. The float leg reference rate for OTC swap settlement. |
| `historical(index_id, start, end)` | `SparsePriceSeries` | Historical index value series. Used to calibrate `sipQuant.index.indexVol()` and backtest OTC structures. |
| `methodology(index_id, version=None)` | `IndexSpec` | Full index methodology specification. Pinned to a version — if `version=None`, returns current. Pass into `sipQuant.index.calculateIndex()` for replication. |
| `constituents(index_id, date)` | `dict` | Index constituent weights and contributions for a given date. Matches the output of `sipQuant.index.constituentsReport()`. |
| `settlement_history(index_id, start, end)` | `list[dict]` | All historical settlement dates, values, and audit references. Used by counterparties to verify settlement prices. |
| `audit_record(index_id, date)` | `dict` | Full audit record for a specific calculation date — all inputs, weights, methodology version, and output. Corresponds to `sipQuant.index.auditLog()` entries. |

---

### `endpoints/markets.py` — market reference data

| Function | Returns | SIP application |
|---|---|---|
| `list_markets()` | `list[dict]` | All markets available under the current API key with metadata: commodity type, region, available grades, history depth. |
| `delivery_points(market)` | `list[dict]` | All valid delivery points for a market with GPS coordinates, storage capacity, and typical transport rates. |
| `available_grades(market)` | `list[dict]` | All grade specifications for a market with quality parameters (moisture, protein, test weight, etc.). |
| `trading_calendar(market, year)` | `dict` | Active trading periods, typical harvest windows, and historical seasonality peaks. Input to `sipQuant.commodity.seasonalIndex()`. |

---

### `endpoints/reference.py` — static reference data

| Function | Returns | SIP application |
|---|---|---|
| `grade_adj_matrix(market)` | `numpy.ndarray` | Grade adjustment matrix for a market — price differentials between quality specifications. Ready to pass into `sipQuant.commodity.gradeAdj()`. |
| `transport_rates(origin, destination)` | `dict` | Current and historical transport rates between delivery points. Input to `sipQuant.commodity.transportBasis()`. |
| `proxy_map(market)` | `dict` | SIP's recommended liquid proxy instruments for each thin market, with historical beta estimates. Input to `sipQuant.book.hedgeRatio()`. |
| `cost_of_production(market, region, year)` | `float` | SIP's estimate of cost of production floor for a market/region. Input to `sipQuant.commodity.physicalPremium()`. |

---

### `exceptions.py` — exception hierarchy

| Exception | Raised when |
|---|---|
| `SIPDataError` | Base class for all sipData exceptions. |
| `AuthError` | API key missing, invalid, expired, or insufficient permissions for the endpoint. |
| `RateLimitError` | Request rate exceeds key tier limit. Includes `retry_after` seconds attribute. |
| `ServiceUnavailableError` | Circuit breaker is open — API endpoint is currently failing. |
| `ValidationError` | API response fails sipQuant schema validation. Includes field name and failing value. |
| `CacheOnlyError` | `offline_mode` is enabled but the requested data is not in cache. |
| `MarketNotFoundError` | Requested market/grade combination does not exist or is not accessible under the current key. |
| `DataGapError` | Requested date range has gaps exceeding the allowable threshold for the requested frequency. |

---

## Authentication Flow

```
User calls sd.configure()
  └── auth.load_key() reads SIP_DATA_KEY from environment
  └── auth.validate_key() confirms key is active
  └── auth.permission_tier() stores tier in KeyStore
  └── client.SIPClient() is initialised with KeyStore reference

User calls sd.prices.spot("alberta_hay_premium")
  └── client.get() attaches key from KeyStore to request header
  └── cache.get() checks local cache first
      ├── Cache hit → return cached PriceSeries immediately
      └── Cache miss → HTTP GET to SIP API
          └── Response parsed and validated against PriceSeries schema
          └── cache.set() stores result with TTL
          └── PriceSeries returned to caller
```

The API key is attached to the request header as `X-SIP-API-Key: {key}`. It never appears in the URL, never in query parameters, never in the response body.

---

## Two Key Tiers

| Tier | Environment variable | Permissions | Used by |
|---|---|---|---|
| `read_data` | `SIP_DATA_KEY` | prices, trades, quotes, reference data | sipData |
| `read_indices` | `SIP_DATA_KEY` (elevated) | all of the above + index values, audit records, methodology | sipData (index endpoints) |
| `execute_trade` | `SIP_TRADE_KEY` | all of the above + order submission | sipTrade |

A user can hold `read_data` without `execute_trade`. An analyst workflow using only sipData and sipQuant never needs a trade key.

---

## Example Workflows

### Proxy regression pricing

```python
import sipData as sd
import sipQuant as sq

sd.configure()

# Fetch target + proxies in one call
series = sd.prices.proxy_series(
    market="alberta_hay_premium",
    proxies=["cme_corn_nearby", "ice_canola_nearby"],
    start="2022-01-01", end="2024-12-31"
)

# Pass directly into sipQuant — no conversion needed
result = sq.econometrics.robustOls(
    X=series["proxies"],
    y=series["alberta_hay_premium"],
    method="huber"
)
```

### OTC swap pricing

```python
sd.configure()

prices = sd.prices.historical("alberta_hay_premium", "premium", start="2022-01-01", end="2024-12-31")
curve  = sd.prices.forward_curve("alberta_hay_premium", "premium")

conv_yield = sq.commodity.convenienceYield(
    spot=prices.values[-1],
    forward=curve.prices[0],
    r=0.045, T=0.25
)

swap = sq.otc.commoditySwap(
    fixed_price=142.0,
    index_curve=curve,
    notional=500,
    schedule=sq.schedule.monthly(3),
    r=0.045
)
```

### Real-time quoting

```python
sd.configure()

def on_quote(quote_sheet):
    swap = sq.otc.commoditySwap(
        fixed_price=quote_sheet.mid,
        index_curve=cached_curve,
        notional=500,
        schedule=schedule,
        r=r
    )
    print(f"Mid: {quote_sheet.mid:.2f}  Swap fair value: {swap['price']:.2f}")

sd.stream.subscribe(
    markets=["alberta_hay_premium"],
    grades=["premium"],
    callback=on_quote
)
```

---

*sipData v1.0 — SIP Global*
