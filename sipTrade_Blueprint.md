# sipTrade — Package Blueprint
### SIP Global — Systematic Index Partners
*Order submission and execution layer for SIP's OTC commodity markets*

| | |
|---|---|
| **Role in stack** | Receive sipQuant pricing outputs → validate → submit orders to SIP |
| **Depends on** | sipQuant (schema.py types), sipData (market context for pre-trade checks) |
| **Consumed by** | Traders, automated execution workflows |
| **Auth** | API key — `SIP_TRADE_KEY` environment variable (elevated vs sipData) |
| **Input** | sipQuant schema objects + trade intent (direction, notional, counterparty) |
| **Version** | 1.0 (proposed) |

---

## Design Philosophy

sipTrade is the execution terminal. It sits at the end of the `sipData → sipQuant → sipTrade` pipeline and does exactly one thing: submit a well-formed, pre-priced order to SIP and track its status through to confirmation or rejection.

Three principles govern every design decision:

**Pricing lives in sipQuant, not sipTrade.** sipTrade never calculates a price. It receives a sipQuant pricing dict and submits it. This means the price that appears in the order is identical to the price the trader saw on their screen, with a full audit trail of the sipQuant inputs that produced it.

**Every order is self-describing.** A sipTrade order carries its complete specification — instrument type, legs, notional, settlement terms, grade, delivery point, and the sipQuant pricing context that generated the price. SIP receives everything needed to confirm or dispute the trade without a follow-up call.

**Execution keys are separate from data keys.** `SIP_TRADE_KEY` is a distinct credential from `SIP_DATA_KEY`. A user can read data and run analytics without holding execution permissions. The trade key requires explicit issuance and carries stricter rate limits and audit logging on SIP's side.

---

## Package Structure

```
sipTrade/
  __init__.py         # configure(), version, top-level imports
  auth.py             # Trade key management — stricter than sipData auth
  client.py           # Signed HTTP client — orders are request-signed, not just key-authenticated
  orders/
    __init__.py
    physical.py       # Physically-settled forward contracts
    swap.py           # Fixed-float commodity swaps
    collar.py         # Collar structures (long put + short call)
    basis_swap.py     # Basis swap between delivery points or vs index
    flex_forward.py   # Forwards with delivery flexibility windows
    swaption.py       # Options to enter a swap at a future date
  validation.py       # Pre-submission checks — price staleness, notional limits, market hours
  status.py           # Order lifecycle tracking — submitted, confirmed, rejected, settled
  blotter.py          # Session blotter — local record of all orders this session
  settlement.py       # Settlement notifications, cash flow records, delivery confirmations
  exceptions.py       # Typed exception hierarchy for execution errors
  utils.py            # Order ID generation, timestamp helpers, serialisation
```

---

## Module Reference

### `__init__.py` — top-level configuration

| Function | What it does | Notes |
|---|---|---|
| `configure(key=None, env_var="SIP_TRADE_KEY", environment="production")` | Initialise sipTrade session. Loads trade key from environment. `environment` supports `"production"` and `"sandbox"`. | Always test in `"sandbox"` first. Sandbox accepts all orders but never routes them to SIP's execution desk. |
| `ping()` | Confirm the execution endpoint is reachable and the key has trade permissions. Returns latency. | Call before any session where timing matters. A 500ms ping indicates a degraded link. |
| `limits()` | Return current order limits for the key: max notional per order, max daily notional, open order count cap. | Check limits before submitting large orders. Exceeding limits raises `OrderLimitError` before the order is sent. |
| `sandbox_mode(enabled)` | Toggle sandbox mode on an active session without reconfiguring. | Use to test new order types without risk. |

---

### `auth.py` — trade key management

| Function | What it does | SIP application |
|---|---|---|
| `load_trade_key(env_var, path)` | Load trade key from `SIP_TRADE_KEY` environment variable or `~/.sip/trade_credentials` file. | Separate file from the data credentials. Restricts who on a team can accidentally submit orders. |
| `validate_trade_key(key)` | Confirm the key has `execute_trade` permission tier and is not suspended. | Called on `configure()`. Raises `TradeAuthError` if insufficient permissions. |
| `sign_request(payload, key)` | Produce an HMAC-SHA256 signature over the order payload using the trade key. | Every order submission is signed — not just authenticated by header. SIP verifies the signature before routing the order. This prevents replay attacks and payload tampering. |
| `request_key(contact_email)` | Initiate the out-of-band process for requesting a new trade key from SIP. Returns a reference number. | Trade keys are not self-serve — they require manual approval from SIP's operations team. |

---

### `client.py` — signed HTTP client

| Function / Class | What it does | Notes |
|---|---|---|
| `TradeClient(base_url, auth, timeout, environment)` | Signed HTTP client for order submission. Attaches auth header AND request signature to every call. | Stricter than sipData's client — every mutating request is signed. |
| `submit(endpoint, order_payload)` | Submit a signed order payload. Returns `OrderReceipt` with order ID and initial status. | Non-idempotent. sipTrade tracks submitted order IDs locally to prevent duplicate submission on retry. |
| `query(endpoint, params)` | Query order status or settlement records. Read-only — no signature required, key auth is sufficient. | |
| `_sign_and_submit(payload)` | Signs the payload with `auth.sign_request()`, attaches signature header, submits. Never retries a signed submission automatically — duplicate orders are a real risk. | Retries require explicit user confirmation via `resubmit(order_id)`. |
| `IdempotencyGuard` class | Tracks order IDs submitted this session. Raises `DuplicateOrderError` if the same order ID is submitted twice. | Protects against double submission from network timeouts where the first request may have succeeded. |

---

### `validation.py` — pre-submission checks

Every order passes through validation before `client.submit()` is called. Validation failures raise exceptions — the order is never sent.

| Function | What it checks | Raises if failed |
|---|---|---|
| `check_price_staleness(pricing_dict, max_age_seconds=60)` | Confirms the sipQuant pricing dict is not older than `max_age_seconds`. Market prices may have moved since the calculation. | `StalePriceError` — includes the age of the pricing dict and how much the market has moved since. |
| `check_notional_limits(notional, instrument_type)` | Confirms the order notional is within the key's per-order and daily limits. | `OrderLimitError` — includes current limit and daily notional already used. |
| `check_market_hours(market, delivery_point)` | Confirms the order is being submitted within SIP's active trading hours for the market. | `MarketClosedError` — includes next open time. |
| `check_grade_spec(grade_spec, market)` | Confirms the grade specification matches a valid grade for the market. | `InvalidGradeError` — includes valid grades for the market. |
| `check_delivery_point(delivery_point, market)` | Confirms the delivery point is valid and active for the market. | `InvalidDeliveryPointError`. |
| `check_counterparty(counterparty_id)` | Confirms the counterparty ID exists in SIP's registered counterparty list and is active. | `UnknownCounterpartyError`. |
| `check_settlement_terms(settlement_dict, instrument_type)` | Confirms settlement terms are consistent with the instrument type (physical vs cash, schedule, grade at delivery). | `InvalidSettlementTermsError` — includes which field is inconsistent and why. |
| `validate_order(order)` | Run all checks above in sequence. Returns `True` or raises the first failing exception. | Used internally — also callable directly for pre-flight checks without committing to submission. |

---

### `orders/physical.py` — physically-settled forward contracts

| Function | What it does | SIP application |
|---|---|---|
| `build(pricing_dict, notional, direction, delivery_point, grade_spec, delivery_date, counterparty_id)` | Construct a physical forward order from a sipQuant `physicalForward()` pricing dict. Returns an `Order` object ready for submission. | The most basic SIP trade — a fixed-price physical delivery contract for hay, stover, or other commodity. |
| `submit(pricing_dict, notional, direction, delivery_point, grade_spec, delivery_date, counterparty_id)` | Build + validate + submit in one call. Returns `OrderReceipt`. | Shorthand for the common case. Use `build()` then `submit_order()` when you need to inspect the order before sending. |
| `submit_order(order)` | Submit a pre-built `Order` object. Runs validation, then calls `client.submit()`. | Use when you built the order manually or want to log it before sending. |
| `amend(order_id, changes_dict)` | Amend an open physical forward (price, notional, delivery date) before counterparty confirmation. | Only valid while order status is `"pending_confirmation"`. Amendments are re-validated and re-signed. |
| `cancel(order_id, reason)` | Cancel an open order before confirmation. | Only valid while status is `"pending_confirmation"`. Reason is logged on both sides. |

---

### `orders/swap.py` — fixed-float commodity swaps

| Function | What it does | SIP application |
|---|---|---|
| `build(pricing_dict, notional, direction, schedule, counterparty_id, settlement_index_id)` | Construct a commodity swap order from a sipQuant `commoditySwap()` pricing dict. `direction` is `"pay_fixed"` or `"receive_fixed"`. `settlement_index_id` is the SIP index ID the float leg settles against. | SIP's primary OTC product — a buyer of Alberta hay locks in a fixed price, SIP quotes both sides. |
| `submit(pricing_dict, notional, direction, schedule, counterparty_id, settlement_index_id)` | Build + validate + submit. Returns `OrderReceipt`. | |
| `submit_order(order)` | Submit a pre-built swap `Order`. | |
| `build_asian(pricing_dict, notional, direction, schedule, avg_method, counterparty_id)` | Construct an Asian average swap order from a sipQuant `asianSwap()` pricing dict. | Monthly-average settlement — the standard for agricultural commodity swaps where fixing manipulation is a concern. |
| `submit_asian(pricing_dict, notional, direction, schedule, avg_method, counterparty_id)` | Build + validate + submit an Asian swap. Returns `OrderReceipt`. | |

---

### `orders/collar.py` — collar structures

| Function | What it does | SIP application |
|---|---|---|
| `build(pricing_dict, notional, direction, underlying_market, grade_spec, expiry, counterparty_id)` | Construct a collar order (long put + short call) from a sipQuant `collar()` pricing dict. | Physical producers who want a price floor with capped upside. SIP quotes the combined structure. |
| `submit(pricing_dict, notional, direction, underlying_market, grade_spec, expiry, counterparty_id)` | Build + validate + submit. Returns `OrderReceipt`. | |
| `submit_order(order)` | Submit a pre-built collar `Order`. | |

---

### `orders/basis_swap.py` — basis swaps

| Function | What it does | SIP application |
|---|---|---|
| `build(pricing_dict, notional, direction, leg1_market, leg2_market, schedule, counterparty_id)` | Construct a basis swap order from a sipQuant `basisSwap()` pricing dict. One leg is the local basis, the other is fixed. | SIP can hedge or monetise basis risk between Alberta and other delivery points. Counterparties seeking basis certainty on transport corridors. |
| `submit(pricing_dict, notional, direction, leg1_market, leg2_market, schedule, counterparty_id)` | Build + validate + submit. Returns `OrderReceipt`. | |

---

### `orders/flex_forward.py` — flexible delivery forwards

| Function | What it does | SIP application |
|---|---|---|
| `build(pricing_dict, notional, direction, delivery_point, grade_spec, flex_window_start, flex_window_end, counterparty_id)` | Construct a flexible forward order from a sipQuant `flexibleForward()` pricing dict. Buyer selects delivery date within the window. | Physical buyers who need delivery flexibility. Standard models underprice this optionality — sipQuant prices it explicitly. |
| `submit(pricing_dict, notional, direction, delivery_point, grade_spec, flex_window_start, flex_window_end, counterparty_id)` | Build + validate + submit. Returns `OrderReceipt`. | |
| `notify_delivery_date(order_id, elected_date)` | After a flex forward is confirmed, notify SIP of the elected delivery date within the window. | Must be called by the buyer at least `n` business days before elected delivery per the contract terms. |

---

### `orders/swaption.py` — options to enter a swap

| Function | What it does | SIP application |
|---|---|---|
| `build(pricing_dict, notional, direction, expiry, underlying_swap_params, counterparty_id)` | Construct a swaption order from a sipQuant `swaptionPrice()` pricing dict. | Allows counterparties to lock in the right to hedge without committing immediately. SIP quotes as dealer. |
| `submit(pricing_dict, notional, direction, expiry, underlying_swap_params, counterparty_id)` | Build + validate + submit. Returns `OrderReceipt`. | |
| `exercise(order_id, exercise_date)` | Exercise an in-the-money swaption. Creates the underlying swap as a new order. | sipTrade automatically constructs and submits the underlying swap on exercise. |

---

### `status.py` — order lifecycle tracking

| Function | What it does | SIP application |
|---|---|---|
| `track(order_id)` | Return current status of an order. Statuses: `pending_validation`, `submitted`, `pending_confirmation`, `confirmed`, `rejected`, `amended`, `cancelled`, `settled`. | Poll to monitor order progression from submission to confirmation. |
| `wait_for_confirmation(order_id, timeout=300)` | Block until the order reaches `confirmed` or `rejected`, or timeout expires. Returns final status dict. | Synchronous workflow helper — use in scripts where you need confirmation before proceeding. |
| `on_status_change(order_id, callback)` | Register a callback function that fires whenever the order status changes. | Asynchronous workflow — use in live trading sessions where you want event-driven status updates. |
| `all_open(session_only=True)` | Return all open orders (not yet settled or cancelled). If `session_only=True`, returns only orders submitted this session. | Monitor the full open order book. `session_only=False` returns all open orders under the key. |
| `order_detail(order_id)` | Return the full order record including original order, amendments, status history, and counterparty confirmation details. | Complete audit trail for a single order. |

---

### `blotter.py` — session blotter

| Function | What it does | SIP application |
|---|---|---|
| `Blotter` class | Local in-memory record of all orders submitted this session. Updated automatically on every `submit()` call. | Session-level order log. Does not persist between sessions — use `settlement.py` for permanent records. |
| `summary()` | Return a summary of session orders: count by instrument type, total notional by market, status breakdown. | End-of-session review before closing out. |
| `export(path, format="json")` | Write the session blotter to a JSON or CSV file. | Persist the session record for reconciliation. |
| `reconcile(confirmed_orders)` | Compare blotter against confirmed orders received from SIP and flag any discrepancies. | Catch any orders that sipTrade submitted but SIP did not receive, or vice versa. |

---

### `settlement.py` — settlement and delivery

| Function | What it does | SIP application |
|---|---|---|
| `pending_settlements(start, end)` | Return all orders approaching their settlement or delivery date within the date range. | Advance warning for physical delivery logistics. |
| `settlement_notification(order_id)` | Return settlement details for a settled order: settlement price (for swaps, the index value used), cash flow, delivery confirmation (for physical forwards). | Post-trade record. For swap settlements, the index value is the SIP published index from `sipData.indices.current()`. |
| `cash_flow_schedule(order_id)` | Return the full projected cash flow schedule for a swap or collar — all settlement dates and projected amounts based on current forward curve. | Forward-looking cash flow planning for the OTC book. |
| `delivery_confirm(order_id, actual_delivery_date, actual_quantity, actual_grade)` | Record that physical delivery occurred and confirm against contract terms. Flags any grade or quantity discrepancies. | Physical delivery confirmation. Discrepancies between contracted and delivered grade trigger the grade adjustment process. |
| `grade_dispute(order_id, claimed_spec, actual_spec, evidence)` | Initiate a grade dispute on a physical delivery. Creates a case with SIP's operations team. | If delivered hay fails to meet contracted specification, this initiates the formal dispute and price adjustment process. |

---

### `exceptions.py` — exception hierarchy

| Exception | Raised when |
|---|---|
| `SIPTradeError` | Base class for all sipTrade exceptions. |
| `TradeAuthError` | Trade key missing, invalid, expired, or lacks `execute_trade` permission. |
| `StalePriceError` | The sipQuant pricing dict is older than the allowed staleness threshold. Includes age and market move since pricing. |
| `OrderLimitError` | Order notional exceeds per-order or daily limit for the key. |
| `MarketClosedError` | Order submitted outside SIP's active trading hours for the market. Includes next open time. |
| `InvalidGradeError` | Grade specification is not valid for the market. |
| `InvalidDeliveryPointError` | Delivery point is not valid or not active for the market. |
| `UnknownCounterpartyError` | Counterparty ID is not in SIP's registered counterparty list. |
| `InvalidSettlementTermsError` | Settlement terms are inconsistent with the instrument type. |
| `DuplicateOrderError` | The same order ID has already been submitted this session. |
| `OrderRejectedError` | SIP rejected the order. Includes rejection reason from SIP's execution desk. |
| `AmendmentError` | Amendment is not permitted (order already confirmed or cancelled). |
| `ExerciseError` | Swaption exercise failed (expired, already exercised, or out-of-the-money exercise not permitted). |
| `DeliveryDisputeError` | Grade dispute submission failed validation. |

---

## Order Lifecycle

```
User calls st.orders.swap.submit(pricing_dict, ...)
  └── validation.validate_order() runs all pre-submission checks
      ├── StalePriceError if pricing_dict older than 60s
      ├── OrderLimitError if notional exceeds key limits
      ├── MarketClosedError if outside trading hours
      └── passes → Order object constructed with full payload

  └── auth.sign_request() produces HMAC-SHA256 signature over payload
  └── IdempotencyGuard checks order ID not already submitted
  └── client.submit() sends signed order to SIP execution endpoint
  └── OrderReceipt returned: { order_id, status: "submitted", timestamp }

  └── blotter.Blotter records the order locally

SIP processes the order:
  └── status: "pending_confirmation" → counterparty must confirm
  └── status: "confirmed" → trade is live, both sides bound
  └── status: "rejected" → raises OrderRejectedError with reason

At settlement:
  └── settlement.settlement_notification() returns index value used + cash flow
  └── For physical: settlement.delivery_confirm() records actual delivery
```

---

## The Signed Order Payload

Every order carries a complete, self-describing payload. Nothing is assumed from prior calls.

```json
{
  "order_id": "SIP-2024-03-15-AHA-00142",
  "instrument_type": "commodity_swap",
  "direction": "pay_fixed",
  "market": "alberta_hay_premium",
  "grade_spec": "premium_bale_14pct_moisture",
  "notional_tonnes": 500,
  "fixed_price": 142.00,
  "currency": "CAD",
  "settlement_index_id": "SIP-AHI-001",
  "schedule": ["2024-04-30", "2024-05-31", "2024-06-30"],
  "counterparty_id": "BUYER_001",
  "pricing_context": {
    "generated_by": "sipQuant",
    "function": "otc.commoditySwap",
    "timestamp": "2024-03-15T14:32:11Z",
    "sipQuant_version": "2.0.1",
    "inputs": {
      "spot_price": 138.50,
      "forward_curve_date": "2024-03-15",
      "convenience_yield": 0.032,
      "discount_rate": 0.045,
      "cost_of_carry": 0.078
    },
    "greeks": {
      "delta": 1.0,
      "dv01": 0.0012
    }
  },
  "submitted_at": "2024-03-15T14:32:18Z",
  "signature": "sha256=a3f9c2..."
}
```

The `pricing_context` block is the direct output of the sipQuant function that priced the trade. SIP can independently verify the pricing using the same sipQuant function with the same inputs. This is the audit trail.

---

## Two-Package Workflow

```python
import sipData as sd
import sipQuant as sq
import sipTrade as st

# Configure both packages
sd.configure()   # loads SIP_DATA_KEY
st.configure()   # loads SIP_TRADE_KEY

# 1. Fetch data
prices = sd.prices.historical("alberta_hay_premium", "premium",
                               start="2022-01-01", end="2024-12-31")
curve  = sd.prices.forward_curve("alberta_hay_premium", "premium")

# 2. Price the trade in sipQuant
swap = sq.otc.commoditySwap(
    fixed_price=142.0,
    index_curve=curve,
    notional=500,
    schedule=sq.schedule.monthly(3),
    r=0.045
)

# 3. Risk check in sipQuant
book = sq.book.Book()
sq.book.addPosition(book, "commodity_swap", swap, direction="pay_fixed")
risk = sq.book.bookReport(book)
print(f"Net delta: {risk['net_greeks']['delta']:.1f}t")
print(f"Book VaR (95%): CAD {risk['var_95']:.0f}")

# 4. Execute in sipTrade
receipt = st.orders.swap.submit(
    pricing_dict=swap,
    notional=500,
    direction="pay_fixed",
    schedule=sq.schedule.monthly(3),
    counterparty_id="BUYER_001",
    settlement_index_id="SIP-AHI-001"
)
print(f"Order submitted: {receipt['order_id']}")

# 5. Wait for confirmation
confirmed = st.status.wait_for_confirmation(receipt['order_id'], timeout=300)
print(f"Status: {confirmed['status']}")
```

---

## Sandbox Testing

Before any new order type goes to production, test it in sandbox:

```python
st.configure(environment="sandbox")

# All orders behave normally — validation, signing, blotter —
# but SIP's sandbox endpoint returns a "confirmed" receipt
# without routing to the execution desk.

receipt = st.orders.physical.submit(
    pricing_dict=forward,
    notional=100,
    direction="sell",
    delivery_point="calgary_hub",
    grade_spec="premium_bale_14pct_moisture",
    delivery_date="2024-06-15",
    counterparty_id="TEST_BUYER"
)

# receipt.order_id will begin with "SANDBOX-"
# settlement.settlement_notification() works in sandbox too
```

---

## Separation of Concerns Across the Stack

| Concern | Package | Module |
|---|---|---|
| API key for data | sipData | `auth.py` — `SIP_DATA_KEY` |
| API key for trading | sipTrade | `auth.py` — `SIP_TRADE_KEY` |
| Data validation | sipData | `exceptions.ValidationError` |
| Pricing | sipQuant | `otc.py`, `commodity.py`, `options.py` |
| Risk | sipQuant | `book.py`, `risk.py`, `liquidity.py` |
| Input type contracts | sipQuant | `schema.py` |
| Order validation | sipTrade | `validation.py` |
| Order signing | sipTrade | `auth.sign_request()` |
| Order status | sipTrade | `status.py` |
| Settlement records | sipTrade | `settlement.py` |
| Index values for settlement | sipData | `endpoints/indices.py` |

No package reaches into another package's concerns. sipTrade never calculates a price. sipQuant never submits an order. sipData never executes a trade.

---

*sipTrade v1.0 — SIP Global*
