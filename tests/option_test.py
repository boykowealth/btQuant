from btQuant.options import blackScholes, binary, spread, barrier, asian, binomial, trinomial, impliedVol, simulate, buildForwardCurve, bootstrapCurve
import numpy as np

print("="*60)
print("TESTING OPTIONS PRICING LIBRARY")
print("="*60)

print("\n1. BLACK-SCHOLES")
result = blackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, optType='call')
print(f"Call: {result['price']:.4f}, Delta: {result['delta']:.4f}, Gamma: {result['gamma']:.4f}")
result = blackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, optType='put')
print(f"Put: {result['price']:.4f}, Delta: {result['delta']:.4f}, Vega: {result['vega']:.4f}")

print("\n2. BINARY")
result = binary(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, optType='call')
print(f"Binary Call: {result['price']:.4f}, Delta: {result['delta']:.4f}")
result = binary(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, optType='put')
print(f"Binary Put: {result['price']:.4f}, Gamma: {result['gamma']:.4f}")

print("\n3. SPREAD")
result = spread(S1=110, S2=100, K=5, T=1, r=0.05, sigma1=0.25, sigma2=0.20, 
                    rho=0.7, q1=0.02, q2=0.03, optType='call')
print(f"Spread Call: {result['price']:.4f}, Delta1: {result['delta1']:.4f}, Delta2: {result['delta2']:.4f}")
result = spread(S1=110, S2=100, K=5, T=1, r=0.05, sigma1=0.25, sigma2=0.20, 
                    rho=0.7, q1=0.02, q2=0.03, optType='put')
print(f"Spread Put: {result['price']:.4f}, Vega1: {result['vega1']:.4f}")

print("\n4. BARRIER")
result = barrier(S=100, K=100, T=1, r=0.05, sigma=0.2, barrierLevel=90, q=0.02, 
                     optType='call', barrierType='down-and-out', rebate=0)
print(f"Down-and-Out Call: {result['price']:.4f}, Delta: {result['delta']:.4f}")
result = barrier(S=100, K=100, T=1, r=0.05, sigma=0.2, barrierLevel=90, q=0.02, 
                     optType='put', barrierType='down-and-in', rebate=5)
print(f"Down-and-In Put (rebate=5): {result['price']:.4f}, Vega: {result['vega']:.4f}")
result = barrier(S=100, K=100, T=1, r=0.05, sigma=0.2, barrierLevel=110, q=0.02, 
                     optType='call', barrierType='up-and-out', rebate=0)
print(f"Up-and-Out Call: {result['price']:.4f}, Gamma: {result['gamma']:.4f}")
result = barrier(S=100, K=100, T=1, r=0.05, sigma=0.2, barrierLevel=110, q=0.02, 
                     optType='put', barrierType='up-and-in', rebate=0)
print(f"Up-and-In Put: {result['price']:.4f}")

print("\n5. ASIAN")
result = asian(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, nSteps=50, 
                   optType='call', avgType='geometric')
print(f"Asian Geometric Call: {result['price']:.4f}, Delta: {result['delta']:.4f}")
result = asian(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, nSteps=50, 
                   optType='put', avgType='geometric')
print(f"Asian Geometric Put: {result['price']:.4f}, Vega: {result['vega']:.4f}")
result = asian(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, nSteps=50, 
                   optType='call', avgType='arithmetic')
print(f"Asian Arithmetic Call: {result['price']}, StdErr: {result['stderr']}")

print("\n6. BINOMIAL")
result = binomial(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, N=100, 
                      optType='call', american=False)
print(f"European Call (binomial): {result['price']:.4f}, Delta: {result['delta']:.4f}")
result = binomial(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, N=100, 
                      optType='put', american=True)
print(f"American Put (binomial): {result['price']:.4f}, Gamma: {result['gamma']:.4f}")

print("\n7. TRINOMIAL")
result = trinomial(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, N=50, 
                       optType='call', american=False)
print(f"European Call (trinomial): {result['price']:.4f}, Delta: {result['delta']:.4f}")
result = trinomial(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02, N=50, 
                       optType='put', american=True)
print(f"American Put (trinomial): {result['price']:.4f}, Gamma: {result['gamma']:.4f}")

print("\n8. SIMULATE")
np.random.seed(42)
paths = np.random.lognormal(mean=np.log(100), sigma=0.2, size=(1000, 101))
result = simulate(pricingModel=blackScholes, paths=paths, r=0.05, T=1, 
                      K=100, sigma=0.2, q=0.02, optType='call')
print(f"Simulated BS Call: {result['price']:.4f}, StdErr: {result['stderr']:.4f}")

print("\n9. IMPLIED VOLATILITY")
target_price = 10.45
iv = impliedVol(price=target_price, S=100, K=100, T=1, r=0.05, 
                    optType='call', q=0.02, tol=1e-6, maxIter=100)
print(f"Implied Vol for price={target_price}: {iv:.4f}")
verification = blackScholes(S=100, K=100, T=1, r=0.05, sigma=iv, q=0.02, optType='call')
print(f"Verification price: {verification['price']:.4f}")

print("\n10. BUILD FORWARD CURVE")
tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
rates = np.array([0.04, 0.045, 0.05, 0.055, 0.06])
storageCosts = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
convYields = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
forwards = buildForwardCurve(spotPrice=100, tenors=tenors, rates=rates, 
                                  storageCosts=storageCosts, convenienceYields=convYields)
print(f"Forward prices: {forwards}")

print("\n11. BOOTSTRAP CURVE")
spotPrice = 100
futuresPrices = np.array([101, 103, 107, 114, 122])
tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
result = bootstrapCurve(spotPrice=spotPrice, futuresPrices=futuresPrices, 
                            tenors=tenors, assumedRate=0.05)
print(f"Conv Yields: {result['convenience_yields']}")
print(f"Storage Costs: {result['storage_costs']}")

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)