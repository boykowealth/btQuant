import numpy as np
import pandas as pd

from btQuant.plot import plot_chart
from btQuant.bootstrap import curve, surface, rbfSurface

# Define input data
x = np.array([1, 2, 3, 5, 7, 10])
y_rate = np.array([0.99, 0.97, 0.94, 0.90, 0.85, 0.80])  # Present values
z_coupon = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])  # Coupons

# === 1. Curve Bootstrapping (Rate Data with Coupons, 'cubic' method) ===
df_curve_rate = curve(x, y_rate, z=z_coupon, method='cubic', data_type='rate', asset_type='generic')
print("\nBootstrapped Rate Curve:")
print(df_curve_rate.head())

# === 2. Volatility Curve ===
x_vol = np.array([1, 2, 3, 5, 7, 10])
y_vol = np.array([0.25, 0.24, 0.23, 0.22, 0.21, 0.20])
df_vol_curve = curve(x_vol, y_vol, method='akima', data_type='volatility')
print("\nVolatility Curve:")
print(df_vol_curve.head())

# === 3. FX Rate Curve with Adjustments ===
x_fx = np.array([1, 2, 3, 5, 7, 10])
y_fx = np.array([1.3, 1.32, 1.34, 1.36, 1.37, 1.38])
z_fx = np.array([0.02, 0.021, 0.022, 0.023, 0.025, 0.027])  # Adjustment factor (e.g., inflation)
df_fx_curve = curve(x_fx, y_fx, z=z_fx, method='pchip', data_type='fx_rate')
print("\nFX Rate Curve:")
print(df_fx_curve.head())

# === 4. Bond Curve ===
df_bond_curve = curve(x, y_rate, z=z_coupon, method='barycentric', asset_type='bond')
print("\nBond Curve:")
print(df_bond_curve.head())

# === 5. Swap Curve ===
y_swap = np.array([0.01, 0.012, 0.013, 0.014, 0.015, 0.016])
df_swap_curve = curve(x, y_swap, method='linear', asset_type='swap')
print("\nSwap Curve:")
print(df_swap_curve.head())

# === 6. Surface Interpolation (e.g., Implied Volatility Surface) ===
x_surf = np.tile([1, 2, 5, 10], 4)
y_surf = np.repeat([80, 90, 100, 110], 4)
z_surf = np.random.uniform(0.15, 0.35, size=16)
df_surface = surface(x_surf, y_surf, z_surf, method='grid', grid_size=(50, 50))
print("\nInterpolated Volatility Surface (grid method):")
print(df_surface.head())

# === 7. RBF Surface Interpolation (e.g., Correlation Surface) ===
z_corr = np.sin(x_surf) * np.cos(y_surf / 10) + np.random.normal(0, 0.02, len(x_surf))
df_rbf_surface = rbfSurface(x_surf, y_surf, z_corr, grid_size=(50, 50))
print("\nRBF Interpolated Correlation Surface:")
print(df_rbf_surface.head())
