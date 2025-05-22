import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, BarycentricInterpolator, RectBivariateSpline, griddata, interp2d, Rbf

def curve(x, y, z=None, method='cubic', num_points=100, data_type='rate', asset_type='generic'):
    """
    Bootstrap a curve for given data points using interpolation, supporting rates, volatility curves,
    swaps, bonds, and foreign exchange (FX) rate curves.

    Parameters:
        x : array-like
            The x-values (e.g., maturities, time).
        y : array-like
            The y-values (e.g., zero rates, swap rates, volatilities, or FX rates).
        z : array-like, optional
            Additional data (e.g., coupon payments, cash flows, or FX adjustments).
        method : str, optional
            The interpolation method, can be 'linear', 'cubic', 'pchip', 'akima', or 'barycentric' (default is 'cubic').
        num_points : int, optional
            The number of points to interpolate between the minimum and maximum x-values (default is 100).
        data_type : str, optional
            Specifies the type of data ('rate', 'volatility', 'fx_rate').
        asset_type : str, optional
            Specifies the type of asset (e.g., 'generic', 'bond', 'swap', 'fx') for extended functionality.

    Returns:
        pd.DataFrame
            DataFrame with Bootstrapped Curve.
    """

    def select_interpolator(method, x, y):
        """
        Helper function to select and apply the interpolation method.

        Parameters:
            method : str
                The interpolation method to use.
            x : array-like
                The x-values (e.g., maturities, time).
            y : array-like
                The y-values to interpolate.

        Returns:
            Callable
                Interpolated function for the given method.
        """
        if method == 'linear':
            return lambda x_new: np.interp(x_new, x, y)
        elif method == 'cubic':
            return CubicSpline(x, y)
        elif method == 'pchip':
            return PchipInterpolator(x, y)
        elif method == 'akima':
            return Akima1DInterpolator(x, y)
        elif method == 'barycentric':
            return BarycentricInterpolator(x, y)
        else:
            raise ValueError("Invalid interpolation method. Choose 'linear', 'cubic', 'pchip', 'akima', or 'barycentric'.")

    if data_type == 'rate':
        zero_rates = []
        for i in range(len(x)):
            if i == 0:
                P = y[i]  # Present value
                T = x[i]  # Maturity
                C = z[i] if z is not None else 0  # Coupon
                zero_rate = (C + P) ** (1 / T) - 1
                zero_rates.append(zero_rate)
            else:
                P = y[i]
                T = x[i]
                C = z[i] if z is not None else 0
                discounted_value = sum(z[j] / (1 + zero_rates[j]) ** x[j] for j in range(i))
                zero_rate = (C + P - discounted_value) ** (1 / T) - 1
                zero_rates.append(zero_rate)

        interpolator = select_interpolator(method, x, zero_rates)
        rate_interp = interpolator(np.linspace(min(x), max(x), num_points))

    elif data_type == 'volatility':
        interpolator = select_interpolator(method, x, y)
        rate_interp = interpolator(np.linspace(min(x), max(x), num_points))

    elif data_type == 'fx_rate':
        fx_adjusted = [y[i] / (1 + z[i]) if z is not None else y[i] for i in range(len(y))]
        interpolator = select_interpolator(method, x, fx_adjusted)
        rate_interp = interpolator(np.linspace(min(x), max(x), num_points))

    elif asset_type == 'swap':
        zero_rates = []
        for i in range(len(x)):
            if i == 0:
                zero_rates.append(y[i]) 
            else:
                swap_rate = y[i]
                discounted_value = sum(zero_rates[j] / (1 + zero_rates[j]) ** x[j] for j in range(i))
                zero_rate = (swap_rate - discounted_value) ** (1 / x[i])
                zero_rates.append(zero_rate)

        interpolator = select_interpolator(method, x, zero_rates)
        rate_interp = interpolator(np.linspace(min(x), max(x), num_points))

    elif asset_type == 'bond':
        zero_rates = []
        for i in range(len(x)):
            if i == 0:
                P = y[i]
                T = x[i]
                C = z[i] if z is not None else 0
                zero_rate = (C + P) ** (1 / T) - 1
                zero_rates.append(zero_rate)
            else:
                P = y[i]
                T = x[i]
                C = z[i] if z is not None else 0
                discounted_value = sum(z[j] / (1 + zero_rates[j]) ** x[j] for j in range(i))
                zero_rate = (C + P - discounted_value) ** (1 / T) - 1
                zero_rates.append(zero_rate)

        interpolator = select_interpolator(method, x, zero_rates)
        rate_interp = interpolator(np.linspace(min(x), max(x), num_points))

    else:
        raise ValueError("Invalid data_type or asset_type. Choose 'rate', 'volatility', 'fx_rate', 'swap', or 'bond'.")

    df = pd.DataFrame({
        'X': np.linspace(min(x), max(x), num_points),
        'Y': rate_interp
    })

    return df

def surface(x, y, z, method='linear', grid_size=(100, 100)):
    """
    Bootstrap a surface for given 2D data points using interpolation, supporting volatility surfaces,
    interest rate volatility, foreign exchange surfaces, and correlation surfaces.

    Parameters:
        x : array-like
            The x-values (e.g., maturities or tenors).
        y : array-like
            The y-values (e.g., strike prices or other second-dimension variables).
        z : array-like
            The z-values (e.g., implied volatility or correlation values).
        method : str, optional
            The interpolation method, can be 'linear', 'cubic', 'bivariate', or 'grid' (default is 'linear').
        grid_size : tuple, optional
            The size of the grid for interpolation (default is (100, 100)).

    Returns:
        pd.DataFrame
            DataFrame with interpolated surface data.
    """
    x_grid = np.linspace(min(x), max(x), grid_size[0])
    y_grid = np.linspace(min(y), max(y), grid_size[1])
    X, Y = np.meshgrid(x_grid, y_grid)

    if method == 'linear':
        interpolator = interp2d(x, y, z, kind='linear')
        Z = interpolator(x_grid, y_grid)
    elif method == 'cubic':
        interpolator = interp2d(x, y, z, kind='cubic')
        Z = interpolator(x_grid, y_grid)
    elif method == 'bivariate':
        interpolator = RectBivariateSpline(np.unique(x), np.unique(y), z.reshape(len(np.unique(x)), len(np.unique(y))))
        Z = interpolator(x_grid, y_grid)
    elif method == 'grid':
        Z = griddata((x, y), z, (X, Y), method='linear')
    else:
        raise ValueError("Invalid interpolation method. Choose 'linear', 'cubic', 'bivariate', or 'grid'.")

    df = pd.DataFrame({
        'X': X.flatten(),
        'Y': Y.flatten(),
        'Z': Z.flatten()
    })

    return df

def rbfSurface(x, y, z, grid_size=(100, 100)):
    """
    Bootstrap a correlation surface using Radial Basis Function (RBF) interpolation.

    Parameters:
        x : array-like
            The x-values (e.g., maturities or time for asset classes).
        y : array-like
            The y-values (e.g., strike prices or second asset class variable).
        z : array-like
            The z-values (e.g., correlation values between asset classes).
        grid_size : tuple, optional
            The size of the grid for interpolation (default is (100, 100)).

    Returns:
        pd.DataFrame
            DataFrame with interpolated correlation surface.
    """
    x_grid = np.linspace(min(x), max(x), grid_size[0])
    y_grid = np.linspace(min(y), max(y), grid_size[1])
    X, Y = np.meshgrid(x_grid, y_grid)

    rbf = Rbf(x, y, z, function='gaussian')
    Z = rbf(X, Y)

    df = pd.DataFrame({
        'X': X.flatten(),
        'Y': Y.flatten(),
        'Z': Z.flatten()
    })

    return df


