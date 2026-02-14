# ============================================================
# Dissertation Data Generator (Corrected & Economically Consistent)
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf

# ==============================
# USER PARAMETERS
# ==============================

ticker = "ETH-USD"
period = "2y"
initial_capital = 10000
window = 30
num_paths = 5000
sim_days = 180

alpha_grid = np.linspace(0.1, 0.9, 41)
gamma_grid = np.linspace(0.5, 5.0, 20)
sigma_grid = np.linspace(0.2, 1.0, 20)

fee_rate_assumed = 0.003  # 0.3%
np.random.seed(42)

# ==============================
# DATA DOWNLOAD
# ==============================

data = yf.download(ticker, period=period, auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

prices = data["Close"].dropna()
returns = prices.pct_change().dropna()

mu_hat = returns.mean() * 252
sigma_hat = returns.std() * np.sqrt(252)

P0 = prices.iloc[0]

# Initial symmetric pool
x0 = initial_capital / (2 * P0)
y0 = initial_capital / 2

# ============================================================
# 1. VOLATILITY REGIME DETECTION
# ============================================================

rolling_vol = returns.rolling(window).std() * np.sqrt(252)
q33, q66 = rolling_vol.quantile([0.33, 0.66])

def classify_regime(vol):
    if np.isnan(vol):
        return np.nan
    elif vol < q33:
        return "Low"
    elif vol < q66:
        return "Medium"
    else:
        return "High"

regimes = rolling_vol.apply(classify_regime)

# IL time series (alpha=0.5)
alpha_base = 0.5
relative_price = prices / P0

x_P = x0 * relative_price ** (-(1 - alpha_base))
y_P = y0 * relative_price ** (alpha_base)

amm_values = x_P * prices + y_P
hodl_values = x0 * prices + y0

impermanent_loss = amm_values / hodl_values - 1

df_regime = pd.DataFrame({
    "date": prices.index,
    "price": prices.values,
    "il": impermanent_loss.values,
    "regime": regimes.reindex(prices.index).values
})

df_regime.to_csv("regime_il_timeseries.csv", index=False)

# ============================================================
# 2. MONTE CARLO TERMINAL PRICES (μ=0 for symmetry)
# ============================================================

dt = 1 / 252
terminal_prices = []

for _ in range(num_paths):
    S = P0
    for _ in range(sim_days):
        z = np.random.normal()
        S *= np.exp((-0.5 * sigma_hat**2) * dt +
                    sigma_hat * np.sqrt(dt) * z)
    terminal_prices.append(S)

terminal_prices = np.array(terminal_prices)

T = sim_days / 252

# ============================================================
# 3. EXPECTED IL VS α
# ============================================================

expected_il = []

for a in alpha_grid:
    il_vals = []
    for S in terminal_prices:
        R = S / P0
        x = x0 * R ** (-(1 - a))
        y = y0 * R ** (a)
        V = x * S + y
        hodl = x0 * S + y0
        il_vals.append(V / hodl - 1)
    expected_il.append(np.mean(il_vals))

df_il_alpha = pd.DataFrame({
    "alpha": alpha_grid,
    "expected_il_mc": expected_il,
    "expected_il_analytic": -0.5 * alpha_grid * (1 - alpha_grid) * sigma_hat**2 * T
})

df_il_alpha.to_csv("expected_il_alpha.csv", index=False)

# ============================================================
# 4. BREAK-EVEN FEE φ*
# ============================================================

# Turnover proxy proportional to absolute return
kappa = np.mean(np.abs(returns)) * sim_days

phi_star = (
    0.5 * alpha_grid * (1 - alpha_grid) * sigma_hat**2 * T
) / (kappa + 1e-12)

df_fee = pd.DataFrame({
    "alpha": alpha_grid,
    "phi_star": phi_star
})

df_fee.to_csv("break_even_fee.csv", index=False)

# ============================================================
# OPTIMAL α* VS γ (THEORETICALLY CORRECT CLOSED FORM)
# ============================================================

optimal_alpha_list = []

for gamma in gamma_grid:

    alpha_star = (
        mu_hat + fee_rate_assumed * sigma_hat**2
    ) / (sigma_hat**2 * (gamma + 2 * fee_rate_assumed))

    # Clamp to feasible region
    alpha_star = np.clip(alpha_star, 0.0, 1.0)

    optimal_alpha_list.append(alpha_star)

df_gamma = pd.DataFrame({
    "gamma": gamma_grid,
    "alpha_star": optimal_alpha_list
})

df_gamma.to_csv("alpha_star_gamma.csv", index=False)


# ============================================================
# 6. IL SURFACE E[IL(α, σ)]
# ============================================================

surface_rows = []

for sigma in sigma_grid:

    for a in alpha_grid:

        il_vals = []

        for _ in range(1000):

            S = P0

            for _ in range(sim_days):
                z = np.random.normal()
                S *= np.exp((-0.5 * sigma**2) * dt +
                            sigma * np.sqrt(dt) * z)

            R = S / P0
            x = x0 * R ** (-(1 - a))
            y = y0 * R ** (a)
            V = x * S + y
            hodl = x0 * S + y0

            il_vals.append(V / hodl - 1)

        surface_rows.append({
            "sigma": sigma,
            "alpha": a,
            "expected_il": np.mean(il_vals)
        })

df_surface = pd.DataFrame(surface_rows)
df_surface.to_csv("il_surface_sigma_alpha.csv", index=False)

print("All dissertation CSV files generated successfully.")
