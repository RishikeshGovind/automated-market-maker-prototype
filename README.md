# Project Overview

This repository contains the simulation engine, empirical analysis pipeline, and interactive dashboard used in the dissertation: **Optimal Invariant Design in Automated Market Makers**.

The project formalizes AMM invariant curvature ($\alpha$) as a portfolio control variable and studies:
* Impermanent Loss (IL)
* Fee compensation equilibrium
* CRRA-optimal curvature choice
* Volatility-regime sensitivity
* Continuous-time theoretical links to HJB control

## Research Question
**Can AMM curvature be derived endogenously from utility maximization rather than chosen heuristically?**

We demonstrate that:
1. Expected IL admits a quadratic approximation: $\mathbb{E}[IL(\alpha)] \approx -\frac{1}{2}\alpha(1-\alpha)\sigma^2 T$
2. Curvature behaves like embedded short gamma exposure.
3. Break-even fee rates scale with variance.
4. Optimal curvature decreases in risk aversion ($\gamma$).

---


# 🌐 Live Application

[https://automated-market-maker.streamlit.app/](https://automated-market-maker.streamlit.app/)

---

# Running the Dashboard

The dashboard facilitates real-time interaction with the model, including Monte Carlo IL simulation, analytic vs. simulated comparisons, and IL surface visualization. To initialize the interactive environment, ensure you have the necessary dependencies installed (`streamlit`, `pandas`, `numpy`) and run the `app.py` script.

# Generating Dissertation Figures

To regenerate the data for all Chapter 5 tables and figures, run the Python simulation script `generate_results.py`. This script produces the following core datasets:

* `expected_il_alpha.csv`
* `il_surface_sigma_alpha.csv`
* `break_even_fee.csv`
* `alpha_star_gamma.csv`
* `regime_il_timeseries.csv`

The final document can then be rendered using Quarto via the `amm_prototype.qmd` file to incorporate these results.

---

# Model Summary

### Asset Dynamics
Geometric Brownian Motion:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

### Impermanent Loss
For constant-mean invariant weight $\alpha$:
$$IL = \frac{V_{AMM}}{V_{HODL}} - 1$$

Small-vol expansion:
$$\mathbb{E}[IL(\alpha)] \approx -\frac{1}{2}\alpha(1-\alpha)\sigma^2 T$$



### Fee Compensation
Break-even fee ($\varphi^*$):
$$\varphi^*(\alpha) \approx \frac{\frac{1}{2}\alpha(1-\alpha)\sigma^2 T}{\kappa}$$
where $\kappa$ is an empirical turnover proxy.

### Optimal Curvature
LPs maximize CRRA utility $U(W) = \frac{W^{1-\gamma}}{1-\gamma}$. The optimal choice is defined as:
$$\alpha^* = \arg\max_\alpha \mathbb{E}[U(W_T(\alpha))]$$

Theoretical results confirm: $\frac{\partial \alpha^*}{\partial \gamma} < 0$.



---

# Main Contributions
* Formal link between AMM curvature and gamma exposure.
* Closed-form IL approximation under GBM.
* Utility-maximizing invariant selection.
* Fee–volatility equilibrium characterization.

# Requirements
Core dependencies include: `numpy`, `pandas`, `scipy`, `matplotlib`, `yfinance`, `streamlit`, and `quarto`.

> **Disclaimer:** This project is for academic research purposes only and does not constitute financial advice.
