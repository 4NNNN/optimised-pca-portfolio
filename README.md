# Portfolio Optimization & Factor Modeling

Optimizes allocations across oil stocks (CVX, XOM, Repsol) using mean-variance optimization, then extracts hidden market factors using PCA.

## What It Does

1. **Finds optimal weights** — Balances risk and return across the three oil majors using historical data
2. **Extracts factors via PCA** — Identifies the main drivers of returns (market-wide moves vs company-specific)
3. **Interprets factors** — Correlates PCA components with sector ETFs to understand what each factor represents

## Quick Start

```bash
pip install -r requirements.txt
python portfolio_optimization.py
```

Outputs a visualization with scree plot, factor loadings, and correlation heatmap.

## Sample Output

| Metric | Value |
|--------|-------|
| CVX Weight | ~35% |
| XOM Weight | ~40% |
| REP Weight | ~25% |
| Sharpe Ratio | ~0.8 |

*Actual values depend on the date range.*

---

## Scaling to n Assets: When to Use Quantum

This classical approach works well for small portfolios. But for **mean-variance optimization across n assets** (think 100+ securities), the problem complexity grows quadratically with the covariance matrix.

**Qiskit Finance** offers quantum-based solvers for these larger combinatorial optimization problems:

```python
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_algorithms import QAOA

# Formulate as QUBO and solve with quantum approximate optimization
portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, budget=n)
```

Quantum approaches like QAOA and VQE can potentially find near-optimal solutions for large-scale portfolio problems where classical solvers become impractical. This is an active research area — not production-ready yet, but worth exploring for institutional-scale portfolios.

---

## Files

- `portfolio_optimization.py` — Main analysis script
- `requirements.txt` — Dependencies
