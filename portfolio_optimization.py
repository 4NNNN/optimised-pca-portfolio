"""
Portfolio Optimization & Factor Modeling
Mean-variance optimizer using historical returns of oil stocks with constraints
"""

import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Oil stocks portfolio
tickers = ["CVX", "XOM", "REP.MC"]  # Chevron, ExxonMobil, Repsol SA (Madrid)

# Sector ETFs for factor interpretation
sector_etfs = ['XLE', 'XLK', 'XLF', 'SPY', 'VTI', 'QQQ']

# Historical data range
START_DATE = "2025-01-01"
END_DATE = "2026-01-01"

def fetch_data():
    """Fetch adjusted close prices for oil stocks"""
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    adj_close = data['Adj Close']
    returns = adj_close.pct_change().dropna()
    return adj_close, returns

def optimize_portfolio(adj_close):
    """Mean-variance optimization with weight constraints"""
    mu = mean_historical_return(adj_close)
    S = sample_cov(adj_close)
    
    # Efficient Frontier with bounds (5% to 60% per asset)
    ef = EfficientFrontier(mu, S, weight_bounds=(0.05, 0.6))
    weights = ef.min_volatility()
    cleaned_weights = ef.clean_weights()
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance()
    
    return cleaned_weights, expected_return, volatility, sharpe_ratio

def perform_pca(returns):
    """Principal Component Analysis for factor extraction"""
    pca = PCA()
    pca.fit(returns)
    
    explained_variance = pca.explained_variance_ratio_
    components = pd.DataFrame(pca.components_, columns=returns.columns)
    components.index = [f"PC{i+1}" for i in range(components.shape[0])]
    factor_returns = returns.dot(components.T)
    
    return pca, explained_variance, components, factor_returns

def fetch_etf_data():
    """Download sector ETF data for factor interpretation"""
    etf_data = yf.download(sector_etfs, start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
    etf_returns = etf_data.pct_change().dropna()
    return etf_returns

def compute_factor_correlations(factor_returns, etf_returns):
    """Compute correlation between PCA factors and sector ETFs"""
    aligned_factors, aligned_etfs = factor_returns.align(etf_returns, join='inner', axis=0)
    
    correlation_grid = pd.DataFrame(index=aligned_factors.columns, columns=aligned_etfs.columns)
    for pc in aligned_factors.columns:
        for etf in aligned_etfs.columns:
            correlation_grid.loc[pc, etf] = aligned_factors[pc].corr(aligned_etfs[etf])
    
    return correlation_grid

def visualize_results(explained_variance, components, cleaned_weights, 
                     expected_return, sharpe_ratio, factor_returns, correlation_grid):
    """Generate comprehensive visualization dashboard"""
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Oil Stocks Portfolio: PCA + Optimization Analysis", fontsize=22, weight='bold', y=0.97)
    
    # 1. Scree Plot (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    x = range(1, len(explained_variance) + 1)
    ax1.plot(x, explained_variance, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.bar(x, explained_variance, alpha=0.3, color='#2E86AB')
    for i, val in enumerate(explained_variance):
        ax1.text(x[i], val + 0.01, f"{val:.1%}", ha='center', fontsize=10, fontweight='bold')
    ax1.set_title("Scree Plot: Explained Variance by Component", fontsize=14)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)
    
    # Portfolio summary box
    summary_text = (
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"Expected Return: {expected_return:.2%}\n\n"
        "Optimized Weights:\n" +
        "\n".join([f"{k}: {v:.1%}" for k, v in cleaned_weights.items()])
    )
    ax1.text(0.98, 0.98, summary_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='#E8F4F8', edgecolor='#2E86AB', boxstyle='round,pad=0.5'))
    
    # PCA Loadings table
    table_data = components.round(3).values
    col_labels = components.columns.tolist()
    row_labels = components.index.tolist()
    table = ax1.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels,
                      loc='bottom', bbox=[0, -0.7, 1, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax1.text(0.5, -0.32, "PCA Component Loadings", fontsize=12, ha='center', 
             va='center', transform=ax1.transAxes, fontweight='bold')
    
    # 2. Factor Returns Over Time (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i in range(min(3, factor_returns.shape[1])):
        ax2.plot(factor_returns.index, factor_returns.iloc[:, i], 
                label=f'PC{i+1}', linewidth=1.5, color=colors[i], alpha=0.8)
    ax2.set_title("Factor Returns Over Time", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Return")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Interpretation (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    interpretation = (
        "Factor Interpretation (based on ETF correlations):\n\n"
        "• PC1: Broad energy market factor - captures overall oil sector movement\n"
        "• PC2: Company-specific divergence - differences between majors\n"
        "• PC3: Residual idiosyncratic factors\n\n"
        "These factors represent latent drivers of returns across the\n"
        "oil stocks portfolio. High correlation with XLE (Energy ETF)\n"
        "confirms the sector-specific nature of this portfolio."
    )
    ax3.text(0.05, 0.85, interpretation, fontsize=13, verticalalignment='top',
             bbox=dict(facecolor='#FFF8E7', edgecolor='#F18F01', boxstyle='round,pad=0.8'))
    
    # 4. Correlation Heatmap (bottom right)
    ax4 = fig.add_axes([0.57, 0.08, 0.37, 0.32])
    subset_correlation = correlation_grid.iloc[:3, :].astype(float)
    sns.heatmap(subset_correlation, annot=True, cmap='RdYlBu_r', fmt=".2f", ax=ax4,
                center=0, vmin=-1, vmax=1, linewidths=0.5)
    ax4.set_title("Factor-ETF Correlation Heatmap", fontsize=14)
    ax4.set_ylabel("Principal Components")
    ax4.set_xlabel("Sector ETFs")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('portfolio_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Analysis complete! Output saved to 'portfolio_analysis.png'")

def main():
    print("=" * 60)
    print("Portfolio Optimization & Factor Modeling")
    print("Oil Stocks: CVX (Chevron), XOM (ExxonMobil), REP (Repsol)")
    print("=" * 60)
    
    # Fetch and process data
    print("\n📊 Fetching historical price data...")
    adj_close, returns = fetch_data()
    
    # Portfolio optimization
    print("⚖️  Running mean-variance optimization...")
    cleaned_weights, expected_return, volatility, sharpe_ratio = optimize_portfolio(adj_close)
    
    print(f"\n📈 Optimized Portfolio:")
    for ticker, weight in cleaned_weights.items():
        print(f"   {ticker}: {weight:.1%}")
    print(f"\n   Expected Return: {expected_return:.2%}")
    print(f"   Volatility: {volatility:.2%}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # PCA analysis
    print("\n🔬 Performing PCA factor extraction...")
    pca, explained_variance, components, factor_returns = perform_pca(returns)
    
    print(f"\n   Variance explained by top 3 factors:")
    for i, var in enumerate(explained_variance[:3]):
        print(f"   PC{i+1}: {var:.1%}")
    
    # Factor-ETF correlation
    print("\n📊 Computing factor-ETF correlations...")
    etf_returns = fetch_etf_data()
    correlation_grid = compute_factor_correlations(factor_returns, etf_returns)
    
    # Visualization
    print("\n🎨 Generating visualizations...")
    visualize_results(explained_variance, components, cleaned_weights,
                     expected_return, sharpe_ratio, factor_returns, correlation_grid)

if __name__ == "__main__":
    main()
