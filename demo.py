# ============================================================================
# ETF Smart Beta - Demo d'utilisation
# ============================================================================
from smart_beta_etf import SmartBetaETF
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Définir les styles des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Créer le répertoire de sortie
output_dir = 'output_demo'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialiser l'ETF Smart Beta
print("Initialisation de l'ETF Smart Beta...")
etf = SmartBetaETF(
    benchmark_ticker="SPY",  # S&P 500 ETF comme benchmark
    start_date="2018-01-01",  # 5 ans de données historiques
    end_date="2023-12-31"
)

# Récupérer les données
print("Récupération des données...")
etf.fetch_data(sample_size=100)  # Utiliser 100 titres pour la démonstration

# Calculer les scores factoriels avec des poids personnalisés
print("Calcul des scores factoriels...")
etf.calculate_factor_scores(
    factors=['value', 'momentum', 'quality', 'low_volatility']
)

# Optimiser le portefeuille
print("Optimisation du portefeuille...")
factor_weights = {
    'value': 0.3,
    'momentum': 0.3,
    'quality': 0.2,
    'low_volatility': 0.2
}
etf._calculate_combined_score(
    factors=['value', 'momentum', 'quality', 'low_volatility'],
    weights=factor_weights
)

# Sélectionner 30 titres avec un maximum de 5% par titre
etf.optimize_portfolio(n_stocks=30, max_weight=0.05)

# Réaliser le backtesting avec rebalancement trimestriel
print("Backtesting de la stratégie...")
etf.backtest(rebalancing_frequency='quarterly')

# Visualiser la performance
print("Visualisation des résultats...")
etf.plot_performance(save_path=os.path.join(output_dir, 'performance.png'))
etf.plot_drawdowns(save_path=os.path.join(output_dir, 'drawdowns.png'))
etf.plot_factor_exposures(save_path=os.path.join(output_dir, 'exposures.png'))
etf.plot_sector_allocation(save_path=os.path.join(output_dir, 'sectors.png'))

# Exporter les résultats
print("Exportation des résultats...")
etf.export_portfolio(file_path=os.path.join(output_dir, 'portfolio.csv'))
etf.export_performance_report(file_path=os.path.join(output_dir, 'report.xlsx'))

print("Démonstration terminée. Les résultats sont disponibles dans le répertoire:", output_dir)

# Afficher un résumé des performances
if etf.performance_metrics:
    print("\nRésumé des performances:")
    print(f"Rendement annualisé de l'ETF: {etf.performance_metrics['ETF Annual Return']:.2%}")
    print(f"Rendement annualisé du benchmark: {etf.performance_metrics['Benchmark Annual Return']:.2%}")
    print(f"Ratio de Sharpe de l'ETF: {etf.performance_metrics['ETF Sharpe Ratio']:.2f}")
    print(f"Ratio de Sharpe du benchmark: {etf.performance_metrics['Benchmark Sharpe Ratio']:.2f}")
    print(f"Alpha: {etf.performance_metrics['Alpha']:.2%}")
    print(f"Beta: {etf.performance_metrics['Beta']:.2f}")
    print(f"Tracking Error: {etf.performance_metrics['Tracking Error']:.2%}")
    print(f"Maximum Drawdown de l'ETF: {etf.performance_metrics['ETF Max Drawdown']:.2%}")
