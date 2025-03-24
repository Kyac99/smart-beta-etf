# ETF Smart Beta - Approche Factorielle

Ce projet implémente un ETF Smart Beta utilisant une approche factorielle basée sur plusieurs critères (Value, Momentum, Quality, Low Volatility, ESG). L'objectif est d'offrir une alternative aux ETF traditionnels en optimisant la performance ajustée au risque.

## 📋 Présentation

L'ETF Smart Beta développé dans ce projet utilise une approche factorielle pour sélectionner et pondérer des actifs selon plusieurs facteurs:

- **Value**: Sélection d'actions sous-évaluées (P/E bas, P/B bas, dividendes élevés)
- **Momentum**: Sélection d'actions avec une forte dynamique de prix
- **Quality**: Sélection d'actions de sociétés avec de solides fondamentaux (ROE, ROA élevés)
- **Low Volatility**: Sélection d'actions moins volatiles
- **ESG**: Prise en compte des critères environnementaux, sociaux et de gouvernance

L'ETF est conçu pour surperformer les indices traditionnels en termes de rendement ajusté au risque, mesuré par des métriques comme le ratio de Sharpe, l'alpha et le tracking error.

## 🔧 Fonctionnalités

- **Construction du Portefeuille**: Définition des critères de sélection des actifs, pondération basée sur les scores factoriels
- **Backtesting**: Simulation historique avec comparaison à un ETF référent (ex: S&P 500)
- **Analyse de Performance**: Calcul du ratio de Sharpe, alpha, beta, tracking error, drawdowns
- **Rebalancement**: Implémentation de différentes fréquences de rebalancement (mensuelle, trimestrielle, etc.)
- **Visualisation**: Graphiques interactifs pour l'analyse des performances et des expositions factorielles
- **Rapports**: Export des résultats sous format CSV et Excel

## 🚀 Installation

1. Cloner le dépôt:
```bash
git clone https://github.com/Kyac99/smart-beta-etf.git
cd smart-beta-etf
```

2. Installer les dépendances requises:
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

Voici un exemple simple d'utilisation de l'ETF Smart Beta:

```python
from smart_beta_etf import SmartBetaETF

# Initialiser l'ETF Smart Beta
etf = SmartBetaETF(benchmark_ticker="SPY", start_date="2018-01-01")

# Exécuter l'analyse complète
etf.run_full_analysis(
    sample_size=100,
    n_stocks=30,
    max_weight=0.05,
    rebalancing_frequency='quarterly',
    factors=['value', 'momentum', 'quality', 'low_volatility'],
    output_dir='output'
)
```

Pour une démonstration plus détaillée, exécutez le script `demo.py`:

```bash
python demo.py
```

## 📊 Exemples de Visualisations

Le projet génère plusieurs visualisations pour analyser la performance de l'ETF:

- Comparaison de la performance avec le benchmark
- Analyse des drawdowns
- Exposition aux différents facteurs
- Allocation sectorielle

## 📂 Structure du Projet

```
├── smart_beta_etf.py       # Classe principale pour l'ETF Smart Beta
├── demo.py                 # Script de démonstration
├── requirements.txt        # Dépendances du projet
├── README.md               # Documentation
└── output/                 # Répertoire pour les résultats
    ├── performance.png     # Graphiques de performance
    ├── portfolio.csv       # Composition du portefeuille
    └── report.xlsx         # Rapport complet de performance
```

## 📝 Notes Techniques

- Les données historiques sont récupérées via l'API Yahoo Finance
- Les calculs de performance utilisent des méthodes standard de l'industrie
- L'optimisation du portefeuille prend en compte la liquidité des titres
- Les coûts de transaction sont modélisés lors du rebalancement

## 🔧 Technologies Utilisées

- **Python**: Langage principal
- **Pandas/NumPy**: Manipulation et analyse des données
- **Matplotlib/Seaborn/Plotly**: Visualisation des données
- **scikit-learn**: Normalisation et traitement des données
- **yfinance**: Récupération des données financières
- **openpyxl**: Export des résultats au format Excel