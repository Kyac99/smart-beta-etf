# ============================================================================
# ETF Smart Beta - Modèle de Construction et Backtesting
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os

class SmartBetaETF:
    """
    Classe pour la construction et l'analyse d'un ETF Smart Beta
    basé sur une approche factorielle (Value, Momentum, Quality, Low Volatility, ESG)
    """
    
    def __init__(self, benchmark_ticker="SPY", start_date="2012-01-01", end_date=None, universe=None):
        """
        Initialisation de l'ETF Smart Beta
        
        Parameters:
        -----------
        benchmark_ticker : str
            Ticker du benchmark (ETF classique) pour la comparaison
        start_date : str
            Date de début pour l'analyse des données
        end_date : str
            Date de fin pour l'analyse des données (par défaut : aujourd'hui)
        universe : list
            Liste des tickers à inclure dans l'univers d'investissement
        """
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        
        # Définition de l'univers d'investissement (par défaut: composants du S&P 500)
        if universe is None:
            self.universe = self._get_sp500_components()
        else:
            self.universe = universe
            
        # Données
        self.data = {}
        self.factor_scores = {}
        self.etf_weights = None
        self.etf_nav = None
        self.benchmark_nav = None
        self.performance_metrics = None
        self.drawdowns = None
        self.sector_allocation = None
        
        print(f"ETF Smart Beta initialisé avec {len(self.universe)} titres potentiels")
    
    def _get_sp500_components(self):
        """Récupère les composants actuels du S&P 500"""
        try:
            # Option 1: Utiliser wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].tolist()
        except:
            # Option 2: Liste prédéfinie (top 100 du S&P 500 par capitalisation)
            return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B", "TSLA", "UNH", 
                    "LLY", "JPM", "V", "XOM", "JNJ", "PG", "MA", "HD", "AVGO", "CVX",
                    "MRK", "COST", "ABT", "PEP", "ABBV", "KO", "BAC", "WMT", "CSCO", "MCD",
                    "CRM", "PFE", "TMO", "ACN", "DHR", "LIN", "ADBE", "TXN", "AMD", "CMCSA",
                    "NFLX", "VZ", "PM", "INTC", "NEE", "RTX", "ORCL", "HON", "UPS", "IBM",
                    "QCOM", "CAT", "GE", "LOW", "T", "BA", "MS", "INTU", "AMAT", "DE",
                    "SPGI", "GS", "BKNG", "ELV", "BLK", "MDLZ", "LMT", "ADP", "AMT", "ADI",
                    "GILD", "TJX", "SBUX", "MMC", "PLD", "SYK", "PYPL", "DIS", "MDT", "AXP",
                    "VRTX", "MO", "SCHW", "C", "ISRG", "CVS", "CI", "LRCX", "SO", "EQIX", 
                    "CL", "CB", "PGR", "ZTS", "BDX", "CME", "DUK", "COP", "AON", "ICE"]
    
    def fetch_data(self, sample_size=100):
        """
        Récupère les données pour l'univers d'investissement
        
        Parameters:
        -----------
        sample_size : int
            Taille de l'échantillon pour l'univers (pour les tests)
        """
        print(f"Récupération des données pour {sample_size} titres et le benchmark {self.benchmark_ticker}...")
        
        # Pour les tests, nous utilisons un échantillon limité
        if sample_size and sample_size < len(self.universe):
            self.universe = self.universe[:sample_size]
        
        # Récupération des prix et des volumes
        try:
            # Données de prix pour les titres de l'univers
            self.data['prices'] = yf.download(self.universe, start=self.start_date, end=self.end_date)['Adj Close']
            
            # Données de volume pour les titres de l'univers
            self.data['volumes'] = yf.download(self.universe, start=self.start_date, end=self.end_date)['Volume']
            
            # Données du benchmark
            benchmark_data = yf.download(self.benchmark_ticker, start=self.start_date, end=self.end_date)
            self.data['benchmark'] = benchmark_data['Adj Close']
            
            # Récupération des données fondamentales (pour les facteurs Value et Quality)
            self.data['fundamentals'] = self._fetch_fundamentals()
            
            # Récupération des données de secteur pour l'allocation sectorielle
            self.data['sectors'] = self._fetch_sectors()
            
            print(f"Données récupérées avec succès pour {self.data['prices'].shape[1]} titres sur {len(self.universe)}")
            
        except Exception as e:
            print(f"Erreur lors de la récupération des données: {e}")
            
        # Nettoyer les données pour supprimer les titres avec trop de valeurs manquantes
        self._clean_data()
    
    def _fetch_fundamentals(self):
        """Récupère les données fondamentales pour les facteurs Value et Quality"""
        fundamentals = {}
        
        for ticker in self.universe:
            try:
                stock = yf.Ticker(ticker)
                
                # Informations financières
                balance_sheet = stock.balance_sheet
                income_stmt = stock.income_stmt
                cash_flow = stock.cashflow
                
                if balance_sheet.empty or income_stmt.empty:
                    continue
                
                # Données fondamentales pour le dernier exercice fiscal
                fundamentals[ticker] = {
                    # Value metrics
                    'P/E': stock.info.get('trailingPE', np.nan),
                    'P/B': stock.info.get('priceToBook', np.nan),
                    'Dividend Yield': stock.info.get('dividendYield', 0),
                    
                    # Quality metrics
                    'ROE': stock.info.get('returnOnEquity', np.nan),
                    'ROA': stock.info.get('returnOnAssets', np.nan),
                    'Debt to Equity': stock.info.get('debtToEquity', np.nan),
                    'Gross Margin': stock.info.get('grossMargins', np.nan),
                    'Net Margin': stock.info.get('profitMargins', np.nan),
                    
                    # ESG score (si disponible)
                    'ESG Score': stock.info.get('esgScore', np.nan)
                }
            except:
                continue
        
        return pd.DataFrame(fundamentals).T
    
    def _fetch_sectors(self):
        """Récupère les données de secteur pour chaque titre"""
        sectors = {}
        
        for ticker in self.universe:
            try:
                stock = yf.Ticker(ticker)
                sectors[ticker] = stock.info.get('sector', 'Unknown')
            except:
                sectors[ticker] = 'Unknown'
        
        return pd.Series(sectors)
    
    def _clean_data(self):
        """Nettoie les données et supprime les titres avec trop de valeurs manquantes"""
        # Identifier les titres avec des données de prix
        valid_tickers = self.data['prices'].dropna(axis=1, thresh=int(len(self.data['prices']) * 0.9)).columns.tolist()
        
        # Mettre à jour l'univers d'investissement
        self.universe = valid_tickers
        
        # Mettre à jour les dataframes
        self.data['prices'] = self.data['prices'][valid_tickers]
        self.data['volumes'] = self.data['volumes'][valid_tickers]
        
        print(f"Après nettoyage, l'univers contient {len(self.universe)} titres")
    
    def calculate_factor_scores(self, factors=None):
        """
        Calcule les scores factoriels pour chaque titre
        
        Parameters:
        -----------
        factors : list
            Liste des facteurs à inclure (par défaut: tous)
        """
        if factors is None:
            factors = ['value', 'momentum', 'quality', 'low_volatility', 'esg']
        
        print(f"Calcul des scores factoriels: {', '.join(factors)}")
        
        # Calculer les scores pour chaque facteur
        if 'value' in factors:
            self._calculate_value_factor()
        
        if 'momentum' in factors:
            self._calculate_momentum_factor()
        
        if 'quality' in factors:
            self._calculate_quality_factor()
        
        if 'low_volatility' in factors:
            self._calculate_low_volatility_factor()
        
        if 'esg' in factors:
            self._calculate_esg_factor()
        
        # Combiner tous les scores en un score global
        self._calculate_combined_score(factors)
    
    def _calculate_value_factor(self):
        """Calcule le score Value basé sur le P/E, P/B et le rendement du dividende"""
        print("Calcul du facteur Value...")
        
        # Initialiser le dataframe pour les scores Value
        value_scores = pd.DataFrame(index=self.universe)
        
        # Récupérer les métriques Value à partir des données fondamentales
        if 'fundamentals' in self.data and not self.data['fundamentals'].empty:
            fund_data = self.data['fundamentals']
            
            # P/E (Price-to-Earnings) - plus bas = meilleur
            if 'P/E' in fund_data.columns:
                value_scores['P/E'] = fund_data['P/E']
                # Inverser le score (1/P/E) pour que les valeurs élevées soient meilleures
                value_scores['P/E'] = 1 / value_scores['P/E']
            
            # P/B (Price-to-Book) - plus bas = meilleur
            if 'P/B' in fund_data.columns:
                value_scores['P/B'] = fund_data['P/B']
                # Inverser le score (1/P/B) pour que les valeurs élevées soient meilleures
                value_scores['P/B'] = 1 / value_scores['P/B']
            
            # Dividend Yield - plus élevé = meilleur
            if 'Dividend Yield' in fund_data.columns:
                value_scores['Dividend Yield'] = fund_data['Dividend Yield']
        
        # Si les données fondamentales ne sont pas disponibles ou incomplètes, utiliser des proxys
        if value_scores.empty or value_scores.isna().all().all():
            # Utiliser le ratio prix/bénéfice des 12 derniers mois comme proxy
            prices = self.data['prices'].iloc[-1]  # Prix actuels
            returns = self.data['prices'].pct_change(252).iloc[-1]  # Rendements sur 1 an
            value_scores = pd.DataFrame(index=self.universe)
            value_scores['Price/Returns'] = prices / returns
            value_scores['Price/Returns'] = 1 / value_scores['Price/Returns']
        
        # Normaliser les scores
        scaler = StandardScaler()
        normalized_scores = pd.DataFrame(
            scaler.fit_transform(value_scores.fillna(value_scores.mean())),
            index=value_scores.index,
            columns=value_scores.columns
        )
        
        # Calculer le score moyen Value
        value_score = normalized_scores.mean(axis=1)
        
        # Stocker le score
        self.factor_scores['value'] = value_score
        
        print(f"Score Value calculé pour {len(value_score)} titres")
    
    def _calculate_momentum_factor(self):
        """Calcule le score Momentum basé sur les rendements historiques"""
        print("Calcul du facteur Momentum...")
        
        # Calcul des rendements
        returns = self.data['prices'].pct_change().dropna()
        
        # Créer le dataframe pour stocker les scores de momentum
        momentum_scores = pd.DataFrame(index=self.universe)
        
        # Momentum 12 mois (en excluant le dernier mois)
        momentum_12m1m = returns.iloc[-252:-21].add(1).prod() - 1
        momentum_scores['12m-1m'] = momentum_12m1m
        
        # Momentum 6 mois
        momentum_6m = returns.iloc[-126:].add(1).prod() - 1
        momentum_scores['6m'] = momentum_6m
        
        # Momentum 3 mois
        momentum_3m = returns.iloc[-63:].add(1).prod() - 1
        momentum_scores['3m'] = momentum_3m
        
        # Normaliser les scores
        scaler = StandardScaler()
        normalized_scores = pd.DataFrame(
            scaler.fit_transform(momentum_scores.fillna(momentum_scores.mean())),
            index=momentum_scores.index,
            columns=momentum_scores.columns
        )
        
        # Calculer le score moyen Momentum
        momentum_score = normalized_scores.mean(axis=1)
        
        # Stocker le score
        self.factor_scores['momentum'] = momentum_score
        
        print(f"Score Momentum calculé pour {len(momentum_score)} titres")
    
    def _calculate_quality_factor(self):
        """Calcule le score Quality basé sur les métriques de rentabilité et de solidité financière"""
        print("Calcul du facteur Quality...")
        
        # Initialiser le dataframe pour les scores Quality
        quality_scores = pd.DataFrame(index=self.universe)
        
        # Récupérer les métriques Quality à partir des données fondamentales
        if 'fundamentals' in self.data and not self.data['fundamentals'].empty:
            fund_data = self.data['fundamentals']
            
            # ROE (Return on Equity) - plus élevé = meilleur
            if 'ROE' in fund_data.columns:
                quality_scores['ROE'] = fund_data['ROE']
            
            # ROA (Return on Assets) - plus élevé = meilleur
            if 'ROA' in fund_data.columns:
                quality_scores['ROA'] = fund_data['ROA']
            
            # Debt to Equity - plus bas = meilleur
            if 'Debt to Equity' in fund_data.columns:
                quality_scores['Debt to Equity'] = fund_data['Debt to Equity']
                # Inverser le score pour que les valeurs élevées soient meilleures
                quality_scores['Debt to Equity'] = 1 / quality_scores['Debt to Equity']
            
            # Gross Margin - plus élevé = meilleur
            if 'Gross Margin' in fund_data.columns:
                quality_scores['Gross Margin'] = fund_data['Gross Margin']
            
            # Net Margin - plus élevé = meilleur
            if 'Net Margin' in fund_data.columns:
                quality_scores['Net Margin'] = fund_data['Net Margin']
        
        # Si les données fondamentales ne sont pas disponibles ou incomplètes, utiliser des proxys
        if quality_scores.empty or quality_scores.isna().all().all():
            # Utiliser la stabilité des rendements comme proxy de qualité
            returns = self.data['prices'].pct_change().dropna()
            quality_scores = pd.DataFrame(index=self.universe)
            
            # Stabilité des rendements (inverse de l'écart-type) - moins volatile = meilleur
            volatility = returns.std()
            quality_scores['Stability'] = 1 / volatility
        
        # Normaliser les scores
        scaler = StandardScaler()
        normalized_scores = pd.DataFrame(
            scaler.fit_transform(quality_scores.fillna(quality_scores.mean())),
            index=quality_scores.index,
            columns=quality_scores.columns
        )
        
        # Calculer le score moyen Quality
        quality_score = normalized_scores.mean(axis=1)
        
        # Stocker le score
        self.factor_scores['quality'] = quality_score
        
        print(f"Score Quality calculé pour {len(quality_score)} titres")
        
    def _calculate_low_volatility_factor(self):
        """Calcule le score Low Volatility basé sur la volatilité historique"""
        print("Calcul du facteur Low Volatility...")
        
        # Calcul des rendements
        returns = self.data['prices'].pct_change().dropna()
        
        # Créer le dataframe pour stocker les scores de volatilité
        volatility_scores = pd.DataFrame(index=self.universe)
        
        # Volatilité sur 1 an (252 jours)
        vol_1y = returns.iloc[-252:].std() * np.sqrt(252)
        volatility_scores['1y'] = 1 / vol_1y  # Inverser pour que les moins volatiles aient un score plus élevé
        
        # Volatilité sur 6 mois (126 jours)
        vol_6m = returns.iloc[-126:].std() * np.sqrt(252)
        volatility_scores['6m'] = 1 / vol_6m
        
        # Volatilité sur 3 mois (63 jours)
        vol_3m = returns.iloc[-63:].std() * np.sqrt(252)
        volatility_scores['3m'] = 1 / vol_3m
        
        # Normaliser les scores
        scaler = StandardScaler()
        normalized_scores = pd.DataFrame(
            scaler.fit_transform(volatility_scores.fillna(volatility_scores.mean())),
            index=volatility_scores.index,
            columns=volatility_scores.columns
        )
        
        # Calculer le score moyen Low Volatility
        low_vol_score = normalized_scores.mean(axis=1)
        
        # Stocker le score
        self.factor_scores['low_volatility'] = low_vol_score
        
        print(f"Score Low Volatility calculé pour {len(low_vol_score)} titres")
        
    def _calculate_esg_factor(self):
        """Calcule le score ESG basé sur les métriques environnementales, sociales et de gouvernance"""
        print("Calcul du facteur ESG...")
        
        # Initialiser le dataframe pour les scores ESG
        esg_scores = pd.DataFrame(index=self.universe)
        
        # Récupérer les métriques ESG à partir des données fondamentales
        if 'fundamentals' in self.data and not self.data['fundamentals'].empty:
            fund_data = self.data['fundamentals']
            
            # ESG Score - plus élevé = meilleur (selon la méthodologie utilisée)
            if 'ESG Score' in fund_data.columns:
                esg_scores['ESG Score'] = fund_data['ESG Score']
        
        # Si les données ESG ne sont pas disponibles, utiliser un score aléatoire (pour démonstration)
        if esg_scores.empty or esg_scores.isna().all().all():
            # Avertissement: Ce bloc est uniquement pour la démonstration
            np.random.seed(42)  # Pour la reproductibilité
            esg_scores['ESG Score'] = pd.Series(
                np.random.uniform(0, 100, size=len(self.universe)),
                index=self.universe
            )
            print("Avertissement: Données ESG non disponibles, utilisation de valeurs aléatoires pour la démonstration")
        
        # Normaliser les scores
        scaler = StandardScaler()
        normalized_scores = pd.DataFrame(
            scaler.fit_transform(esg_scores.fillna(esg_scores.mean())),
            index=esg_scores.index,
            columns=esg_scores.columns
        )
        
        # Calculer le score moyen ESG
        esg_score = normalized_scores.mean(axis=1)
        
        # Stocker le score
        self.factor_scores['esg'] = esg_score
        
        print(f"Score ESG calculé pour {len(esg_score)} titres")
        
    def _calculate_combined_score(self, factors, weights=None):
        """
        Calcule le score combiné en utilisant une pondération des facteurs
        
        Parameters:
        -----------
        factors : list
            Liste des facteurs à inclure
        weights : dict
            Dictionnaire avec les poids pour chaque facteur (somme = 1)
        """
        print("Calcul du score combiné...")
        
        # Vérifier si les poids sont fournis, sinon utiliser des poids égaux
        if weights is None:
            weights = {factor: 1/len(factors) for factor in factors}
        
        # Normaliser les poids pour qu'ils somment à 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Créer un dataframe pour le score combiné
        combined_score = pd.Series(0, index=self.universe)
        
        # Additionner les scores pondérés
        for factor, weight in weights.items():
            if factor in self.factor_scores:
                combined_score += self.factor_scores[factor] * weight
        
        # Stocker le score combiné
        self.factor_scores['combined'] = combined_score
        
        print(f"Score combiné calculé avec les poids suivants: {weights}")
        
        return combined_score
    
    def optimize_portfolio(self, n_stocks=50, max_weight=0.05, min_liquidity_percentile=0.2):
        """
        Optimise le portefeuille en sélectionnant les titres avec les meilleurs scores
        
        Parameters:
        -----------
        n_stocks : int
            Nombre de titres à inclure dans le portefeuille
        max_weight : float
            Poids maximum pour un titre (contrainte de diversification)
        min_liquidity_percentile : float
            Percentile minimum de liquidité pour inclure un titre
        """
        print(f"Optimisation du portefeuille avec {n_stocks} titres...")
        
        # Vérifier que le score combiné est calculé
        if 'combined' not in self.factor_scores:
            raise ValueError("Le score combiné n'a pas été calculé. Exécutez d'abord calculate_factor_scores().")
        
        # Filtrer les titres avec une liquidité suffisante
        if 'volumes' in self.data:
            # Calculer la liquidité moyenne sur 30 jours
            liquidity = self.data['volumes'].iloc[-30:].mean()
            
            # Déterminer le seuil de liquidité
            liquidity_threshold = liquidity.quantile(min_liquidity_percentile)
            
            # Filtrer les titres avec une liquidité suffisante
            liquid_tickers = liquidity[liquidity >= liquidity_threshold].index.tolist()
            print(f"{len(liquid_tickers)} titres passent le filtre de liquidité")
            
            # Filtrer les scores combinés
            filtered_scores = self.factor_scores['combined'][liquid_tickers]
        else:
            filtered_scores = self.factor_scores['combined']
        
        # Sélectionner les n_stocks titres avec les meilleurs scores
        selected_tickers = filtered_scores.nlargest(n_stocks).index.tolist()
        
        # Calculer les poids initiaux basés sur les scores
        initial_weights = filtered_scores[selected_tickers]
        initial_weights = initial_weights / initial_weights.sum()
        
        # Appliquer la contrainte de poids maximum
        capped_weights = np.minimum(initial_weights, max_weight)
        
        # Renormaliser les poids pour qu'ils somment à 1
        etf_weights = capped_weights / capped_weights.sum()
        
        # Stocker les poids de l'ETF
        self.etf_weights = etf_weights
        
        # Calculer l'allocation sectorielle
        if 'sectors' in self.data:
            self._calculate_sector_allocation()
        
        print(f"Portefeuille optimisé avec {len(etf_weights)} titres")
        
        return etf_weights
    
    def _calculate_sector_allocation(self):
        """Calcule l'allocation sectorielle du portefeuille"""
        if self.etf_weights is None or 'sectors' not in self.data:
            return
        
        # Récupérer les secteurs des titres du portefeuille
        portfolio_sectors = self.data['sectors'][self.etf_weights.index]
        
        # Calculer l'allocation par secteur
        sector_allocation = pd.Series(0.0, index=portfolio_sectors.unique())
        
        for ticker in self.etf_weights.index:
            sector = self.data['sectors'][ticker]
            sector_allocation[sector] += self.etf_weights[ticker]
        
        # Stocker l'allocation sectorielle
        self.sector_allocation = sector_allocation.sort_values(ascending=False)
    
    def backtest(self, rebalancing_frequency='quarterly'):
        """
        Effectue un backtesting de la stratégie
        
        Parameters:
        -----------
        rebalancing_frequency : str
            Fréquence de rebalancement ('monthly', 'quarterly', 'semi-annually', 'annually')
        """
        print(f"Backtesting avec rebalancement {rebalancing_frequency}...")
        
        # Vérifier que les poids sont calculés
        if self.etf_weights is None:
            raise ValueError("Les poids du portefeuille n'ont pas été calculés. Exécutez d'abord optimize_portfolio().")
        
        # Définir la fenêtre de rebalancement en jours
        rebalancing_days = {
            'monthly': 21,
            'quarterly': 63,
            'semi-annually': 126,
            'annually': 252
        }
        
        window = rebalancing_days.get(rebalancing_frequency, 63)  # 63 jours par défaut (trimestriel)
        
        # Récupérer les prix
        prices = self.data['prices']
        
        # Initialiser le dataframe pour la NAV de l'ETF
        etf_nav = pd.Series(index=prices.index, dtype=float)
        
        # Valeur initiale
        initial_value = 100
        etf_nav.iloc[0] = initial_value
        
        # Calculer les rendements journaliers
        daily_returns = prices.pct_change().fillna(0)
        
        # Initialiser les poids du portefeuille
        current_weights = self.etf_weights.copy()
        
        # Simulation
        for i in range(1, len(prices)):
            # Calculer la NAV pour le jour i
            if i == 1:
                # Premier jour après l'initialisation
                etf_nav.iloc[i] = initial_value * (1 + (daily_returns.iloc[i][current_weights.index] * current_weights).sum())
            else:
                # Appliquer les rendements du jour
                etf_nav.iloc[i] = etf_nav.iloc[i-1] * (1 + (daily_returns.iloc[i][current_weights.index] * current_weights).sum())
            
            # Rebalancement du portefeuille selon la fréquence définie
            if i % window == 0:
                # Dans un cas réel, nous recalculerions les scores et les poids ici
                # Pour cette simulation, nous gardons les poids initiaux
                
                # Nous pourrions modéliser les coûts de rebalancement si nécessaire
                rebalance_cost = 0.001  # 0.1% de coûts de transaction
                etf_nav.iloc[i] = etf_nav.iloc[i] * (1 - rebalance_cost)
        
        # Stocker la NAV de l'ETF
        self.etf_nav = etf_nav
        
        # Calculer la NAV du benchmark pour la même période
        benchmark_prices = self.data['benchmark']
        benchmark_returns = benchmark_prices.pct_change().fillna(0)
        
        benchmark_nav = pd.Series(index=benchmark_prices.index, dtype=float)
        benchmark_nav.iloc[0] = initial_value
        
        for i in range(1, len(benchmark_prices)):
            benchmark_nav.iloc[i] = benchmark_nav.iloc[i-1] * (1 + benchmark_returns.iloc[i])
        
        # Stocker la NAV du benchmark
        self.benchmark_nav = benchmark_nav
        
        # Calculer les drawdowns
        self._calculate_drawdowns()
        
        print(f"Backtesting terminé sur {len(etf_nav)} jours")
        
        # Calculer les métriques de performance
        self.calculate_performance_metrics()
        
        return etf_nav, benchmark_nav
    
    def _calculate_drawdowns(self):
        """Calcule les drawdowns pour l'ETF et le benchmark"""
        # Drawdowns de l'ETF
        if self.etf_nav is None:
            return
            
        # Calcul des drawdowns de l'ETF
        etf_peak = self.etf_nav.cummax()
        etf_drawdowns = (self.etf_nav - etf_peak) / etf_peak
        
        # Calcul des drawdowns du benchmark
        benchmark_peak = self.benchmark_nav.cummax()
        benchmark_drawdowns = (self.benchmark_nav - benchmark_peak) / benchmark_peak
        
        # Stockage des drawdowns
        self.drawdowns = pd.DataFrame({
            'ETF': etf_drawdowns,
            'Benchmark': benchmark_drawdowns
        })
    
    def calculate_performance_metrics(self):
        """Calcule les métriques de performance de l'ETF et du benchmark"""
        if self.etf_nav is None or self.benchmark_nav is None:
            print("Aucune donnée de NAV disponible. Exécutez d'abord backtest().")
            return
        
        # Calcul des rendements journaliers
        etf_returns = self.etf_nav.pct_change().dropna()
        benchmark_returns = self.benchmark_nav.pct_change().dropna()
        
        # Rendements annualisés
        etf_annual_return = ((self.etf_nav.iloc[-1] / self.etf_nav.iloc[0]) ** (252 / len(self.etf_nav))) - 1
        benchmark_annual_return = ((self.benchmark_nav.iloc[-1] / self.benchmark_nav.iloc[0]) ** (252 / len(self.benchmark_nav))) - 1
        
        # Volatilité annualisée
        etf_volatility = etf_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Ratio de Sharpe (avec taux sans risque = 0 pour simplifier)
        etf_sharpe = etf_annual_return / etf_volatility
        benchmark_sharpe = benchmark_annual_return / benchmark_volatility
        
        # Alpha et Beta (régression sur les rendements du benchmark)
        X = benchmark_returns.values.reshape(-1, 1)
        y = etf_returns.values
        beta, alpha = np.polyfit(X.flatten(), y, 1)
        alpha = alpha * 252  # Annualisé
        
        # Tracking Error
        tracking_diff = etf_returns - benchmark_returns
        tracking_error = tracking_diff.std() * np.sqrt(252)
        
        # Maximum Drawdown
        etf_max_drawdown = self.drawdowns['ETF'].min()
        benchmark_max_drawdown = self.drawdowns['Benchmark'].min()
        
        # Nombre de mois positifs / négatifs
        monthly_returns = self.etf_nav.resample('M').last().pct_change().dropna()
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns <= 0).sum()
        
        # Ratio d'information
        information_ratio = (etf_annual_return - benchmark_annual_return) / tracking_error
        
        # Stocker les métriques de performance
        self.performance_metrics = {
            'ETF Annual Return': etf_annual_return,
            'Benchmark Annual Return': benchmark_annual_return,
            'ETF Volatility': etf_volatility,
            'Benchmark Volatility': benchmark_volatility,
            'ETF Sharpe Ratio': etf_sharpe,
            'Benchmark Sharpe Ratio': benchmark_sharpe,
            'Alpha': alpha,
            'Beta': beta,
            'Tracking Error': tracking_error,
            'Information Ratio': information_ratio,
            'ETF Max Drawdown': etf_max_drawdown,
            'Benchmark Max Drawdown': benchmark_max_drawdown,
            'Positive Months': positive_months,
            'Negative Months': negative_months,
            'Positive/Negative Ratio': positive_months / negative_months if negative_months > 0 else float('inf')
        }
        
        return self.performance_metrics
    
    def plot_performance(self, save_path=None):
        """
        Affiche un graphique de performance comparant l'ETF Smart Beta et le benchmark
        
        Parameters:
        -----------
        save_path : str
            Chemin où enregistrer le graphique (optionnel)
        """
        if self.etf_nav is None or self.benchmark_nav is None:
            print("Aucune donnée de NAV disponible. Exécutez d'abord backtest().")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Normaliser les séries pour qu'elles commencent à 100
        etf_normalized = self.etf_nav / self.etf_nav.iloc[0] * 100
        benchmark_normalized = self.benchmark_nav / self.benchmark_nav.iloc[0] * 100
        
        # Tracer les performances
        plt.plot(etf_normalized.index, etf_normalized, 'b-', linewidth=2, label=f'ETF Smart Beta (↑{self.performance_metrics["ETF Annual Return"]:.2%}/an)')
        plt.plot(benchmark_normalized.index, benchmark_normalized, 'g--', linewidth=2, label=f'{self.benchmark_ticker} (↑{self.performance_metrics["Benchmark Annual Return"]:.2%}/an)')
        
        # Ajouter titre et légendes
        plt.title(f'Performance de l\'ETF Smart Beta vs {self.benchmark_ticker}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Valeur (base 100)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Formatter l'axe Y
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x)}" if x % 1 == 0 else f"{x:.1f}"))
        
        # Ajouter des informations de performance
        plt.figtext(0.15, 0.15, 
                  f'Alpha: {self.performance_metrics["Alpha"]:.2%}\n'
                  f'Beta: {self.performance_metrics["Beta"]:.2f}\n'
                  f'Sharpe Ratio: {self.performance_metrics["ETF Sharpe Ratio"]:.2f}',
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # Sauvegarder si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique de performance enregistré: {save_path}")
        
        plt.show()

    def plot_drawdowns(self, save_path=None):
        """
        Affiche un graphique des drawdowns
        
        Parameters:
        -----------
        save_path : str
            Chemin où enregistrer le graphique (optionnel)
        """
        if self.drawdowns is None:
            print("Aucune donnée de drawdowns disponible. Exécutez d'abord backtest().")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Tracer les drawdowns
        plt.plot(self.drawdowns.index, self.drawdowns['ETF'] * 100, 'b-', linewidth=2, label='ETF Smart Beta')
        plt.plot(self.drawdowns.index, self.drawdowns['Benchmark'] * 100, 'g--', linewidth=2, label=self.benchmark_ticker)
        
        # Ajouter titre et légendes
        plt.title(f'Drawdowns de l\'ETF Smart Beta vs {self.benchmark_ticker}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Inverser l'axe Y pour que les drawdowns soient vers le bas
        plt.gca().invert_yaxis()
        
        # Ajouter une ligne horizontale à zéro
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Ajouter des informations de performance
        plt.figtext(0.15, 0.15, 
                  f'Max Drawdown ETF: {self.performance_metrics["ETF Max Drawdown"]*100:.2f}%\n'
                  f'Max Drawdown {self.benchmark_ticker}: {self.performance_metrics["Benchmark Max Drawdown"]*100:.2f}%',
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # Sauvegarder si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique des drawdowns enregistré: {save_path}")
        
        plt.show()
    
    def plot_factor_exposures(self, save_path=None):
        """
        Affiche un graphique radar des expositions factorielles
        
        Parameters:
        -----------
        save_path : str
            Chemin où enregistrer le graphique (optionnel)
        """
        if self.etf_weights is None or not self.factor_scores:
            print("Aucune donnée de portefeuille ou de facteurs disponible.")
            return
        
        # Calculer les expositions factorielles du portefeuille
        exposures = {}
        factors = [f for f in self.factor_scores.keys() if f != 'combined']
        
        for factor in factors:
            if factor in self.factor_scores:
                # Exposition = score moyen pondéré des titres du portefeuille
                exposures[factor] = (self.factor_scores[factor][self.etf_weights.index] * self.etf_weights).sum()
        
        # Normaliser les expositions sur une échelle de 0 à 1
        max_exposure = max(exposures.values())
        min_exposure = min(exposures.values())
        range_exposure = max_exposure - min_exposure
        
        normalized_exposures = {k: (v - min_exposure) / range_exposure if range_exposure > 0 else 0.5 for k, v in exposures.items()}
        
        # Créer le graphique radar
        categories = list(normalized_exposures.keys())
        values = list(normalized_exposures.values())
        
        # Répéter le premier élément pour fermer le graphique radar
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        # Convertir en radians pour le tracé
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Fermer le graphique radar
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Tracer les expositions factorielles
        ax.plot(angles, values, 'o-', linewidth=2, label='ETF Smart Beta')
        ax.fill(angles, values, alpha=0.25)
        
        # Ajouter les labels des facteurs
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1], fontsize=12)
        
        # Ajouter des cercles de référence
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.set_rlabel_position(0)
        
        # Ajouter titre
        plt.title('Expositions Factorielles de l\'ETF Smart Beta', fontsize=14, y=1.1)
        
        plt.tight_layout()
        
        # Sauvegarder si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique des expositions factorielles enregistré: {save_path}")
        
        plt.show()
    
    def plot_sector_allocation(self, save_path=None):
        """
        Affiche un graphique de l'allocation sectorielle
        
        Parameters:
        -----------
        save_path : str
            Chemin où enregistrer le graphique (optionnel)
        """
        if self.sector_allocation is None:
            print("Aucune donnée d'allocation sectorielle disponible.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Créer le graphique en barres horizontales
        sector_allocation_pct = self.sector_allocation * 100
        sector_allocation_pct = sector_allocation_pct.sort_values(ascending=True)
        
        bars = plt.barh(sector_allocation_pct.index, sector_allocation_pct, color=sns.color_palette("viridis", len(sector_allocation_pct)))
        
        # Ajouter des étiquettes avec les pourcentages
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                    ha='left', va='center', fontsize=10)
        
        # Ajouter titre et légendes
        plt.title('Allocation Sectorielle de l\'ETF Smart Beta', fontsize=14)
        plt.xlabel('Allocation (%)', fontsize=12)
        plt.ylabel('Secteur', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Sauvegarder si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique de l'allocation sectorielle enregistré: {save_path}")
        
        plt.show()
    
    def export_portfolio(self, file_path):
        """
        Exporte la composition du portefeuille vers un fichier CSV
        
        Parameters:
        -----------
        file_path : str
            Chemin où enregistrer le fichier CSV
        """
        if self.etf_weights is None:
            print("Aucune donnée de portefeuille disponible. Exécutez d'abord optimize_portfolio().")
            return
        
        # Créer un DataFrame avec les poids du portefeuille
        portfolio_df = pd.DataFrame({'Weight': self.etf_weights})
        
        # Ajouter les scores factoriels
        for factor, scores in self.factor_scores.items():
            portfolio_df[f'{factor.capitalize()} Score'] = scores[portfolio_df.index]
        
        # Ajouter le secteur si disponible
        if 'sectors' in self.data:
            portfolio_df['Sector'] = self.data['sectors'][portfolio_df.index]
        
        # Ajouter les prix actuels si disponibles
        if 'prices' in self.data:
            portfolio_df['Last Price'] = self.data['prices'].iloc[-1][portfolio_df.index]
        
        # Trier par poids décroissant
        portfolio_df = portfolio_df.sort_values(by='Weight', ascending=False)
        
        # Convertir les poids en pourcentage
        portfolio_df['Weight'] = portfolio_df['Weight'] * 100
        
        # Enregistrer en CSV
        portfolio_df.to_csv(file_path)
        print(f"Composition du portefeuille exportée: {file_path}")
    
    def export_performance_report(self, file_path):
        """
        Exporte un rapport de performance complet vers un fichier Excel
        
        Parameters:
        -----------
        file_path : str
            Chemin où enregistrer le fichier Excel
        """
        if self.performance_metrics is None:
            print("Aucune métrique de performance disponible. Exécutez d'abord backtest().")
            return
        
        # Créer un writer Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Feuille 1: Métriques de performance
            metrics_df = pd.DataFrame({
                'Métrique': list(self.performance_metrics.keys()),
                'Valeur': list(self.performance_metrics.values())
            })
            metrics_df.to_excel(writer, sheet_name='Performance', index=False)
            
            # Feuille 2: Composition du portefeuille
            if self.etf_weights is not None:
                portfolio_df = pd.DataFrame({'Poids (%)': self.etf_weights * 100})
                if 'sectors' in self.data:
                    portfolio_df['Secteur'] = self.data['sectors'][portfolio_df.index]
                portfolio_df = portfolio_df.sort_values(by='Poids (%)', ascending=False)
                portfolio_df.to_excel(writer, sheet_name='Portefeuille')
            
            # Feuille 3: NAV historique
            if self.etf_nav is not None and self.benchmark_nav is not None:
                nav_df = pd.DataFrame({
                    'ETF Smart Beta': self.etf_nav,
                    f'{self.benchmark_ticker}': self.benchmark_nav
                })
                nav_df.to_excel(writer, sheet_name='NAV Historique')
            
            # Feuille 4: Drawdowns
            if self.drawdowns is not None:
                self.drawdowns.to_excel(writer, sheet_name='Drawdowns')
            
            # Feuille 5: Allocation sectorielle
            if self.sector_allocation is not None:
                sector_df = pd.DataFrame({'Allocation (%)': self.sector_allocation * 100})
                sector_df = sector_df.sort_values(by='Allocation (%)', ascending=False)
                sector_df.to_excel(writer, sheet_name='Allocation Sectorielle')
        
        print(f"Rapport de performance exporté: {file_path}")
    
    def run_full_analysis(self, sample_size=100, n_stocks=30, max_weight=0.05, 
                         rebalancing_frequency='quarterly', factors=None, output_dir='output'):
        """
        Exécute l'analyse complète en une seule méthode
        
        Parameters:
        -----------
        sample_size : int
            Taille de l'échantillon pour l'univers d'investissement
        n_stocks : int
            Nombre de titres à inclure dans le portefeuille
        max_weight : float
            Poids maximum pour un titre
        rebalancing_frequency : str
            Fréquence de rebalancement ('monthly', 'quarterly', 'semi-annually', 'annually')
        factors : list
            Liste des facteurs à inclure (par défaut: tous)
        output_dir : str
            Répertoire de sortie pour les résultats
        """
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Répertoire de sortie créé: {output_dir}")
        
        # Étape 1: Récupération des données
        print("\n=== Étape 1: Récupération des données ===")
        self.fetch_data(sample_size=sample_size)
        
        # Étape 2: Calcul des scores factoriels
        print("\n=== Étape 2: Calcul des scores factoriels ===")
        self.calculate_factor_scores(factors=factors)
        
        # Étape 3: Optimisation du portefeuille
        print("\n=== Étape 3: Optimisation du portefeuille ===")
        self.optimize_portfolio(n_stocks=n_stocks, max_weight=max_weight)
        
        # Étape 4: Backtesting
        print("\n=== Étape 4: Backtesting ===")
        self.backtest(rebalancing_frequency=rebalancing_frequency)
        
        # Étape 5: Visualisation des résultats
        print("\n=== Étape 5: Visualisation des résultats ===")
        
        # Graphique de performance
        self.plot_performance(save_path=os.path.join(output_dir, 'performance.png'))
        
        # Graphique des drawdowns
        self.plot_drawdowns(save_path=os.path.join(output_dir, 'drawdowns.png'))
        
        # Graphique des expositions factorielles
        self.plot_factor_exposures(save_path=os.path.join(output_dir, 'factor_exposures.png'))
        
        # Graphique de l'allocation sectorielle
        self.plot_sector_allocation(save_path=os.path.join(output_dir, 'sector_allocation.png'))
        
        # Étape 6: Exportation des résultats
        print("\n=== Étape 6: Exportation des résultats ===")
        
        # Exporter la composition du portefeuille
        self.export_portfolio(file_path=os.path.join(output_dir, 'portfolio.csv'))
        
        # Exporter le rapport de performance
        self.export_performance_report(file_path=os.path.join(output_dir, 'performance_report.xlsx'))
        
        print("\n=== Analyse complète terminée ===")
        print(f"Les résultats sont disponibles dans le répertoire: {output_dir}")
        
        # Afficher un résumé des performances
        if self.performance_metrics:
            print("\nRésumé des performances:")
            print(f"Rendement annualisé de l'ETF: {self.performance_metrics['ETF Annual Return']:.2%}")
            print(f"Rendement annualisé du benchmark: {self.performance_metrics['Benchmark Annual Return']:.2%}")
            print(f"Ratio de Sharpe de l'ETF: {self.performance_metrics['ETF Sharpe Ratio']:.2f}")
            print(f"Alpha: {self.performance_metrics['Alpha']:.2%}")
            print(f"Beta: {self.performance_metrics['Beta']:.2f}")
            print(f"Maximum Drawdown de l'ETF: {self.performance_metrics['ETF Max Drawdown']:.2%}")
