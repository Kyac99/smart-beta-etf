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
