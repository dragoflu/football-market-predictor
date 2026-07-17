"""
Football Prediction Model: ансамбль Dixon-Coles + ELO + XGBoost.

Архитектура:
  1. Dixon-Coles (Bivariate Poisson): baseline
  2. ELO logistic: быстрый рейтинговый прогноз
  3. XGBoost на фичах: нелинейные паттерны
  4. Ensemble: взвешенная комбинация → Isotonic Regression калибровка
  5. Value detector: model_prob vs Polymarket price → Quarter Kelly sizing

Метрики оптимизации: log_loss / Brier Score (НЕ accuracy!)
Валидация: walk-forward по сезонам
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning('xgboost not installed. Run: pip install xgboost')

log = logging.getLogger(__name__)

MODELS_DIR = Path('data/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _dc_tau(x, y, lambda_, mu, rho):
    """Dixon-Coles correction for low-scoring matches (scalar version)."""
    if x == 0 and y == 0:
        return 1.0 - lambda_ * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_ * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _dc_tau_vec(hg, ag, lambda_, mu, rho):
    """Vectorized Dixon-Coles tau correction."""
    tau = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lambda_[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lambda_[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


def _prepare_match_arrays(matches: pd.DataFrame, teams: list[str]):
    """Pre-compute numpy arrays from DataFrame (called once, not per iteration)."""
    team_idx = {t: i for i, t in enumerate(teams)}
    df = matches.dropna(subset=['FTHG', 'FTAG']).copy()

    h_idx = df['HomeTeam'].map(team_idx)
    a_idx = df['AwayTeam'].map(team_idx)
    valid = h_idx.notna() & a_idx.notna()
    df = df[valid]

    return {
        'h_idx': h_idx[valid].values.astype(np.int32),
        'a_idx': a_idx[valid].values.astype(np.int32),
        'hg': df['FTHG'].values.astype(np.int32),
        'ag': df['FTAG'].values.astype(np.int32),
        'days_ago': df['days_ago'].values.astype(np.float64) if 'days_ago' in df.columns else None,
        'n': len(df),
    }


def _dc_log_likelihood_vec(params, arrays: dict, n_teams: int, xi: float = 0.0) -> float:
    """Vectorized negative log-likelihood (100-1000x faster than iterrows)."""
    attack = params[:n_teams]
    defence = params[n_teams:2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    h_idx = arrays['h_idx']
    a_idx = arrays['a_idx']
    hg = arrays['hg']
    ag = arrays['ag']

    lambda_ = np.exp(attack[h_idx] - defence[a_idx] + home_adv)
    mu = np.exp(attack[a_idx] - defence[h_idx])

    # без клампа rho уводит tau в отрицательные значения
    lambda_ = np.clip(lambda_, 1e-10, 20.0)
    mu = np.clip(mu, 1e-10, 20.0)

    tau = _dc_tau_vec(hg, ag, lambda_, mu, rho)
    tau = np.clip(tau, 1e-10, None)

    ll = np.log(tau) + poisson.logpmf(hg, lambda_) + poisson.logpmf(ag, mu)

    if xi > 0 and arrays['days_ago'] is not None:
        weights = np.exp(-xi * arrays['days_ago'])
        ll *= weights

    return -ll.sum()


class DixonColesModel:
    """
    Bivariate Poisson модель Dixon-Coles с time-weighting.

    Использование:
        model = DixonColesModel()
        model.fit(df)
        probs = model.predict_proba(home_team, away_team)
        # → {'home': 0.45, 'draw': 0.27, 'away': 0.28}
    """

    def __init__(self, xi: float = 0.002, max_goals: int = 10):
        self.xi = xi           # time decay (0.002 ≈ Maher 1982 recommendation)
        self.max_goals = max_goals
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defence_: dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, ref_date: Optional[pd.Timestamp] = None) -> 'DixonColesModel':
        """
        Обучает модель на исторических матчах.

        Args:
            df: датафрейм с колонками Date, HomeTeam, AwayTeam, FTHG, FTAG
            ref_date: дата относительно которой считать time-weighting
                      (None = max(Date) в датафрейме)
        """
        df = df.dropna(subset=['FTHG', 'FTAG']).copy()
        df['Date'] = pd.to_datetime(df['Date'])

        if ref_date is None:
            ref_date = df['Date'].max()

        df['days_ago'] = (ref_date - df['Date']).dt.days

        self.teams_ = sorted(
            set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        )
        n = len(self.teams_)

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.1    # home_adv
        x0[2 * n + 1] = -0.1  # rho

        # сумма атак = 0, иначе параметры не идентифицируются
        constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[:n])}]

        bounds = (
            [(-3, 3)] * n +           # attack
            [(-3, 3)] * n +           # defence
            [(-0.5, 2.0)] +           # home_adv
            [(-0.5, 0.5)]             # rho
        )

        # один раз, а не на каждой итерации оптимизатора
        arrays = _prepare_match_arrays(df, self.teams_)

        result = minimize(
            _dc_log_likelihood_vec,
            x0,
            args=(arrays, n, self.xi),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9},
        )

        params = result.x
        for i, team in enumerate(self.teams_):
            self.attack_[team] = params[i]
            self.defence_[team] = params[n + i]

        self.home_adv_ = params[2 * n]
        self.rho_ = params[2 * n + 1]
        self.is_fitted = True

        log.debug(f'Dixon-Coles fitted: {n} teams, home_adv={self.home_adv_:.3f}, rho={self.rho_:.3f}')
        return self

    def _get_lambdas(self, home: str, away: str) -> tuple[float, float]:
        """Ожидаемые голы: (lambda_home, mu_away)."""
        a_h = self.attack_.get(home, 0.0)
        d_h = self.defence_.get(home, 0.0)
        a_a = self.attack_.get(away, 0.0)
        d_a = self.defence_.get(away, 0.0)

        lambda_ = np.exp(a_h - d_a + self.home_adv_)
        mu = np.exp(a_a - d_h)
        return lambda_, mu

    def predict_score_matrix(self, home: str, away: str) -> np.ndarray:
        """
        Матрица вероятностей счетов [0..max_goals] x [0..max_goals].
        score_matrix[i, j] = P(home_goals=i, away_goals=j)
        """
        lambda_, mu = self._get_lambdas(home, away)
        mg = self.max_goals

        score_matrix = np.outer(
            poisson.pmf(range(mg + 1), lambda_),
            poisson.pmf(range(mg + 1), mu),
        )

        # коррекция Dixon-Coles для счетов 0-0, 1-0, 0-1, 1-1
        for i in range(min(2, mg + 1)):
            for j in range(min(2, mg + 1)):
                tau = _dc_tau(i, j, lambda_, mu, self.rho_)
                score_matrix[i, j] *= tau

        score_matrix /= score_matrix.sum()
        return score_matrix

    def predict_proba(self, home: str, away: str) -> dict[str, float]:
        """
        Вероятности исхода матча.
        Returns: {'home': p_h, 'draw': p_d, 'away': p_a}
        """
        if not self.is_fitted:
            raise RuntimeError('Model not fitted. Call fit() first.')

        sm = self.predict_score_matrix(home, away)

        p_home = float(np.tril(sm, -1).sum())   # home goals > away goals
        p_away = float(np.triu(sm, 1).sum())     # away goals > home goals
        p_draw = float(np.trace(sm))             # equal goals

        return {'home': p_home, 'draw': p_draw, 'away': p_away}

class EloModel:
    """
    Простая логистическая регрессия на ELO difference → H/D/A вероятности.
    Быстрый, интерпретируемый baseline.
    """

    def __init__(self):
        self.model_home = LogisticRegression(C=1.0)
        self.model_away = LogisticRegression(C=1.0)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'EloModel':
        """
        df должен содержать: elo_diff, target (H/D/A)
        """
        df = df.dropna(subset=['elo_diff', 'target'])
        X = df[['elo_diff']].values

        y_home = (df['target'] == 'H').astype(int).values
        y_away = (df['target'] == 'A').astype(int).values

        self.model_home.fit(X, y_home)
        self.model_away.fit(X, y_away)
        self.is_fitted = True
        return self

    def predict_proba(self, elo_diff: float) -> dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted.')
        X = np.array([[elo_diff]])
        p_home = self.model_home.predict_proba(X)[0, 1]
        p_away = self.model_away.predict_proba(X)[0, 1]
        p_draw = max(0.0, 1.0 - p_home - p_away)
        total = p_home + p_draw + p_away
        return {'home': p_home / total, 'draw': p_draw / total, 'away': p_away / total}

# Фичи в том порядке, в котором они идут в матрице
FEATURE_COLS = [
    # ELO
    'elo_diff', 'elo_home', 'elo_away',
    # форма
    'home_goals_scored_last5', 'home_goals_scored_last10', 'home_goals_scored_last20',
    'home_goals_conceded_last5', 'home_goals_conceded_last10', 'home_goals_conceded_last20',
    'home_points_last5', 'home_points_last10', 'home_points_last20',
    'away_goals_scored_last5', 'away_goals_scored_last10', 'away_goals_scored_last20',
    'away_goals_conceded_last5', 'away_goals_conceded_last10', 'away_goals_conceded_last20',
    'away_points_last5', 'away_points_last10', 'away_points_last20',
    # rolling xG, Understat есть только с ~2014, дальше NaN
    'home_xg_scored_last5', 'home_xg_scored_last10',
    'home_xg_conceded_last5', 'home_xg_conceded_last10',
    'away_xg_scored_last5', 'away_xg_scored_last10',
    'away_xg_conceded_last5', 'away_xg_conceded_last10',
    # Pinnacle implied probs. Таргета не видят, но в бэктесте Pinnacle же
    # выступает рынком (96% ставок), так что edge частично self-referential.
    # Ablation без этих колонок: см. README, P1 держится (+11.1%).
    'implied_home', 'implied_draw', 'implied_away',
    'overround',
    # удары в створ
    'home_shots_on_target_last5', 'home_shots_on_target_last10',
    'away_shots_on_target_last5', 'away_shots_on_target_last10',
    # отдых и плотность календаря
    'home_days_rest', 'away_days_rest', 'rest_diff',
    'home_matches_last14', 'away_matches_last14', 'congestion_diff',
    # H2H
    'h2h_home_winrate', 'h2h_away_winrate', 'h2h_draw_rate',
    'h2h_home_goals_avg', 'h2h_away_goals_avg',
    # позиция в таблице
    'home_league_pos_pct', 'away_league_pos_pct', 'league_pos_diff',
    # склонность к ничьим, ключевое для draw-стратегий
    'home_draw_rate_last5', 'home_draw_rate_last10', 'home_draw_rate_last20',
    'away_draw_rate_last5', 'away_draw_rate_last10', 'away_draw_rate_last20',
    # сухие матчи дают больше 0-0
    'home_clean_sheet_rate_last5', 'home_clean_sheet_rate_last10',
    'away_clean_sheet_rate_last5', 'away_clean_sheet_rate_last10',
    # низкая результативность коррелирует с ничьими
    'home_total_goals_last5', 'home_total_goals_last10',
    'away_total_goals_last5', 'away_total_goals_last10',
    # в росте или в спаде: уровень ELO этого не показывает
    'home_elo_momentum', 'away_elo_momentum',
    # серия подряд
    'home_streak', 'away_streak',
    # стадия сезона
    'season_stage', 'home_season_matches_played', 'away_season_matches_played',
]


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Строит X (фичи), y (0=away, 1=draw, 2=home), и список колонок.
    Убирает строки где нет target или нет ключевых фичей.
    """
    df = df.dropna(subset=['target']).copy()

    available_cols = [c for c in FEATURE_COLS if c in df.columns]

    X = df[available_cols].values.astype(np.float32)

    label_map = {'H': 2, 'D': 1, 'A': 0}
    y = df['target'].map(label_map).values.astype(np.int32)

    return X, y, available_cols


class XGBoostModel:
    """XGBoost классификатор, оптимизированный на log_loss."""

    def __init__(self, n_estimators: int = 382, max_depth: int = 4,
                 learning_rate: float = 0.0163, subsample: float = 0.606,
                 colsample_bytree: float = 0.986, min_child_weight: int = 4,
                 reg_alpha: float = 0.816, reg_lambda: float = 1.423):
        if not HAS_XGB:
            raise ImportError('pip install xgboost')

        self.clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
        )
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_cols: list[str],
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'XGBoostModel':

        self.feature_cols = feature_cols
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.clf.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        return self

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, 3) array: [p_away, p_draw, p_home]."""
        return self.clf.predict_proba(X)

    def predict_proba(self, feat_dict: dict) -> dict[str, float]:
        """Inference для одного матча."""
        x = np.array([[feat_dict.get(c, np.nan) for c in self.feature_cols]], dtype=np.float32)
        proba = self.clf.predict_proba(x)[0]
        return {'away': float(proba[0]), 'draw': float(proba[1]), 'home': float(proba[2])}

    def feature_importance(self) -> pd.Series:
        imp = self.clf.feature_importances_
        return pd.Series(imp, index=self.feature_cols).sort_values(ascending=False)

@dataclass
class EnsembleWeights:
    dixon_coles: float = 0.000
    elo: float = 0.000
    xgboost: float = 1.000


class EnsembleModel:
    """
    Взвешенный ансамбль Dixon-Coles + ELO + XGBoost.
    Финальная калибровка через Isotonic Regression.

    Использование:
        model = EnsembleModel()
        model.fit(train_df)
        probs = model.predict_proba(home_team, away_team, feat_dict)
        value = model.compute_edge(probs, market_probs)
    """

    def __init__(self, weights: Optional[EnsembleWeights] = None):
        self.weights = weights or EnsembleWeights()
        self.dc = DixonColesModel(xi=0.002)
        self.elo_model = EloModel()
        self.xgb = XGBoostModel() if HAS_XGB else None
        self.calibrator_home: Optional[IsotonicRegression] = None
        self.calibrator_away: Optional[IsotonicRegression] = None
        self.calibrator_draw: Optional[IsotonicRegression] = None
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame,
            val_df: Optional[pd.DataFrame] = None) -> 'EnsembleModel':
        """
        Обучает все компоненты на train_df.
        Калибровка выполняется на val_df (если есть) или на train_df.

        Args:
            train_df: матчи с фичами (из football_features.build_features)
            val_df:   валидационный набор для калибровки
        """
        log.info('Fitting Dixon-Coles...')
        self.dc.fit(train_df)

        log.info('Fitting ELO logistic model...')
        self.elo_model.fit(train_df)

        if self.xgb is not None:
            log.info('Fitting XGBoost...')
            X, y, self.feature_cols = build_feature_matrix(train_df)

            if val_df is not None and len(val_df) > 0:
                X_val, y_val, _ = build_feature_matrix(val_df)
                self.xgb.fit(X, y, self.feature_cols, X_val, y_val)
            else:
                self.xgb.fit(X, y, self.feature_cols)

        calib_df = val_df if val_df is not None and len(val_df) > 0 else train_df
        log.info(f'Calibrating on {len(calib_df)} matches...')
        self._fit_calibration(calib_df)

        self.is_fitted = True
        log.info('Ensemble fitted.')
        return self

    def _raw_ensemble_proba(self, home: str, away: str,
                             feat_dict: dict) -> dict[str, float]:
        """Взвешенная комбинация без калибровки."""
        w = self.weights
        probs = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
        total_w = 0.0

        try:
            dc_p = self.dc.predict_proba(home, away)
            for k in probs:
                probs[k] += w.dixon_coles * dc_p[k]
            total_w += w.dixon_coles
        except Exception:
            pass

        elo_diff = feat_dict.get('elo_diff', 0.0)
        if not np.isnan(elo_diff):
            elo_p = self.elo_model.predict_proba(elo_diff)
            for k in probs:
                probs[k] += w.elo * elo_p[k]
            total_w += w.elo

        if self.xgb is not None and self.xgb.is_fitted:
            try:
                xgb_p = self.xgb.predict_proba(feat_dict)
                for k in probs:
                    probs[k] += w.xgboost * xgb_p[k]
                total_w += w.xgboost
            except Exception:
                pass

        if total_w > 0:
            for k in probs:
                probs[k] /= total_w

        return probs

    def _fit_calibration(self, df: pd.DataFrame):
        """Обучает Isotonic Regression калибраторы."""
        raw_home, raw_draw, raw_away = [], [], []
        true_home, true_draw, true_away = [], [], []

        for _, row in df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            feat = row.to_dict()
            target = row.get('target')

            if target not in ('H', 'D', 'A'):
                continue

            try:
                p = self._raw_ensemble_proba(home, away, feat)
            except Exception:
                continue

            raw_home.append(p['home'])
            raw_draw.append(p['draw'])
            raw_away.append(p['away'])
            true_home.append(1 if target == 'H' else 0)
            true_draw.append(1 if target == 'D' else 0)
            true_away.append(1 if target == 'A' else 0)

        if len(raw_home) < 50:
            log.warning('Too few samples for calibration, skipping.')
            return

        self.calibrator_home = IsotonicRegression(out_of_bounds='clip').fit(raw_home, true_home)
        self.calibrator_draw = IsotonicRegression(out_of_bounds='clip').fit(raw_draw, true_draw)
        self.calibrator_away = IsotonicRegression(out_of_bounds='clip').fit(raw_away, true_away)
        log.debug(f'Calibrators fitted on {len(raw_home)} samples.')

    def predict_proba(self, home: str, away: str,
                       feat_dict: dict) -> dict[str, float]:
        """
        Финальный прогноз с калибровкой.

        Returns: {'home': p_h, 'draw': p_d, 'away': p_a}
        """
        if not self.is_fitted:
            raise RuntimeError('Model not fitted.')

        raw = self._raw_ensemble_proba(home, away, feat_dict)

        if self.calibrator_home is not None:
            p_h = float(self.calibrator_home.predict([raw['home']])[0])
            p_d = float(self.calibrator_draw.predict([raw['draw']])[0])
            p_a = float(self.calibrator_away.predict([raw['away']])[0])
        else:
            p_h, p_d, p_a = raw['home'], raw['draw'], raw['away']

        total = p_h + p_d + p_a
        if total <= 0:
            return {'home': 1/3, 'draw': 1/3, 'away': 1/3}

        return {'home': p_h / total, 'draw': p_d / total, 'away': p_a / total}

    def compute_edge(self, model_probs: dict[str, float],
                      market_probs: dict[str, float]) -> dict[str, float]:
        """
        Вычисляет edge = model_prob - market_prob для каждого исхода.

        Returns: {'home_edge': e_h, 'draw_edge': e_d, 'away_edge': e_a,
                  'best_outcome': 'home'/'draw'/'away', 'best_edge': float}
        """
        edges = {
            'home_edge':  model_probs['home']  - market_probs.get('home', 0.33),
            'draw_edge':  model_probs['draw']  - market_probs.get('draw', 0.33),
            'away_edge':  model_probs['away']  - market_probs.get('away', 0.33),
        }
        best = max(edges, key=edges.get)
        edges['best_outcome'] = best.replace('_edge', '')
        edges['best_edge'] = edges[best]
        return edges

    def kelly_size(self, model_prob: float, market_prob: float,
                   bankroll: float = 1.0, fraction: float = 0.25) -> float:
        """
        Quarter Kelly position size.

        Args:
            model_prob:  наша вероятность победы
            market_prob: вероятность рынка (implied от цены)
            bankroll:    банкролл
            fraction:    доля Kelly (0.25 = Quarter Kelly)

        Returns:
            Размер позиции в $ (0 если нет edge)
        """
        if market_prob <= 0 or market_prob >= 1:
            return 0.0

        odds = 1.0 / market_prob  # decimal odds
        edge = model_prob * odds - 1.0  # Kelly edge = p * odds - 1

        if edge <= 0:
            return 0.0

        kelly = edge / (odds - 1.0)
        return bankroll * kelly * fraction

    def save(self, path: str | Path):
        joblib.dump(self, path)
        log.info(f'Model saved to {path}')

    @classmethod
    def load(cls, path: str | Path) -> 'EnsembleModel':
        model = joblib.load(path)
        log.info(f'Model loaded from {path}')
        return model

def walk_forward_validate(df: pd.DataFrame,
                           n_train_seasons: int = 5,
                           min_test_matches: int = 200,
                           edge_threshold: float = 0.08,
                           kelly_fraction: float = 0.25) -> pd.DataFrame:
    """
    Walk-forward валидация по сезонам.

    Train: сезоны 1..N → Test: сезон N+1
    Expanding window.

    Returns:
        DataFrame с результатами для каждого тестового матча:
        Date, HomeTeam, AwayTeam, target,
        pred_home, pred_draw, pred_away,
        market_home, market_draw, market_away,
        best_outcome, best_edge, kelly_size, profit
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Season'] = df['Season'].astype(str)
    seasons = sorted(df['Season'].unique())

    if len(seasons) < n_train_seasons + 1:
        log.warning(f'Need at least {n_train_seasons + 1} seasons, got {len(seasons)}')
        return pd.DataFrame()

    all_results = []

    for test_idx in range(n_train_seasons, len(seasons)):
        train_seasons = seasons[:test_idx]
        test_season = seasons[test_idx]

        train_df = df[df['Season'].isin(train_seasons)]
        test_df = df[df['Season'] == test_season]

        if len(test_df) < min_test_matches:
            log.info(f'  Skip {test_season}: only {len(test_df)} matches')
            continue

        log.info(f'  Train: {train_seasons[0]}..{train_seasons[-1]} ({len(train_df)} matches) '
                 f'→ Test: {test_season} ({len(test_df)} matches)')

        # Используем последний сезон трейна как val для калибровки
        val_df = train_df[train_df['Season'] == train_seasons[-1]]
        actual_train_df = train_df[train_df['Season'] != train_seasons[-1]]

        model = EnsembleModel()
        try:
            model.fit(actual_train_df, val_df)
        except Exception as e:
            log.warning(f'  Model fit failed: {e}')
            continue

        for _, row in test_df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            target = row.get('target')

            if target not in ('H', 'D', 'A'):
                continue

            try:
                probs = model.predict_proba(home, away, row.to_dict())
            except Exception:
                continue

            # Market prices: Betfair exchange (prediction market proxy) если есть,
            # иначе Pinnacle implied probs.
            # Betfair: peer-to-peer, шарпов не банят, ближайший аналог Polymarket
            bf_h = row.get('betfair_home', np.nan)
            if pd.notna(bf_h):
                market = {
                    'home': bf_h,
                    'draw': row.get('betfair_draw', np.nan),
                    'away': row.get('betfair_away', np.nan),
                    'source': 'betfair',
                }
            else:
                market = {
                    'home': row.get('implied_home', np.nan),
                    'draw': row.get('implied_draw', np.nan),
                    'away': row.get('implied_away', np.nan),
                    'source': 'pinnacle',
                }

            # Вычисляем edge только если есть рыночные цены
            if any(pd.isna(v) for k, v in market.items() if k != 'source'):
                continue

            edges = model.compute_edge(probs, market)
            best = edges['best_outcome']
            best_edge = edges['best_edge']

            # Стратегия: ставим только при edge > threshold
            if best_edge < edge_threshold:
                profit = 0.0
                bet_size = 0.0
            else:
                bet_size = model.kelly_size(
                    probs[best],
                    market[best],
                    bankroll=1.0,
                    fraction=kelly_fraction,
                )
                # Прибыль: если угадали → (1/market_prob - 1) * bet, иначе -bet
                correct = (
                    (best == 'home' and target == 'H') or
                    (best == 'draw' and target == 'D') or
                    (best == 'away' and target == 'A')
                )
                if correct:
                    profit = bet_size * (1.0 / market[best] - 1.0)
                else:
                    profit = -bet_size

            all_results.append({
                'Season': test_season,
                'Date': row['Date'],
                'League': row.get('League', ''),
                'HomeTeam': home,
                'AwayTeam': away,
                'target': target,
                'pred_home': probs['home'],
                'pred_draw': probs['draw'],
                'pred_away': probs['away'],
                'market_home': market['home'],
                'market_draw': market['draw'],
                'market_away': market['away'],
                'market_source': market.get('source', 'pinnacle'),
                'best_outcome': best,
                'best_edge': best_edge,
                'kelly_size': bet_size,
                'profit': profit,
                'correct': int(profit > 0) if bet_size > 0 else None,
            })

    results = pd.DataFrame(all_results)

    if len(results) > 0:
        # Суммарная статистика
        bets = results[results['kelly_size'] > 0]
        total_bets = len(bets)
        total_profit = bets['profit'].sum()
        total_staked = bets['kelly_size'].sum()
        roi = total_profit / total_staked if total_staked > 0 else 0.0
        win_rate = (bets['profit'] > 0).mean() if total_bets > 0 else 0.0

        log.info(f'\n=== Walk-Forward Results ===')
        log.info(f'  Total bets: {total_bets}')
        log.info(f'  Win rate: {win_rate:.1%}')
        log.info(f'  ROI: {roi:.1%}')
        log.info(f'  Total profit: {total_profit:.3f} units')

    return results

def evaluate_predictions(df: pd.DataFrame) -> dict:
    """
    Вычисляет метрики качества прогнозов.

    Args:
        df: результат walk_forward_validate()

    Returns:
        dict с метриками
    """
    df = df.dropna(subset=['target', 'pred_home', 'pred_draw', 'pred_away'])

    # Brier Score (по каждому исходу, потом среднее)
    bs_home = brier_score_loss(df['target'] == 'H', df['pred_home'])
    bs_draw = brier_score_loss(df['target'] == 'D', df['pred_draw'])
    bs_away = brier_score_loss(df['target'] == 'A', df['pred_away'])
    brier_avg = (bs_home + bs_draw + bs_away) / 3.0

    # Ranked Probability Score (RPS)
    # RPS = mean over matches of mean over ordered outcomes of (cum_pred - cum_actual)^2
    def rps(row):
        pred = [row['pred_home'], row['pred_draw'], row['pred_away']]
        actual = [
            1 if row['target'] == 'H' else 0,
            1 if row['target'] == 'D' else 0,
            1 if row['target'] == 'A' else 0,
        ]
        # Упорядочение: away=0, draw=1, home=2
        pred_ord = [row['pred_away'], row['pred_draw'], row['pred_home']]
        actual_ord = [
            1 if row['target'] == 'A' else 0,
            1 if row['target'] == 'D' else 0,
            1 if row['target'] == 'H' else 0,
        ]
        rps_val = 0.0
        cum_p, cum_a = 0.0, 0.0
        for p, a in zip(pred_ord[:-1], actual_ord[:-1]):
            cum_p += p
            cum_a += a
            rps_val += (cum_p - cum_a) ** 2
        return rps_val / (len(pred_ord) - 1)

    rps_score = df.apply(rps, axis=1).mean()

    # Log loss
    y_true = df['target'].map({'H': 2, 'D': 1, 'A': 0}).values
    y_pred = df[['pred_away', 'pred_draw', 'pred_home']].values
    ll = log_loss(y_true, y_pred)

    # Accuracy
    pred_outcome = df[['pred_home', 'pred_draw', 'pred_away']].idxmax(axis=1)
    pred_outcome = pred_outcome.map({'pred_home': 'H', 'pred_draw': 'D', 'pred_away': 'A'})
    accuracy = (pred_outcome == df['target']).mean()

    # Betting stats (если есть)
    bets = df[df['kelly_size'] > 0] if 'kelly_size' in df.columns else pd.DataFrame()
    roi = 0.0
    if len(bets) > 0:
        total_staked = bets['kelly_size'].sum()
        roi = bets['profit'].sum() / total_staked if total_staked > 0 else 0.0

    metrics = {
        'brier_score': round(brier_avg, 4),
        'rps': round(rps_score, 4),
        'log_loss': round(ll, 4),
        'accuracy': round(accuracy, 4),
        'roi': round(roi, 4),
        'n_matches': len(df),
        'n_bets': len(bets),
        # Benchmarks для сравнения
        'brier_benchmark': 0.210,    # уровень букмекера
        'rps_benchmark': 0.202,      # XGBoost + pi-ratings (Challenge 2017)
        'accuracy_benchmark': 0.535, # Pinnacle
    }

    return metrics

def tune_xgboost(features_df: pd.DataFrame, n_trials: int = 100) -> dict:
    """
    Optuna hyperparameter search для XGBoost.

    Использует последние 2 сезона как val, остальные как train.
    Оптимизирует log_loss.

    Returns:
        dict с лучшими параметрами (передать в XGBoostModel)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError('pip install optuna')

    if not HAS_XGB:
        raise ImportError('pip install xgboost')

    df = features_df.copy()
    df['Season'] = df['Season'].astype(str)
    seasons = sorted(df['Season'].unique())

    if len(seasons) < 4:
        raise ValueError(f'Need at least 4 seasons, got {len(seasons)}')

    val_seasons = seasons[-2:]
    train_seasons = seasons[:-2]

    train = df[df['Season'].isin(train_seasons)]
    val = df[df['Season'].isin(val_seasons)]

    X_train, y_train, cols = build_feature_matrix(train)
    X_val, y_val, _ = build_feature_matrix(val)

    log.info(f'Optuna: train={len(X_train)}, val={len(X_val)}, n_trials={n_trials}')

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        }
        clf = XGBClassifier(
            **params,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = clf.predict_proba(X_val)
        return log_loss(y_val, proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f'Best log_loss: {study.best_value:.4f}')
    log.info(f'Best params: {study.best_params}')
    return study.best_params


def tune_ensemble_weights(features_df: pd.DataFrame) -> EnsembleWeights:
    """
    Оптимизирует веса DC/ELO/XGBoost через scipy minimize на Brier Score.

    Returns:
        EnsembleWeights с оптимальными весами
    """
    from scipy.optimize import minimize as sp_minimize

    df = features_df.copy()
    df['Season'] = df['Season'].astype(str)
    seasons = sorted(df['Season'].unique())

    val_season = seasons[-1]
    train_seasons = seasons[:-1]
    train = df[df['Season'].isin(train_seasons)]
    val = df[df['Season'] == val_season]

    # Обучаем три компоненты по отдельности
    dc = DixonColesModel(xi=0.002)
    dc.fit(train)

    elo = EloModel()
    elo.fit(train)

    xgb = None
    if HAS_XGB:
        X_train, y_train, cols = build_feature_matrix(train)
        xgb = XGBoostModel()
        xgb.fit(X_train, y_train, cols)

    # Собираем прогнозы на val
    dc_preds, elo_preds, xgb_preds, targets = [], [], [], []
    for _, row in val.iterrows():
        if row.get('target') not in ('H', 'D', 'A'):
            continue
        try:
            dp = dc.predict_proba(row['HomeTeam'], row['AwayTeam'])
            ep = elo.predict_proba(row.get('elo_diff', 0.0))
        except Exception:
            continue

        dc_preds.append([dp['home'], dp['draw'], dp['away']])
        elo_preds.append([ep['home'], ep['draw'], ep['away']])

        if xgb is not None:
            xp = xgb.predict_proba(row.to_dict())
            xgb_preds.append([xp['home'], xp['draw'], xp['away']])
        else:
            xgb_preds.append([1/3, 1/3, 1/3])

        targets.append(row['target'])

    dc_arr = np.array(dc_preds)
    elo_arr = np.array(elo_preds)
    xgb_arr = np.array(xgb_preds)
    y_arr = np.array([(1 if t == 'H' else 0, 1 if t == 'D' else 0, 1 if t == 'A' else 0)
                      for t in targets], dtype=float)

    def brier(w):
        w = np.abs(w) / np.abs(w).sum()  # нормализуем, нет отрицательных
        blend = w[0] * dc_arr + w[1] * elo_arr + w[2] * xgb_arr
        blend = blend / blend.sum(axis=1, keepdims=True)
        return float(np.mean((blend - y_arr) ** 2))

    x0 = np.array([0.25, 0.15, 0.60])
    res = sp_minimize(brier, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-4})
    w_opt = np.abs(res.x) / np.abs(res.x).sum()

    log.info(f'Optimal weights: DC={w_opt[0]:.3f}, ELO={w_opt[1]:.3f}, XGB={w_opt[2]:.3f} '
             f'(Brier={res.fun:.4f})')
    return EnsembleWeights(dixon_coles=float(w_opt[0]),
                           elo=float(w_opt[1]),
                           xgboost=float(w_opt[2]))


if __name__ == '__main__':
    # Быстрый тест
    import glob

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    parquets = glob.glob('data/raw/football/matches_*.parquet')
    if not parquets:
        print('Нет данных.')
    else:
        print('Загружаем данные...')
        dfs = [pd.read_parquet(p) for p in parquets[:1]]   # только EPL для быстрого теста
        df = pd.concat(dfs, ignore_index=True)
        print(f'  {len(df)} matches loaded')

        print('\nТестируем Dixon-Coles...')
        train = df[df['Season'].isin(['2022/2023', '2023/2024'])].dropna(subset=['FTHG', 'FTAG'])
        dc = DixonColesModel()
        dc.fit(train)
        p = dc.predict_proba('Arsenal', 'Chelsea')
        print(f'  Arsenal vs Chelsea: {p}')
        assert abs(sum(p.values()) - 1.0) < 0.01, 'Probabilities must sum to 1'
        print('  Dixon-Coles OK')

        print('\nМодель работает корректно.')
