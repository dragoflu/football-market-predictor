"""
Football Feature Engineering Pipeline.

Строит все фичи для ML-модели предсказания футбольных матчей.
Все фичи walk-forward корректны: на дату T используются данные только до T-1.

Фичи (по приоритету):
  Tier 1: ELO, rolling xG/xGA, implied probs, home/away, form
  Tier 2: goal diff, H2H, rest days, shots on target, league position
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# ELO Rating
# ============================================================

ELO_INIT = 1500      # стартовый рейтинг
ELO_K = 20           # скорость обновления
ELO_HOME_ADV = 100   # бонус хозяев в единицах ELO


def _expected_elo(rating_a: float, rating_b: float) -> float:
    """Ожидаемый результат команды A против B (0-1)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating: float, expected: float, actual: float, k: float = ELO_K) -> float:
    return rating + k * (actual - expected)


def build_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет ELO рейтинги к датафрейму матчей.

    Args:
        df: датафрейм с колонками Date, HomeTeam, AwayTeam, FTR, League
            отсортированный по дате внутри каждой лиги

    Returns:
        df с новыми колонками:
            elo_home_before, elo_away_before  — рейтинг ДО матча (для фич)
            elo_home_after, elo_away_after    — рейтинг ПОСЛЕ (для обновления)
            elo_diff                          — elo_home_before - elo_away_before + ELO_HOME_ADV
    """
    df = df.copy().sort_values(['League', 'Date']).reset_index(drop=True)
    ratings: dict[str, float] = {}

    elo_home_before = []
    elo_away_before = []
    elo_home_after = []
    elo_away_after = []

    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']

        r_home = ratings.get(home, ELO_INIT)
        r_away = ratings.get(away, ELO_INIT)

        elo_home_before.append(r_home)
        elo_away_before.append(r_away)

        # Ожидаемое с учётом домашнего преимущества
        exp_home = _expected_elo(r_home + ELO_HOME_ADV, r_away)
        exp_away = 1.0 - exp_home

        # Фактический результат: H=1, D=0.5, A=0
        result = row.get('FTR', None)
        if result == 'H':
            actual_home, actual_away = 1.0, 0.0
        elif result == 'A':
            actual_home, actual_away = 0.0, 1.0
        elif result == 'D':
            actual_home, actual_away = 0.5, 0.5
        else:
            # Нет результата — не обновляем (будущий матч)
            elo_home_after.append(r_home)
            elo_away_after.append(r_away)
            continue

        new_home = _elo_update(r_home, exp_home, actual_home)
        new_away = _elo_update(r_away, exp_away, actual_away)

        ratings[home] = new_home
        ratings[away] = new_away

        elo_home_after.append(new_home)
        elo_away_after.append(new_away)

    df['elo_home_before'] = elo_home_before
    df['elo_away_before'] = elo_away_before
    df['elo_home_after'] = elo_home_after
    df['elo_away_after'] = elo_away_after
    df['elo_diff'] = df['elo_home_before'] - df['elo_away_before'] + ELO_HOME_ADV

    return df


# ============================================================
# Rolling Stats (walk-forward safe)
# ============================================================

def _rolling_team_stats(df: pd.DataFrame, team: str, date: pd.Timestamp,
                         windows: list[int] = [5, 10, 20],
                         home_only: bool = False,
                         away_only: bool = False) -> dict:
    """
    Вычисляет rolling статистику команды ДО даты матча.
    Использует только прошлые матчи (строго < date).
    """
    # Все матчи команды до этой даты
    home_mask = (df['HomeTeam'] == team) & (df['Date'] < date)
    away_mask = (df['AwayTeam'] == team) & (df['Date'] < date)

    if home_only:
        mask = home_mask
    elif away_only:
        mask = away_mask
    else:
        mask = home_mask | away_mask

    past = df[mask].sort_values('Date')

    result = {}

    for w in windows:
        recent = past.tail(w)
        n = len(recent)

        if n == 0:
            for stat in ['goals_scored', 'goals_conceded', 'points', 'shots_on_target']:
                result[f'{stat}_last{w}'] = np.nan
            continue

        goals_scored = []
        goals_conceded = []
        points = []
        sot = []

        for _, row in recent.iterrows():
            if row['HomeTeam'] == team:
                gs = row.get('FTHG', np.nan)
                gc = row.get('FTAG', np.nan)
                ftr = row.get('FTR')
                s = row.get('HST', np.nan)
            else:
                gs = row.get('FTAG', np.nan)
                gc = row.get('FTHG', np.nan)
                ftr = row.get('FTR')
                s = row.get('AST', np.nan)

            goals_scored.append(gs)
            goals_conceded.append(gc)
            sot.append(s)

            if ftr == 'H' and row['HomeTeam'] == team:
                points.append(3)
            elif ftr == 'A' and row['AwayTeam'] == team:
                points.append(3)
            elif ftr == 'D':
                points.append(1)
            else:
                points.append(0)

        result[f'goals_scored_last{w}'] = np.nanmean(goals_scored)
        result[f'goals_conceded_last{w}'] = np.nanmean(goals_conceded)
        result[f'points_last{w}'] = np.nanmean(points)
        result[f'shots_on_target_last{w}'] = np.nanmean(sot)

        # Total goals (proxy for match openness — низкое значение → больше ничьих)
        total_goals = [g + c for g, c in zip(goals_scored, goals_conceded)
                       if not np.isnan(g) and not np.isnan(c)]
        result[f'total_goals_last{w}'] = np.mean(total_goals) if total_goals else np.nan

        # Draw rate (некоторые команды систематически чаще ничьи)
        draws_list = [1 if p == 1 else 0 for p in points]
        result[f'draw_rate_last{w}'] = np.mean(draws_list) if draws_list else np.nan

        # Clean sheet rate (нет пропущенных голов → больше 0-0 ничьих)
        cs_list = [1 if not np.isnan(c) and c == 0 else 0 for c in goals_conceded]
        result[f'clean_sheet_rate_last{w}'] = np.mean(cs_list) if cs_list else np.nan

        # Rolling xG (Understat данные, доступны с ~2014)
        xg_scored, xg_conceded = [], []
        for _, xrow in recent.iterrows():
            if xrow['HomeTeam'] == team:
                xg_scored.append(xrow.get('xg_home', np.nan))
                xg_conceded.append(xrow.get('xg_away', np.nan))
            else:
                xg_scored.append(xrow.get('xg_away', np.nan))
                xg_conceded.append(xrow.get('xg_home', np.nan))
        result[f'xg_scored_last{w}'] = np.nanmean(xg_scored) if xg_scored else np.nan
        result[f'xg_conceded_last{w}'] = np.nanmean(xg_conceded) if xg_conceded else np.nan

    # ELO momentum: изменение ELO за последние 5 матчей
    # (команда в росте vs в спаде — нелинейный сигнал)
    if 'elo_home_after' in past.columns or 'elo_away_after' in past.columns:
        recent5 = past.tail(5)
        elo_vals = []
        for _, xrow in recent5.iterrows():
            if xrow.get('HomeTeam') == team:
                v = xrow.get('elo_home_after', np.nan)
            else:
                v = xrow.get('elo_away_after', np.nan)
            if not np.isnan(v):
                elo_vals.append(v)
        if len(elo_vals) >= 2:
            result['elo_momentum'] = elo_vals[-1] - elo_vals[0]
        else:
            result['elo_momentum'] = np.nan
    else:
        result['elo_momentum'] = np.nan

    # Win/loss/draw streak (серия подряд — нелинейный психологический сигнал)
    if len(past) > 0:
        last_result = None
        streak = 0
        for _, xrow in reversed(list(past.tail(10).iterrows())):
            ftr = xrow.get('FTR')
            if ftr is None:
                break
            if xrow.get('HomeTeam') == team:
                r = ftr  # H=win, D=draw, A=loss
            else:
                r = {'H': 'A', 'A': 'H', 'D': 'D'}.get(ftr, None)
            if last_result is None:
                last_result = r
                streak = 1
            elif r == last_result:
                streak += 1
            else:
                break
        result['streak'] = streak if last_result == 'H' else (-streak if last_result == 'A' else 0)
    else:
        result['streak'] = 0

    return result


def _matches_last_n_days(df: pd.DataFrame, team: str, date: pd.Timestamp, days: int = 14) -> int:
    """Количество матчей команды за последние N дней до date (fixture congestion)."""
    cutoff = date - pd.Timedelta(days=days)
    mask = (
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date) &
        (df['Date'] >= cutoff)
    )
    return int(mask.sum())


def _days_since_last_match(df: pd.DataFrame, team: str, date: pd.Timestamp) -> float:
    """Количество дней с последнего матча команды до date."""
    mask = ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & (df['Date'] < date)
    past = df[mask]
    if len(past) == 0:
        return np.nan
    last_date = past['Date'].max()
    return (date - last_date).days


def _precompute_league_positions(df: pd.DataFrame) -> dict:
    """
    Предвычисляет позиции в таблице для всех матчей за один проход O(n).

    Возвращает dict: match_id -> {
        'home_league_pos': int, 'home_league_pos_pct': float, 'home_league_pts': float,
        'away_league_pos': int, 'away_league_pos_pct': float, 'away_league_pts': float,
        'league_pos_diff': float,
    }
    """
    result = {}

    for league in df['League'].unique():
        league_df = df[df['League'] == league].sort_values('Date')
        pts: dict = {}
        gd: dict = {}

        for row in league_df.itertuples(index=False):
            mid = row.match_id
            home = row.HomeTeam
            away = row.AwayTeam

            # Инициализируем новые команды
            for t in (home, away):
                if t not in pts:
                    pts[t] = 0
                    gd[t] = 0

            # Позиция ДО этого матча
            all_teams = list(pts.keys())
            n_teams = len(all_teams)
            if n_teams > 0:
                sorted_teams = sorted(all_teams, key=lambda t: (pts[t], gd[t]), reverse=True)
                home_pos = sorted_teams.index(home) + 1
                away_pos = sorted_teams.index(away) + 1
                result[mid] = {
                    'home_league_pos': home_pos,
                    'home_league_pos_pct': home_pos / n_teams,
                    'home_league_pts': float(pts[home]),
                    'away_league_pos': away_pos,
                    'away_league_pos_pct': away_pos / n_teams,
                    'away_league_pts': float(pts[away]),
                    'league_pos_diff': float(home_pos - away_pos),
                }
            else:
                result[mid] = {
                    'home_league_pos': np.nan, 'home_league_pos_pct': np.nan, 'home_league_pts': np.nan,
                    'away_league_pos': np.nan, 'away_league_pos_pct': np.nan, 'away_league_pts': np.nan,
                    'league_pos_diff': np.nan,
                }

            # Обновляем таблицу ПОСЛЕ записи позиции
            ftr = getattr(row, 'FTR', None)
            gh = getattr(row, 'FTHG', 0) or 0
            ga = getattr(row, 'FTAG', 0) or 0
            if ftr == 'H':
                pts[home] += 3
            elif ftr == 'A':
                pts[away] += 3
            elif ftr == 'D':
                pts[home] += 1
                pts[away] += 1
            gd[home] += gh - ga
            gd[away] += ga - gh

    return result


def _h2h_stats(df: pd.DataFrame, home_team: str, away_team: str,
               date: pd.Timestamp, n: int = 10) -> dict:
    """
    H2H статистика между двумя командами.
    Минимум 3 встречи за 5 лет — иначе NaN.
    """
    cutoff = date - pd.Timedelta(days=5 * 365)

    mask = (
        (df['Date'] < date) &
        (df['Date'] >= cutoff) &
        (
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
            ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
        )
    )
    past = df[mask].sort_values('Date').tail(n)

    if len(past) < 3:
        return {
            'h2h_home_winrate': np.nan,
            'h2h_away_winrate': np.nan,
            'h2h_draw_rate': np.nan,
            'h2h_home_goals_avg': np.nan,
            'h2h_away_goals_avg': np.nan,
        }

    home_wins, away_wins, draws = 0, 0, 0
    home_goals, away_goals = [], []

    for _, row in past.iterrows():
        if row['HomeTeam'] == home_team:
            hg = row.get('FTHG', np.nan)
            ag = row.get('FTAG', np.nan)
            ftr = row.get('FTR')
            if ftr == 'H': home_wins += 1
            elif ftr == 'A': away_wins += 1
            elif ftr == 'D': draws += 1
        else:
            # Перевёрнутый матч
            hg = row.get('FTAG', np.nan)
            ag = row.get('FTHG', np.nan)
            ftr = row.get('FTR')
            if ftr == 'A': home_wins += 1
            elif ftr == 'H': away_wins += 1
            elif ftr == 'D': draws += 1

        home_goals.append(hg)
        away_goals.append(ag)

    total = len(past)
    return {
        'h2h_home_winrate': home_wins / total,
        'h2h_away_winrate': away_wins / total,
        'h2h_draw_rate': draws / total,
        'h2h_home_goals_avg': np.nanmean(home_goals),
        'h2h_away_goals_avg': np.nanmean(away_goals),
    }


def _implied_probs(row: pd.Series) -> dict:
    """
    Implied probabilities для ФИЧЕЙ модели (вход в XGBoost/ELO).
    Приоритет: Pinnacle > B365 > Avg.
    НЕ используем Betfair здесь — он нужен только для расчёта edge (market comparison).
    Если использовать Betfair как фичу И как market → circular dependency.
    """
    for prefix, h, d, a in [
        ('ps', 'PSH', 'PSD', 'PSA'),
        ('b365', 'B365H', 'B365D', 'B365A'),
        ('avg', 'AvgH', 'AvgD', 'AvgA'),
    ]:
        if all(c in row.index and pd.notna(row[c]) and row[c] > 0 for c in [h, d, a]):
            raw_h = 1.0 / row[h]
            raw_d = 1.0 / row[d]
            raw_a = 1.0 / row[a]
            total = raw_h + raw_d + raw_a
            return {
                'implied_home': raw_h / total,
                'implied_draw': raw_d / total,
                'implied_away': raw_a / total,
                'overround': total,
                'odds_source': prefix,
            }

    return {
        'implied_home': np.nan,
        'implied_draw': np.nan,
        'implied_away': np.nan,
        'overround': np.nan,
        'odds_source': None,
    }


# ============================================================
# Main Feature Builder
# ============================================================

ROLLING_WINDOWS = [5, 10, 20]


def build_features(df: pd.DataFrame, xg_df: pd.DataFrame | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Строит полную матрицу фичей для обучения модели.

    Args:
        df:     датафрейм матчей (из collect_football_history.py)
        xg_df:  xG данные (опционально, из collect_xg_data.py)
        verbose: логировать прогресс

    Returns:
        DataFrame с колонками:
            match_id, Date, League, HomeTeam, AwayTeam
            + все фичи Tier1 + Tier2
            + target: FTR (H/D/A) и target_num (1/0.5/0)
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['League', 'Date']).reset_index(drop=True)
    df['match_id'] = df.index

    if verbose:
        print(f'Building ELO ratings for {len(df):,} matches...')
    df = build_elo_ratings(df)

    # Если есть xG данные — мержим
    if xg_df is not None:
        xg_merge_cols = ['Date', 'HomeTeam', 'AwayTeam', 'League']
        xg_keep = xg_merge_cols + [c for c in xg_df.columns if c not in df.columns]
        xg_df = xg_df[[c for c in xg_keep if c in xg_df.columns]]
        df = df.merge(xg_df, on=xg_merge_cols, how='left')

    # Предвычисляем позиции в таблице для всех матчей за один проход O(n)
    if verbose:
        print('Precomputing league positions...')
    league_positions = _precompute_league_positions(df)

    rows = []
    total = len(df)

    if verbose:
        print(f'Computing rolling features...')

    for i, row in df.iterrows():
        if verbose and i % 2000 == 0:
            print(f'  {i:,} / {total:,} ({i/total*100:.0f}%)')

        date = row['Date']
        home = row['HomeTeam']
        away = row['AwayTeam']
        league = row['League']

        feat: dict = {
            'match_id': row['match_id'],
            'Date': date,
            'League': league,
            'HomeTeam': home,
            'AwayTeam': away,
            'Season': row.get('Season', ''),
            # Нужны для Dixon-Coles fit
            'FTHG': row.get('FTHG', np.nan),
            'FTAG': row.get('FTAG', np.nan),
            'FTR': row.get('FTR', None),
        }

        # --- Tier 1: ELO ---
        feat['elo_diff'] = row['elo_diff']
        feat['elo_home'] = row['elo_home_before']
        feat['elo_away'] = row['elo_away_before']

        # --- Tier 1: Rolling stats (home team) ---
        home_stats = _rolling_team_stats(df, home, date, ROLLING_WINDOWS)
        for k, v in home_stats.items():
            feat[f'home_{k}'] = v

        # --- Rolling stats (home team, away only) ---
        home_away_stats = _rolling_team_stats(df, home, date, ROLLING_WINDOWS, away_only=True)
        for k, v in home_away_stats.items():
            feat[f'home_away_{k}'] = v

        # --- Rolling stats (away team) ---
        away_stats = _rolling_team_stats(df, away, date, ROLLING_WINDOWS)
        for k, v in away_stats.items():
            feat[f'away_{k}'] = v

        # --- Rolling stats (away team, home only) ---
        away_home_stats = _rolling_team_stats(df, away, date, ROLLING_WINDOWS, home_only=True)
        for k, v in away_home_stats.items():
            feat[f'away_home_{k}'] = v

        # --- xG features (если есть) ---
        for xg_col in ['xg_home', 'xg_away', 'xga_home', 'xga_away',
                        'npxg_home', 'npxg_away']:
            feat[xg_col] = row.get(xg_col, np.nan)

        # --- Tier 1: Implied probabilities ---
        probs = _implied_probs(row)
        feat.update(probs)

        # --- Tier 2: Days rest + Fixture congestion ---
        feat['home_days_rest'] = _days_since_last_match(df, home, date)
        feat['away_days_rest'] = _days_since_last_match(df, away, date)
        feat['rest_diff'] = (
            (feat['home_days_rest'] or 0) - (feat['away_days_rest'] or 0)
        )
        feat['home_matches_last14'] = _matches_last_n_days(df, home, date, 14)
        feat['away_matches_last14'] = _matches_last_n_days(df, away, date, 14)
        feat['congestion_diff'] = feat['home_matches_last14'] - feat['away_matches_last14']

        # --- Tier 2: H2H ---
        h2h = _h2h_stats(df, home, away, date)
        feat.update(h2h)

        # --- Tier 2: League position (из предвычисленного словаря) ---
        pos_data = league_positions.get(row['match_id'], {})
        feat.update(pos_data)

        # --- Tier 2: Season stage ---
        # Ранний сезон (август-октябрь) = больше ничьих, команды не дифференцированы
        # Поздний сезон (апрель-май) = давление на relegated/champions → меньше ничьих
        month = date.month
        # Нормализуем в 0-1: начало сезона (август=0) → конец (май=1)
        # Футбольный сезон: авг(8) → май(5)
        season_months = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
        feat['season_stage'] = season_months.index(month) / (len(season_months) - 1) \
            if month in season_months else 0.5

        # Количество матчей команды в текущем сезоне (proxy для усталости/сыгранности)
        season = row.get('Season', '')
        home_season_matches = int((((df['HomeTeam'] == home) | (df['AwayTeam'] == home)) &
                                    (df['Season'] == season) & (df['Date'] < date)).sum()) \
            if season else 0
        feat['home_season_matches_played'] = home_season_matches
        away_season_matches = int((((df['HomeTeam'] == away) | (df['AwayTeam'] == away)) &
                                    (df['Season'] == season) & (df['Date'] < date)).sum()) \
            if season else 0
        feat['away_season_matches_played'] = away_season_matches

        # --- Betfair exchange implied probs (для edge calculation, НЕ фичи модели) ---
        # Betfair closing price = prediction market proxy, используется в walk-forward
        # для сравнения model_prob vs market_prob вместо Pinnacle
        for out, col in [('home', 'BFEH'), ('draw', 'BFED'), ('away', 'BFEA')]:
            val = row.get(col, np.nan)
            if pd.notna(val) and val > 0:
                feat[f'betfair_{out}'] = 1.0 / val
            else:
                feat[f'betfair_{out}'] = np.nan
        # Нормализуем Betfair implied probs (убираем overround)
        bf_h, bf_d, bf_a = feat.get('betfair_home'), feat.get('betfair_draw'), feat.get('betfair_away')
        if all(pd.notna(x) for x in [bf_h, bf_d, bf_a]):
            bf_total = bf_h + bf_d + bf_a
            feat['betfair_home'] = bf_h / bf_total
            feat['betfair_draw'] = bf_d / bf_total
            feat['betfair_away'] = bf_a / bf_total

        # --- Target ---
        ftr = row.get('FTR')
        feat['target'] = ftr
        if ftr == 'H':
            feat['target_num'] = 1.0
        elif ftr == 'D':
            feat['target_num'] = 0.5
        elif ftr == 'A':
            feat['target_num'] = 0.0
        else:
            feat['target_num'] = np.nan

        rows.append(feat)

    result = pd.DataFrame(rows)

    if verbose:
        non_null = result.notna().mean()
        print(f'\nFeature coverage (% non-null):')
        for col, pct in non_null.items():
            if pct < 0.5 and col not in ('xg_home', 'xg_away', 'xga_home', 'xga_away',
                                          'npxg_home', 'npxg_away'):
                print(f'  WARNING: {col} = {pct:.1%}')

    return result


if __name__ == '__main__':
    # Быстрый тест
    from pathlib import Path
    import glob

    parquets = glob.glob('data/raw/football/matches_*.parquet')
    if not parquets:
        print('Нет данных. Сначала запусти collect_football_history.py')
    else:
        dfs = [pd.read_parquet(p) for p in parquets]
        df = pd.concat(dfs, ignore_index=True)
        print(f'Загружено {len(df):,} матчей из {len(parquets)} лиг')

        # Тест на малой выборке
        sample = df[df['League'] == 'E0'].tail(500)
        features = build_features(sample, verbose=True)
        print(f'\nFeature matrix shape: {features.shape}')
        print(f'Columns: {list(features.columns[:20])}...')
        print(f'\nSample (last 3 rows):')
        print(features[['Date', 'HomeTeam', 'AwayTeam', 'elo_diff',
                         'home_goals_scored_last5', 'implied_home', 'target']].tail(3))
