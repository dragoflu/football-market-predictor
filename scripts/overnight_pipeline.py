"""
Ночной пайплайн: новые лиги → фичи → Optuna → walk-forward → валидация стратегий.

Шаги:
  1. Скачать новые лиги (NEW_LEAGUES) поверх собранных collect_football_history.py
  2. Пересобрать feature matrix по всем лигам (ALL_LEAGUES)
  3. Optuna тюнинг на OPTUNA_LEAGUES (100 trials)
  4. Обновить дефолтные параметры в football_model.py
  5. Walk-forward на всех лигах
  6. Strategy validation (selection 2006-2018 vs holdout 2019-2026)
  7. Финальный отчёт

Время: ~3-4 часа
Результаты:
  data/results/walkforward_results.parquet
  data/results/strategy_grid_full.csv
  data/results/strategy_validation_report.txt
  data/results/overnight_summary.txt

Запуск:
    python scripts/overnight_pipeline.py

Сервер отца:
    scp scripts/overnight_pipeline.py "Snoop Dog"@192.168.0.35:"C:\\first_project\\scripts\\overnight_pipeline.py"
    ssh "Snoop Dog"@192.168.0.35
    cd C:\\first_project && python scripts\\overnight_pipeline.py
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('cp1251', 'cp866'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import re
import io
import json
import time
import logging
import itertools
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path('data/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'overnight_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# Конфигурация

# Новые лиги для скачивания
NEW_LEAGUES = {
    'E1': 'Championship',
    'N1': 'Eredivisie',
    'B1': 'Belgian_Pro',
    'P1': 'Primeira_Liga',
    'T1': 'Super_Lig',
    'SC0': 'Scottish_Premiership',
    'G1': 'Super_League_Greece',
}

# Все лиги (старые + новые). A1 (Австрия) убрана: football-data.co.uk не отдаёт по ней данные.
ALL_LEAGUES = ['E0', 'SP1', 'D1', 'I1', 'F1', 'E1', 'N1', 'B1', 'P1', 'T1', 'SC0', 'G1']

# Лиги для Optuna тюнинга (самые качественные данные)
OPTUNA_LEAGUES = ['E0', 'D1', 'E1']

OPTUNA_TRIALS = 100

BASE_URL = 'https://www.football-data.co.uk/mmz4281'

CORE_COLUMNS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
ODDS_COLUMNS = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA',
                'PSH', 'PSD', 'PSA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA']
STATS_COLUMNS = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']


# Шаг 1: Сбор новых лиг

def generate_season_codes(start=2000, end=2025):
    return [f'{y % 100:02d}{(y+1) % 100:02d}' for y in range(start, end + 1)]


def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    for fmt in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d']:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(s, dayfirst=True)
    except Exception:
        return pd.NaT


def download_new_leagues():
    """Скачивает данные для новых лиг. Пропускает если файл уже существует и свежий."""
    log.info('\n=== ШАГ 1: Скачивание новых лиг ===')
    season_codes = generate_season_codes(2000, 2025)
    out_dir = Path('data/raw/football')
    out_dir.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(
        headers={'User-Agent': 'Mozilla/5.0 (research)'},
        follow_redirects=True,
    )

    downloaded = []
    skipped = []

    try:
        for code, name in NEW_LEAGUES.items():
            out_path = out_dir / f'matches_{code}.parquet'

            # Пропускаем если файл свежий (< 7 дней)
            if out_path.exists():
                age_days = (time.time() - out_path.stat().st_mtime) / 86400
                if age_days < 7:
                    log.info(f'  [{code}] уже есть ({age_days:.0f} дней назад), пропускаем')
                    skipped.append(code)
                    continue

            log.info(f'  [{code}] {name}: скачиваем...')
            all_seasons = []

            for sc in season_codes:
                url = f'{BASE_URL}/{sc}/{code}.csv'
                try:
                    resp = client.get(url, timeout=15)
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()

                    content = resp.content
                    for enc in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            text = content.decode(enc)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        text = content.decode('latin-1', errors='replace')

                    df = pd.read_csv(io.StringIO(text))
                    df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
                    if len(df) == 0:
                        continue

                    # Выбираем нужные колонки
                    wanted = CORE_COLUMNS + ODDS_COLUMNS + STATS_COLUMNS
                    available = [c for c in wanted if c in df.columns]
                    df = df[available].copy()

                    df['Date'] = df['Date'].apply(parse_date)
                    season_label = f'20{sc[:2]}/20{sc[2:]}'
                    df['Season'] = season_label
                    df['League'] = code

                    for col in ['FTHG', 'FTAG', 'HTHG', 'HTAG']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    for col in ODDS_COLUMNS + STATS_COLUMNS:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    all_seasons.append(df)

                except Exception as e:
                    log.warning(f'    {url}: {e}')
                    continue

            if not all_seasons:
                log.warning(f'  [{code}] нет данных!')
                continue

            combined = pd.concat(all_seasons, ignore_index=True)
            combined = combined.sort_values('Date').reset_index(drop=True)
            combined = combined.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first')

            # Implied probabilities из B365
            if all(c in combined.columns for c in ['B365H', 'B365D', 'B365A']):
                rh = 1.0 / combined['B365H']
                rd = 1.0 / combined['B365D']
                ra = 1.0 / combined['B365A']
                total = rh + rd + ra
                combined['implied_prob_home'] = rh / total
                combined['implied_prob_draw'] = rd / total
                combined['implied_prob_away'] = ra / total
                combined['overround'] = total

            combined.to_parquet(out_path, index=False)
            log.info(f'  [{code}] сохранено: {len(combined):,} матчей → {out_path}')
            downloaded.append(code)

    finally:
        client.close()

    log.info(f'\n  Скачано: {downloaded}')
    log.info(f'  Пропущено (актуальны): {skipped}')
    return downloaded + skipped


# Шаг 2: Rebuild feature matrix

def build_feature_matrix_all():
    log.info('\n=== ШАГ 2: Пересборка feature matrix (все лиги) ===')
    from src.football_features import build_features

    data_dir = Path('data/raw/football')
    feat_path = Path('data/processed/football_features.parquet')

    all_dfs = []
    for league in ALL_LEAGUES:
        p = data_dir / f'matches_{league}.parquet'
        if not p.exists():
            log.warning(f'  [{league}] файл не найден, пропускаем')
            continue
        table = pq.read_table(str(p))
        df = table.to_pandas()
        df['Date'] = pd.to_datetime(df['Date'])
        log.info(f'  [{league}] {len(df):,} матчей')
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError('Нет данных для построения фичей')

    df_all = pd.concat(all_dfs, ignore_index=True)
    log.info(f'  Итого: {len(df_all):,} матчей из {df_all["League"].nunique()} лиг')

    # xG данные (если есть)
    xg_all = None
    xg_dfs = []
    for p in data_dir.glob('xg_*.parquet'):
        xg_df = pq.read_table(str(p)).to_pandas()
        xg_dfs.append(xg_df)
    if xg_dfs:
        xg_all = pd.concat(xg_dfs, ignore_index=True)
        log.info(f'  xG данные: {len(xg_all):,} матчей')

    all_features = []
    for league in sorted(df_all['League'].unique()):
        log.info(f'\n  [{league}] строим фичи...')
        t0 = time.time()
        ldf = df_all[df_all['League'] == league].copy()

        lxg = None
        if xg_all is not None and 'League' in xg_all.columns:
            lxg = xg_all[xg_all['League'] == league]
            if len(lxg) == 0:
                lxg = None

        features = build_features(ldf, xg_df=lxg, verbose=False)
        all_features.append(features)
        log.info(f'  [{league}] {len(features):,} строк × {len(features.columns)} колонок ({time.time()-t0:.0f}s)')

    final = pd.concat(all_features, ignore_index=True)
    final = final.sort_values(['League', 'Date']).reset_index(drop=True)
    final.to_parquet(feat_path, index=False)
    log.info(f'\n  Сохранено: {final.shape} → {feat_path}')
    return final


# Шаг 3: Optuna тюнинг

def run_optuna(features_df: pd.DataFrame):
    log.info(f'\n=== ШАГ 3: Optuna тюнинг ({OPTUNA_TRIALS} trials, лиги: {OPTUNA_LEAGUES}) ===')
    from src.football_model import tune_xgboost, tune_ensemble_weights

    df_tune = features_df[features_df['League'].isin(OPTUNA_LEAGUES)].copy()
    log.info(f'  Данные для тюнинга: {len(df_tune):,} матчей')

    best_params = tune_xgboost(df_tune, n_trials=OPTUNA_TRIALS)
    params_path = RESULTS_DIR / 'xgboost_best_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    log.info(f'  XGBoost params сохранены: {params_path}')
    log.info(f'  Params: {best_params}')

    log.info('\n  Оптимизация весов ensemble...')
    best_weights = tune_ensemble_weights(df_tune)
    weights_path = RESULTS_DIR / 'ensemble_best_weights.json'
    with open(weights_path, 'w') as f:
        json.dump({
            'dixon_coles': best_weights.dixon_coles,
            'elo': best_weights.elo,
            'xgboost': best_weights.xgboost,
        }, f, indent=2)
    log.info(f'  Ensemble weights: DC={best_weights.dixon_coles:.3f}, '
             f'ELO={best_weights.elo:.3f}, XGB={best_weights.xgboost:.3f}')

    return best_params, best_weights


# Шаг 4: Обновить параметры в football_model.py

def patch_model_params(best_params: dict, best_weights):
    """Патчит дефолтные параметры в src/football_model.py."""
    log.info('\n=== ШАГ 4: Обновление параметров в football_model.py ===')
    model_path = Path('src/football_model.py')
    text = model_path.read_text(encoding='utf-8')

    # Патчим XGBoostModel.__init__ defaults
    p = best_params
    old_xgb = re.search(
        r'def __init__\(self, n_estimators: int = \d+, max_depth: int = \d+,\s*'
        r'learning_rate: float = [\d.]+, subsample: float = [\d.]+,\s*'
        r'colsample_bytree: float = [\d.]+, min_child_weight: int = \d+,\s*'
        r'reg_alpha: float = [\d.]+, reg_lambda: float = [\d.]+\)',
        text
    )
    if old_xgb:
        new_xgb = (
            f'def __init__(self, n_estimators: int = {p["n_estimators"]}, '
            f'max_depth: int = {p["max_depth"]},\n'
            f'                 learning_rate: float = {p["learning_rate"]:.4f}, '
            f'subsample: float = {p["subsample"]:.3f},\n'
            f'                 colsample_bytree: float = {p["colsample_bytree"]:.3f}, '
            f'min_child_weight: int = {p["min_child_weight"]},\n'
            f'                 reg_alpha: float = {p["reg_alpha"]:.3f}, '
            f'reg_lambda: float = {p["reg_lambda"]:.3f})'
        )
        text = text[:old_xgb.start()] + new_xgb + text[old_xgb.end():]
        log.info('  XGBoostModel defaults обновлены')
    else:
        log.warning('  Не нашли XGBoostModel.__init__, пропускаем патч XGB')

    # Патчим EnsembleWeights
    old_ew = re.search(
        r'class EnsembleWeights:\s*\n'
        r'    dixon_coles: float = [\d.]+\s*.*?\n'
        r'    elo: float = [\d.]+\s*\n'
        r'    xgboost: float = [\d.]+',
        text
    )
    if old_ew:
        new_ew = (
            f'class EnsembleWeights:\n'
            f'    dixon_coles: float = {best_weights.dixon_coles:.3f}\n'
            f'    elo: float = {best_weights.elo:.3f}\n'
            f'    xgboost: float = {best_weights.xgboost:.3f}'
        )
        text = text[:old_ew.start()] + new_ew + text[old_ew.end():]
        log.info('  EnsembleWeights defaults обновлены')
    else:
        log.warning('  Не нашли EnsembleWeights, пропускаем патч весов')

    model_path.write_text(text, encoding='utf-8')
    log.info(f'  {model_path} обновлён')


# Шаг 5: Walk-forward

def run_walkforward(features_df: pd.DataFrame) -> pd.DataFrame:
    log.info('\n=== ШАГ 5: Walk-forward validation (все лиги) ===')
    from src.football_model import walk_forward_validate

    all_results = []
    for league in sorted(features_df['League'].unique()):
        log.info(f'\n  [{league}]')
        ldf = features_df[features_df['League'] == league].copy()
        seasons = sorted(ldf['Season'].astype(str).unique())
        if len(seasons) < 7:
            log.warning(f'  [{league}] только {len(seasons)} сезонов, пропускаем')
            continue

        results = walk_forward_validate(ldf, n_train_seasons=5, min_test_matches=200, edge_threshold=0.08)
        if len(results) > 0:
            results['League'] = league
            all_results.append(results)
            bets = results[results['kelly_size'] > 0]
            if len(bets) > 0:
                staked = bets['kelly_size'].sum()
                roi = bets['profit'].sum() / staked
                log.info(f'  [{league}] {len(bets)} bets, ROI={roi:+.1%}')

    if not all_results:
        raise RuntimeError('Нет результатов walk-forward')

    combined = pd.concat(all_results, ignore_index=True)
    results_path = RESULTS_DIR / 'walkforward_results.parquet'
    combined.to_parquet(results_path, index=False)
    log.info(f'\n  Сохранено: {results_path} ({len(combined):,} строк)')
    return combined


# Шаг 6: Strategy validation

def calc_roi(df):
    bets = df[df['kelly_size'] > 0]
    n = len(bets)
    if n == 0:
        return {'n_bets': 0, 'roi': np.nan, 'win_rate': np.nan, 'profit': 0.0}
    staked = bets['kelly_size'].sum()
    profit = bets['profit'].sum()
    return {'n_bets': n, 'roi': profit / staked,
            'win_rate': (bets['profit'] > 0).mean(), 'profit': profit}


def apply_strategy(df, leagues=None, outcomes=None, edge_min=0.08):
    mask = pd.Series(True, index=df.index)
    if leagues:
        mask &= df['League'].isin(leagues)
    if outcomes:
        mask &= df['best_outcome'].isin(outcomes)
    mask &= df['best_edge'] >= edge_min
    return df[mask]


def run_strategy_validation(results_df: pd.DataFrame) -> str:
    log.info('\n=== ШАГ 6: Strategy validation (selection vs holdout) ===')

    results_df = results_df.copy()
    results_df['year'] = results_df['Season'].str[:4].astype(int)
    df_sel = results_df[results_df['year'] < 2019]
    df_hld = results_df[results_df['year'] >= 2019]

    all_leagues = sorted(results_df['League'].unique())
    log.info(f'  Selection: {df_sel["Season"].min()} - {df_sel["Season"].max()} '
             f'({df_sel["Season"].nunique()} сезонов)')
    log.info(f'  Holdout:   {df_hld["Season"].min()} - {df_hld["Season"].max()} '
             f'({df_hld["Season"].nunique()} сезонов)')

    # Grid search на selection
    leagues_grid = [all_leagues] + [[l] for l in all_leagues]
    leagues_grid += [
        [l for l in ['E0', 'D1', 'E1'] if l in all_leagues],
        [l for l in ['E0', 'D1'] if l in all_leagues],
        [l for l in ['E0', 'D1', 'SP1'] if l in all_leagues],
        [l for l in ['E0', 'D1', 'E1', 'N1'] if l in all_leagues],
        # Новые лиги (гегемоны + малоизвестные для американцев)
        [l for l in ['SC0'] if l in all_leagues],
        [l for l in ['G1'] if l in all_leagues],
        [l for l in ['P1', 'SC0'] if l in all_leagues],
        [l for l in ['D1', 'SC0'] if l in all_leagues],
        [l for l in ['P1', 'D1', 'SC0'] if l in all_leagues],
        [l for l in ['P1', 'D1'] if l in all_leagues],
        [l for l in ['SC0', 'G1'] if l in all_leagues],
    ]
    # убираем дубликаты
    seen = set()
    leagues_grid_clean = []
    for lg in leagues_grid:
        key = tuple(sorted(lg))
        if key not in seen and len(lg) > 0:
            seen.add(key)
            leagues_grid_clean.append(lg)

    outcomes_grid = [None, ['draw'], ['away'], ['home'], ['draw', 'away']]
    edge_grid = [0.08, 0.10, 0.15, 0.20, 0.25]

    grid_results = []
    for leagues, outcomes, edge_min in itertools.product(leagues_grid_clean, outcomes_grid, edge_grid):
        filtered_sel = apply_strategy(df_sel, leagues=leagues, outcomes=outcomes, edge_min=edge_min)
        m_sel = calc_roi(filtered_sel)
        if m_sel['n_bets'] < 20 or np.isnan(m_sel['roi']):
            continue

        filtered_hld = apply_strategy(df_hld, leagues=leagues, outcomes=outcomes, edge_min=edge_min)
        m_hld = calc_roi(filtered_hld)

        grid_results.append({
            'leagues': '+'.join(sorted(leagues)),
            'outcomes': '+'.join(outcomes) if outcomes else 'all',
            'edge_min': edge_min,
            'sel_bets': m_sel['n_bets'],
            'sel_roi': m_sel['roi'],
            'hld_bets': m_hld['n_bets'],
            'hld_roi': m_hld['roi'] if not np.isnan(m_hld['roi']) else 0.0,
            'hld_win': m_hld['win_rate'] if not np.isnan(m_hld.get('win_rate', np.nan)) else 0.0,
        })

    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(RESULTS_DIR / 'strategy_grid_full.csv', index=False)

    # Сортируем по holdout ROI (честная метрика)
    grid_df_valid = grid_df[grid_df['hld_bets'] >= 10].sort_values('hld_roi', ascending=False)

    lines = []
    lines.append('\n' + '='*70)
    lines.append('STRATEGY VALIDATION REPORT')
    lines.append('='*70)
    lines.append(f'\nSelection: {df_sel["Season"].min()} - {df_sel["Season"].max()}')
    lines.append(f'Holdout:   {df_hld["Season"].min()} - {df_hld["Season"].max()}')
    lines.append(f'Лиги в данных: {all_leagues}')

    top = grid_df_valid.head(15).rename(columns={
        'leagues': 'Лиги', 'outcomes': 'Исходы', 'edge_min': 'Edge',
        'sel_bets': 'SEL_bets', 'sel_roi': 'SEL_ROI',
        'hld_bets': 'HLD_bets', 'hld_roi': 'HLD_ROI', 'hld_win': 'HLD_Win%',
    })[['Лиги', 'Исходы', 'Edge', 'SEL_bets', 'SEL_ROI', 'HLD_bets', 'HLD_ROI', 'HLD_Win%']].copy()
    for col in ('SEL_ROI', 'HLD_ROI'):
        top[col] = top[col].map('{:+.1%}'.format)
    top['HLD_Win%'] = top['HLD_Win%'].map('{:.1%}'.format)

    lines.append('\nТОП-15 по Holdout ROI (min 10 ставок на holdout)')
    lines.append(top.to_string(index=False))

    # Отдельно: сезонная разбивка топ-1 стратегии на holdout
    if len(grid_df_valid) > 0:
        top1 = grid_df_valid.iloc[0]
        leagues = top1['leagues'].split('+')
        outcomes = top1['outcomes'].split('+') if top1['outcomes'] != 'all' else None
        filtered = apply_strategy(df_hld, leagues=leagues, outcomes=outcomes, edge_min=top1['edge_min'])
        bets = filtered[filtered['kelly_size'] > 0]

        lines.append(f'\n\nТоп-1 стратегия по сезонам (holdout):')
        lines.append(f'{top1["leagues"]} | {top1["outcomes"]} | edge≥{top1["edge_min"]}')
        for season in sorted(bets['Season'].unique()):
            sb = bets[bets['Season'] == season]
            staked = sb['kelly_size'].sum()
            roi = sb['profit'].sum() / staked if staked > 0 else 0
            lines.append(f'  {season}: ROI={roi:+.1%} ({len(sb)} bets)')

        bets_per_season = len(bets) / df_hld['Season'].nunique()
        lines.append(f'\n  Ставок/сезон на holdout: {bets_per_season:.1f}')

    report = '\n'.join(lines)
    print(report)

    report_path = RESULTS_DIR / 'strategy_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log.info(f'  Отчёт: {report_path}')

    return report


# Финальный отчёт

def print_walkforward_summary(results_df: pd.DataFrame):
    log.info('\n=== Walk-Forward Summary ===')
    from src.football_model import evaluate_predictions, FEATURE_COLS

    metrics = evaluate_predictions(results_df)
    lines = [
        '\n' + '='*60,
        'WALK-FORWARD RESULTS',
        '='*60,
        f'  Лиги:        {sorted(results_df["League"].unique())}',
        f'  Matches:     {metrics["n_matches"]:,}',
        f'  Brier Score: {metrics["brier_score"]:.4f} (target < {metrics["brier_benchmark"]})',
        f'  RPS:         {metrics["rps"]:.4f} (target < {metrics["rps_benchmark"]})',
        f'  Accuracy:    {metrics["accuracy"]:.1%} (benchmark {metrics["accuracy_benchmark"]:.1%})',
        f'  Bets:        {metrics["n_bets"]:,}',
        f'  ROI:         {metrics["roi"]:.1%}',
        '\nПо лигам:',
    ]

    for league in sorted(results_df['League'].unique()):
        lr = results_df[results_df['League'] == league]
        m = evaluate_predictions(lr)
        bets = lr[lr['kelly_size'] > 0]
        staked = bets['kelly_size'].sum()
        roi = bets['profit'].sum() / staked if staked > 0 else 0
        lines.append(f'  {league}: Brier={m["brier_score"]:.4f} | ROI={roi:+.1%} | Bets={len(bets):,}')

    report = '\n'.join(lines)
    print(report)

    summary_path = RESULTS_DIR / 'overnight_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(report)


# Main

def main():
    t_total = time.time()
    log.info('='*60)
    log.info('OVERNIGHT PIPELINE START')
    log.info('='*60)

    summary = []

    try:
        # 1. Данные
        t0 = time.time()
        download_new_leagues()
        summary.append(f'Шаг 1 (данные):   {(time.time()-t0)/60:.1f} мин')

        # 2. Фичи
        t0 = time.time()
        features_df = build_feature_matrix_all()
        summary.append(f'Шаг 2 (фичи):     {(time.time()-t0)/60:.1f} мин, '
                       f'{len(features_df):,} матчей × {len(features_df.columns)} фичей')

        # 3. Optuna
        t0 = time.time()
        best_params, best_weights = run_optuna(features_df)
        summary.append(f'Шаг 3 (Optuna):   {(time.time()-t0)/60:.1f} мин')

        # 4. Патч модели
        patch_model_params(best_params, best_weights)
        summary.append('Шаг 4 (патч):     готово')

        # Перезагружаем модуль чтобы новые параметры применились
        import importlib
        import src.football_model as fm_module
        importlib.reload(fm_module)
        log.info('  src.football_model перезагружен с новыми параметрами')

        # 5. Walk-forward
        t0 = time.time()
        results_df = run_walkforward(features_df)
        summary.append(f'Шаг 5 (WF):       {(time.time()-t0)/60:.1f} мин, {len(results_df):,} матчей')

        # 6. Strategy validation
        t0 = time.time()
        run_strategy_validation(results_df)
        summary.append(f'Шаг 6 (стратег.): {(time.time()-t0)/60:.1f} мин')

        # Финальный отчёт
        print_walkforward_summary(results_df)

    except Exception as e:
        log.error(f'\nОШИБКА на шаге: {e}', exc_info=True)
        summary.append(f'ОШИБКА: {e}')

    elapsed = (time.time() - t_total) / 60
    log.info('\n' + '='*60)
    log.info(f'OVERNIGHT PIPELINE DONE за {elapsed:.1f} мин')
    log.info('='*60)
    for s in summary:
        log.info(f'  {s}')
    log.info(f'\nРезультаты в: {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
