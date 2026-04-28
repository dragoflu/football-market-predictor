"""
Pipeline без Pinnacle implied probs в фичах — ablation study.

Единственное отличие от run_with_importance.py:
  implied_home, implied_draw, implied_away убраны из FEATURE_COLS перед запуском.

Результаты сохраняются с суффиксом _no_pinnacle — НЕ перезаписывают оригинал.
  data/results/walkforward_results_no_pinnacle.parquet
  data/results/feature_importance_no_pinnacle.csv
  data/results/strategy_validation_no_pinnacle.txt
  data/results/xgboost_params_no_pinnacle.json

Запуск:
    python scripts/run_without_pinnacle.py

Сервер:
    scp scripts/run_without_pinnacle.py "Snoop Dog"@192.168.0.35:"C:\\first_project\\scripts\\run_without_pinnacle.py"
    ssh "Snoop Dog"@192.168.0.35
    cd C:\\first_project && python scripts\\run_without_pinnacle.py
"""

import sys
import json
import itertools
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Патчим FEATURE_COLS ДО импорта модели
import src.football_model as _fm
PINNACLE_COLS = {'implied_home', 'implied_draw', 'implied_away'}
_fm.FEATURE_COLS = [c for c in _fm.FEATURE_COLS if c not in PINNACLE_COLS]

from src.football_model import (
    walk_forward_validate, tune_xgboost, tune_ensemble_weights,
    evaluate_predictions, FEATURE_COLS,
)

RESULTS_DIR = Path('data/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'run_no_pinnacle.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

OPTUNA_LEAGUES = ['E0', 'D1', 'E1']
OPTUNA_TRIALS  = 100
ALL_LEAGUES    = ['E0', 'SP1', 'D1', 'I1', 'F1', 'E1', 'N1', 'B1', 'P1', 'T1', 'SC0', 'G1']


def load_features():
    path = Path('data/processed/football_features.parquet')
    df = pd.read_parquet(path)
    log.info(f'Загружено: {len(df):,} матчей')
    log.info(f'FEATURE_COLS ({len(FEATURE_COLS)} фичей, Pinnacle убран): {FEATURE_COLS[:10]}...')
    assert 'implied_home' not in FEATURE_COLS, 'implied_home всё ещё в фичах!'
    return df


def run_optuna(df):
    log.info(f'\n=== Optuna без Pinnacle ({OPTUNA_TRIALS} trials) ===')
    df_tune = df[df['League'].isin(OPTUNA_LEAGUES)].copy()
    best_params = tune_xgboost(df_tune, n_trials=OPTUNA_TRIALS)
    with open(RESULTS_DIR / 'xgboost_params_no_pinnacle.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    log.info(f'Params: {best_params}')

    best_weights = tune_ensemble_weights(df_tune)
    log.info(f'Weights: DC={best_weights.dixon_coles:.3f}, ELO={best_weights.elo:.3f}, XGB={best_weights.xgboost:.3f}')
    return best_params, best_weights


def run_walkforward(df):
    log.info('\n=== Walk-forward (без Pinnacle) ===')
    all_results = []
    for league in sorted(df['League'].unique()):
        if league not in ALL_LEAGUES:
            continue
        ldf = df[df['League'] == league].copy()
        if ldf['Season'].nunique() < 7:
            continue
        log.info(f'[{league}]...')
        results = walk_forward_validate(ldf, n_train_seasons=5, min_test_matches=200, edge_threshold=0.08)
        if len(results) > 0:
            results['League'] = league
            all_results.append(results)
            bets = results[results['kelly_size'] > 0]
            if len(bets) > 0:
                roi = bets['profit'].sum() / bets['kelly_size'].sum()
                log.info(f'[{league}] {len(bets)} bets, ROI={roi:+.1%}')

    combined = pd.concat(all_results, ignore_index=True)
    out = RESULTS_DIR / 'walkforward_results_no_pinnacle.parquet'
    combined.to_parquet(out, index=False)
    log.info(f'Сохранено: {out}')
    return combined


def extract_feature_importance(df, params):
    log.info('\n=== Feature Importance (без Pinnacle) ===')
    df_clean = df.dropna(subset=['target']).copy()
    avail = [c for c in FEATURE_COLS if c in df_clean.columns]

    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    le = LabelEncoder()
    X = df_clean[avail].fillna(0).values
    y = le.fit_transform(df_clean['target'].values)

    clf = XGBClassifier(**params, random_state=42, eval_metric='mlogloss', verbosity=0)
    clf.fit(X, y)

    imp = pd.Series(clf.feature_importances_, index=avail).sort_values(ascending=False)
    imp.to_csv(RESULTS_DIR / 'feature_importance_no_pinnacle.csv', header=['importance'])

    log.info('Top-20:')
    for feat, val in imp.head(20).items():
        log.info(f'  {feat:<40} {val:.4f}')
    return imp


def run_strategy_validation(results_df):
    log.info('\n=== Strategy Validation (без Pinnacle) ===')
    results_df = results_df.copy()
    results_df['year'] = results_df['Season'].str[:4].astype(int)
    df_sel = results_df[results_df['year'] < 2019]
    df_hld = results_df[results_df['year'] >= 2019]

    def calc_roi(df):
        bets = df[df['kelly_size'] > 0]
        if len(bets) == 0:
            return {'n_bets': 0, 'roi': np.nan}
        return {'n_bets': len(bets), 'roi': bets['profit'].sum() / bets['kelly_size'].sum()}

    def apply_strategy(df, leagues, outcomes, edge_min):
        mask = df['League'].isin(leagues) & (df['best_edge'] >= edge_min)
        if outcomes:
            mask &= df['best_outcome'].isin(outcomes)
        return df[mask]

    all_leagues = sorted(results_df['League'].unique())
    leagues_grid = [all_leagues] + [[l] for l in all_leagues] + [
        [l for l in ['P1', 'G1', 'E1'] if l in all_leagues],
        [l for l in ['P1', 'G1']       if l in all_leagues],
        [l for l in ['P1', 'E1']       if l in all_leagues],
        [l for l in ['G1', 'E1']       if l in all_leagues],
        [l for l in ['E0', 'D1', 'E1'] if l in all_leagues],
        [l for l in ['E0', 'D1']       if l in all_leagues],
    ]
    seen = set()
    leagues_grid = [lg for lg in leagues_grid
                    if len(lg) > 0 and tuple(sorted(lg)) not in seen
                    and not seen.add(tuple(sorted(lg)))]

    outcomes_grid = [None, ['home'], ['draw'], ['away'], ['home', 'draw'], ['draw', 'away']]
    edge_grid     = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

    rows = []
    for leagues, outcomes, edge_min in itertools.product(leagues_grid, outcomes_grid, edge_grid):
        m_sel = calc_roi(apply_strategy(df_sel, leagues, outcomes, edge_min))
        if m_sel['n_bets'] < 20 or np.isnan(m_sel['roi']):
            continue
        m_hld = calc_roi(apply_strategy(df_hld, leagues, outcomes, edge_min))
        rows.append({
            'leagues': '+'.join(sorted(leagues)),
            'outcomes': '+'.join(outcomes) if outcomes else 'all',
            'edge_min': edge_min,
            'sel_bets': m_sel['n_bets'], 'sel_roi': m_sel['roi'],
            'hld_bets': m_hld['n_bets'], 'hld_roi': m_hld['roi'] if not np.isnan(m_hld.get('roi', np.nan)) else 0.0,
        })

    grid_df = pd.DataFrame(rows)
    valid = grid_df[grid_df['hld_bets'] >= 10].sort_values('hld_roi', ascending=False)

    log.info(f'\nТоп-10 (без Pinnacle):')
    log.info(f'{"Лиги":<20} {"Исходы":<15} {"Edge":>5}  {"SEL":>6}  {"SEL_ROI":>8}  {"HLD":>5}  {"HLD_ROI":>8}')
    log.info('-' * 75)
    for _, row in valid.head(10).iterrows():
        log.info(f'{row["leagues"]:<20} {row["outcomes"]:<15} {row["edge_min"]:>5.2f}  '
                 f'{row["sel_bets"]:>6.0f}  {row["sel_roi"]:>+8.1%}  '
                 f'{row["hld_bets"]:>5.0f}  {row["hld_roi"]:>+8.1%}')

    report_path = RESULTS_DIR / 'strategy_validation_no_pinnacle.txt'
    valid.head(15).to_string(buf=open(report_path, 'w'))
    log.info(f'Отчёт: {report_path}')


def main():
    t0 = time.time()
    log.info('=== RUN WITHOUT PINNACLE START ===')

    df = load_features()
    best_params, _ = run_optuna(df)
    results_df = run_walkforward(df)
    extract_feature_importance(df, best_params)
    run_strategy_validation(results_df)

    elapsed = (time.time() - t0) / 3600
    log.info(f'\n=== DONE в {elapsed:.1f}h ===')


if __name__ == '__main__':
    main()
