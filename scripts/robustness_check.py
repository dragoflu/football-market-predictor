"""
Robustness Check: устойчивость стратегий к смене параметров модели.

Логика:
  1. Запускаем Optuna N раз с разными random_seed
  2. Для каждого набора параметров делаем walk-forward
  3. Проверяем, держится ли ROI топ-стратегий во всех прогонах

Стратегии для проверки (топ из grid search):
  - F1+I1+T1  | draw | edge≥0.10
  - F1+G1+I1  | draw | edge≥0.10
  - I1+T1     | draw | edge≥0.10
  - G1+P1     | away | edge≥0.12
  - D1+P1     | home+draw | edge≥0.18  (старая стратегия, для сравнения)

Запуск:
    python scripts/robustness_check.py

Сервер:
    scp scripts/robustness_check.py "Snoop Dog"@192.168.0.35:"C:\\first_project\\scripts\\robustness_check.py"
    ssh "Snoop Dog"@192.168.0.35
    cd C:\\first_project && python scripts\\robustness_check.py
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('cp1251', 'cp866'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import time
import logging
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path('data/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'robustness_check.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# Конфигурация

# разные сиды Optuna: если стратегия держится на всех, это не переобучение под сид
RANDOM_SEEDS = [42, 7, 13, 99, 2025]

# меньше чем в overnight: нужна скорость, 5 прогонов подряд
OPTUNA_TRIALS = 30

# Лиги для тюнинга Optuna (те же что в overnight)
OPTUNA_LEAGUES = ['E0', 'D1', 'E1']

# Все лиги для walk-forward
ALL_LEAGUES = ['E0', 'SP1', 'D1', 'I1', 'F1', 'E1', 'N1', 'B1', 'P1', 'T1', 'SC0', 'G1']

# Стратегии для проверки
STRATEGIES = [
    {'name': 'F1+I1+T1 draw≥0.10',       'leagues': ['F1','I1','T1'],    'outcomes': ['draw'],         'edge': 0.10},
    {'name': 'F1+G1+I1 draw≥0.10',       'leagues': ['F1','G1','I1'],    'outcomes': ['draw'],         'edge': 0.10},
    {'name': 'I1+T1 draw≥0.10',          'leagues': ['I1','T1'],         'outcomes': ['draw'],         'edge': 0.10},
    {'name': 'G1+P1 away≥0.12',          'leagues': ['G1','P1'],         'outcomes': ['away'],         'edge': 0.12},
    {'name': 'D1+P1 home+draw≥0.18',     'leagues': ['D1','P1'],         'outcomes': ['home','draw'],  'edge': 0.18},
]

N_HLD_SEASONS = 7  # 2019/2020 .. 2025/2026


# Утилиты

def calc_strategy_roi(df_hld: pd.DataFrame, strategy: dict) -> dict:
    mask = (
        df_hld['League'].isin(strategy['leagues']) &
        df_hld['best_outcome'].isin(strategy['outcomes']) &
        (df_hld['best_edge'] >= strategy['edge'])
    )
    bets = df_hld[mask & (df_hld['kelly_size'] > 0)]
    n = len(bets)
    if n == 0:
        return {'n': 0, 'roi': np.nan, 'per_season': 0.0}
    staked = bets['kelly_size'].sum()
    roi = bets['profit'].sum() / staked
    return {'n': n, 'roi': roi, 'per_season': n / N_HLD_SEASONS}


def tune_with_seed(features_df: pd.DataFrame, seed: int) -> dict:
    """Запускает Optuna с конкретным seed, возвращает лучшие параметры."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError('pip install optuna')

    from xgboost import XGBClassifier
    from sklearn.metrics import log_loss
    from src.football_model import build_feature_matrix

    df = features_df.copy()
    df['Season'] = df['Season'].astype(str)
    seasons = sorted(df['Season'].unique())
    val_seasons = seasons[-2:]
    train_seasons = seasons[:-2]

    train = df[df['Season'].isin(train_seasons)]
    val = df[df['Season'].isin(val_seasons)]
    X_train, y_train, _ = build_feature_matrix(train)
    X_val, y_val, _ = build_feature_matrix(val)

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
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = clf.predict_proba(X_val)
        return log_loss(y_val, proba)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params


def patch_and_reload(best_params: dict, best_weights=None):
    """Патчит football_model.py и перезагружает модуль."""
    import re
    model_path = Path('src/football_model.py')
    text = model_path.read_text(encoding='utf-8')
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
        model_path.write_text(text, encoding='utf-8')

    import src.football_model as fm_module
    importlib.reload(fm_module)


def run_walkforward_all(features_df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward по всем доступным лигам."""
    import src.football_model as fm_module

    all_results = []
    for league in sorted(features_df['League'].unique()):
        ldf = features_df[features_df['League'] == league].copy()
        seasons = sorted(ldf['Season'].astype(str).unique())
        if len(seasons) < 7:
            continue
        results = fm_module.walk_forward_validate(ldf, n_train_seasons=5, min_test_matches=200, edge_threshold=0.08)
        if len(results) > 0:
            results['League'] = league
            all_results.append(results)

    if not all_results:
        return pd.DataFrame()
    combined = pd.concat(all_results, ignore_index=True)
    return combined


# Main

def main():
    t_total = time.time()
    log.info('=' * 60)
    log.info('ROBUSTNESS CHECK START')
    log.info(f'Seeds: {RANDOM_SEEDS}, {OPTUNA_TRIALS} trials each')
    log.info('=' * 60)

    # Загружаем feature matrix (уже собрана overnight pipeline)
    feat_path = Path('data/processed/football_features.parquet')
    if not feat_path.exists():
        log.error('Нет data/processed/football_features.parquet, сначала запусти overnight_pipeline.py')
        return

    features_df = pd.read_parquet(feat_path)
    log.info(f'Feature matrix: {features_df.shape}')

    # Лиги для тюнинга
    df_tune = features_df[features_df['League'].isin(OPTUNA_LEAGUES)].copy()
    log.info(f'Тюнинг на: {OPTUNA_LEAGUES} ({len(df_tune):,} матчей)')

    # Результаты по всем прогонам
    all_runs = []  # список DataFrame с результатами каждого прогона

    for i, seed in enumerate(RANDOM_SEEDS):
        log.info(f'\n{"="*60}')
        log.info(f'ПРОГОН {i+1}/{len(RANDOM_SEEDS)}, seed={seed}')
        log.info(f'{"="*60}')

        # 1. Optuna с этим seed
        t0 = time.time()
        log.info(f'  Optuna ({OPTUNA_TRIALS} trials)...')
        best_params = tune_with_seed(df_tune, seed)
        log.info(f'  Params: {best_params} ({time.time()-t0:.0f}s)')

        # 2. Патчим модель и перезагружаем
        patch_and_reload(best_params)

        # 3. Walk-forward
        t0 = time.time()
        log.info(f'  Walk-forward...')
        results_df = run_walkforward_all(features_df)
        log.info(f'  Walk-forward: {len(results_df):,} строк ({time.time()-t0:.0f}s)')

        if len(results_df) == 0:
            log.warning(f'  Нет результатов для seed={seed}')
            continue

        results_df['year'] = results_df['Season'].str[:4].astype(int)
        df_hld = results_df[results_df['year'] >= 2019]

        # 4. Считаем ROI для каждой стратегии
        run_result = {'seed': seed, 'params': best_params}
        for strat in STRATEGIES:
            m = calc_strategy_roi(df_hld, strat)
            run_result[strat['name']] = m['roi']
            run_result[strat['name'] + '_n'] = m['n']
            log.info(f"  {strat['name']}: ROI={m['roi']:+.1%} ({m['n']} bets, {m['per_season']:.1f}/сезон)")

        all_runs.append(run_result)

    log.info(f'\n{"="*60}')
    log.info('ROBUSTNESS REPORT')
    log.info(f'{"="*60}')

    rows = []
    for strat in STRATEGIES:
        name = strat['name']
        rois = [r[name] for r in all_runs if name in r and not np.isnan(r[name])]
        if not rois:
            rows.append({'Стратегия': name})
            continue
        roi_arr = np.array(rois)
        row = {'Стратегия': name}
        for s, r in zip(RANDOM_SEEDS, rois):
            row[f'seed={s}'] = f'{r:+.1%}'
        row['mean'] = f'{roi_arr.mean():+.1%}'
        row['std'] = f'±{roi_arr.std():.1%}'
        row['min'] = f'{roi_arr.min():+.1%}'
        row['стаб.'] = f'{(roi_arr > 0).sum()}/{len(rois)}'
        rows.append(row)

    table = pd.DataFrame(rows).fillna('нет данных').to_string(index=False)

    lines = [
        f'\nROBUSTNESS CHECK REPORT',
        f'Seeds: {RANDOM_SEEDS} | {OPTUNA_TRIALS} Optuna trials each\n',
        table,
        '\nСтабильная стратегия: mean > 0, std < |mean|, все прогоны > 0',
        'Нестабильная: высокий std или есть отрицательные прогоны',
    ]
    report = '\n'.join(lines)
    print(report)

    report_path = RESULTS_DIR / 'robustness_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log.info(f'Отчёт: {report_path}')

    # Сохраняем детальные данные
    runs_df = pd.DataFrame(all_runs)
    runs_df.to_csv(RESULTS_DIR / 'robustness_runs.csv', index=False)

    elapsed = (time.time() - t_total) / 60
    log.info(f'\nГотово за {elapsed:.1f} мин')


if __name__ == '__main__':
    main()
