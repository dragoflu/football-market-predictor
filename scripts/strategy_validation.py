"""
Честная валидация стратегий: разделение на selection и holdout периоды.

  Selection:  2006–2018 (grid search, выбираем лучшую стратегию)
  Holdout:    2019–2026 (трогаем ОДИН раз, финальная проверка)

Запуск:
    python scripts/strategy_validation.py

Сервер:
    scp scripts/strategy_validation.py "Snoop Dog"@192.168.0.35:"C:\\first_project\\scripts\\strategy_validation.py"
    python scripts\\strategy_validation.py
"""

import sys
import itertools
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_PATH = Path('data/results/walkforward_results.parquet')
SPLIT_YEAR = 2019  # до — selection, с — holdout


def calc_roi(df: pd.DataFrame) -> dict:
    bets = df[df['kelly_size'] > 0]
    n = len(bets)
    if n == 0:
        return {'n_bets': 0, 'roi': np.nan, 'win_rate': np.nan}
    staked = bets['kelly_size'].sum()
    profit = bets['profit'].sum()
    return {
        'n_bets': n,
        'roi': profit / staked,
        'win_rate': (bets['profit'] > 0).mean(),
        'profit': profit,
    }


def apply_strategy(df, leagues=None, outcomes=None, edge_min=0.08, model_prob_min=0.0):
    mask = pd.Series(True, index=df.index)
    if leagues:
        mask &= df['League'].isin(leagues)
    if outcomes:
        mask &= df['best_outcome'].isin(outcomes)
    mask &= df['best_edge'] >= edge_min
    if model_prob_min > 0:
        prob_map = {'home': 'pred_home', 'draw': 'pred_draw', 'away': 'pred_away'}
        model_probs = df.apply(lambda r: r[prob_map.get(r['best_outcome'], 'pred_home')], axis=1)
        mask &= model_probs >= model_prob_min
    return df[mask]


def grid_search_on(df: pd.DataFrame, min_bets_per_season: float = 15, label: str = '') -> pd.DataFrame:
    """Grid search на переданном датасете."""
    all_leagues = sorted(df['League'].unique())
    n_seasons = df['Season'].nunique()

    leagues_grid = [
        all_leagues,
        ['E0', 'D1'],
        ['E0', 'D1', 'SP1'],
        ['E0', 'D1', 'E1'],
        ['E0', 'D1', 'E1', 'B1'],
        ['E0', 'D1', 'E1', 'N1'],
        ['E0'], ['D1'], ['SP1'], ['I1'], ['F1'],
        ['E1'], ['B1'], ['P1'], ['N1'], ['T1'],
        ['E0', 'E1'],
        ['D1', 'E1'],
        ['B1', 'P1'],
        ['E0', 'D1', 'B1', 'P1'],
        ['E0', 'SP1'],
        ['D1', 'SP1'],
    ]
    outcomes_grid = [None, ['draw'], ['away'], ['home'], ['draw', 'away']]
    edge_grid = [0.08, 0.10, 0.15, 0.20, 0.25]

    min_bets_total = int(min_bets_per_season * n_seasons)

    results = []
    for leagues, outcomes, edge_min in itertools.product(leagues_grid, outcomes_grid, edge_grid):
        filtered = apply_strategy(df, leagues=leagues, outcomes=outcomes, edge_min=edge_min)
        m = calc_roi(filtered)
        if m['n_bets'] < min_bets_total or np.isnan(m['roi']):
            continue
        bets = filtered[filtered['kelly_size'] > 0]
        bets_per_season = m['n_bets'] / n_seasons
        results.append({
            'leagues': '+'.join(sorted(leagues)) if leagues else 'all',
            'outcomes': '+'.join(outcomes) if outcomes else 'all',
            'edge_min': edge_min,
            'n_bets': m['n_bets'],
            'bets_per_season': round(bets_per_season, 1),
            'roi': m['roi'],
            'win_rate': m['win_rate'],
        })

    grid_df = pd.DataFrame(results).sort_values('roi', ascending=False)
    print(f'\n{"="*70}')
    print(f'ТОП-10 на {label} (min {min_bets_per_season:.0f} ставок/сезон = {min_bets_total} всего)')
    print(f'{"="*70}')
    print(f'{"Лиги":<22} {"Исходы":<14} {"Edge":<7} {"Ставок":<8} {"/сезон":<8} {"ROI":<10} {"Win%"}')
    print('-' * 75)
    for _, row in grid_df.head(10).iterrows():
        print(f'{row["leagues"]:<22} {row["outcomes"]:<14} {row["edge_min"]:<7.2f} '
              f'{row["n_bets"]:<8.0f} {row["bets_per_season"]:<8.1f} {row["roi"]:+.1%}     {row["win_rate"]:.1%}')
    return grid_df


def validate_strategy(df_holdout: pd.DataFrame,
                      leagues: list, outcomes: list | None,
                      edge_min: float, label: str):
    """Применяет одну стратегию к holdout и печатает результат."""
    filtered = apply_strategy(df_holdout, leagues=leagues, outcomes=outcomes, edge_min=edge_min)
    m = calc_roi(filtered)
    bets = filtered[filtered['kelly_size'] > 0]

    print(f'\n{"="*55}')
    print(f'  HOLDOUT: {label}')
    print(f'{"="*55}')
    if m['n_bets'] == 0:
        print('  Нет ставок на holdout периоде')
        return

    print(f'  Ставок: {m["n_bets"]} ({m["n_bets"] / len(df_holdout["Season"].unique()):.1f}/сезон)')
    print(f'  ROI:    {m["roi"]:+.1%}')
    print(f'  Win%:   {m["win_rate"]:.1%}')
    print(f'  Profit: {m["profit"]:+.3f} units')

    print('\n  По сезонам:')
    for season in sorted(bets['Season'].unique()):
        sb = bets[bets['Season'] == season]
        staked = sb['kelly_size'].sum()
        roi = sb['profit'].sum() / staked if staked > 0 else 0
        print(f'    {season}: ROI={roi:+.1%} ({len(sb)} bets)')


def main():
    if not RESULTS_PATH.exists():
        print(f'Не найден {RESULTS_PATH}')
        print('Сначала: python scripts/run_full_pipeline.py')
        return

    df = pd.read_parquet(RESULTS_PATH)
    df['year'] = df['Season'].str[:4].astype(int)

    df_sel = df[df['year'] < SPLIT_YEAR].copy()
    df_hld = df[df['year'] >= SPLIT_YEAR].copy()

    sel_seasons = sorted(df_sel['Season'].unique())
    hld_seasons = sorted(df_hld['Season'].unique())

    print(f'Загружено: {len(df):,} матчей')
    print(f'Selection: {sel_seasons[0]} — {sel_seasons[-1]} ({len(sel_seasons)} сезонов, {len(df_sel):,} матчей)')
    print(f'Holdout:   {hld_seasons[0]} — {hld_seasons[-1]} ({len(hld_seasons)} сезонов, {len(df_hld):,} матчей)')

    # Шаг 1: grid search ТОЛЬКО на selection
    print('\n\n--- ШАГ 1: Grid search на selection периоде (2006-2018) ---')
    grid_sel = grid_search_on(df_sel, min_bets_per_season=15, label='2006–2018')

    # Шаг 2: берём топ-3 стратегии из selection и проверяем на holdout
    print('\n\n--- ШАГ 2: Валидация топ стратегий на holdout (2019–2026) ---')
    print('(Этот раздел трогаем только ОДИН раз — финальная честная проверка)\n')

    top3 = grid_sel.head(3)
    for _, row in top3.iterrows():
        leagues = row['leagues'].split('+') if row['leagues'] != 'all' else None
        outcomes = row['outcomes'].split('+') if row['outcomes'] != 'all' else None
        label = f'{row["leagues"]} | {row["outcomes"]} | edge≥{row["edge_min"]}'
        validate_strategy(df_hld, leagues=leagues, outcomes=outcomes,
                          edge_min=row['edge_min'], label=label)

    # Шаг 3: проверяем конкретные гипотезы на holdout
    print('\n\n--- ШАГ 3: Проверка гипотез на holdout ---')

    print('(A) D1+E0, draw, edge≥0.25 — гипотеза из предыдущего анализа')
    validate_strategy(df_hld,
                      leagues=['D1', 'E0'],
                      outcomes=['draw'],
                      edge_min=0.25,
                      label='D1+E0 | draw | edge≥0.25  [pre-registered]')

    print('\n(B) D1+E0+E1, home, edge≥0.25 — новая гипотеза из grid')
    validate_strategy(df_hld,
                      leagues=['D1', 'E0', 'E1'],
                      outcomes=['home'],
                      edge_min=0.25,
                      label='D1+E0+E1 | home | edge≥0.25  [new]')

    print('\n(C) B1+P1, draw+away, edge≥0.25 — новые лиги')
    validate_strategy(df_hld,
                      leagues=['B1', 'P1'],
                      outcomes=['draw', 'away'],
                      edge_min=0.25,
                      label='B1+P1 | draw+away | edge≥0.25  [new]')

    print('\n(D) SP1, away, edge≥0.20 — лучшая SP1 стратегия')
    validate_strategy(df_hld,
                      leagues=['SP1'],
                      outcomes=['away'],
                      edge_min=0.20,
                      label='SP1 | away | edge≥0.20  [new]')

    # Шаг 4: сравнение selection vs holdout для топ-1 стратегии
    if len(grid_sel) > 0:
        top1 = grid_sel.iloc[0]
        leagues = top1['leagues'].split('+') if top1['leagues'] != 'all' else None
        outcomes = top1['outcomes'].split('+') if top1['outcomes'] != 'all' else None

        sel_m = calc_roi(apply_strategy(df_sel, leagues=leagues,
                                         outcomes=outcomes, edge_min=top1['edge_min']))
        hld_m = calc_roi(apply_strategy(df_hld, leagues=leagues,
                                         outcomes=outcomes, edge_min=top1['edge_min']))

        print(f'\n\n--- ШАГ 4: Деградация топ-1 стратегии ---')
        print(f'Стратегия: {top1["leagues"]} | {top1["outcomes"]} | edge≥{top1["edge_min"]}')
        print(f'  Selection ROI: {sel_m["roi"]:+.1%} ({sel_m["n_bets"]} bets)')
        if not np.isnan(hld_m["roi"]):
            print(f'  Holdout ROI:   {hld_m["roi"]:+.1%} ({hld_m["n_bets"]} bets)')
            decay = sel_m["roi"] - hld_m["roi"]
            print(f'  Деградация:    {decay:+.1%} п.п.')
            if hld_m["roi"] > 0:
                print('  ✓ Edge сохранился на holdout')
            else:
                print('  ✗ Edge не подтвердился — likely overfitting')
        else:
            print(f'  Holdout: нет ставок (мало данных)')


if __name__ == '__main__':
    main()
