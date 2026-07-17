"""
Strategy Tester: фильтрует готовые walk-forward результаты без перезапуска модели.

Загружает data/results/walkforward_results.parquet и тестирует комбинации:
  - leagues:       какие лиги включать
  - outcomes:      какие исходы торговать (home/draw/away)
  - edge_min:      минимальный edge
  - model_prob_min: минимальная вероятность модели на выбранный исход
  - market_prob_max: максимальная рыночная цена (отсечь фаворитов)

Запуск:
    python scripts/strategy_tester.py
    python scripts/strategy_tester.py --top 20   # показать топ-20 стратегий
    python scripts/strategy_tester.py --grid      # полный grid search

Сервер отца:
    scp scripts/strategy_tester.py "Snoop Dog"@192.168.0.35:"C:\\first_project\\scripts\\"
    ssh "Snoop Dog"@192.168.0.35
    cd C:\\first_project && python scripts\\strategy_tester.py
"""

import sys
import argparse
import itertools
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


RESULTS_PATH = Path('data/results/walkforward_results.parquet')


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f'Не найден {RESULTS_PATH}\n'
            'Сначала запусти: python scripts/run_full_pipeline.py'
        )
    df = pd.read_parquet(RESULTS_PATH)
    print(f'Загружено {len(df):,} матчей из {RESULTS_PATH}')
    print(f'Лиги: {sorted(df["League"].unique())}')
    print(f'Сезоны: {df["Season"].min()} - {df["Season"].max()}')
    print(f'Колонки: {list(df.columns)}\n')
    return df


def calc_roi(df: pd.DataFrame) -> dict:
    """Считает метрики по отфильтрованному датасету."""
    bets = df[df['kelly_size'] > 0]
    n = len(bets)
    if n == 0:
        return {'n_bets': 0, 'roi': np.nan, 'win_rate': np.nan, 'profit': 0.0}

    staked = bets['kelly_size'].sum()
    profit = bets['profit'].sum()
    roi = profit / staked
    win_rate = (bets['profit'] > 0).mean()
    return {
        'n_bets': n,
        'roi': roi,
        'win_rate': win_rate,
        'profit': profit,
    }


def apply_strategy(df: pd.DataFrame,
                   leagues: list[str] | None = None,
                   outcomes: list[str] | None = None,
                   edge_min: float = 0.08,
                   model_prob_min: float = 0.0,
                   market_prob_max: float = 1.0) -> pd.DataFrame:
    """Применяет фильтры стратегии к датафрейму результатов."""
    mask = pd.Series(True, index=df.index)

    if leagues:
        mask &= df['League'].isin(leagues)

    if outcomes:
        mask &= df['best_outcome'].isin(outcomes)

    mask &= df['best_edge'] >= edge_min

    # вероятность модели на выбранный исход
    if model_prob_min > 0:
        prob_col_map = {'home': 'pred_home', 'draw': 'pred_draw', 'away': 'pred_away'}
        model_prob = df['best_outcome'].map(prob_col_map).map(lambda c: df[c] if isinstance(c, str) else np.nan)
        # Вычисляем prob для каждой строки
        model_probs = df.apply(
            lambda r: r[prob_col_map.get(r['best_outcome'], 'pred_home')], axis=1
        )
        mask &= model_probs >= model_prob_min

    # отсекаем случаи где рынок уже оценил высоко
    if market_prob_max < 1.0:
        market_prob_map = {'home': 'market_home', 'draw': 'market_draw', 'away': 'market_away'}
        market_probs = df.apply(
            lambda r: r[market_prob_map.get(r['best_outcome'], 'market_home')], axis=1
        )
        mask &= market_probs <= market_prob_max

    return df[mask].copy()


def print_strategy_report(df: pd.DataFrame, label: str = ''):
    """Детальный отчёт по одной стратегии."""
    bets = df[df['kelly_size'] > 0]
    if len(bets) == 0:
        print(f'{label}: нет ставок')
        return

    metrics = calc_roi(df)
    print(f'\n{"="*55}')
    print(f'  {label}')
    print(f'{"="*55}')
    print(f'  Ставок: {metrics["n_bets"]:,}')
    print(f'  ROI:    {metrics["roi"]:+.1%}')
    print(f'  Win%:   {metrics["win_rate"]:.1%}')
    print(f'  Profit: {metrics["profit"]:+.3f} units')

    # По лигам
    print('\n  По лигам:')
    for league in sorted(bets['League'].unique()):
        lb = bets[bets['League'] == league]
        staked = lb['kelly_size'].sum()
        roi = lb['profit'].sum() / staked if staked > 0 else 0
        print(f'    {league}: ROI={roi:+.1%} ({len(lb)} bets)')

    # По исходам
    print('\n  По исходам:')
    for outcome in ['home', 'draw', 'away']:
        ob = bets[bets['best_outcome'] == outcome]
        if len(ob) == 0:
            continue
        staked = ob['kelly_size'].sum()
        roi = ob['profit'].sum() / staked if staked > 0 else 0
        print(f'    {outcome}: ROI={roi:+.1%} ({len(ob)} bets)')

    # По сезонам
    print('\n  По сезонам:')
    for season in sorted(bets['Season'].unique()):
        sb = bets[bets['Season'] == season]
        staked = sb['kelly_size'].sum()
        roi = sb['profit'].sum() / staked if staked > 0 else 0
        print(f'    {season}: ROI={roi:+.1%} ({len(sb)} bets)')


def grid_search(df: pd.DataFrame, top_n: int = 20):
    """Grid search по всем комбинациям параметров."""
    all_leagues = sorted(df['League'].unique())

    # Параметры для перебора
    leagues_grid = [
        all_leagues,
        ['E0', 'D1'],
        ['E0', 'D1', 'SP1'],
        ['E0', 'D1', 'E1'],
        ['E0', 'D1', 'E1', 'B1'],
        ['E0', 'D1', 'E1', 'N1'],
        ['E0'],
        ['SP1'],
        ['D1'],
        ['E1'],
        ['B1'],
        ['P1'],
        ['N1'],
        ['T1'],
        ['E0', 'E1'],
        ['D1', 'E1'],
        ['B1', 'P1'],
        ['E0', 'D1', 'B1', 'P1'],
    ]
    outcomes_grid = [
        None,               # все исходы
        ['draw'],
        ['away'],
        ['home'],
        ['draw', 'away'],
    ]
    edge_grid = [0.08, 0.10, 0.15, 0.20, 0.25]
    model_prob_grid = [0.0, 0.35, 0.40, 0.45]

    results = []
    total = len(leagues_grid) * len(outcomes_grid) * len(edge_grid) * len(model_prob_grid)
    print(f'Grid search: {total} комбинаций...\n')

    for leagues, outcomes, edge_min, model_prob_min in itertools.product(
        leagues_grid, outcomes_grid, edge_grid, model_prob_grid
    ):
        filtered = apply_strategy(
            df,
            leagues=leagues,
            outcomes=outcomes,
            edge_min=edge_min,
            model_prob_min=model_prob_min,
        )
        m = calc_roi(filtered)
        if m['n_bets'] < 30 or np.isnan(m['roi']):
            continue

        results.append({
            'leagues': '+'.join(sorted(leagues)) if leagues else 'all',
            'outcomes': '+'.join(outcomes) if outcomes else 'all',
            'edge_min': edge_min,
            'model_prob_min': model_prob_min,
            'n_bets': m['n_bets'],
            'roi': m['roi'],
            'win_rate': m['win_rate'],
        })

    if not results:
        print('Нет результатов с n_bets >= 30')
        return

    grid_df = pd.DataFrame(results).sort_values('roi', ascending=False)

    top = grid_df.head(top_n).rename(columns={
        'leagues': 'Лиги', 'outcomes': 'Исходы', 'edge_min': 'Edge',
        'model_prob_min': 'P_min', 'n_bets': 'Ставок',
        'roi': 'ROI', 'win_rate': 'Win%',
    }).copy()
    top['ROI'] = top['ROI'].map('{:+.1%}'.format)
    top['Win%'] = top['Win%'].map('{:.1%}'.format)

    print(f'\nТОП-{top_n} СТРАТЕГИЙ (min 30 ставок)')
    print(top.to_string(index=False))

    out_path = Path('data/results/strategy_grid.csv')
    grid_df.to_csv(out_path, index=False)
    print(f'\nПолная таблица сохранена: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Football strategy tester')
    parser.add_argument('--top', type=int, default=20, help='Топ N стратегий в grid')
    parser.add_argument('--grid', action='store_true', help='Запустить полный grid search')
    args = parser.parse_args()

    df = load_results()

    # 1. Базовый отчёт (вся выборка, текущий threshold=0.08)
    print_strategy_report(df, label='Baseline: все лиги, все исходы, edge≥0.08')

    # 2. Стратегия из предыдущего анализа (E0+D1, только ничьи, edge≥0.20)
    filtered_draw = apply_strategy(
        df,
        leagues=['E0', 'D1'],
        outcomes=['draw'],
        edge_min=0.20,
        model_prob_min=0.35,
    )
    print_strategy_report(filtered_draw, label='E0+D1 | draw only | edge≥0.20 | p_model≥0.35')

    # 3. Расширенная: E0+D1+SP1, draw+away
    filtered_ext = apply_strategy(
        df,
        leagues=['E0', 'D1', 'SP1'],
        outcomes=['draw', 'away'],
        edge_min=0.15,
        model_prob_min=0.35,
    )
    print_strategy_report(filtered_ext, label='E0+D1+SP1 | draw+away | edge≥0.15 | p_model≥0.35')

    # 4. SP1 отдельно: лучшая лига по ROI
    filtered_sp1 = apply_strategy(
        df,
        leagues=['SP1'],
        outcomes=None,
        edge_min=0.10,
        model_prob_min=0.0,
    )
    print_strategy_report(filtered_sp1, label='SP1 only | all outcomes | edge≥0.10')

    # 5. Grid search если запрошен
    if args.grid:
        print('\n\n')
        grid_search(df, top_n=args.top)
    else:
        print('\n\nДля полного grid search запусти с флагом --grid')
        print('Пример: python scripts/strategy_tester.py --grid --top 30')


if __name__ == '__main__':
    main()
