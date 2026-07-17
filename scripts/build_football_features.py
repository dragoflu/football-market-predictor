"""
Строит полную матрицу фичей для всех 5 лиг и сохраняет в parquet.
Запускать на сервере отца (Dell OptiPlex, 32GB RAM).

Время: ~20-40 мин, самое медленное это league_position O(n^2)
Результат: data/processed/football_features.parquet

Запуск: python scripts/build_football_features.py
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('cp1251', 'cp866'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import logging
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd


def read_parquet_safe(path):
    """Read parquet with pyarrow directly to avoid pandas wrapper bug."""
    table = pq.read_table(str(path))
    return table.to_pandas()

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.football_features import build_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('data/processed/build_features.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

DATA_DIR = Path('data/raw/football')
OUT_PATH = Path('data/processed/football_features.parquet')
OUT_PATH.parent.mkdir(exist_ok=True)


def main():
    log.info('=== Building Football Feature Matrix ===')

    # Загружаем все лиги
    parquets = sorted(DATA_DIR.glob('matches_*.parquet'))
    if not parquets:
        log.error('No data found. Run collect_football_history.py first.')
        return

    dfs = []
    for p in parquets:
        df = read_parquet_safe(p)
        dfs.append(df)
        log.info(f'  Loaded {p.name}: {len(df):,} matches')

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    log.info(f'\nTotal: {len(df_all):,} matches')

    # Пробуем загрузить xG данные
    xg_dfs = []
    for p in DATA_DIR.glob('xg_*.parquet'):
        xg_df = read_parquet_safe(p)
        xg_dfs.append(xg_df)
        log.info(f'  xG: {p.name} - {len(xg_df):,} matches')

    xg_all = pd.concat(xg_dfs, ignore_index=True) if xg_dfs else None

    # Строим фичи по лигам (быстрее, чем весь датасет сразу)
    all_features = []

    for league in df_all['League'].unique():
        log.info(f'\n=== {league} ===')
        league_df = df_all[df_all['League'] == league].copy()

        league_xg = None
        if xg_all is not None and 'League' in xg_all.columns:
            league_xg = xg_all[xg_all['League'] == league]
            if len(league_xg) == 0:
                league_xg = None

        features = build_features(league_df, xg_df=league_xg, verbose=True)
        all_features.append(features)
        log.info(f'  {league}: {len(features):,} rows × {len(features.columns)} features')

    # Объединяем и сохраняем
    final = pd.concat(all_features, ignore_index=True)
    final = final.sort_values(['League', 'Date']).reset_index(drop=True)

    log.info(f'\nFinal feature matrix: {final.shape}')
    log.info(f'Saving to {OUT_PATH}...')
    final.to_parquet(OUT_PATH, index=False)
    log.info('Done.')

    # Статистика покрытия
    coverage = final.notna().mean()
    low_coverage = coverage[coverage < 0.7]
    if len(low_coverage) > 0:
        log.info('\nLow coverage columns (< 70%):')
        for col, pct in low_coverage.items():
            log.info(f'  {col}: {pct:.1%}')


if __name__ == '__main__':
    main()
