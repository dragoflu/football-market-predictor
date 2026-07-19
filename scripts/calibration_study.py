"""
Генерирует сырые и калиброванные вероятности на holdout для анализа калибровки.

Модель нигде не сохранена, а walk_forward хранит только финальные (калиброванные)
вероятности. Для reliability-диаграмм до/после калибровки нужны обе версии, поэтому
здесь один фит: train на сезонах до SPLIT, val-сезон для калибровки, holdout от SPLIT.
Для каждого holdout-матча берём _raw_ensemble_proba (до) и predict_proba (после).

Выход:
  data/results/calibration_probs.parquet
  data/models/ensemble_holdout.joblib   (модель, чтобы не переобучать)

Запуск на сервере отца (один фит, минуты):
  scp scripts/calibration_study.py "Snoop Dog"@192.168.0.35:"C:\\first_project\\scripts\\calibration_study.py"
  ssh "Snoop Dog"@192.168.0.35
  cd C:\\first_project && python scripts\\calibration_study.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.football_model import EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

FEATURES_PATH = ROOT / 'data/processed/football_features.parquet'
OUT_PARQUET = ROOT / 'data/results/calibration_probs.parquet'
OUT_MODEL = ROOT / 'data/models/ensemble_holdout.joblib'

SPLIT_YEAR = 2019      # holdout начинается с сезона 2019/2020


def main():
    if not FEATURES_PATH.exists():
        log.error(f'Нет {FEATURES_PATH}, сначала собери фичи (build_football_features.py)')
        return

    df = pd.read_parquet(FEATURES_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Season'] = df['Season'].astype(str)
    df['year'] = df['Season'].str[:4].astype(int)

    holdout = df[df['year'] >= SPLIT_YEAR]
    train_all = df[df['year'] < SPLIT_YEAR]

    train_seasons = sorted(train_all['Season'].unique())
    val_season = train_seasons[-1]                       # последний трейновый сезон под калибратор
    val_df = train_all[train_all['Season'] == val_season]
    actual_train = train_all[train_all['Season'] != val_season]

    log.info(f'Train: {train_seasons[0]}..{train_seasons[-2]} ({len(actual_train)} матчей)')
    log.info(f'Val (калибратор): {val_season} ({len(val_df)} матчей)')
    log.info(f'Holdout: {SPLIT_YEAR}+ ({len(holdout)} матчей)')

    model = EnsembleModel()
    model.fit(actual_train, val_df)

    rows = []
    for _, row in holdout.iterrows():
        target = row.get('target')
        if target not in ('H', 'D', 'A'):
            continue
        home, away = row['HomeTeam'], row['AwayTeam']
        feat = row.to_dict()
        try:
            raw = model._raw_ensemble_proba(home, away, feat)
            cal = model.predict_proba(home, away, feat)
        except Exception:
            continue
        rows.append({
            'Date': row['Date'], 'Season': row['Season'], 'League': row['League'],
            'HomeTeam': home, 'AwayTeam': away, 'target': target,
            'raw_home': raw['home'], 'raw_draw': raw['draw'], 'raw_away': raw['away'],
            'cal_home': cal['home'], 'cal_draw': cal['draw'], 'cal_away': cal['away'],
            # implied Pinnacle для опционального сравнения model vs market
            'implied_home': row.get('implied_home', np.nan),
            'implied_draw': row.get('implied_draw', np.nan),
            'implied_away': row.get('implied_away', np.nan),
        })

    out = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    log.info(f'Сохранено {len(out)} строк в {OUT_PARQUET}')

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUT_MODEL)

    # быстрая сверка: Brier сырых vs калиброванных на home-исходе
    y_home = (out['target'] == 'H').astype(int)
    brier_raw = ((out['raw_home'] - y_home) ** 2).mean()
    brier_cal = ((out['cal_home'] - y_home) ** 2).mean()
    log.info(f'Brier home  raw={brier_raw:.4f}  cal={brier_cal:.4f}')


if __name__ == '__main__':
    main()
