"""
Feature importance analysis: XGBoost with vs without Pinnacle implied probs.
Outputs:
  data/results/feature_importance_with_pinnacle.csv
  data/results/feature_importance_without_pinnacle.csv
  data/results/ablation_report.txt
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
FEATURES_PATH = ROOT / 'data/processed/football_features.parquet'
PARAMS_PATH   = ROOT / 'data/results/xgboost_best_params.json'
RESULTS_PATH  = ROOT / 'data/results/walkforward_results.parquet'
OUT_DIR       = ROOT / 'data/results'

PINNACLE_COLS = ['implied_home', 'implied_draw', 'implied_away']

FEATURE_COLS = [
    'elo_diff', 'elo_home', 'elo_away',
    'home_goals_scored_last5', 'home_goals_scored_last10', 'home_goals_scored_last20',
    'home_goals_conceded_last5', 'home_goals_conceded_last10', 'home_goals_conceded_last20',
    'away_goals_scored_last5', 'away_goals_scored_last10', 'away_goals_scored_last20',
    'away_goals_conceded_last5', 'away_goals_conceded_last10', 'away_goals_conceded_last20',
    'home_points_last5', 'home_points_last10', 'home_points_last20',
    'away_points_last5', 'away_points_last10', 'away_points_last20',
    'home_xg_scored_last5', 'home_xg_scored_last10',
    'home_xg_conceded_last5', 'home_xg_conceded_last10',
    'away_xg_scored_last5', 'away_xg_scored_last10',
    'away_xg_conceded_last5', 'away_xg_conceded_last10',
    'draw_rate_last5', 'draw_rate_last10', 'draw_rate_last20',
    'home_clean_sheet_rate_last5', 'home_clean_sheet_rate_last10',
    'away_clean_sheet_rate_last5', 'away_clean_sheet_rate_last10',
    'home_total_goals_last5', 'home_total_goals_last10',
    'away_total_goals_last5', 'away_total_goals_last10',
    'elo_momentum', 'home_streak', 'away_streak',
    'season_stage', 'home_season_matches_played', 'away_season_matches_played',
    'implied_home', 'implied_draw', 'implied_away',
]


def load_params():
    with open(PARAMS_PATH) as f:
        return json.load(f)


def walk_forward_roi(df, feature_cols, params, strategy_leagues, edge_thresh=0.18):
    seasons = sorted(df['Season'].unique())
    all_preds = []

    for i, test_season in enumerate(seasons[6:], start=6):
        train_seasons = seasons[:i]
        train = df[df['Season'].isin(train_seasons)].dropna(subset=['target']).copy()
        test  = df[df['Season'] == test_season].dropna(subset=['target']).copy()

        avail = [c for c in feature_cols if c in train.columns]
        X_train = train[avail].fillna(0).values
        y_train = train['target'].values
        X_test  = test[avail].fillna(0).values

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(test['target'].values)

        clf = XGBClassifier(**params, random_state=42,
                            eval_metric='mlogloss', verbosity=0)
        clf.fit(X_train, y_train_enc)
        clf._le = le

        proba = clf.predict_proba(X_test)
        classes = list(le.classes_)
        h_idx = classes.index('H')
        d_idx = classes.index('D')
        a_idx = classes.index('A')

        test = test.copy()
        test['pred_home'] = proba[:, h_idx]
        test['pred_draw'] = proba[:, d_idx]
        test['pred_away'] = proba[:, a_idx]
        test['feature_cols'] = str(avail)
        all_preds.append(test)

    pred_df = pd.concat(all_preds)
    return pred_df, clf, avail


def compute_roi(pred_df, strategy_leagues, edge_thresh):
    # Market odds всегда из walkforward_results (Pinnacle closing line)
    results_df = pd.read_parquet(RESULTS_PATH)
    merge_cols = ['Season', 'Date', 'League', 'HomeTeam', 'AwayTeam',
                  'market_home', 'market_draw', 'market_away']
    pred_df = pred_df.drop(columns=['market_home','market_draw','market_away'], errors='ignore')
    pred_df = pred_df.merge(
        results_df[merge_cols].drop_duplicates(),
        on=['Season','Date','League','HomeTeam','AwayTeam'], how='left'
    )

    pred_df['edge_home'] = pred_df['pred_home'] - pred_df['market_home']
    pred_df['edge_draw'] = pred_df['pred_draw'] - pred_df['market_draw']
    pred_df['edge_away'] = pred_df['pred_away'] - pred_df['market_away']

    edge_cols = pred_df[['edge_home','edge_draw','edge_away']].copy().fillna(-999)
    pred_df['best_outcome'] = edge_cols.idxmax(axis=1).str.replace('edge_','')
    pred_df['best_edge']    = edge_cols.max(axis=1)

    strat = pred_df[
        pred_df['League'].isin(strategy_leagues) &
        pred_df['best_outcome'].isin(['home','draw']) &
        (pred_df['best_edge'] >= edge_thresh) &
        pred_df['market_home'].notna()
    ].copy()

    def calc_profit(row):
        if row['best_outcome'] == 'home':
            win  = row['target'] == 'H'
            odds = 1 / row['market_home']
        else:
            win  = row['target'] == 'D'
            odds = 1 / row['market_draw']
        return (odds - 1) if win else -1.0

    strat['profit'] = strat.apply(calc_profit, axis=1)
    roi = strat['profit'].sum() / len(strat) if len(strat) > 0 else 0

    season_roi = strat.groupby('Season')['profit'].mean()
    sharpe = season_roi.mean() / season_roi.std() if len(season_roi) > 1 and season_roi.std() > 0 else 0

    valid = pred_df.dropna(subset=['pred_home','pred_draw','pred_away'])
    brier = (
        brier_score_loss((valid['target']=='H').astype(int), valid['pred_home']) +
        brier_score_loss((valid['target']=='D').astype(int), valid['pred_draw']) +
        brier_score_loss((valid['target']=='A').astype(int), valid['pred_away'])
    ) / 3

    return {
        'brier': brier,
        'roi': roi,
        'sharpe': sharpe,
        'n_bets': len(strat),
        'positive_seasons': int((season_roi > 0).sum()),
        'total_seasons': len(season_roi),
    }


def get_feature_importance(clf, feature_cols):
    imp = clf.feature_importances_
    return pd.Series(imp, index=feature_cols).sort_values(ascending=False)


def main():
    print('Loading features...')
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df['Season'] >= '2006/2007'].copy()
    print(f'  {len(df):,} matches, {df["Season"].nunique()} seasons')

    params = load_params()
    strategy_leagues = ['P1', 'G1', 'E1']
    edge_thresh = 0.18

    report_lines = []

    for label, use_pinnacle in [('WITH Pinnacle', True), ('WITHOUT Pinnacle', False)]:
        print(f'\n=== {label} ===')

        if use_pinnacle:
            feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        else:
            feat_cols = [c for c in FEATURE_COLS if c in df.columns and c not in PINNACLE_COLS]

        print(f'  Features: {len(feat_cols)} ({", ".join(PINNACLE_COLS) if use_pinnacle else "no Pinnacle cols"})')

        pred_df, clf, used_cols = walk_forward_roi(df, feat_cols, params, strategy_leagues, edge_thresh)
        metrics = compute_roi(pred_df, strategy_leagues, edge_thresh)
        importance = get_feature_importance(clf, used_cols)

        tag = 'with_pinnacle' if use_pinnacle else 'without_pinnacle'
        importance.to_csv(OUT_DIR / f'feature_importance_{tag}.csv', header=['importance'])

        print(f'  Brier:    {metrics["brier"]:.4f}')
        print(f'  ROI:      {metrics["roi"]:.1%}')
        print(f'  Sharpe:   {metrics["sharpe"]:.3f}')
        print(f'  Bets:     {metrics["n_bets"]}')
        print(f'  Pos seas: {metrics["positive_seasons"]}/{metrics["total_seasons"]}')
        print(f'\n  Top-15 features:')
        print(importance.head(15).to_string())

        report_lines.append(f'\n{"="*60}')
        report_lines.append(f'{label}')
        report_lines.append(f'{"="*60}')
        report_lines.append(f'Features used: {len(used_cols)}')
        report_lines.append(f'Brier Score:   {metrics["brier"]:.4f}')
        report_lines.append(f'ROI (P1+G1+E1, home+draw, edge>=0.18): {metrics["roi"]:.1%}')
        report_lines.append(f'Sharpe:        {metrics["sharpe"]:.3f}')
        report_lines.append(f'Bets:          {metrics["n_bets"]}')
        report_lines.append(f'Positive seasons: {metrics["positive_seasons"]}/{metrics["total_seasons"]}')
        report_lines.append(f'\nTop-20 feature importance:')
        report_lines.append(importance.head(20).to_string())

    report_path = OUT_DIR / 'ablation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'\nReport saved to {report_path}')


if __name__ == '__main__':
    main()
