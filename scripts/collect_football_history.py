"""
Сбор исторических данных футбольных матчей с football-data.co.uk.

Скачивает CSV за 20+ лет для 5 лиг (EPL, La Liga, Bundesliga, Serie A, Ligue 1).
Включает результаты, коэффициенты букмекеров, статистику ударов.

Результат: data/raw/football/matches_{league}.parquet (5 файлов)
Время: ~5 минут (HTTP downloads)
"""

import sys

# Windows cp1251 fix
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('cp1251', 'cp866'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import logging
import io
from pathlib import Path

import httpx
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('data/raw/football/collect_history.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# === Конфигурация ===

BASE_URL = 'https://www.football-data.co.uk/mmz4281'

# Лиги: код -> название
LEAGUES = {
    'E0': 'EPL',
    'SP1': 'La_Liga',
    'D1': 'Bundesliga',
    'I1': 'Serie_A',
    'F1': 'Ligue_1',
    'E1': 'Championship',
    'N1': 'Eredivisie',
    'B1': 'Belgium_Pro',
    'P1': 'Primeira_Liga',
    'T1': 'Super_Lig',
}

# Сезоны: от 0001 (2000/01) до 2526 (2025/26)
# football-data.co.uk формат: 0001, 0102, 0203, ..., 2425, 2526
def generate_season_codes(start_year=2000, end_year=2025):
    """Генерирует коды сезонов для football-data.co.uk URL."""
    codes = []
    for y in range(start_year, end_year + 1):
        y1 = y % 100
        y2 = (y + 1) % 100
        code = f'{y1:02d}{y2:02d}'
        codes.append(code)
    return codes

SEASON_CODES = generate_season_codes(2000, 2025)

# Колонки, которые нас интересуют (не все есть в каждом сезоне)
CORE_COLUMNS = [
    'Date', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',     # Full Time: Home Goals, Away Goals, Result (H/D/A)
    'HTHG', 'HTAG', 'HTR',     # Half Time
]

ODDS_COLUMNS = [
    'B365H', 'B365D', 'B365A',  # Bet365
    'BWH', 'BWD', 'BWA',        # Betway
    'PSH', 'PSD', 'PSA',        # Pinnacle
    'MaxH', 'MaxD', 'MaxA',     # Market max
    'AvgH', 'AvgD', 'AvgA',     # Market average
    'BFH', 'BFD', 'BFA',        # Betfair opening odds
    'BFEH', 'BFED', 'BFEA',     # Betfair exchange closing price (prediction market proxy)
]

STATS_COLUMNS = [
    'HS', 'AS',                  # Shots
    'HST', 'AST',               # Shots on Target
    'HC', 'AC',                  # Corners
    'HF', 'AF',                  # Fouls
    'HY', 'AY', 'HR', 'AR',    # Yellow/Red cards
]

OUT_DIR = Path('data/raw/football')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Основная логика ===

def parse_date(date_str):
    """Парсит дату из разных форматов football-data.co.uk."""
    if pd.isna(date_str):
        return pd.NaT
    date_str = str(date_str).strip()
    for fmt in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d']:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except Exception:
        return pd.NaT


def download_season(league_code: str, season_code: str, client: httpx.Client) -> pd.DataFrame | None:
    """Скачивает CSV одного сезона одной лиги."""
    url = f'{BASE_URL}/{season_code}/{league_code}.csv'
    try:
        resp = client.get(url, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        # football-data.co.uk отдаёт CSV в разных кодировках
        content = resp.content
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = content.decode('latin-1', errors='replace')

        df = pd.read_csv(io.StringIO(text))

        # Убираем пустые строки (football-data.co.uk иногда добавляет мусор в конце)
        df = df.dropna(subset=['HomeTeam', 'AwayTeam'], how='any')

        if len(df) == 0:
            return None

        return df

    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            log.warning(f'  HTTP {e.response.status_code} for {url}')
        return None
    except Exception as e:
        log.warning(f'  Error downloading {url}: {e}')
        return None


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Выбирает нужные колонки, пропуская отсутствующие."""
    all_wanted = CORE_COLUMNS + ODDS_COLUMNS + STATS_COLUMNS
    available = [c for c in all_wanted if c in df.columns]
    return df[available].copy()


def process_league(league_code: str, league_name: str, client: httpx.Client) -> pd.DataFrame:
    """Скачивает и объединяет все сезоны для одной лиги."""
    log.info(f'=== {league_name} ({league_code}) ===')

    all_seasons = []

    for season_code in SEASON_CODES:
        season_label = f'20{season_code[:2]}/20{season_code[2:]}'

        df = download_season(league_code, season_code, client)
        if df is None:
            continue

        df = select_columns(df)

        # Парсим даты
        df['Date'] = df['Date'].apply(parse_date)

        # Добавляем метаданные
        df['Season'] = season_label
        df['League'] = league_code

        # Числовые колонки
        for col in ['FTHG', 'FTAG', 'HTHG', 'HTAG']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in ODDS_COLUMNS + STATS_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        all_seasons.append(df)
        log.info(f'  {season_label}: {len(df)} matches')

    if not all_seasons:
        log.warning(f'  No data found for {league_name}!')
        return pd.DataFrame()

    combined = pd.concat(all_seasons, ignore_index=True)

    # Сортируем по дате
    combined = combined.sort_values('Date').reset_index(drop=True)

    # Убираем дубликаты (если перекачали)
    combined = combined.drop_duplicates(
        subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first'
    )

    log.info(f'  Total: {len(combined)} matches, '
             f'{combined["Date"].min():%Y-%m-%d} to {combined["Date"].max():%Y-%m-%d}')

    return combined


def compute_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет implied probabilities из трёх источников:
      - Pinnacle (PSH/D/A) — sharp bookmaker, основной бенчмарк
      - Betfair exchange closing (BFEH/D/A) — prediction market proxy (ближайший аналог Polymarket)
      - B365 — fallback если нет Pinnacle
    """
    def _implied(h_col, d_col, a_col, prefix, df):
        if not all(c in df.columns for c in [h_col, d_col, a_col]):
            return df
        raw_h = 1.0 / df[h_col].replace(0, np.nan)
        raw_d = 1.0 / df[d_col].replace(0, np.nan)
        raw_a = 1.0 / df[a_col].replace(0, np.nan)
        total = raw_h + raw_d + raw_a
        df[f'{prefix}_home'] = raw_h / total
        df[f'{prefix}_draw'] = raw_d / total
        df[f'{prefix}_away'] = raw_a / total
        df[f'{prefix}_overround'] = total
        return df

    # Pinnacle — sharp bookie (нет margin bias)
    df = _implied('PSH', 'PSD', 'PSA', 'pinnacle', df)

    # Betfair exchange closing — prediction market, аналог Polymarket
    df = _implied('BFEH', 'BFED', 'BFEA', 'betfair', df)

    # B365 — fallback
    df = _implied('B365H', 'B365D', 'B365A', 'b365', df)

    # implied_prob_* — основной источник для модели: Betfair если есть, иначе Pinnacle, иначе B365
    for outcome in ['home', 'draw', 'away']:
        df[f'implied_prob_{outcome}'] = (
            df.get(f'betfair_{outcome}',
            df.get(f'pinnacle_{outcome}',
            df.get(f'b365_{outcome}')))
        )
        # Удобно иметь оба варианта для сравнения
        if f'pinnacle_{outcome}' not in df.columns:
            df[f'pinnacle_{outcome}'] = np.nan
        if f'betfair_{outcome}' not in df.columns:
            df[f'betfair_{outcome}'] = np.nan

    # Overround из основного источника
    if 'betfair_overround' in df.columns:
        df['overround'] = df['betfair_overround']
    elif 'pinnacle_overround' in df.columns:
        df['overround'] = df['pinnacle_overround']
    elif 'b365_overround' in df.columns:
        df['overround'] = df['b365_overround']

    bf_cov = df['betfair_home'].notna().mean() * 100 if 'betfair_home' in df.columns else 0
    pn_cov = df['pinnacle_home'].notna().mean() * 100 if 'pinnacle_home' in df.columns else 0
    log.info(f'  Betfair coverage: {bf_cov:.0f}%  Pinnacle coverage: {pn_cov:.0f}%')
    return df


def main():
    log.info('=== Football History Collector ===')
    log.info(f'Leagues: {list(LEAGUES.values())}')
    log.info(f'Seasons: {SEASON_CODES[0]} to {SEASON_CODES[-1]}')

    client = httpx.Client(
        headers={'User-Agent': 'Mozilla/5.0 (research project)'},
        follow_redirects=True,
    )

    total_matches = 0

    try:
        for league_code, league_name in LEAGUES.items():
            df = process_league(league_code, league_name, client)
            if len(df) == 0:
                continue

            # Implied probabilities
            df = compute_implied_probs(df)

            # Сохраняем
            out_path = OUT_DIR / f'matches_{league_code}.parquet'
            df.to_parquet(out_path, index=False)
            log.info(f'  Saved: {out_path} ({len(df)} rows)')

            total_matches += len(df)

    finally:
        client.close()

    log.info(f'\n=== DONE: {total_matches} total matches across {len(LEAGUES)} leagues ===')

    # Быстрая проверка
    log.info('\n--- Quick sanity check ---')
    for league_code, league_name in LEAGUES.items():
        path = OUT_DIR / f'matches_{league_code}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            seasons = df['Season'].nunique()
            teams = df['HomeTeam'].nunique()
            odds_pct = df['B365H'].notna().mean() * 100 if 'B365H' in df.columns else 0
            log.info(f'  {league_name}: {len(df)} matches, {seasons} seasons, '
                     f'{teams} unique teams, {odds_pct:.0f}% with B365 odds')


if __name__ == '__main__':
    main()
