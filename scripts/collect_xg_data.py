"""
Сбор xG данных с Understat (бесплатно, без API ключа).

Understat покрывает: EPL, La Liga, Bundesliga, Serie A, Ligue 1
Данные доступны с сезона 2014/15.

Результат: data/raw/football/xg_{league}.parquet
Время: ~10-15 минут (медленный скрейпинг)
"""

import sys

# Windows cp1251 fix
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('cp1251', 'cp866'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('data/raw/football/collect_xg.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

OUT_DIR = Path('data/raw/football')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Understat league codes → наши коды
LEAGUES = {
    'EPL':          'E0',
    'La_liga':      'SP1',
    'Bundesliga':   'D1',
    'Serie_A':      'I1',
    'Ligue_1':      'F1',
}

SEASONS = list(range(2014, 2026))   # 2014 = сезон 2014/15

BASE_URL = 'https://understat.com'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}


def fetch_league_data_api(league: str, season: int, client: httpx.Client) -> list | None:
    """Получает данные матчей через AJAX API Understat."""
    url = f'{BASE_URL}/getLeagueData/{league}/{season}'
    try:
        resp = client.get(url, timeout=30, headers={
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': f'{BASE_URL}/league/{league}/{season}',
        })
        resp.raise_for_status()
        data = resp.json()
        return data.get('dates', [])
    except Exception as e:
        log.warning(f'  API error for {league}/{season}: {e}')
        return None


def parse_matches(data: list) -> list[dict]:
    """Парсит список матчей из Understat API JSON."""
    rows = []
    for m in data:
        try:
            # API возвращает isResult=true для сыгранных матчей
            if not m.get('isResult', False):
                continue
            goals_h = int(m.get('goals', {}).get('h', 0) or 0)
            goals_a = int(m.get('goals', {}).get('a', 0) or 0)
            if goals_h > goals_a:
                result = 'h'
            elif goals_a > goals_h:
                result = 'a'
            else:
                result = 'd'
            rows.append({
                'understat_id': m.get('id'),
                'Date': pd.to_datetime(m.get('datetime', '')[:10]),
                'HomeTeam_understat': m.get('h', {}).get('title', ''),
                'AwayTeam_understat': m.get('a', {}).get('title', ''),
                'xg_home': float(m.get('xG', {}).get('h', 0) or 0),
                'xg_away': float(m.get('xG', {}).get('a', 0) or 0),
                'goals_home': goals_h,
                'goals_away': goals_a,
                'result': result,
                'forecast_win': float(m.get('forecast', {}).get('w', 0) or 0),
                'forecast_draw': float(m.get('forecast', {}).get('d', 0) or 0),
                'forecast_loss': float(m.get('forecast', {}).get('l', 0) or 0),
            })
        except Exception as e:
            log.warning(f'  Error parsing match: {e}')
    return rows


def fetch_league_season(league: str, season: int, client: httpx.Client) -> list[dict]:
    """Скачивает и парсит один сезон одной лиги через API."""
    data = fetch_league_data_api(league, season, client)
    if data is None or len(data) == 0:
        return []
    return parse_matches(data)


# Маппинг названий команд Understat → football-data.co.uk
# (нужен для мержа датасетов)
TEAM_NAME_MAP = {
    # EPL
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Wolverhampton Wanderers': 'Wolves',
    'West Bromwich Albion': 'West Brom',
    'Queens Park Rangers': 'QPR',
    'Brighton & Hove Albion': 'Brighton',
    'Sheffield United': 'Sheffield United',
    'Nottingham Forest': "Nott'm Forest",
    'AFC Bournemouth': 'Bournemouth',
    'Leeds United': 'Leeds',
    'Swansea City': 'Swansea',
    'Stoke City': 'Stoke',
    'Huddersfield Town': 'Huddersfield',
    'Cardiff City': 'Cardiff',
    'Brentford': 'Brentford',
    # Bundesliga
    'Bayern Munich': 'Bayern Munich',
    'Borussia Dortmund': 'Dortmund',
    'Borussia M.Gladbach': "M'gladbach",
    'RB Leipzig': 'RB Leipzig',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'Bayer Leverkusen': 'Leverkusen',
    'TSG 1899 Hoffenheim': 'Hoffenheim',
    'Werder Bremen': 'Werder Bremen',
    'FC Augsburg': 'Augsburg',
    'VfB Stuttgart': 'Stuttgart',
    'SC Freiburg': 'Freiburg',
    'Hannover 96': 'Hannover',
    'FC Schalke 04': 'Schalke 04',
    'FC Cologne': 'FC Koln',
    'Hamburger SV': 'Hamburg',
    # Serie A
    'Internazionale': 'Inter',
    'AC Milan': 'Milan',
    'Hellas Verona': 'Verona',
    # La Liga
    'Atletico Madrid': 'Ath Madrid',
    'Athletic Club': 'Ath Bilbao',
    'Deportivo Alaves': 'Alaves',
    'Real Betis': 'Betis',
    'Espanyol': 'Espanol',
    # Ligue 1
    'Paris Saint-Germain': 'Paris SG',
    'Olympique Lyonnais': 'Lyon',
    'Olympique Marseille': 'Marseille',
    'AS Saint-Etienne': 'St Etienne',
    'Stade Rennais FC': 'Rennes',
    'RC Strasbourg Alsace': 'Strasbourg',
    'Girondins Bordeaux': 'Bordeaux',
    'FC Nantes': 'Nantes',
    'Stade Brestois 29': 'Brest',
    'Le Havre AC': 'Le Havre',
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def main():
    log.info('=== xG Data Collector (Understat) ===')

    client = httpx.Client(headers=HEADERS, follow_redirects=True)

    try:
        for understat_league, our_code in LEAGUES.items():
            log.info(f'\n=== {understat_league} ({our_code}) ===')
            all_matches = []

            for season in SEASONS:
                season_label = f'{season}/{season+1}'
                matches = fetch_league_season(understat_league, season, client)

                if not matches:
                    log.info(f'  {season_label}: no data')
                    continue

                for m in matches:
                    m['League'] = our_code
                    m['Season'] = season_label
                    m['HomeTeam'] = normalize_team(m['HomeTeam_understat'])
                    m['AwayTeam'] = normalize_team(m['AwayTeam_understat'])

                all_matches.extend(matches)
                log.info(f'  {season_label}: {len(matches)} matches')

                # Вежливая задержка
                time.sleep(1.0)

            if not all_matches:
                log.warning(f'  No data for {understat_league}')
                continue

            df = pd.DataFrame(all_matches)
            df = df.sort_values('Date').reset_index(drop=True)

            # Убираем дубликаты
            df = df.drop_duplicates(
                subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first'
            )

            # Добавляем разность xG
            df['xg_diff'] = df['xg_home'] - df['xg_away']

            out_path = OUT_DIR / f'xg_{our_code}.parquet'
            df.to_parquet(out_path, index=False)

            log.info(f'  Total: {len(df)} matches → {out_path}')
            log.info(f'  xG range: home {df["xg_home"].mean():.2f} avg, '
                     f'away {df["xg_away"].mean():.2f} avg')

    finally:
        client.close()

    log.info('\n=== DONE ===')

    # Sanity check
    log.info('\n--- Sanity check ---')
    for code in LEAGUES.values():
        path = OUT_DIR / f'xg_{code}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            seasons = df['Season'].nunique() if 'Season' in df.columns else '?'
            log.info(f'  {code}: {len(df)} matches, {seasons} seasons, '
                     f'avg xG home={df["xg_home"].mean():.2f}, away={df["xg_away"].mean():.2f}')


if __name__ == '__main__':
    main()
