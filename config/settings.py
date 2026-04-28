"""
Параметры стратегий и конфигурация.
Меняй здесь — не хардкодь в notebooks.
"""

# === Пороги стратегий ===

# High-Probability Bond
HIGH_PROB_MIN = 0.92        # минимальная вероятность для входа
HIGH_PROB_MAX = 0.97        # выше этого — нет смысла (мало ROI)

# Cross-platform arb
ARB_MIN_SPREAD = 0.03       # минимальный спред для сигнала (3%)
ARB_MIN_VOLUME = 1000       # минимальный объём рынка в $ (чтобы не торговать мусор)

# Calibration signal
CALIB_MIN_EDGE = 0.05       # минимальный edge (model_prob - market_prob) для входа
CALIB_MIN_CONFIDENCE = 0.6  # минимальная уверенность модели

# === Размеры позиций ===
MAX_POSITION_USD = 100      # максимум на одну позицию (стартовый режим)
MAX_TOTAL_EXPOSURE = 500    # максимум всех открытых позиций

# === API настройки ===
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

REQUEST_TIMEOUT = 10        # секунд
REQUEST_RETRY = 3           # попыток при ошибке

# === Данные ===
DATA_RAW_DIR = "../data/raw"
DATA_PROCESSED_DIR = "../data/processed"
SIGNALS_DIR = "../data/signals"

# =============================================
# === Football Model Settings ===
# =============================================

# --- Модель ---
FOOTBALL_LEAGUES = ['E0', 'SP1', 'D1', 'I1', 'F1']   # Top-5 Европы
FOOTBALL_ROLLING_WINDOWS = [5, 10, 20]                  # окна rolling stats
FOOTBALL_ELO_K = 20                                      # скорость обновления ELO
FOOTBALL_ELO_HOME_ADV = 100                              # домашнее преимущество в ELO
FOOTBALL_DC_XI = 0.002                                   # time decay Dixon-Coles

# --- Бэктест ---
FOOTBALL_EDGE_THRESHOLD = 0.05      # минимальный edge для ставки (5%)
FOOTBALL_KELLY_FRACTION = 0.25      # Quarter Kelly
FOOTBALL_MIN_VOLUME_USD = 5_000     # минимальный объём рынка Polymarket для входа
FOOTBALL_MIN_BETS_SEASON = 10       # минимум ставок в тестовом сезоне для статистики

# --- Position sizing ---
FOOTBALL_MAX_POSITION_USD = 50      # максимальная ставка на матч (paper trading)
FOOTBALL_MAX_TOTAL_EXPOSURE = 200   # максимум всех открытых позиций (football)

# --- Бенчмарки (из научных работ) ---
FOOTBALL_BRIER_BENCHMARK = 0.210    # уровень Pinnacle
FOOTBALL_RPS_BENCHMARK = 0.202      # XGBoost + pi-ratings, Soccer Prediction Challenge 2017
FOOTBALL_ACCURACY_BENCHMARK = 0.535 # Pinnacle closing line accuracy

# --- Scanner ---
FOOTBALL_SCAN_INTERVAL_MIN = 30     # частота сканирования Polymarket (минут)

# --- Telegram NLP ---
TELEGRAM_MIN_IMPACT = 0.08          # минимальный estimated impact для алерта
TELEGRAM_MIN_CONFIDENCE = 0.70      # минимальная уверенность Claude для сигнала
TELEGRAM_LOOKBACK_HOURS = 24        # горизонт для backfill
