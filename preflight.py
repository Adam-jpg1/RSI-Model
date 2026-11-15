"""
Pre-flight check before paper trading.
Verifies: models loaded, data fetching, signal generation, Alpaca connection.

Usage: python preflight.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"


def check(label: str, fn):
    try:
        result = fn()
        status = PASS if result is not True else PASS
        note = result if isinstance(result, str) else ""
        print(f"  {PASS}  {label}{('  ' + note) if note else ''}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {label} — {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  RSI-MODEL PRE-FLIGHT CHECK")
    print("=" * 60)
    errors = 0
    warnings = 0

    # ---- 1. Config ----
    print("\n[1] Config")
    from config import CONFIG
    pos_pct = CONFIG.bot.max_position_pct
    max_pos = CONFIG.bot.max_positions
    adx_min = CONFIG.bot.adx_filter_min
    allow_short = CONFIG.bot.allow_short
    print(f"  {PASS}  Config loaded")
    print(f"        position_pct={pos_pct*100:.0f}%  max_positions={max_pos}  "
          f"adx_min={adx_min}  shorts={'on' if allow_short else 'off'}")

    # ---- 2. Models ----
    print("\n[2] Models")
    from data.universe import get_trained_tickers, get_model_age_hours
    trained = get_trained_tickers()
    if not trained:
        print(f"  {FAIL}  No trained models found in models/")
        errors += 1
    else:
        stale = []
        for t in trained:
            age = get_model_age_hours(t)
            if age and age > CONFIG.bot.model_staleness_hours:
                stale.append(f"{t}({age:.0f}h)")
        print(f"  {PASS}  {len(trained)} models found")
        if stale:
            print(f"  {WARN}  {len(stale)} stale (>{CONFIG.bot.model_staleness_hours}h): {', '.join(stale[:5])}")
            warnings += 1
        else:
            print(f"        All models fresh (within {CONFIG.bot.model_staleness_hours}h staleness limit)")

    # ---- 3. Model load test ----
    print("\n[3] Model load")
    from ml.model import RSIModel
    sample = trained[:3] if trained else []
    loaded = 0
    for t in sample:
        try:
            m = RSIModel()
            m.load(ticker=t)
            loaded += 1
        except Exception as e:
            print(f"  {FAIL}  {t} load failed: {e}")
            errors += 1
    if loaded == len(sample):
        print(f"  {PASS}  Loaded {loaded}/{len(sample)} sample models (AAPL/MSFT/NVDA)")

    # ---- 4. Data fetching ----
    print("\n[4] Live data fetch (AAPL, 5d/1h)")
    try:
        from data.fetcher import get_price_data
        df = get_price_data("AAPL", period="5d", interval="1h")
        from data.indicators import add_all_indicators
        df = add_all_indicators(df)
        print(f"  {PASS}  Fetched {len(df)} bars, {len(df.columns)} columns")
    except Exception as e:
        print(f"  {FAIL}  Data fetch failed: {e}")
        errors += 1

    # ---- 5. Signal generation ----
    print("\n[5] Signal generation")
    try:
        from data.fetcher import get_price_data, get_daily_data
        from data.indicators import add_all_indicators
        from bot.strategy import generate_signal
        df2 = get_price_data("AAPL", period="30d", interval="1h")
        df2 = add_all_indicators(df2)
        try:
            daily = get_daily_data("AAPL", period="1y")
        except Exception:
            daily = None
        m2 = RSIModel()
        m2.load(ticker="AAPL")
        sig = generate_signal(df2, m2, 0.0, daily_df=daily)
        if sig:
            print(f"  {PASS}  Signal: {sig['signal']}  conf={sig['confidence']:.3f}  "
                  f"adx={sig.get('adx', 0):.1f}  price=${sig['price']:.2f}")
        else:
            print(f"  {WARN}  No signal generated (FLAT/filtered — normal)")
            warnings += 1
    except Exception as e:
        print(f"  {FAIL}  Signal generation failed: {e}")
        errors += 1

    # ---- 6. Alpaca keys ----
    print("\n[6] Alpaca keys")
    api_key = CONFIG.bot.alpaca_api_key
    secret_key = CONFIG.bot.alpaca_secret_key
    if not api_key or not secret_key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            print(f"  {FAIL}  .env file found but keys are empty — fill in your Alpaca paper keys")
        else:
            print(f"  {FAIL}  No .env file. Create it:")
            print(f"        copy .env.example .env  (then fill in your Alpaca paper keys)")
            print(f"        Get keys at: alpaca.markets -> Paper Trading -> API Keys")
        errors += 1
    else:
        key_prefix = api_key[:6] + "..."
        print(f"  {PASS}  Keys loaded (api_key={key_prefix})")

        # Try connecting
        print("\n[7] Alpaca connection")
        try:
            from bot.alpaca_client import AlpacaClient
            client = AlpacaClient()
            acct = client.get_account()
            is_open = client.is_market_open()
            print(f"  {PASS}  Connected — equity=${acct['equity']:,.0f}  "
                  f"buying_power=${acct['buying_power']:,.0f}  "
                  f"market={'OPEN' if is_open else 'CLOSED'}")
        except Exception as e:
            print(f"  {FAIL}  Alpaca connection failed: {e}")
            errors += 1

    # ---- Summary ----
    print("\n" + "=" * 60)
    if errors == 0 and warnings == 0:
        print("  READY - no issues found.")
        print("  Start paper trading: python main.py run --dashboard")
    elif errors == 0:
        print(f"  READY with {warnings} warning(s) - review above.")
        print("  Start paper trading: python main.py run --dashboard")
    else:
        print(f"  NOT READY - {errors} error(s) must be fixed first.")
        if api_key:
            print("  After fixing: python preflight.py")
        else:
            print("  After adding Alpaca keys: python preflight.py")
    print("=" * 60 + "\n")

    return errors == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
