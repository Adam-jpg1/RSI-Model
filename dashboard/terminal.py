import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    raise ImportError("Run: pip install rich")

from config import CONFIG
from data.universe import get_model_age_hours, get_trained_tickers
from ml.optimizer import load_best_params


class Dashboard:
    """
    Rich terminal dashboard for the autonomous trader.
    Call render_once() for a one-shot print, or run_live() for a live-updating view.
    """

    def __init__(
        self,
        client=None,
        scanner_fn: Optional[Callable] = None,
        scan_interval_seconds: int = 300,
    ):
        self.console = Console()
        self.client = client
        self.scanner_fn = scanner_fn
        self.scan_interval = scan_interval_seconds
        self._opportunities: List[Dict] = []
        self._last_scan: float = 0.0
        self._trade_log: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def render_once(self):
        acct, positions = self._fetch_live_state()
        self._maybe_refresh_scan()

        self.console.clear()
        self.console.rule(f"[bold cyan]RSI-Model Autonomous Trader[/]  {datetime.now():%H:%M:%S}")
        self.console.print(self._account_panel(acct))
        self.console.print(self._positions_panel(positions))
        self.console.print(self._opportunities_panel(self._opportunities))
        self.console.print(self._models_panel())

    def run_live(self, refresh_seconds: int = 30):
        """Continuously refresh the dashboard until Ctrl+C."""
        self.console.print("[bold green]Dashboard running — press Ctrl+C to stop[/]")
        try:
            while True:
                self.render_once()
                time.sleep(refresh_seconds)
        except KeyboardInterrupt:
            self.console.print("\n[dim]Dashboard stopped.[/]")

    def log_trade(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._trade_log.append(f"[{ts}] {msg}")
        if len(self._trade_log) > 50:
            self._trade_log = self._trade_log[-50:]

    # ------------------------------------------------------------------ #
    # Panels                                                              #
    # ------------------------------------------------------------------ #

    def _account_panel(self, acct: Dict) -> Panel:
        pv = acct.get("portfolio_value", 0)
        cash = acct.get("cash", 0)
        bp = acct.get("buying_power", 0)
        t = Text()
        t.append(f"Portfolio  ${pv:>12,.2f}", style="bold green")
        t.append(f"    Cash  ${cash:>12,.2f}")
        t.append(f"    Buying Power  ${bp:>12,.2f}")
        t.append(f"    Paper={CONFIG.bot.paper_trading}", style="dim")
        return Panel(t, title="[cyan]Account[/]", border_style="cyan")

    def _positions_panel(self, positions: Dict) -> Panel:
        tbl = Table(show_header=True, header_style="bold magenta", box=None)
        tbl.add_column("Ticker", width=8)
        tbl.add_column("Side", width=6)
        tbl.add_column("Qty", justify="right", width=10)
        tbl.add_column("Value", justify="right", width=14)
        tbl.add_column("PnL %", justify="right", width=10)
        tbl.add_column("Entry", justify="right", width=10)

        for ticker, pos in sorted(positions.items()):
            qty = pos["qty"]
            side = "LONG" if qty >= 0 else "SHORT"
            pnl = pos["unrealized_pnl_pct"]
            clr = "green" if pnl >= 0 else "red"
            tbl.add_row(
                ticker,
                f"[{'green' if side == 'LONG' else 'red'}]{side}[/]",
                f"{abs(qty):.2f}",
                f"${abs(pos['market_value']):,.2f}",
                f"[{clr}]{pnl:+.2f}%[/]",
                f"${pos['avg_entry']:.2f}",
            )

        title = f"Positions  {len(positions)}/{CONFIG.bot.max_positions}"
        return Panel(tbl, title=f"[magenta]{title}[/]", border_style="magenta")

    def _opportunities_panel(self, opps: List[Dict]) -> Panel:
        tbl = Table(show_header=True, header_style="bold yellow", box=None)
        tbl.add_column("Ticker", width=8)
        tbl.add_column("Signal", width=7)
        tbl.add_column("Conf", justify="right", width=7)
        tbl.add_column("ADX", justify="right", width=7)
        tbl.add_column("RSI", justify="right", width=7)
        tbl.add_column("Trend", justify="right", width=7)
        tbl.add_column("Price", justify="right", width=10)

        for opp in opps[:15]:
            sig = opp["signal"]
            sig_clr = "green" if sig == "UP" else ("red" if sig == "DOWN" else "dim")
            conf = opp["confidence"]
            conf_clr = "green" if conf >= 0.60 else ("yellow" if conf >= 0.52 else "dim")
            adx = opp["adx"]
            adx_clr = "green" if adx >= 25 else ("yellow" if adx >= 15 else "dim")
            d_trend = opp.get("d_trend", 0.0)
            trend_clr = "green" if d_trend >= 1 else ("red" if d_trend <= -1 else "dim")
            tbl.add_row(
                opp["ticker"],
                f"[{sig_clr}]{sig}[/]",
                f"[{conf_clr}]{conf:.3f}[/]",
                f"[{adx_clr}]{adx:.1f}[/]",
                f"{opp['rsi']:.1f}",
                f"[{trend_clr}]{d_trend:+.0f}[/]",
                f"${opp['price']:.2f}",
            )

        age_s = ""
        if self._last_scan:
            age_s = f"  (scan {(time.time()-self._last_scan)/60:.0f}m ago)"
        return Panel(tbl, title=f"[yellow]Top Opportunities{age_s}[/]", border_style="yellow")

    def _models_panel(self) -> Panel:
        tickers = get_trained_tickers()
        tbl = Table(show_header=True, header_style="bold blue", box=None)
        tbl.add_column("Ticker", width=8)
        tbl.add_column("Age", justify="right", width=10)
        tbl.add_column("Status", width=8)
        tbl.add_column("ADX", justify="right", width=7)
        tbl.add_column("Opt.Score", justify="right", width=10)

        for ticker in tickers:
            age = get_model_age_hours(ticker)
            if age is None:
                continue
            if age < 48:
                status, clr = "FRESH", "green"
            elif age < 168:
                status, clr = "STALE", "yellow"
            else:
                status, clr = "OLD", "red"
            best = load_best_params(ticker)
            if best:
                adx_str = f"{best['adx_filter_min']:.0f}*"
                score_str = f"{best['best_score']:.4f}"
            else:
                adx_str = f"{CONFIG.bot.per_ticker_adx.get(ticker, CONFIG.bot.adx_filter_min):.0f}"
                score_str = "—"
            tbl.add_row(ticker, f"{age:.1f}h", f"[{clr}]{status}[/]", adx_str, score_str)

        return Panel(tbl, title=f"[blue]Models ({len(tickers)} trained)[/]  [dim]* = Optuna-tuned[/]",
                     border_style="blue")

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _fetch_live_state(self):
        if self.client:
            try:
                acct = self.client.get_account()
                positions = self.client.get_positions()
                return acct, positions
            except Exception:
                pass
        return {"portfolio_value": 0, "cash": 0, "buying_power": 0}, {}

    def _maybe_refresh_scan(self):
        if self.scanner_fn and time.time() - self._last_scan >= self.scan_interval:
            try:
                self._opportunities = self.scanner_fn()
                self._last_scan = time.time()
            except Exception:
                pass
