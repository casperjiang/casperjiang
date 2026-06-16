"""
decision_engine.py — Five-Dimension Trading Decision Engine for Taiwan Futures
Integrates all five engines into a single unified signal generator.

Five Dimensions:
  Dim 1 - Wave Position      (WaveEngine)
  Dim 2 - Fibonacci Price    (FibPriceEngine)
  Dim 3 - Fibonacci Time     (FibTimeEngine)
  Dim 4 - Gann Swing         (SwingEngine)
  Dim 5 - Market Regime      (RegimeEngine)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd

from swing_engine    import SwingEngine, SwingResult
from fib_price_engine import FibPriceEngine
from fib_time_engine  import FibTimeEngine
from wave_engine      import WaveEngine, WavePattern
from regime_engine    import RegimeEngine, RegimeState, Regime


class Signal(Enum):
    STRONG_LONG  = "strong_long"
    LONG         = "long"
    NEUTRAL      = "neutral"
    SHORT        = "short"
    STRONG_SHORT = "strong_short"


@dataclass
class TradingSignal:
    signal:          Signal
    score:           int            # -100 to +100
    confidence:      float          # 0.0 to 1.0
    entry_price:     Optional[float] = None
    stop_loss:       Optional[float] = None
    targets:         list[float] = field(default_factory=list)
    regime:          Optional[Regime] = None
    wave_notes:      list[str] = field(default_factory=list)
    fib_levels:      dict = field(default_factory=dict)
    time_windows:    list = field(default_factory=list)
    swing_direction: int = 0        # +1 or -1
    dim_scores:      dict = field(default_factory=dict)
    notes:           list[str] = field(default_factory=list)


class DecisionEngine:
    """
    Five-dimension decision engine for Taiwan Futures (TXFPM1).

    Scoring logic:
    - All 5 dimensions agree (+/-): ±100 → STRONG signal
    - 4/5 agree:                    ±80  → normal signal
    - 3/5 agree:                    ±60  → weak signal
    - ≤2/5:                         NEUTRAL / WAIT

    Thresholds:
    ≥ 70 → STRONG_LONG
    ≥ 40 → LONG
    ≤-40 → SHORT
    ≤-70 → STRONG_SHORT
    else  → NEUTRAL
    """

    def __init__(
        self,
        swing_days: int   = 2,
        peak_order: int   = 5,
        fib_tol:    float = 0.08,
        atr_sl_mult: float = 1.5,
    ):
        self.swing_eng  = SwingEngine(swing_days=swing_days, inside_down=True)
        self.fib_eng    = FibPriceEngine()
        self.time_eng   = FibTimeEngine()
        self.wave_eng   = WaveEngine(peak_order=peak_order, fib_tol=fib_tol)
        self.regime_eng = RegimeEngine()
        self.atr_sl_mult = atr_sl_mult

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        ohlc_daily:  pd.DataFrame,
        ohlc_60min:  Optional[pd.DataFrame] = None,
        ohlc_15min:  Optional[pd.DataFrame] = None,
    ) -> TradingSignal:
        """
        Run all five engines and synthesize a trading signal.
        """
        notes: list[str] = []
        dim_scores: dict[str, int] = {}

        # ── Dimension 4: Gann Swing ──────────────────────────────────
        swing_result = self.swing_eng.calculate_swings(ohlc_daily)
        pivots_df    = self.swing_eng.pivots_to_dataframe(swing_result.pivots)
        swing_dir    = self._swing_direction(swing_result)
        dim_scores["swing"] = swing_dir * 20
        notes.append(f"Swing: {'↑' if swing_dir>0 else '↓' if swing_dir<0 else '→'}")

        # ── Dimension 5: Regime ──────────────────────────────────────
        tf_states = self.regime_eng.classify_multi_tf(
            ohlc_daily, ohlc_60min, ohlc_15min
        )
        consensus   = self.regime_eng.multi_tf_consensus(tf_states)
        regime_dir  = consensus["direction"]
        regime_conf = consensus["avg_confluence"]
        regime_score = int(regime_dir * min(20, regime_conf / 5))
        dim_scores["regime"] = regime_score
        notes.append(f"Regime: {consensus['label']}")

        # ── Dimension 1: Wave Position ───────────────────────────────
        wave_score, wave_notes, best_wave = self._analyze_waves(
            ohlc_daily, pivots_df
        )
        dim_scores["wave"] = wave_score
        notes.extend(wave_notes)

        # ── Dimension 2: Fibonacci Price ─────────────────────────────
        fib_score, fib_levels = self._analyze_fib_price(
            ohlc_daily, swing_result
        )
        dim_scores["fib_price"] = fib_score
        notes.append(f"Fib Price confluence score: {fib_score}")

        # ── Dimension 3: Fibonacci Time ──────────────────────────────
        time_score, time_windows = self._analyze_fib_time(
            ohlc_daily, swing_result
        )
        dim_scores["fib_time"] = time_score
        notes.append(f"Fib Time zones near current bar: {time_score}")

        # ── Synthesis ────────────────────────────────────────────────
        total = sum(dim_scores.values())
        signal = self._score_to_signal(total)
        confidence = min(1.0, abs(total) / 100)

        # Entry / SL / TP
        last_close = _last_close(ohlc_daily)
        atr = _last_atr(ohlc_daily)
        entry, sl, targets = self._calc_entry_sl_tp(
            last_close, atr, signal, fib_levels
        )

        daily_regime = tf_states.get("daily")

        return TradingSignal(
            signal=signal,
            score=total,
            confidence=round(confidence, 2),
            entry_price=entry,
            stop_loss=sl,
            targets=targets,
            regime=daily_regime.regime if daily_regime else None,
            wave_notes=wave_notes,
            fib_levels=fib_levels,
            time_windows=time_windows,
            swing_direction=swing_dir,
            dim_scores=dim_scores,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Dimension sub-analyzers
    # ------------------------------------------------------------------

    def _swing_direction(self, result: SwingResult) -> int:
        pivots = result.pivots
        if len(pivots) < 2:
            return 0
        last = pivots[-1]
        from swing_engine import SwingDir
        if last.direction == SwingDir.UP:
            return 1
        elif last.direction == SwingDir.DOWN:
            return -1
        return 0

    def _analyze_waves(
        self, ohlc: pd.DataFrame, pivots_df: pd.DataFrame
    ) -> tuple[int, list[str], Optional[WavePattern]]:
        notes = []
        # Use wave engine's own pivot detection (produces pivot_type column)
        wave_pivots = self.wave_eng.detect_pivots(ohlc)
        if len(wave_pivots) < 6:
            return 0, ["Insufficient pivots for wave analysis"], None

        patterns = self.wave_eng.find_impulse_waves(wave_pivots, direction="up")
        if not patterns:
            patterns = self.wave_eng.find_impulse_waves(wave_pivots, direction="down")
            direction_mult = -1
        else:
            direction_mult = 1

        if not patterns:
            notes.append("No valid impulse wave found")
            return 0, notes, None

        best = patterns[0]
        wave_quality = min(20, best.score.total // 20)
        score = direction_mult * wave_quality
        notes.append(f"Best wave score: {best.score.total}, notes: {'; '.join(best.score.notes)}")
        return score, notes, best

    def _analyze_fib_price(
        self, ohlc: pd.DataFrame, swing_result: SwingResult
    ) -> tuple[int, dict]:
        pivots = swing_result.pivots
        if len(pivots) < 2:
            return 0, {}

        df = _norm_df(ohlc)
        last_close = df["Close"].iloc[-1]

        # Find most recent significant high and low
        from swing_engine import SwingDir
        highs = [p for p in pivots if p.direction == SwingDir.UP]
        lows  = [p for p in pivots if p.direction == SwingDir.DOWN]
        if not highs or not lows:
            return 0, {}

        recent_high = highs[-1].price
        recent_low  = lows[-1].price

        linear_levels = self.fib_eng.calc_retracement(recent_high, recent_low)
        semilog_levels = self.fib_eng.calc_semilog_retracement(recent_high, recent_low)

        # How close is current price to a Fibonacci level?
        all_prices = [lv.price for lv in linear_levels + semilog_levels]
        min_dist = min(abs(last_close - p) / last_close for p in all_prices)
        proximity_score = max(0, 20 - int(min_dist * 1000))

        fib_dict = {
            "linear":  {lv.label: round(lv.price, 0) for lv in linear_levels},
            "semilog": {lv.label: round(lv.price, 0) for lv in semilog_levels},
            "recent_high": recent_high,
            "recent_low":  recent_low,
            "last_close":  last_close,
        }
        return proximity_score, fib_dict

    def _analyze_fib_time(
        self, ohlc: pd.DataFrame, swing_result: SwingResult
    ) -> tuple[int, list]:
        pivots = swing_result.pivots
        if len(pivots) < 2:
            return 0, []

        dates = None
        df = _norm_df(ohlc)
        if hasattr(df.index, 'year'):
            dates = df.index

        last_pivot = pivots[-1]
        zones = self.time_eng.fibonacci_time_zones(
            last_pivot.index, dates=dates, max_zones=8
        )

        total_bars = len(df)
        near_zones = [z for z in zones if abs(z.bar_index - total_bars) <= 3]
        score = min(20, len(near_zones) * 10)
        return score, [{"fib": z.fib_number, "bar": z.bar_index,
                        "date": str(z.date_approx)[:10] if z.date_approx else None}
                       for z in near_zones]

    # ------------------------------------------------------------------
    # Entry / SL / TP calculation
    # ------------------------------------------------------------------

    def _calc_entry_sl_tp(
        self, last_close: float, atr: float,
        signal: Signal, fib_levels: dict
    ) -> tuple[Optional[float], Optional[float], list[float]]:
        if signal == Signal.NEUTRAL:
            return None, None, []

        is_long = signal in (Signal.LONG, Signal.STRONG_LONG)
        entry   = round(last_close, 0)
        sl      = round(entry - atr * self.atr_sl_mult, 0) if is_long \
                  else round(entry + atr * self.atr_sl_mult, 0)

        targets = []
        ext_levels = self.fib_eng.calc_extension(
            entry - atr * 5, entry, entry - atr * 2
        ) if is_long else self.fib_eng.calc_extension(
            entry + atr * 5, entry, entry + atr * 2
        )
        for lv in ext_levels[:3]:
            targets.append(round(lv.price, 0))

        return entry, sl, targets

    @staticmethod
    def _score_to_signal(score: int) -> Signal:
        if score >= 70:   return Signal.STRONG_LONG
        if score >= 40:   return Signal.LONG
        if score <= -70:  return Signal.STRONG_SHORT
        if score <= -40:  return Signal.SHORT
        return Signal.NEUTRAL


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open","開盤"):    rename[col] = "Open"
        elif lc in ("high","最高"):  rename[col] = "High"
        elif lc in ("low","最低"):   rename[col] = "Low"
        elif lc in ("close","收盤"): rename[col] = "Close"
    return df.rename(columns=rename)


def _last_close(ohlc: pd.DataFrame) -> float:
    df = _norm_df(ohlc)
    return float(df["Close"].iloc[-1])


def _last_atr(ohlc: pd.DataFrame, period: int = 14) -> float:
    df = _norm_df(ohlc)
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])
