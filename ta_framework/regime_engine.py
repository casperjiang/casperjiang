"""
regime_engine.py — Market Regime & State Engine
Multi-timeframe regime classification inspired by fibonacci_ml's 3-memory concept.
Integrates swing structure and Elliott Wave position.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


class Regime(Enum):
    TRENDING_UP       = "trending_up"
    TRENDING_DOWN     = "trending_down"
    CORRECTING_UP     = "correcting_up"
    CORRECTING_DOWN   = "correcting_down"
    RANGING           = "ranging"
    BREAKOUT_PENDING  = "breakout_pending"


class WavePosition(Enum):
    IMPULSE_W1 = "impulse_w1"
    IMPULSE_W2 = "impulse_w2"
    IMPULSE_W3 = "impulse_w3"
    IMPULSE_W4 = "impulse_w4"
    IMPULSE_W5 = "impulse_w5"
    CORRECTION_A = "correction_a"
    CORRECTION_B = "correction_b"
    CORRECTION_C = "correction_c"
    UNKNOWN      = "unknown"


@dataclass
class RegimeState:
    regime:            Regime
    wave_position:     WavePosition
    trend_direction:   int          # +1 up, -1 down, 0 neutral
    higher_highs:      bool
    higher_lows:       bool
    above_ma20:        bool
    above_ma60:        bool
    volatility_pct:    float        # 20-bar ATR / close in %
    confluence_score:  int          # 0-100
    notes:             list[str]


class RegimeEngine:
    """
    Market regime classifier for Taiwan Futures.

    Uses three timeframe inputs (inspired by fibonacci_ml 3-memory architecture):
    - Short-term  (15min / hourly)
    - Medium-term (daily)
    - Long-term   (weekly / monthly — from daily data downsampled)
    """

    def __init__(
        self,
        ma_short:  int = 20,
        ma_medium: int = 60,
        ma_long:   int = 120,
        atr_period: int = 14,
    ):
        self.ma_short  = ma_short
        self.ma_medium = ma_medium
        self.ma_long   = ma_long
        self.atr_period = atr_period

    # ------------------------------------------------------------------
    # Core regime detection
    # ------------------------------------------------------------------

    def classify(
        self,
        ohlc: pd.DataFrame,
        pivots_df: Optional[pd.DataFrame] = None,
    ) -> RegimeState:
        """
        Classify current market regime from OHLC data.
        pivots_df: optional swing pivot DataFrame for structure analysis.
        """
        df = _norm(ohlc)
        close = df["Close"]
        notes = []

        # Moving averages
        ma20  = close.rolling(self.ma_short).mean()
        ma60  = close.rolling(self.ma_medium).mean()
        ma120 = close.rolling(self.ma_long).mean()
        last  = close.iloc[-1]

        above_ma20 = last > ma20.iloc[-1]
        above_ma60 = last > ma60.iloc[-1]

        # ATR-based volatility
        atr = _calc_atr(df, self.atr_period)
        vol_pct = (atr.iloc[-1] / last * 100) if last > 0 else 0.0

        # Swing structure
        hh = hl = False
        if pivots_df is not None and len(pivots_df) >= 4:
            hh, hl = self._check_swing_structure(pivots_df)

        # MA alignment for trend direction
        ma_aligned_up   = above_ma20 and above_ma60
        ma_aligned_down = (not above_ma20) and (not above_ma60)

        if ma_aligned_up and hh and hl:
            regime = Regime.TRENDING_UP
            direction = 1
            notes.append("MA aligned up + HH + HL → Trending Up")
        elif ma_aligned_down and (not hh) and (not hl):
            regime = Regime.TRENDING_DOWN
            direction = -1
            notes.append("MA aligned down + LH + LL → Trending Down")
        elif ma_aligned_up and not hl:
            regime = Regime.CORRECTING_DOWN
            direction = 1
            notes.append("Uptrend structure but lower lows → Correcting")
        elif ma_aligned_down and not (not hh):
            regime = Regime.CORRECTING_UP
            direction = -1
            notes.append("Downtrend but higher highs → Correcting Up")
        elif vol_pct < 0.8:
            regime = Regime.RANGING
            direction = 0
            notes.append(f"Low ATR ({vol_pct:.2f}%) → Ranging")
        else:
            regime = Regime.BREAKOUT_PENDING
            direction = 0
            notes.append("Mixed signals → Breakout Pending")

        confluence = self._calc_confluence_score(
            above_ma20, above_ma60, hh, hl, vol_pct
        )

        return RegimeState(
            regime=regime,
            wave_position=WavePosition.UNKNOWN,
            trend_direction=direction,
            higher_highs=hh,
            higher_lows=hl,
            above_ma20=above_ma20,
            above_ma60=above_ma60,
            volatility_pct=round(vol_pct, 2),
            confluence_score=confluence,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Multi-timeframe regime (fibonacci_ml 3-memory concept)
    # ------------------------------------------------------------------

    def classify_multi_tf(
        self,
        ohlc_daily:   pd.DataFrame,
        ohlc_60min:   Optional[pd.DataFrame] = None,
        ohlc_15min:   Optional[pd.DataFrame] = None,
    ) -> dict[str, RegimeState]:
        """
        Run regime classification across three timeframes.
        Returns dict with keys: 'daily', '60min', '15min'.
        """
        results: dict[str, RegimeState] = {}
        results["daily"] = self.classify(ohlc_daily)

        if ohlc_60min is not None and len(ohlc_60min) > self.ma_short:
            results["60min"] = self.classify(ohlc_60min)

        if ohlc_15min is not None and len(ohlc_15min) > self.ma_short:
            results["15min"] = self.classify(ohlc_15min)

        return results

    def multi_tf_consensus(self, states: dict[str, RegimeState]) -> dict:
        """
        Determine consensus direction across timeframes.
        Returns direction (+1/-1/0) and confidence score.
        """
        directions = [s.trend_direction for s in states.values()]
        scores     = [s.confluence_score for s in states.values()]

        up_votes   = sum(1 for d in directions if d > 0)
        down_votes = sum(1 for d in directions if d < 0)
        total_tf   = len(directions)

        if up_votes == total_tf:
            consensus = 1; label = "Strong Uptrend (all TF aligned)"
        elif down_votes == total_tf:
            consensus = -1; label = "Strong Downtrend (all TF aligned)"
        elif up_votes > down_votes:
            consensus = 1; label = f"Uptrend ({up_votes}/{total_tf} TF)"
        elif down_votes > up_votes:
            consensus = -1; label = f"Downtrend ({down_votes}/{total_tf} TF)"
        else:
            consensus = 0; label = "Mixed (no consensus)"

        return {
            "direction": consensus,
            "label": label,
            "avg_confluence": round(sum(scores) / len(scores), 0) if scores else 0,
            "by_tf": {k: v.regime.value for k, v in states.items()},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_swing_structure(self, pivots_df: pd.DataFrame) -> tuple[bool, bool]:
        highs = pivots_df[pivots_df["direction"] == 1]["price"].tolist()[-4:]
        lows  = pivots_df[pivots_df["direction"] == -1]["price"].tolist()[-4:]
        hh = len(highs) >= 2 and highs[-1] > highs[-2]
        hl = len(lows)  >= 2 and lows[-1]  > lows[-2]
        return hh, hl

    def _calc_confluence_score(
        self, above_ma20, above_ma60, hh, hl, vol_pct
    ) -> int:
        score = 0
        if above_ma20: score += 20
        if above_ma60: score += 20
        if hh:         score += 20
        if hl:         score += 20
        if vol_pct > 0.5: score += 20
        return score


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open","開盤"):   rename[col] = "Open"
        elif lc in ("high","最高"): rename[col] = "High"
        elif lc in ("low","最低"):  rename[col] = "Low"
        elif lc in ("close","收盤"):rename[col] = "Close"
    return df.rename(columns=rename)


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
