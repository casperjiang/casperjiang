"""
swing_engine.py — Gann Swing + Multi-level Pivot Engine
Core logic adapted from: github.com/monch1962/gann-swing
Multi-level pivot concept from: github.com/rluu/pricechartingtool
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


class BarType(Enum):
    UP_DAY      = "up"
    DOWN_DAY    = "down"
    INSIDE_DAY  = "inside"
    OUTSIDE_DAY = "outside"


class SwingDir(Enum):
    UP   = 1
    DOWN = -1
    NONE = 0


class PivotStrength(Enum):
    """H/HH/HHH pivot classification"""
    H   = 1   # minor pivot
    HH  = 2   # intermediate pivot
    HHH = 3   # major pivot
    L   = 1
    LL  = 2
    LLL = 3


@dataclass
class SwingPoint:
    index: int
    date: pd.Timestamp
    price: float
    direction: SwingDir
    strength: int = 1          # 1=minor, 2=intermediate, 3=major
    bar_type: Optional[BarType] = None


@dataclass
class SwingResult:
    bars: pd.DataFrame          # original OHLC + bar_type + swing_dir columns
    pivots: list[SwingPoint]    # detected swing points
    swing_days: int



class SwingEngine:
    """
    Gann Swing detection engine for Taiwan Futures (TXFPM1).

    Parameters
    ----------
    swing_days : int
        Number of consecutive reversal bars required to confirm a swing.
        1 = most sensitive (1-bar reversal)
        2 = standard Gann swing (recommended for TXFM daily)
        3 = conservative
    inside_down : bool
        Treat inside days as bearish (tightens continuation logic).
    ignore_threshold : float
        Minimum price move (in points) to register a swing. 0 = all swings.
    """

    FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]

    def __init__(
        self,
        swing_days: int = 2,
        inside_down: bool = True,
        ignore_threshold: float = 0.0,
    ):
        self.swing_days = swing_days
        self.inside_down = inside_down
        self.ignore_threshold = ignore_threshold

    # ------------------------------------------------------------------
    # Bar classification (from gann-swing)
    # ------------------------------------------------------------------

    def classify_bar(self, bar: pd.Series, prev: pd.Series) -> BarType:
        higher_high = bar["High"] > prev["High"]
        lower_low   = bar["Low"]  < prev["Low"]
        if higher_high and not lower_low:
            return BarType.UP_DAY
        if lower_low and not higher_high:
            return BarType.DOWN_DAY
        if not higher_high and not lower_low:
            return BarType.INSIDE_DAY
        return BarType.OUTSIDE_DAY

    def annotate_bar_types(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        df = ohlc.copy()
        types = [None]
        for i in range(1, len(df)):
            types.append(self.classify_bar(df.iloc[i], df.iloc[i - 1]))
        df["bar_type"] = types
        return df

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    def calculate_swings(self, ohlc: pd.DataFrame) -> SwingResult:
        """
        Detect Gann swings from OHLC DataFrame.
        Expects columns: Open, High, Low, Close (case-insensitive).
        Returns SwingResult with annotated bars and pivot list.
        """
        df = _normalize_columns(ohlc)
        df = self.annotate_bar_types(df)
        df["swing_dir"] = SwingDir.NONE.value

        direction = SwingDir.UP
        reversal_count = 0
        last_swing_high_idx = 0
        last_swing_low_idx  = 0
        pivots: list[SwingPoint] = []

        for i in range(1, len(df)):
            bt = df.iloc[i]["bar_type"]
            if bt is None:
                continue

            if direction == SwingDir.UP:
                if bt == BarType.DOWN_DAY or (self.inside_down and bt == BarType.INSIDE_DAY):
                    reversal_count += 1
                else:
                    reversal_count = 0
                if reversal_count >= self.swing_days:
                    # confirm swing high
                    high_idx = df.iloc[last_swing_low_idx:i]["High"].idxmax()
                    hi = df.loc[high_idx]
                    move = hi["High"] - df.iloc[last_swing_low_idx]["Low"]
                    if move >= self.ignore_threshold:
                        sp = SwingPoint(
                            index=int(df.index.get_loc(high_idx)) if not isinstance(df.index, pd.RangeIndex) else int(high_idx),
                            date=hi.name if hasattr(hi, "name") else pd.Timestamp("now"),
                            price=hi["High"],
                            direction=SwingDir.UP,
                        )
                        pivots.append(sp)
                    last_swing_high_idx = i
                    direction = SwingDir.DOWN
                    reversal_count = 0
            else:  # DOWN
                if bt == BarType.UP_DAY or (not self.inside_down and bt == BarType.OUTSIDE_DAY):
                    reversal_count += 1
                elif bt == BarType.DOWN_DAY:
                    reversal_count = 0
                if reversal_count >= self.swing_days:
                    # confirm swing low
                    low_idx = df.iloc[last_swing_high_idx:i]["Low"].idxmin()
                    lo = df.loc[low_idx]
                    move = df.iloc[last_swing_high_idx]["High"] - lo["Low"]
                    if move >= self.ignore_threshold:
                        sp = SwingPoint(
                            index=int(df.index.get_loc(low_idx)) if not isinstance(df.index, pd.RangeIndex) else int(low_idx),
                            date=lo.name if hasattr(lo, "name") else pd.Timestamp("now"),
                            price=lo["Low"],
                            direction=SwingDir.DOWN,
                        )
                        pivots.append(sp)
                    last_swing_low_idx = i
                    direction = SwingDir.UP
                    reversal_count = 0

        # classify pivot strength
        pivots = _classify_pivot_strength(pivots)
        return SwingResult(bars=df, pivots=pivots, swing_days=self.swing_days)

    # ------------------------------------------------------------------
    # Structure analysis
    # ------------------------------------------------------------------

    def is_higher_high(self, pivots: list[SwingPoint]) -> bool:
        highs = [p for p in pivots if p.direction == SwingDir.UP][-4:]
        if len(highs) < 2:
            return False
        return highs[-1].price > highs[-2].price

    def is_lower_low(self, pivots: list[SwingPoint]) -> bool:
        lows = [p for p in pivots if p.direction == SwingDir.DOWN][-4:]
        if len(lows) < 2:
            return False
        return lows[-1].price < lows[-2].price

    def detect_structure_break(self, pivots: list[SwingPoint]) -> list[dict]:
        breaks = []
        highs = [p for p in pivots if p.direction == SwingDir.UP]
        lows  = [p for p in pivots if p.direction == SwingDir.DOWN]
        for i in range(1, len(highs)):
            if highs[i].price > highs[i-1].price:
                breaks.append({"type": "BOS_UP", "date": highs[i].date, "price": highs[i].price})
        for i in range(1, len(lows)):
            if lows[i].price < lows[i-1].price:
                breaks.append({"type": "BOS_DOWN", "date": lows[i].date, "price": lows[i].price})
        return sorted(breaks, key=lambda x: x["date"])

    def pivots_to_dataframe(self, pivots: list[SwingPoint]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "date": p.date,
                "price": p.price,
                "direction": p.direction.value,
                "strength": p.strength,
            }
            for p in pivots
        ])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "開盤"):
            rename[col] = "Open"
        elif lc in ("high", "最高"):
            rename[col] = "High"
        elif lc in ("low", "最低"):
            rename[col] = "Low"
        elif lc in ("close", "收盤"):
            rename[col] = "Close"
        elif lc in ("volume", "成交量"):
            rename[col] = "Volume"
    return df.rename(columns=rename)


def _classify_pivot_strength(pivots: list[SwingPoint]) -> list[SwingPoint]:
    """
    Classify pivots as H/HH/HHH (strength 1/2/3) using fractal logic.
    A pivot is stronger if it exceeds N surrounding same-direction pivots.
    """
    ups   = [p for p in pivots if p.direction == SwingDir.UP]
    downs = [p for p in pivots if p.direction == SwingDir.DOWN]

    def _assign_strength(pts: list[SwingPoint], higher: bool) -> None:
        prices = [p.price for p in pts]
        for i, p in enumerate(pts):
            lo, hi = max(0, i - 3), min(len(pts), i + 4)
            neighbourhood = prices[lo:hi]
            if higher:
                rank = sum(1 for x in neighbourhood if p.price >= x)
            else:
                rank = sum(1 for x in neighbourhood if p.price <= x)
            p.strength = 1 if rank < 5 else (2 if rank < 7 else 3)

    _assign_strength(ups,   higher=True)
    _assign_strength(downs, higher=False)
    return pivots
