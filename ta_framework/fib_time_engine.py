"""
fib_time_engine.py — Fibonacci Time Cycle Engine
Core time-ratio logic adapted from: github.com/DrEdwardPCB/python-taew (Alternative method)
Fibonacci time zones (1,1,2,3,5,8,13,21,34,55...) added independently.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np


# Ratios used by python-taew Alternative method for wave timing constraints
WAVE_TIME_RATIOS = {
    "w2_min": 0.4011, "w2_max": 1.6989,
    "w3_min": 1.0000, "w3_max": 4.2360,
    "w4_min": 0.4011, "w4_max": 1.6989,
    "w5_min": 0.4011, "w5_max": 1.6989,
}

# Fibonacci number sequence for time zone projection
FIB_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


@dataclass
class TimeTarget:
    bar_offset: int          # bars from reference point
    date_approx: pd.Timestamp | None
    ratio: float
    label: str
    wave: str = ""


@dataclass
class TimeZone:
    bar_index: int
    date_approx: pd.Timestamp | None
    fib_number: int


class FibTimeEngine:
    """
    Fibonacci time cycle projector for Taiwan Futures.

    Two modes:
    1. Wave timing validation — checks if a wave's bar count falls within
       Fibonacci ratio constraints (from python-taew Alternative method).
    2. Fibonacci time zones — projects future turning-point windows from
       a reference bar using the Fibonacci number sequence.
    """

    def __init__(
        self,
        wave_time_ratios: dict = None,
        fib_sequence: list[int] = None,
        bars_per_year: int = 252,
    ):
        self.wtr = wave_time_ratios or WAVE_TIME_RATIOS
        self.fib_seq = fib_sequence or FIB_SEQUENCE
        self.bars_per_year = bars_per_year

    # ------------------------------------------------------------------
    # Wave timing validation (python-taew Alternative logic)
    # ------------------------------------------------------------------

    def validate_wave_timing(
        self,
        wave_bars: int,
        ref_wave_bars: int,
        wave_num: int,
    ) -> dict:
        """
        Check whether wave_bars for Wave N is within Fibonacci time constraints
        relative to Wave 1's bar count (ref_wave_bars).

        Returns dict with: valid, ratio, min_ratio, max_ratio, wave_label
        """
        key_min = f"w{wave_num}_min"
        key_max = f"w{wave_num}_max"
        min_r = self.wtr.get(key_min, 0.0)
        max_r = self.wtr.get(key_max, 9999.0)
        ratio = wave_bars / ref_wave_bars if ref_wave_bars > 0 else 0.0
        return {
            "wave":      f"Wave {wave_num}",
            "valid":     min_r <= ratio <= max_r,
            "ratio":     round(ratio, 4),
            "min_ratio": min_r,
            "max_ratio": max_r,
            "wave_bars": wave_bars,
            "ref_bars":  ref_wave_bars,
        }

    def project_wave_time_targets(
        self,
        wave1_start_bar: int,
        wave1_end_bar: int,
        dates: pd.DatetimeIndex | None = None,
    ) -> list[TimeTarget]:
        """
        Given Wave 1's bar range, project expected bar ranges for Waves 2–5.
        """
        w1_bars = wave1_end_bar - wave1_start_bar
        targets: list[TimeTarget] = []
        for wave_num in range(2, 6):
            min_r = self.wtr.get(f"w{wave_num}_min", 0.4)
            max_r = self.wtr.get(f"w{wave_num}_max", 2.0)
            # project from previous wave end (simplified: stack from wave1_end)
            for ratio, lbl in [(min_r, "min"), (max_r, "max")]:
                offset = int(round(ratio * w1_bars))
                ref_bar = wave1_end_bar + offset
                date = dates[ref_bar] if (dates is not None and ref_bar < len(dates)) else None
                targets.append(TimeTarget(
                    bar_offset=ref_bar,
                    date_approx=date,
                    ratio=ratio,
                    label=f"W{wave_num} {lbl} ({ratio}x)",
                    wave=f"Wave {wave_num}",
                ))
        return targets

    # ------------------------------------------------------------------
    # Fibonacci time zones (classic Gann / EW tool)
    # ------------------------------------------------------------------

    def fibonacci_time_zones(
        self,
        start_bar: int,
        dates: pd.DatetimeIndex | None = None,
        max_zones: int = 13,
    ) -> list[TimeZone]:
        """
        Project future time zones from start_bar using Fibonacci sequence.
        These mark potential turning-point windows.
        """
        zones: list[TimeZone] = []
        for fib_n in self.fib_seq[:max_zones]:
            bar_idx = start_bar + fib_n
            date = dates[bar_idx] if (dates is not None and bar_idx < len(dates)) else None
            zones.append(TimeZone(
                bar_index=bar_idx,
                date_approx=date,
                fib_number=fib_n,
            ))
        return zones

    # ------------------------------------------------------------------
    # Swing-based time analysis
    # ------------------------------------------------------------------

    def measure_swing_durations(
        self, pivots_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate bar/day duration between consecutive swing pivots.
        Input: DataFrame with columns [date, price, direction]
        """
        df = pivots_df.copy().reset_index(drop=True)
        df["prev_date"]  = df["date"].shift(1)
        df["prev_price"] = df["price"].shift(1)
        df["duration_days"] = (df["date"] - df["prev_date"]).dt.days
        df["price_move"] = df["price"] - df["prev_price"]
        df["ratio_to_prev"] = df["duration_days"] / df["duration_days"].shift(1)
        df["nearest_fib_time"] = df["duration_days"].apply(self._nearest_fib_number)
        return df.dropna(subset=["duration_days"])

    def _nearest_fib_number(self, days: float) -> int:
        if pd.isna(days):
            return 0
        return min(self.fib_seq, key=lambda f: abs(f - days))

    # ------------------------------------------------------------------
    # Convergence window detection
    # ------------------------------------------------------------------

    def find_time_convergence(
        self,
        zones_list: list[list[TimeZone]],
        window_bars: int = 3,
    ) -> list[dict]:
        """
        Find bar ranges where multiple time zone sets converge.
        High-confluence time windows suggest major turning points.
        """
        all_bars = []
        for zones in zones_list:
            for z in zones:
                all_bars.append(z.bar_index)

        convergences = []
        checked = set()
        for bar in sorted(set(all_bars)):
            if bar in checked:
                continue
            nearby = [b for b in all_bars if abs(b - bar) <= window_bars]
            if len(nearby) >= 2:
                convergences.append({
                    "center_bar": bar,
                    "count":      len(nearby),
                    "bars":       sorted(set(nearby)),
                })
                for b in nearby:
                    checked.add(b)
        return sorted(convergences, key=lambda x: x["count"], reverse=True)
