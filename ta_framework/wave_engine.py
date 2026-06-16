"""
wave_engine.py — Elliott Wave Detection & Scoring Engine
Fibonacci validation from:  github.com/DrEdwardPCB/python-taew/taew/ew.py
Three-rule hard validation from: github.com/ESJavadex/elliot-waves-auto/pattern_recognition.py
Multi-criteria scoring from: github.com/ESJavadex/elliot-waves-auto/improved_elliott.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


# Fibonacci ratios for wave validation (from python-taew)
FIB_RETRACE_RATIOS   = [0.146, 0.236, 0.382, 0.500, 0.618, 0.764, 0.854]
FIB_EXTENSION_RATIOS = [1.000, 1.236, 1.272, 1.382, 1.500, 1.618, 2.000, 2.272, 2.618, 3.618, 4.236]
FIB_TOLERANCE        = 0.08   # 8% tolerance for ratio matching


class WaveType(Enum):
    IMPULSE    = "impulse"
    CORRECTIVE = "corrective"
    UNKNOWN    = "unknown"


class CorrectionType(Enum):
    ZIGZAG   = "zigzag"
    FLAT     = "flat"
    TRIANGLE = "triangle"
    UNKNOWN  = "unknown"


@dataclass
class WavePoint:
    bar_index: int
    date: Optional[pd.Timestamp]
    price: float
    label: str        # "0","1","2","3","4","5","A","B","C"


@dataclass
class WaveScore:
    total:          int = 0
    rule1_ok:       bool = False   # Wave2 never retraces >100% of Wave1
    rule2_ok:       bool = False   # Wave3 never shortest
    rule3_ok:       bool = False   # Wave4 doesn't enter Wave1 price range
    w3_longest:     bool = False   # guideline: Wave3 is longest
    fib_score:      int = 0        # Fibonacci ratio hit score
    time_score:     int = 0        # timing constraint score
    notes:          list[str] = field(default_factory=list)


@dataclass
class WavePattern:
    wave_type:   WaveType
    points:      list[WavePoint]
    score:       WaveScore
    correction:  Optional[CorrectionType] = None


class WaveEngine:
    """
    Elliott Wave detection engine.

    Strategy:
    1. Detect pivot highs/lows via scipy.find_peaks (elliot-waves-auto approach).
    2. Apply Fibonacci ratio validation from python-taew.
    3. Score against three hard rules + guidelines (elliot-waves-auto scoring).
    """

    def __init__(
        self,
        peak_order:   int   = 5,
        fib_tol:      float = FIB_TOLERANCE,
        min_wave_pts: int   = 3,
    ):
        self.peak_order   = peak_order
        self.fib_tol      = fib_tol
        self.min_wave_pts = min_wave_pts

    # ------------------------------------------------------------------
    # Pivot detection
    # ------------------------------------------------------------------

    def detect_pivots(
        self, ohlc: pd.DataFrame, order: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Detect swing highs and lows using scipy.signal.find_peaks.
        Returns DataFrame with columns: date, price, pivot_type ('high'/'low').
        """
        df = _normalize(ohlc)
        ord_ = order or self.peak_order
        highs_idx, _ = find_peaks(df["High"].values, distance=ord_)
        lows_idx,  _ = find_peaks(-df["Low"].values,  distance=ord_)

        rows = []
        for i in highs_idx:
            rows.append({
                "bar_index":  i,
                "date":       df.index[i] if hasattr(df.index[i], 'year') else None,
                "price":      df["High"].iloc[i],
                "pivot_type": "high",
            })
        for i in lows_idx:
            rows.append({
                "bar_index":  i,
                "date":       df.index[i] if hasattr(df.index[i], 'year') else None,
                "price":      df["Low"].iloc[i],
                "pivot_type": "low",
            })
        result = pd.DataFrame(rows).sort_values("bar_index").reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Fibonacci checks (from python-taew)
    # ------------------------------------------------------------------

    def _fib_match(self, ratio: float, targets: list[float]) -> bool:
        return any(abs(ratio - t) / t < self.fib_tol for t in targets)

    def wave2_fib_ok(self, w1: float, w2: float) -> bool:
        ratio = abs(w2) / abs(w1) if w1 != 0 else 0
        return self._fib_match(ratio, FIB_RETRACE_RATIOS)

    def wave3_fib_ok(self, w1: float, w3: float) -> bool:
        ratio = abs(w3) / abs(w1) if w1 != 0 else 0
        return self._fib_match(ratio, FIB_EXTENSION_RATIOS)

    def wave4_fib_ok(self, w3: float, w4: float) -> bool:
        ratio = abs(w4) / abs(w3) if w3 != 0 else 0
        return self._fib_match(ratio, FIB_RETRACE_RATIOS)

    def wave5_fib_ok(self, w1: float, w5: float) -> bool:
        ratio = abs(w5) / abs(w1) if w1 != 0 else 0
        all_targets = FIB_RETRACE_RATIOS + FIB_EXTENSION_RATIOS
        return self._fib_match(ratio, all_targets)

    # ------------------------------------------------------------------
    # Three hard rules (from elliot-waves-auto/pattern_recognition.py)
    # ------------------------------------------------------------------

    def _rule1(self, p0, p1, p2) -> bool:
        """Wave 2 never retraces more than 100% of Wave 1."""
        w1 = abs(p1 - p0)
        w2 = abs(p2 - p1)
        return w2 < w1 * 1.001   # 0.1% tolerance

    def _rule2(self, p0, p1, p2, p3, p4, p5) -> bool:
        """Wave 3 is never the shortest among Waves 1, 3, 5."""
        w1 = abs(p1 - p0)
        w3 = abs(p3 - p2)
        w5 = abs(p5 - p4)
        return not (w3 < w1 and w3 < w5)

    def _rule3_up(self, p0, p1, p4) -> bool:
        """Wave 4 does not enter Wave 1 territory (uptrend)."""
        return p4 > p1   # Wave 4 low stays above Wave 1 high

    # ------------------------------------------------------------------
    # Impulse wave search
    # ------------------------------------------------------------------

    def find_impulse_waves(
        self, pivots: pd.DataFrame, direction: str = "up"
    ) -> list[WavePattern]:
        """
        Search for 5-wave impulse sequences in pivot list.
        direction: 'up' or 'down'
        """
        patterns: list[WavePattern] = []
        pts = pivots.reset_index(drop=True)
        n = len(pts)

        # We need at least 6 pivot points: 0(start),1,2,3,4,5
        if n < 6:
            return patterns

        for start in range(n - 5):
            # Require alternating high/low
            seq = pts.iloc[start:start + 6]
            prices = seq["price"].tolist()
            types  = seq["pivot_type"].tolist()

            if direction == "up":
                # Expected pattern: low,high,low,high,low,high
                expected = ["low","high","low","high","low","high"]
            else:
                expected = ["high","low","high","low","high","low"]

            if types != expected:
                continue

            p0,p1,p2,p3,p4,p5 = prices
            score = WaveScore()
            notes = []

            # Three hard rules
            w1 = abs(p1-p0); w2 = abs(p2-p1); w3 = abs(p3-p2)
            w4 = abs(p4-p3); w5 = abs(p5-p4)

            score.rule1_ok = w2 < w1 * 1.001
            if score.rule1_ok:
                score.total += 100; notes.append("Rule1 ✓ (W2 < W1)")
            else:
                notes.append("Rule1 ✗ (W2 > W1)")

            score.rule2_ok = not (w3 < w1 and w3 < w5)
            if score.rule2_ok:
                score.total += 100; notes.append("Rule2 ✓ (W3 not shortest)")
            else:
                notes.append("Rule2 ✗ (W3 shortest)")

            if direction == "up":
                score.rule3_ok = p4 > p1
            else:
                score.rule3_ok = p4 < p1
            if score.rule3_ok:
                score.total += 100; notes.append("Rule3 ✓ (W4 no overlap)")
            else:
                notes.append("Rule3 ✗ (W4 overlaps W1)")

            # Must pass all 3 rules to be a valid impulse
            if not (score.rule1_ok and score.rule2_ok and score.rule3_ok):
                continue

            # Guidelines
            score.w3_longest = (w3 >= w1 and w3 >= w5)
            if score.w3_longest:
                score.total += 15; notes.append("Guideline ✓ (W3 longest)")

            # Fibonacci scoring
            fib_pts = 0
            if self.wave2_fib_ok(w1, w2): fib_pts += 25; notes.append("W2 fib ✓")
            if self.wave3_fib_ok(w1, w3): fib_pts += 25; notes.append("W3 fib ✓")
            if self.wave4_fib_ok(w3, w4): fib_pts += 25; notes.append("W4 fib ✓")
            if self.wave5_fib_ok(w1, w5): fib_pts += 25; notes.append("W5 fib ✓")
            score.fib_score = fib_pts
            score.total += fib_pts

            score.notes = notes

            dates = seq["date"].tolist()
            wave_pts = [
                WavePoint(int(seq.iloc[i]["bar_index"]), dates[i], prices[i], str(i))
                for i in range(6)
            ]
            patterns.append(WavePattern(
                wave_type=WaveType.IMPULSE,
                points=wave_pts,
                score=score,
            ))

        return sorted(patterns, key=lambda p: p.score.total, reverse=True)

    # ------------------------------------------------------------------
    # Corrective wave (ABC) detection
    # ------------------------------------------------------------------

    def find_corrective_waves(
        self, pivots: pd.DataFrame, direction: str = "down"
    ) -> list[WavePattern]:
        """
        Search for ABC (3-wave) corrective sequences.
        """
        patterns: list[WavePattern] = []
        pts = pivots.reset_index(drop=True)
        n = len(pts)
        if n < 4:
            return patterns

        for start in range(n - 3):
            seq = pts.iloc[start:start + 4]
            prices = seq["price"].tolist()
            types  = seq["pivot_type"].tolist()

            if direction == "down":
                expected = ["high","low","high","low"]
            else:
                expected = ["low","high","low","high"]

            if types != expected:
                continue

            p0,pA,pB,pC = prices
            score = WaveScore()
            notes = []

            wA = abs(pA - p0)
            wB = abs(pB - pA)
            wC = abs(pC - pB)

            # Rule: Wave B should not exceed Wave A's start
            if direction == "down":
                b_ok = pB < p0
            else:
                b_ok = pB > p0
            if b_ok:
                score.total += 50; notes.append("B-rule ✓")

            # Fibonacci: C often equals A or 1.618*A
            ratio_ca = wC / wA if wA > 0 else 0
            if self._fib_match(ratio_ca, [1.0, 1.236, 1.618, 2.618]):
                score.fib_score = 30; score.total += 30
                notes.append(f"C/A fib ✓ ({ratio_ca:.3f})")

            # B retracement of A
            ratio_ba = wB / wA if wA > 0 else 0
            if self._fib_match(ratio_ba, FIB_RETRACE_RATIOS):
                score.total += 20; notes.append(f"B/A retrace fib ✓ ({ratio_ba:.3f})")

            correction_type = _classify_correction(wA, wB, wC, direction)
            score.notes = notes

            dates = seq["date"].tolist()
            wave_pts = [
                WavePoint(int(seq.iloc[i]["bar_index"]), dates[i], prices[i], lbl)
                for i, lbl in enumerate(["0","A","B","C"])
            ]
            patterns.append(WavePattern(
                wave_type=WaveType.CORRECTIVE,
                points=wave_pts,
                score=score,
                correction=correction_type,
            ))

        return sorted(patterns, key=lambda p: p.score.total, reverse=True)

    # ------------------------------------------------------------------
    # Next wave target projection
    # ------------------------------------------------------------------

    def project_wave_target(
        self, pattern: WavePattern, next_wave: int
    ) -> dict:
        """
        Given a completed n-wave pattern, project the next wave's price target.
        Uses Fibonacci ratios relative to the preceding wave.
        """
        pts = pattern.points
        if len(pts) < next_wave:
            return {}

        p_prev = pts[-1].price
        ref_wave = abs(pts[-1].price - pts[-2].price)
        direction = 1 if pts[-1].price > pts[-2].price else -1

        targets = {}
        for r in [0.618, 1.0, 1.272, 1.618, 2.0, 2.618]:
            targets[f"{r:.3f}"] = round(p_prev + direction * r * ref_wave, 0)
        return {"from": p_prev, "ref_wave": ref_wave, "targets": targets}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "開盤"):   rename[col] = "Open"
        elif lc in ("high", "最高"): rename[col] = "High"
        elif lc in ("low", "最低"):  rename[col] = "Low"
        elif lc in ("close","收盤"): rename[col] = "Close"
    return df.rename(columns=rename)


def _classify_correction(
    wA: float, wB: float, wC: float, direction: str
) -> CorrectionType:
    """Simple correction type classifier."""
    if wB / wA < 0.618 and abs(wC / wA - 1.0) < 0.2:
        return CorrectionType.ZIGZAG
    if 0.618 <= wB / wA <= 1.0 and abs(wC / wA - 1.0) < 0.2:
        return CorrectionType.FLAT
    if wC < wA and wB < wA:
        return CorrectionType.TRIANGLE
    return CorrectionType.UNKNOWN
