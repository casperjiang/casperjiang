"""
fib_price_engine.py — Fibonacci Price Retracement / Extension Engine
Supports both linear-scale and semi-logarithmic scale calculations.

Linear Fibonacci adapted from: github.com/ESJavadex/elliot-waves-auto/fibonacci_analysis.py
Semi-log formula adapted from: github.com/Louisli0515/Stock-market/high_low.py (bug-fixed)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math


RETRACE_LEVELS  = [0.0, 0.146, 0.236, 0.382, 0.500, 0.618, 0.764, 0.854, 1.0]
EXTEND_LEVELS   = [1.0, 1.236, 1.272, 1.382, 1.500, 1.618, 2.0, 2.272, 2.618, 3.618, 4.236]
CLUSTER_TOL     = 0.003   # 0.3% cluster tolerance


@dataclass
class FibLevel:
    ratio: float
    price: float
    label: str
    is_semilog: bool = False


@dataclass
class FibCluster:
    center_price: float
    levels: list[FibLevel]
    strength: int   # number of coincident levels


class FibPriceEngine:
    """
    Fibonacci price target calculator for Taiwan Futures.

    Provides:
    - Linear retracement / extension
    - Semi-logarithmic retracement / extension  (key differentiator for long-term TXF)
    - Price cluster analysis
    """

    def __init__(
        self,
        retrace_levels: list[float] = None,
        extend_levels:  list[float] = None,
        cluster_tol:    float = CLUSTER_TOL,
    ):
        self.retrace_levels = retrace_levels or RETRACE_LEVELS
        self.extend_levels  = extend_levels  or EXTEND_LEVELS
        self.cluster_tol    = cluster_tol

    # ------------------------------------------------------------------
    # Linear Fibonacci
    # ------------------------------------------------------------------

    def calc_retracement(self, high: float, low: float) -> list[FibLevel]:
        """Standard linear Fibonacci retracement from high to low."""
        span = high - low
        return [
            FibLevel(r, high - r * span, f"{r:.1%}")
            for r in self.retrace_levels
        ]

    def calc_extension(self, wave_start: float, wave_end: float,
                       retrace_end: float) -> list[FibLevel]:
        """
        Linear Fibonacci extension projection.
        wave_start → wave_end = Wave 1 (or A)
        retrace_end           = end of Wave 2 (or B) retracement
        """
        wave_len = abs(wave_end - wave_start)
        direction = 1 if wave_end > wave_start else -1
        return [
            FibLevel(r, retrace_end + direction * r * wave_len, f"{r:.1%} ext")
            for r in self.extend_levels
        ]

    # ------------------------------------------------------------------
    # Semi-logarithmic Fibonacci
    # Semi-log interpretation: equal percentage moves appear equal in size.
    # On a log chart, a Fibonacci retracement of ratio r between price A and B
    # corresponds to the geometric mean: A^(1-r) * B^r
    # ------------------------------------------------------------------

    @staticmethod
    def semilog_price(low: float, high: float, ratio: float) -> float:
        """
        Compute Fibonacci price at `ratio` between low and high
        on a semi-logarithmic scale.

        Formula (from Stock-market/high_low.py, bug-fixed):
            price = low^(1 - ratio) * high^ratio

        This is the geometric interpolation, equivalent to:
            log(price) = (1-ratio)*log(low) + ratio*log(high)
        """
        if low <= 0 or high <= 0:
            raise ValueError("Prices must be positive for semi-log calculation.")
        return (low ** (1.0 - ratio)) * (high ** ratio)

    def calc_semilog_retracement(self, high: float, low: float) -> list[FibLevel]:
        """
        Semi-log Fibonacci retracement from high to low.
        Ratio=0 → high, Ratio=1 → low.
        """
        levels = []
        for r in self.retrace_levels:
            # retracement: high → low direction, ratio r of the log-range
            price = self.semilog_price(low, high, 1.0 - r)
            levels.append(FibLevel(r, price, f"{r:.1%} SL", is_semilog=True))
        return levels

    def calc_semilog_extension(self, wave_start: float, wave_end: float,
                                retrace_end: float) -> list[FibLevel]:
        """
        Semi-log Fibonacci extension.
        Treats the wave's log-range as the unit, projects from retrace_end.
        """
        if any(p <= 0 for p in [wave_start, wave_end, retrace_end]):
            raise ValueError("Prices must be positive for semi-log calculation.")

        log_wave = math.log(wave_end) - math.log(wave_start)   # log amplitude of wave
        direction = 1 if wave_end > wave_start else -1
        levels = []
        for r in self.extend_levels:
            log_target = math.log(retrace_end) + direction * r * log_wave
            price = math.exp(log_target)
            levels.append(FibLevel(r, price, f"{r:.1%} SL ext", is_semilog=True))
        return levels

    # ------------------------------------------------------------------
    # Compare linear vs semi-log
    # ------------------------------------------------------------------

    def compare_linear_vs_semilog(
        self, high: float, low: float
    ) -> list[dict]:
        """
        Side-by-side comparison for research/debugging.
        Returns list of dicts with ratio, linear_price, semilog_price, diff_pct.
        """
        lin  = {lv.ratio: lv.price for lv in self.calc_retracement(high, low)}
        semi = {lv.ratio: lv.price for lv in self.calc_semilog_retracement(high, low)}
        rows = []
        for r in self.retrace_levels:
            lp, sp = lin[r], semi[r]
            rows.append({
                "ratio": r,
                "linear": round(lp, 0),
                "semilog": round(sp, 0),
                "diff_pts": round(sp - lp, 0),
                "diff_pct": f"{(sp - lp) / lp:.2%}",
            })
        return rows

    # ------------------------------------------------------------------
    # Cluster analysis (from elliot-waves-auto/fibonacci_analysis.py)
    # ------------------------------------------------------------------

    def find_price_clusters(
        self, all_levels: list[FibLevel], tolerance: Optional[float] = None
    ) -> list[FibCluster]:
        """
        Find price zones where multiple Fibonacci levels coincide.
        These are high-confluence support/resistance areas.
        """
        tol = tolerance or self.cluster_tol
        sorted_levels = sorted(all_levels, key=lambda x: x.price)
        clusters: list[FibCluster] = []
        used = [False] * len(sorted_levels)

        for i, base in enumerate(sorted_levels):
            if used[i]:
                continue
            group = [base]
            used[i] = True
            for j in range(i + 1, len(sorted_levels)):
                if used[j]:
                    continue
                if abs(sorted_levels[j].price - base.price) / base.price <= tol:
                    group.append(sorted_levels[j])
                    used[j] = True
            if len(group) >= 2:
                center = sum(lv.price for lv in group) / len(group)
                clusters.append(FibCluster(
                    center_price=round(center, 0),
                    levels=group,
                    strength=len(group),
                ))

        return sorted(clusters, key=lambda c: c.strength, reverse=True)

    # ------------------------------------------------------------------
    # Wave relationship analysis
    # ------------------------------------------------------------------

    def wave_ratio(self, wave_a: float, wave_b: float) -> float:
        """Compute the ratio |wave_b| / |wave_a|."""
        if wave_a == 0:
            return float("inf")
        return abs(wave_b) / abs(wave_a)

    def nearest_fib(self, ratio: float) -> tuple[float, str]:
        """Find the nearest Fibonacci ratio and its label."""
        all_ratios = self.retrace_levels + self.extend_levels
        nearest = min(all_ratios, key=lambda r: abs(r - ratio))
        diff_pct = abs(nearest - ratio) / nearest if nearest else 0
        return nearest, f"{nearest:.3f} ({diff_pct:.1%} off)"
