"""
demo_analysis.py — 台指期五維操盤決策模型 Demo
Uses real TXFPM1 data from Excel.
Run: python demo_analysis.py <path_to_xlsx>
"""
from __future__ import annotations
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from data_adapter    import load_all_timeframes
from swing_engine    import SwingEngine
from fib_price_engine import FibPriceEngine
from fib_time_engine  import FibTimeEngine
from wave_engine      import WaveEngine
from regime_engine    import RegimeEngine
from decision_engine  import DecisionEngine


EXCEL_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/root/.claude/uploads/e1776b00-bd7f-4331-a579-76e7b8ed68ba/e2b7b0e0-____20260401.xlsx"
)


def sep(title: str = "", char: str = "─", width: int = 60) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(char * pad + f" {title} " + char * (width - pad - len(title) - 2))
    else:
        print(char * width)


def main() -> None:
    # ──────────────────────────────────────────────
    # 1. Load data
    # ──────────────────────────────────────────────
    sep("載入資料")
    tfs = load_all_timeframes(EXCEL_PATH)
    for name, df in tfs.items():
        print(f"  {name:8s}: {len(df)} 筆  {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"           收盤範圍 {df['Close'].min():,} ~ {df['Close'].max():,}")

    daily  = tfs.get("day")
    h60min = tfs.get("60min")
    m15min = tfs.get("15min")
    print()

    # ──────────────────────────────────────────────
    # 2. Gann Swing Analysis
    # ──────────────────────────────────────────────
    sep("Gann Swing 分析（日線）")
    swing_eng = SwingEngine(swing_days=2, inside_down=True, ignore_threshold=200)
    swing_result = swing_eng.calculate_swings(daily)
    pivots_df = swing_eng.pivots_to_dataframe(swing_result.pivots)
    print(f"  偵測到 {len(swing_result.pivots)} 個 Swing Pivot")

    # Show last 10 pivots
    print(f"\n  最近 10 個 Pivot：")
    for p in swing_result.pivots[-10:]:
        dir_lbl = "▲高點" if p.direction.value == 1 else "▼低點"
        str_lbl = "★" * p.strength
        print(f"    {str(p.date)[:10]}  {dir_lbl}  {p.price:>7,.0f}  {str_lbl}")

    hh = swing_eng.is_higher_high(swing_result.pivots)
    ll = swing_eng.is_lower_low(swing_result.pivots)
    print(f"\n  Higher High: {hh}  |  Lower Low: {ll}")

    bos = swing_eng.detect_structure_break(swing_result.pivots)
    print(f"  結構突破(BOS)次數: {len(bos)}")
    if bos:
        last_bos = bos[-1]
        print(f"  最近 BOS: {last_bos['type']}  {str(last_bos['date'])[:10]}  {last_bos['price']:,.0f}")
    print()

    # ──────────────────────────────────────────────
    # 3. Fibonacci Price Analysis
    # ──────────────────────────────────────────────
    sep("Fibonacci 價格分析")
    fib_eng = FibPriceEngine()

    from swing_engine import SwingDir
    highs = [p for p in swing_result.pivots if p.direction == SwingDir.UP]
    lows  = [p for p in swing_result.pivots if p.direction == SwingDir.DOWN]

    if highs and lows:
        recent_high = highs[-1].price
        recent_low  = lows[-1].price
        print(f"  最近高點: {recent_high:,.0f}  最近低點: {recent_low:,.0f}")

        print(f"\n  【線性 Fibonacci 回撤】")
        lin_levels = fib_eng.calc_retracement(recent_high, recent_low)
        for lv in lin_levels:
            marker = " ←" if abs(daily["Close"].iloc[-1] - lv.price) / lv.price < 0.005 else ""
            print(f"    {lv.label:>6s}  {lv.price:>8,.0f}{marker}")

        print(f"\n  【半對數 Fibonacci 回撤】")
        sl_levels = fib_eng.calc_semilog_retracement(recent_high, recent_low)
        for lv in sl_levels:
            marker = " ←" if abs(daily["Close"].iloc[-1] - lv.price) / lv.price < 0.005 else ""
            print(f"    {lv.label:>10s}  {lv.price:>8,.0f}{marker}")

        print(f"\n  【線性 vs 半對數 差異】")
        cmp = fib_eng.compare_linear_vs_semilog(recent_high, recent_low)
        print(f"  {'比例':>6}  {'線性':>8}  {'半對數':>8}  {'差距(點)':>10}  {'差距%':>7}")
        for row in cmp:
            print(f"  {row['ratio']:>6.1%}  {row['linear']:>8,.0f}  {row['semilog']:>8,.0f}"
                  f"  {row['diff_pts']:>10,.0f}  {row['diff_pct']:>7}")

        # Clusters
        all_lv = lin_levels + sl_levels
        clusters = fib_eng.find_price_clusters(all_lv)
        if clusters:
            print(f"\n  【Fibonacci 群集（高度重疊支撐/阻力）】")
            for c in clusters[:5]:
                print(f"    {c.center_price:,.0f}  ({c.strength} 層重疊)")
    print()

    # ──────────────────────────────────────────────
    # 4. Fibonacci Time Analysis
    # ──────────────────────────────────────────────
    sep("Fibonacci 時間週期")
    time_eng = FibTimeEngine()

    if swing_result.pivots:
        last_pivot = swing_result.pivots[-1]
        dates = daily.index
        zones = time_eng.fibonacci_time_zones(
            last_pivot.index, dates=dates, max_zones=10
        )
        print(f"  從 {str(last_pivot.date)[:10]} 起算的 Fibonacci 時間窗口：")
        for z in zones:
            date_str = str(z.date_approx)[:10] if z.date_approx else "─"
            is_near  = " ◀ 近期" if z.bar_index >= len(daily) - 3 else ""
            print(f"    Fib {z.fib_number:>3d} bar → 預計日期 {date_str}{is_near}")

        # Swing duration analysis
        print(f"\n  Swing 持續時間分析：")
        swing_dur = time_eng.measure_swing_durations(pivots_df)
        if len(swing_dur) > 0:
            print(swing_dur[["date","price","direction","duration_days","nearest_fib_time"]].tail(8).to_string(index=False))
    print()

    # ──────────────────────────────────────────────
    # 5. Elliott Wave Analysis
    # ──────────────────────────────────────────────
    sep("Elliott Wave 分析")
    wave_eng = WaveEngine(peak_order=5, fib_tol=0.10)
    wave_pivots = wave_eng.detect_pivots(daily, order=5)
    print(f"  偵測到 {len(wave_pivots)} 個波浪 Pivot（order=5）")

    up_patterns   = wave_eng.find_impulse_waves(wave_pivots, direction="up")
    down_patterns = wave_eng.find_impulse_waves(wave_pivots, direction="down")
    print(f"  上升衝擊波形: {len(up_patterns)}  下降衝擊波形: {len(down_patterns)}")

    for label, patterns in [("上升", up_patterns), ("下降", down_patterns)]:
        if patterns:
            best = patterns[0]
            print(f"\n  最佳{label}衝擊波（得分 {best.score.total}）：")
            for wp in best.points:
                date_str = str(wp.date)[:10] if wp.date else "─"
                print(f"    Wave {wp.label}: {date_str}  {wp.price:>8,.0f}")
            print(f"  規則驗證: {'; '.join(best.score.notes[:5])}")
    print()

    # ──────────────────────────────────────────────
    # 6. Regime Analysis
    # ──────────────────────────────────────────────
    sep("市場狀態分析（多時間框架）")
    regime_eng = RegimeEngine()
    tf_states = regime_eng.classify_multi_tf(daily, h60min, m15min)
    for tf, state in tf_states.items():
        print(f"  {tf:6s}: {state.regime.value:<22s}  "
              f"MA20={'上' if state.above_ma20 else '下'}  "
              f"MA60={'上' if state.above_ma60 else '下'}  "
              f"HH={'✓' if state.higher_highs else '✗'}  "
              f"HL={'✓' if state.higher_lows else '✗'}  "
              f"ATR%={state.volatility_pct:.2f}%  "
              f"Conf={state.confluence_score}")

    consensus = regime_eng.multi_tf_consensus(tf_states)
    print(f"\n  多框架共識: {consensus['label']}")
    print(f"  平均匯聚分: {consensus['avg_confluence']:.0f}")
    print()

    # ──────────────────────────────────────────────
    # 7. Five-Dimension Decision
    # ──────────────────────────────────────────────
    sep("五維整合決策")
    dec_eng = DecisionEngine(swing_days=2, peak_order=5, fib_tol=0.10)
    signal  = dec_eng.analyze(daily, h60min, m15min)

    signal_emoji = {
        "strong_long": "🟢 強力做多",
        "long":        "🔵 做多",
        "neutral":     "⚪ 觀望",
        "short":       "🔴 做空",
        "strong_short":"⛔ 強力做空",
    }.get(signal.signal.value, signal.signal.value)

    print(f"  信號: {signal_emoji}")
    print(f"  得分: {signal.score:+d} / ±100")
    print(f"  信心: {signal.confidence:.0%}")
    if signal.entry_price:
        print(f"  進場: {signal.entry_price:,.0f}")
        print(f"  停損: {signal.stop_loss:,.0f}")
        if signal.targets:
            print(f"  目標: {' / '.join(f'{t:,.0f}' for t in signal.targets)}")

    print(f"\n  各維度得分：")
    for dim, sc in signal.dim_scores.items():
        bar = "█" * abs(sc // 2) if sc != 0 else "─"
        print(f"    {dim:12s}: {sc:+4d}  {bar}")

    print(f"\n  分析備注：")
    for note in signal.notes:
        print(f"    • {note}")

    sep()
    print("  分析完成。")
    sep()


if __name__ == "__main__":
    main()
