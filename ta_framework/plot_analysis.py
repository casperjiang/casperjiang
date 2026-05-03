"""
plot_analysis.py — 台指期五維分析 K 線圖（提示詞 A 完整版）
執行：python plot_analysis.py
輸出：txf_analysis.html（互動式）
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_adapter     import load_all_timeframes
from swing_engine     import SwingEngine, SwingDir
from fib_price_engine import FibPriceEngine
from fib_time_engine  import FibTimeEngine
from wave_engine      import WaveEngine
from regime_engine    import RegimeEngine
from decision_engine  import DecisionEngine

# ── 顏色主題 ────────────────────────────────────────────────────────
BG        = "#0d1117"
GRID      = "#1e2433"
TEXT      = "#c9d1d9"
UP_COLOR  = "#26a69a"
DN_COLOR  = "#ef5350"
MA20_CLR  = "#f59e0b"
MA60_CLR  = "#60a5fa"
SWING_CLR = "#a78bfa"
EW_CLR    = "#38bdf8"
FIB_LIN   = "#6b7280"
FIB_SL    = "#fb923c"
FIB_TIME  = "rgba(250,204,21,0.12)"
FIB_TIME_BORDER = "rgba(250,204,21,0.6)"

EXCEL = "/root/.claude/uploads/e1776b00-bd7f-4331-a579-76e7b8ed68ba/e2b7b0e0-____20260401.xlsx"


def main():
    # ================================================================
    # 1. 載入資料
    # ================================================================
    tfs    = load_all_timeframes(EXCEL)
    daily  = tfs["day"]
    h60min = tfs.get("60min")
    m15min = tfs.get("15min")

    # ================================================================
    # 2. 執行各引擎
    # ================================================================
    swing_eng  = SwingEngine(swing_days=2, inside_down=True, ignore_threshold=200)
    fib_eng    = FibPriceEngine()
    time_eng   = FibTimeEngine()
    wave_eng   = WaveEngine(peak_order=5, fib_tol=0.10)
    regime_eng = RegimeEngine()
    dec_eng    = DecisionEngine(swing_days=2, peak_order=5, fib_tol=0.10)

    swing_result = swing_eng.calculate_swings(daily)
    pivots_df    = swing_eng.pivots_to_dataframe(swing_result.pivots)
    signal       = dec_eng.analyze(daily, h60min, m15min)
    tf_states    = regime_eng.classify_multi_tf(daily, h60min, m15min)

    # Fibonacci 價格水準（最近高低）
    highs = [p for p in swing_result.pivots if p.direction == SwingDir.UP]
    lows  = [p for p in swing_result.pivots if p.direction == SwingDir.DOWN]
    recent_high = highs[-1].price if highs else daily["High"].max()
    recent_low  = lows[-1].price  if lows  else daily["Low"].min()
    lin_fibs    = fib_eng.calc_retracement(recent_high, recent_low)
    sl_fibs     = fib_eng.calc_semilog_retracement(recent_high, recent_low)

    # Elliott Wave 最佳波形
    wave_pivots   = wave_eng.detect_pivots(daily, order=5)
    up_patterns   = wave_eng.find_impulse_waves(wave_pivots, direction="up")
    best_wave     = up_patterns[0] if up_patterns else None

    # Fibonacci 時間窗口（從最後一個 pivot 起算）
    last_pivot   = swing_result.pivots[-1]
    fib_zones    = time_eng.fibonacci_time_zones(
        last_pivot.index, dates=daily.index, max_zones=10
    )

    # MA
    close = daily["Close"]
    ma20  = close.rolling(20).mean()
    ma60  = close.rolling(60).mean()

    # ================================================================
    # 3. 建立子圖（主圖 + 量能）
    # ================================================================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
    )

    dates_arr = daily.index

    # ── Candlestick ──────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates_arr,
        open=daily["Open"], high=daily["High"],
        low=daily["Low"],   close=daily["Close"],
        increasing_line_color=UP_COLOR,
        decreasing_line_color=DN_COLOR,
        increasing_fillcolor=UP_COLOR,
        decreasing_fillcolor=DN_COLOR,
        line_width=1,
        name="TXFPM1",
        showlegend=False,
    ), row=1, col=1)

    # ── MA20 / MA60 ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates_arr, y=ma20,
        line=dict(color=MA20_CLR, width=1.2),
        name="MA20", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_arr, y=ma60,
        line=dict(color=MA60_CLR, width=1.2),
        name="MA60", showlegend=True,
    ), row=1, col=1)

    # ================================================================
    # 4. Gann Swing 折線 + 標記
    # ================================================================
    swing_dates  = [p.date for p in swing_result.pivots]
    swing_prices = [p.price for p in swing_result.pivots]

    # 折線（Swing Line）
    fig.add_trace(go.Scatter(
        x=swing_dates, y=swing_prices,
        mode="lines",
        line=dict(color=SWING_CLR, width=1, dash="dot"),
        name="Gann Swing", showlegend=True,
        opacity=0.7,
    ), row=1, col=1)

    # 高點標記
    high_pivots = [p for p in swing_result.pivots if p.direction == SwingDir.UP]
    fig.add_trace(go.Scatter(
        x=[p.date for p in high_pivots],
        y=[p.price + (recent_high - recent_low) * 0.005 for p in high_pivots],
        mode="markers+text",
        marker=dict(symbol="triangle-down", color=DN_COLOR, size=8),
        text=[f"{p.price:,.0f}" for p in high_pivots],
        textposition="top center",
        textfont=dict(size=8, color=DN_COLOR),
        name="Swing High", showlegend=False,
    ), row=1, col=1)

    # 低點標記
    low_pivots = [p for p in swing_result.pivots if p.direction == SwingDir.DOWN]
    fig.add_trace(go.Scatter(
        x=[p.date for p in low_pivots],
        y=[p.price - (recent_high - recent_low) * 0.005 for p in low_pivots],
        mode="markers+text",
        marker=dict(symbol="triangle-up", color=UP_COLOR, size=8),
        text=[f"{p.price:,.0f}" for p in low_pivots],
        textposition="bottom center",
        textfont=dict(size=8, color=UP_COLOR),
        name="Swing Low", showlegend=False,
    ), row=1, col=1)

    # ================================================================
    # 5. Fibonacci 水平線
    # ================================================================
    key_ratios = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0}
    x_start    = dates_arr[max(0, len(dates_arr) - 80)]
    x_end      = dates_arr[-1]

    # 線性 Fibonacci
    for lv in lin_fibs:
        if lv.ratio not in key_ratios:
            continue
        fig.add_shape(type="line",
            x0=x_start, x1=x_end, y0=lv.price, y1=lv.price,
            line=dict(color=FIB_LIN, width=1, dash="dash"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=x_end, y=lv.price,
            text=f"<b>{lv.ratio:.1%}</b> {lv.price:,.0f}",
            showarrow=False, xanchor="left",
            font=dict(size=9, color=FIB_LIN),
            xshift=4, row=1, col=1,
        )

    # 半對數 Fibonacci（偏移一點讓兩組不重疊）
    shown_sl = set()
    for lv in sl_fibs:
        if lv.ratio not in key_ratios:
            continue
        label = f"{lv.ratio:.1%} SL {lv.price:,.0f}"
        if label in shown_sl:
            continue
        shown_sl.add(label)
        fig.add_shape(type="line",
            x0=x_start, x1=x_end, y0=lv.price, y1=lv.price,
            line=dict(color=FIB_SL, width=1, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=x_start, y=lv.price,
            text=f"SL {lv.ratio:.1%} {lv.price:,.0f}",
            showarrow=False, xanchor="right",
            font=dict(size=8, color=FIB_SL),
            xshift=-4, row=1, col=1,
        )

    # ================================================================
    # 6. Elliott Wave 折線與標籤
    # ================================================================
    if best_wave:
        ew_dates  = [wp.date for wp in best_wave.points if wp.date is not None]
        ew_prices = [wp.price for wp in best_wave.points if wp.date is not None]
        ew_labels = [wp.label  for wp in best_wave.points if wp.date is not None]

        fig.add_trace(go.Scatter(
            x=ew_dates, y=ew_prices,
            mode="lines+markers+text",
            line=dict(color=EW_CLR, width=2),
            marker=dict(color=EW_CLR, size=10,
                        line=dict(color="white", width=1)),
            text=ew_labels,
            textposition="top center",
            textfont=dict(size=12, color=EW_CLR, family="Arial Black"),
            name=f"EW Impulse (score={best_wave.score.total})",
            showlegend=True,
        ), row=1, col=1)

    # ================================================================
    # 7. Fibonacci 時間窗口（黃色半透明垂直帶）
    # ================================================================
    show_fib_nums = {5, 8, 13, 21}
    for zone in fib_zones:
        if zone.fib_number not in show_fib_nums:
            continue
        if zone.date_approx is None:
            continue
        bar_idx = zone.bar_index
        d0 = dates_arr[max(0, bar_idx - 1)] if bar_idx < len(dates_arr) else dates_arr[-1]
        d1 = dates_arr[min(len(dates_arr)-1, bar_idx + 1)]
        fig.add_vrect(
            x0=d0, x1=d1,
            fillcolor=FIB_TIME, opacity=1.0,
            line_color=FIB_TIME_BORDER, line_width=1,
            row=1, col=1,
        )
        label_date = dates_arr[min(bar_idx, len(dates_arr)-1)]
        fig.add_annotation(
            x=label_date,
            y=recent_high * 0.998,
            text=f"T{zone.fib_number}",
            showarrow=False,
            font=dict(size=9, color="#fde047"),
            row=1, col=1,
        )

    # ================================================================
    # 8. 成交量副圖
    # ================================================================
    vol_colors = [
        UP_COLOR if c >= o else DN_COLOR
        for c, o in zip(daily["Close"], daily["Open"])
    ]
    fig.add_trace(go.Bar(
        x=dates_arr, y=daily["Volume"],
        marker_color=vol_colors,
        opacity=0.6,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # ================================================================
    # 9. 右上角資訊方塊
    # ================================================================
    daily_state = tf_states.get("daily")
    regime_text = daily_state.regime.value if daily_state else "─"

    sig_map = {
        "strong_long":  "🟢 強力做多",
        "long":         "🔵 做多",
        "neutral":      "⚪ 觀望",
        "short":        "🔴 做空",
        "strong_short": "⛔ 強力做空",
    }
    sig_label = sig_map.get(signal.signal.value, signal.signal.value)
    wave_score_text = f"{best_wave.score.total}" if best_wave else "─"

    last_close = daily["Close"].iloc[-1]

    info_lines = [
        f"<b>TXFPM1 台指期 日線</b>",
        f"資料截至：{daily.index[-1].strftime('%Y-%m-%d')}",
        f"收盤：{last_close:,.0f}",
        "──────────────────",
        f"五維信號：{sig_label}",
        f"總得分：{signal.score:+d} / ±100",
        f"信心度：{signal.confidence:.0%}",
        "──────────────────",
        f"市場狀態：{regime_text}（三框架共識）",
        f"Swing 方向：{'↓ 最近低點' if signal.swing_direction < 0 else '↑ 最近高點'} {recent_low:,.0f}",
        "──────────────────",
        f"Fib 基準區間",
        f"  高點：{recent_high:,.0f}",
        f"  低點：{recent_low:,.0f}",
        f"  50% 線性：{lin_fibs[4].price:,.0f}",
        f"  50% 半對數：{sl_fibs[4].price:,.0f}",
        "──────────────────",
        f"EW 波浪品質分：{wave_score_text}",
        f"Fib 時間窗口：T5 / T8 / T13 / T21",
        "──────────────────",
        f"各維得分",
        *[f"  {k}：{v:+d}" for k, v in signal.dim_scores.items()],
    ]

    fig.add_annotation(
        x=1.0, y=0.99,
        xref="paper", yref="paper",
        text="<br>".join(info_lines),
        showarrow=False,
        align="left",
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(13,17,23,0.85)",
        bordercolor="#30363d",
        borderwidth=1,
        borderpad=8,
        font=dict(size=10, color=TEXT, family="monospace"),
    )

    # ================================================================
    # 10. 圖表樣式
    # ================================================================
    axis_style = dict(
        gridcolor=GRID, gridwidth=1,
        color=TEXT, showline=True,
        linecolor="#30363d",
        tickfont=dict(color=TEXT, size=10),
    )

    fig.update_layout(
        title=dict(
            text="TXFPM1 台指期 日線分析｜五維決策框架",
            font=dict(size=16, color=TEXT, family="sans-serif"),
            x=0.03,
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        width=1500,
        height=900,
        margin=dict(l=60, r=220, t=60, b=40),
        xaxis=dict(**axis_style, rangeslider_visible=False,
                   rangebreaks=[dict(bounds=["sat","mon"])]),
        xaxis2=dict(**axis_style, rangebreaks=[dict(bounds=["sat","mon"])]),
        yaxis=dict(**axis_style, title="點位", tickformat=","),
        yaxis2=dict(**axis_style, title="成交量", tickformat=".2s"),
        legend=dict(
            orientation="h", x=0.01, y=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=10),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e2433", font_color=TEXT, font_size=11),
    )

    # ================================================================
    # 11. 輸出
    # ================================================================
    out = os.path.join(os.path.dirname(__file__), "txf_analysis.html")
    fig.write_html(out, include_plotlyjs=True)
    print(f"圖表已儲存：{out}")
    fig.show()


if __name__ == "__main__":
    main()
