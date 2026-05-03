"""
plot_analysis.py — TAIEX / 台指期 K 線分析圖（六大 Repo 組合版）
資料來源：本機 CSV（yfinance ^TWII 格式）
執行：python plot_analysis.py
輸出：txf_analysis.html（手機 / 桌機皆可開啟）

分析模組來源：
  Gann Swing       ← gann-swing (monch1962)
  Fibonacci 價格   ← elliot-waves-auto + Stock-market/high_low.py
  Fibonacci 時間   ← python-taew (Alternative method)
  Elliott Wave     ← python-taew + elliot-waves-auto
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from swing_engine     import SwingEngine, SwingDir
from fib_price_engine import FibPriceEngine
from fib_time_engine  import FibTimeEngine
from wave_engine      import WaveEngine

# ── 顏色主題 ────────────────────────────────────────────────────────
BG             = "#0d1117"
GRID           = "#1e2433"
TEXT           = "#c9d1d9"
UP_COLOR       = "#26a69a"
DN_COLOR       = "#ef5350"
MA20_CLR       = "#f59e0b"
MA60_CLR       = "#60a5fa"
SWING_CLR      = "#a78bfa"
EW_CLR         = "#38bdf8"
FIB_LIN        = "#6b7280"
FIB_SL         = "#fb923c"
FIB_TIME_FILL  = "rgba(250,204,21,0.12)"
FIB_TIME_BORDER= "rgba(250,204,21,0.6)"

CSV_PATH = (
    sys.argv[1] if len(sys.argv) > 1 else
    "/root/.claude/uploads/3edc58e3-79bb-41d6-8c29-d7cb4b1a91c1/"
    "7c1ff448-TAIEX_History_yfinance_20260503_1.csv"
)
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "txf_analysis.html")


def load_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # 移除 ^TWII metadata 列（第一列 Date=NaN）
    raw = raw.dropna(subset=[raw.columns[0]])
    raw = raw[~raw.iloc[:, 0].astype(str).str.startswith("^")]
    raw.columns = [c.strip() for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")
    return raw[["Open", "High", "Low", "Close", "Volume"]]


def main():
    # ================================================================
    # 1. 載入資料
    # ================================================================
    daily = load_csv(CSV_PATH)
    print(f"資料範圍：{daily.index[0].date()} ~ {daily.index[-1].date()}（{len(daily)} 筆）")

    # ================================================================
    # 2. 分析引擎（六大 Repo 組合）
    # ================================================================

    # ── Gann Swing（gann-swing repo）────────────────────────────────
    swing_eng    = SwingEngine(swing_days=2, inside_down=True, ignore_threshold=200)
    swing_result = swing_eng.calculate_swings(daily)

    from swing_engine import SwingDir
    highs = [p for p in swing_result.pivots if p.direction == SwingDir.UP]
    lows  = [p for p in swing_result.pivots if p.direction == SwingDir.DOWN]
    recent_high = highs[-1].price if highs else daily["High"].max()
    recent_low  = lows[-1].price  if lows  else daily["Low"].min()

    # ── Fibonacci 價格（elliot-waves-auto + Stock-market 半對數）──────
    fib_eng  = FibPriceEngine()
    lin_fibs = fib_eng.calc_retracement(recent_high, recent_low)
    sl_fibs  = fib_eng.calc_semilog_retracement(recent_high, recent_low)

    # ── Elliott Wave（python-taew + elliot-waves-auto）───────────────
    wave_eng    = WaveEngine(peak_order=5, fib_tol=0.10)
    wave_pivots = wave_eng.detect_pivots(daily, order=5)
    up_patterns = wave_eng.find_impulse_waves(wave_pivots, direction="up")
    best_wave   = up_patterns[0] if up_patterns else None

    # ── Fibonacci 時間週期（python-taew Alternative method）──────────
    time_eng   = FibTimeEngine()
    last_pivot = swing_result.pivots[-1]
    fib_zones  = time_eng.fibonacci_time_zones(
        last_pivot.index, dates=daily.index, max_zones=12
    )

    close = daily["Close"]
    ma20  = close.rolling(20).mean()
    ma60  = close.rolling(60).mean()

    # ================================================================
    # 3. 建立圖表
    # ================================================================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
    )
    dates_arr  = daily.index
    price_rng  = recent_high - recent_low

    # ── Candlestick ──────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates_arr,
        open=daily["Open"], high=daily["High"],
        low=daily["Low"],   close=close,
        increasing_line_color=UP_COLOR, decreasing_line_color=DN_COLOR,
        increasing_fillcolor=UP_COLOR,  decreasing_fillcolor=DN_COLOR,
        line_width=1, name="TAIEX", showlegend=False,
    ), row=1, col=1)

    # ── MA20 / MA60 ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=dates_arr, y=ma20,
        line=dict(color=MA20_CLR, width=1.2), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates_arr, y=ma60,
        line=dict(color=MA60_CLR, width=1.2), name="MA60"), row=1, col=1)

    # ── Gann Swing 折線 ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[p.date for p in swing_result.pivots],
        y=[p.price for p in swing_result.pivots],
        mode="lines",
        line=dict(color=SWING_CLR, width=1, dash="dot"),
        name="Gann Swing", opacity=0.7,
    ), row=1, col=1)

    high_pivots = [p for p in swing_result.pivots if p.direction == SwingDir.UP]
    low_pivots  = [p for p in swing_result.pivots if p.direction == SwingDir.DOWN]

    fig.add_trace(go.Scatter(
        x=[p.date for p in high_pivots],
        y=[p.price + price_rng * 0.004 for p in high_pivots],
        mode="markers+text",
        marker=dict(symbol="triangle-down", color=DN_COLOR, size=7),
        text=[f"{p.price:,.0f}" for p in high_pivots],
        textposition="top center",
        textfont=dict(size=8, color=DN_COLOR),
        name="Swing H", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[p.date for p in low_pivots],
        y=[p.price - price_rng * 0.004 for p in low_pivots],
        mode="markers+text",
        marker=dict(symbol="triangle-up", color=UP_COLOR, size=7),
        text=[f"{p.price:,.0f}" for p in low_pivots],
        textposition="bottom center",
        textfont=dict(size=8, color=UP_COLOR),
        name="Swing L", showlegend=False,
    ), row=1, col=1)

    # ── Fibonacci 水平線（最近 80 根 K 線區間）────────────────────────
    key_ratios = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0}
    x_start    = dates_arr[max(0, len(dates_arr) - 80)]
    x_end      = dates_arr[-1]

    for lv in lin_fibs:
        if lv.ratio not in key_ratios:
            continue
        fig.add_shape(type="line",
            x0=x_start, x1=x_end, y0=lv.price, y1=lv.price,
            line=dict(color=FIB_LIN, width=1, dash="dash"), row=1, col=1)
        fig.add_annotation(x=x_end, y=lv.price,
            text=f"<b>{lv.ratio:.1%}</b> {lv.price:,.0f}",
            showarrow=False, xanchor="left",
            font=dict(size=9, color=FIB_LIN), xshift=4, row=1, col=1)

    shown_sl = set()
    for lv in sl_fibs:
        if lv.ratio not in key_ratios:
            continue
        key = f"{lv.ratio:.3f}"
        if key in shown_sl:
            continue
        shown_sl.add(key)
        fig.add_shape(type="line",
            x0=x_start, x1=x_end, y0=lv.price, y1=lv.price,
            line=dict(color=FIB_SL, width=1, dash="dot"), row=1, col=1)
        fig.add_annotation(x=x_start, y=lv.price,
            text=f"SL {lv.ratio:.1%} {lv.price:,.0f}",
            showarrow=False, xanchor="right",
            font=dict(size=8, color=FIB_SL), xshift=-4, row=1, col=1)

    # ── Elliott Wave ─────────────────────────────────────────────────
    if best_wave:
        ew_pts    = [wp for wp in best_wave.points if wp.date is not None]
        fig.add_trace(go.Scatter(
            x=[wp.date  for wp in ew_pts],
            y=[wp.price for wp in ew_pts],
            mode="lines+markers+text",
            line=dict(color=EW_CLR, width=2),
            marker=dict(color=EW_CLR, size=10, line=dict(color="white", width=1)),
            text=[wp.label for wp in ew_pts],
            textposition="top center",
            textfont=dict(size=12, color=EW_CLR, family="Arial Black"),
            name=f"EW Impulse (score={best_wave.score.total})",
        ), row=1, col=1)

    # ── Fibonacci 時間窗口（XTAI 台股開市日曆）────────────────────────
    show_fib_nums = {5, 8, 13, 21}
    last_hist_date = dates_arr[-1]

    for zone in fib_zones:
        if zone.fib_number not in show_fib_nums:
            continue
        if zone.date_approx is None:
            continue

        zone_date = pd.Timestamp(zone.date_approx)
        weekday_zh = ["一","二","三","四","五","六","日"][zone_date.weekday()]
        label_text = (f"T{zone.fib_number}<br>"
                      f"{zone_date.strftime('%m/%d')}<br>"
                      f"週{weekday_zh}")

        if zone.bar_index < len(dates_arr):
            # 歷史資料內：用 K 線 x 軸位置
            bi = zone.bar_index
            d0 = dates_arr[max(0, bi - 1)]
            d1 = dates_arr[min(len(dates_arr) - 1, bi + 1)]
            x_label = dates_arr[bi]
        else:
            # 未來投影：直接用日期（Plotly 會正確定位在 x 軸）
            d0 = zone_date - pd.Timedelta(days=1)
            d1 = zone_date + pd.Timedelta(days=1)
            x_label = zone_date

        fig.add_vrect(x0=d0, x1=d1,
            fillcolor=FIB_TIME_FILL, opacity=1.0,
            line_color=FIB_TIME_BORDER, line_width=1, row=1, col=1)
        fig.add_annotation(
            x=x_label, y=recent_high * 0.998,
            text=label_text,
            showarrow=False, align="center",
            font=dict(size=8, color="#fde047"), row=1, col=1)

    # ── 成交量副圖 ───────────────────────────────────────────────────
    vol_colors = [
        UP_COLOR if c >= o else DN_COLOR
        for c, o in zip(daily["Close"], daily["Open"])
    ]
    fig.add_trace(go.Bar(
        x=dates_arr, y=daily["Volume"],
        marker_color=vol_colors, opacity=0.6,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── 右下角資訊方塊（不遮主圖）────────────────────────────────────
    last_close = close.iloc[-1]
    prev_close = close.iloc[-2] if len(close) > 1 else last_close
    chg        = last_close - prev_close
    chg_pct    = chg / prev_close * 100

    fib50_lin = lin_fibs[4].price  # 50% 線性
    fib50_sl  = sl_fibs[4].price   # 50% 半對數
    wave_score_text = f"{best_wave.score.total}" if best_wave else "─"

    # 時間窗口說明（含台股開市日）
    tz_lines = []
    for z in fib_zones:
        if z.fib_number not in show_fib_nums or z.date_approx is None:
            continue
        dt = pd.Timestamp(z.date_approx)
        wd = ["一","二","三","四","五","六","日"][dt.weekday()]
        suffix = "" if z.bar_index < len(dates_arr) else "（投影）"
        tz_lines.append(f"  T{z.fib_number:>2d}  {dt.strftime('%m/%d')} 週{wd}{suffix}")

    info_lines = [
        f"<b>TAIEX 日線（^TWII）</b>",
        f"資料截至：{daily.index[-1].strftime('%Y-%m-%d')}",
        f"收盤 {last_close:,.0f}　{chg:+,.0f}（{chg_pct:+.2f}%）",
        "──────────────────",
        f"Swing 高點：{recent_high:,.0f}",
        f"Swing 低點：{recent_low:,.0f}",
        f"50% 線性　：{fib50_lin:,.0f}",
        f"50% 半對數：{fib50_sl:,.0f}",
        "──────────────────",
        f"EW 波浪分：{wave_score_text}",
        "Fib 時間窗（XTAI）",
        *tz_lines,
    ]

    fig.add_annotation(
        x=0.985, y=0.035,
        xref="paper", yref="paper",
        text="<br>".join(info_lines),
        showarrow=False, align="left",
        xanchor="right", yanchor="bottom",
        bgcolor="rgba(13,17,23,0.88)",
        bordercolor="#30363d",
        borderwidth=1, borderpad=8,
        font=dict(size=10, color=TEXT, family="monospace"),
    )

    # ================================================================
    # 4. 樣式（手機 responsive）
    # ================================================================
    axis_style = dict(
        gridcolor=GRID, gridwidth=1,
        color=TEXT, showline=True, linecolor="#30363d",
        tickfont=dict(color=TEXT, size=10),
    )

    fig.update_layout(
        title=dict(
            text=f"TAIEX 日線分析（{daily.index[-1].strftime('%Y-%m-%d')}）",
            font=dict(size=15, color=TEXT),
            x=0.03,
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        autosize=True,            # 手機自動填滿寬度
        height=700,
        margin=dict(l=50, r=10, t=50, b=40),
        xaxis=dict(**axis_style, rangeslider_visible=False,
                   rangebreaks=[dict(bounds=["sat", "mon"])]),
        xaxis2=dict(**axis_style,
                    rangebreaks=[dict(bounds=["sat", "mon"])]),
        yaxis=dict(**axis_style, title="點位", tickformat=","),
        yaxis2=dict(**axis_style, title="量", tickformat=".2s"),
        legend=dict(
            orientation="h", x=0.01, y=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=10),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e2433", font_color=TEXT, font_size=11),
    )

    # ================================================================
    # 5. 輸出（手機 responsive config）
    # ================================================================
    config = dict(responsive=True, scrollZoom=True,
                  displayModeBar=True, displaylogo=False)

    fig.write_html(
        OUTPUT,
        include_plotlyjs=True,
        config=config,
        full_html=True,
    )
    print(f"圖表已儲存：{OUTPUT}（{os.path.getsize(OUTPUT)//1024} KB）")


if __name__ == "__main__":
    main()
