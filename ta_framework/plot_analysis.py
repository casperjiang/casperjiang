"""
plot_analysis.py — TAIEX / 台指期 K 線分析圖（日線 / 週線 / 月線）
資料來源：本機 CSV（yfinance ^TWII 格式）
執行：python plot_analysis.py [csv_path]
輸出：txf_analysis.html（日線）
      txf_analysis_weekly.html（週線）
      txf_analysis_monthly.html（月線）

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

# ── 顏色主題 ─────────────────────────────────────────────────────────
BG              = "#0d1117"
GRID            = "#1e2433"
TEXT            = "#c9d1d9"
UP_COLOR        = "#26a69a"
DN_COLOR        = "#ef5350"
SWING_CLR       = "#a78bfa"
EW_CLR          = "#38bdf8"
FIB_LIN         = "#6b7280"
FIB_SL          = "#fb923c"
FIB_TIME_FILL   = "rgba(250,204,21,0.12)"
FIB_TIME_BORDER = "rgba(250,204,21,0.6)"

# EMA 均線規格：(週期, 顏色, 線寬)  短→長：黃→橘→綠→青→藍→紫
EMA_SPECS = [
    (13,  "#fbbf24", 1.0),
    (21,  "#f97316", 1.0),
    (55,  "#4ade80", 1.2),
    (89,  "#22d3ee", 1.2),
    (144, "#818cf8", 1.5),
    (233, "#c084fc", 1.5),
]

CSV_PATH = (
    sys.argv[1] if len(sys.argv) > 1 else
    "/root/.claude/uploads/d7ceab03-ddf2-421f-affd-d3cc5286d85b/"
    "beda256f-TAIEX_History_yfinance_20260504_updated.csv"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.dropna(subset=[raw.columns[0]])
    raw = raw[~raw.iloc[:, 0].astype(str).str.startswith("^")]
    raw.columns = [c.strip() for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")
    return raw[["Open", "High", "Low", "Close", "Volume"]]


def resample_ohlcv(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """將日線資料重新取樣為週線或月線。"""
    return daily.resample(rule).agg(
        Open=("Open",   "first"),
        High=("High",   "max"),
        Low =("Low",    "min"),
        Close=("Close", "last"),
        Volume=("Volume","sum"),
    ).dropna(subset=["Close"])


def build_chart(
    ohlc:            pd.DataFrame,
    tf_label:        str,
    output_path:     str,
    swing_days:      int   = 2,
    ignore_thresh:   float = 200,
    peak_order:      int   = 5,
    show_fib_time:   bool  = True,
    use_rangebreaks: bool  = True,
) -> None:
    """
    建立單一時框的 K 線分析圖並輸出 HTML。

    Parameters
    ----------
    ohlc            : OHLCV DataFrame（DatetimeIndex）
    tf_label        : 顯示名稱，如 "日線" / "週線" / "月線"
    output_path     : 輸出 HTML 路徑
    swing_days      : Gann Swing 確認 bar 數
    ignore_thresh   : Gann Swing 最小幅度門檻（點）
    peak_order      : Elliott Wave 峰值偵測距離
    show_fib_time   : 是否顯示 Fib 時間窗口（僅日線有意義）
    use_rangebreaks : 是否移除週末空白（日線用 True，週/月用 False）
    """
    close = ohlc["Close"]

    # ── EMA 計算 ────────────────────────────────────────────────────────
    emas = {
        period: close.ewm(span=period, adjust=False).mean()
        for period, _, _ in EMA_SPECS
    }

    # ── Gann Swing ─────────────────────────────────────────────────────
    swing_eng    = SwingEngine(swing_days=swing_days, inside_down=True,
                               ignore_threshold=ignore_thresh)
    swing_result = swing_eng.calculate_swings(ohlc)
    highs        = [p for p in swing_result.pivots if p.direction == SwingDir.UP]
    lows         = [p for p in swing_result.pivots if p.direction == SwingDir.DOWN]
    recent_high  = highs[-1].price if highs else ohlc["High"].max()
    recent_low   = lows[-1].price  if lows  else ohlc["Low"].min()

    # ── Fibonacci 價格 ──────────────────────────────────────────────────
    fib_eng  = FibPriceEngine()
    lin_fibs = fib_eng.calc_retracement(recent_high, recent_low)
    sl_fibs  = fib_eng.calc_semilog_retracement(recent_high, recent_low)

    # ── Elliott Wave ────────────────────────────────────────────────────
    wave_eng    = WaveEngine(peak_order=peak_order, fib_tol=0.10)
    wave_pivots = wave_eng.detect_pivots(ohlc, order=peak_order)
    up_patterns = wave_eng.find_impulse_waves(wave_pivots, direction="up")
    best_wave   = up_patterns[0] if up_patterns else None

    # ── Fibonacci 時間週期（僅日線啟用）───────────────────────────────
    fib_zones = []
    if show_fib_time and swing_result.pivots:
        time_eng  = FibTimeEngine()
        fib_zones = time_eng.fibonacci_time_zones(
            swing_result.pivots[-1].index, dates=ohlc.index, max_zones=12
        )

    # ================================================================
    # 建立圖表
    # ================================================================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
    )
    dates_arr = ohlc.index
    price_rng = recent_high - recent_low

    # ── Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates_arr,
        open=ohlc["Open"], high=ohlc["High"],
        low=ohlc["Low"],   close=close,
        increasing_line_color=UP_COLOR, decreasing_line_color=DN_COLOR,
        increasing_fillcolor=UP_COLOR,  decreasing_fillcolor=DN_COLOR,
        line_width=1, name="TAIEX", showlegend=False,
    ), row=1, col=1)

    # ── EMA 均線（13/21/55/89/144/233）─────────────────────────────────
    for period, color, width in EMA_SPECS:
        fig.add_trace(go.Scatter(
            x=dates_arr, y=emas[period],
            line=dict(color=color, width=width),
            name=f"EMA{period}",
        ), row=1, col=1)

    # ── Gann Swing 折線 ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[p.date  for p in swing_result.pivots],
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

    # ── Fibonacci 水平線（最近 80 根 K 線區間）─────────────────────────
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

    # ── Elliott Wave ────────────────────────────────────────────────────
    if best_wave:
        ew_pts = [wp for wp in best_wave.points if wp.date is not None]
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

    # ── Fibonacci 時間窗口（僅日線）────────────────────────────────────
    show_fib_nums = {5, 8, 13, 21}
    tz_lines = []
    for zone in fib_zones:
        if zone.fib_number not in show_fib_nums or zone.date_approx is None:
            continue
        zone_date  = pd.Timestamp(zone.date_approx)
        weekday_zh = ["一","二","三","四","五","六","日"][zone_date.weekday()]
        label_text = (f"T{zone.fib_number}<br>"
                      f"{zone_date.strftime('%m/%d')}<br>"
                      f"週{weekday_zh}")

        if zone.bar_index < len(dates_arr):
            bi  = zone.bar_index
            d0  = dates_arr[max(0, bi - 1)]
            d1  = dates_arr[min(len(dates_arr) - 1, bi + 1)]
            x_l = dates_arr[bi]
        else:
            d0  = zone_date - pd.Timedelta(days=1)
            d1  = zone_date + pd.Timedelta(days=1)
            x_l = zone_date

        fig.add_vrect(x0=d0, x1=d1,
            fillcolor=FIB_TIME_FILL, opacity=1.0,
            line_color=FIB_TIME_BORDER, line_width=1, row=1, col=1)
        fig.add_annotation(
            x=x_l, y=recent_high * 0.998,
            text=label_text,
            showarrow=False, align="center",
            font=dict(size=8, color="#fde047"), row=1, col=1)

        suffix = "" if zone.bar_index < len(dates_arr) else "（投影）"
        tz_lines.append(
            f"  T{zone.fib_number:>2d}  {zone_date.strftime('%m/%d')} 週{weekday_zh}{suffix}"
        )

    # ── 成交量副圖 ──────────────────────────────────────────────────────
    vol_colors = [
        UP_COLOR if c >= o else DN_COLOR
        for c, o in zip(ohlc["Close"], ohlc["Open"])
    ]
    fig.add_trace(go.Bar(
        x=dates_arr, y=ohlc["Volume"],
        marker_color=vol_colors, opacity=0.6,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── 右下角資訊方塊 ──────────────────────────────────────────────────
    last_close = close.iloc[-1]
    prev_close = close.iloc[-2] if len(close) > 1 else last_close
    chg        = last_close - prev_close
    chg_pct    = chg / prev_close * 100
    fib50_lin  = lin_fibs[4].price
    fib50_sl   = sl_fibs[4].price
    wave_score = f"{best_wave.score.total}" if best_wave else "─"

    latest_pivot = swing_result.pivots[-1] if swing_result.pivots else None
    if latest_pivot:
        arrow_dir   = "▼ 向下（高點確認）" if latest_pivot.direction == SwingDir.UP \
                      else "▲ 向上（低點確認）"
        arrow_date  = pd.Timestamp(latest_pivot.date).strftime("%Y-%m-%d")
        arrow_price = f"{latest_pivot.price:,.0f}"
        arrow_line  = f"  {arrow_date}  {arrow_price}  {arrow_dir}"
    else:
        arrow_line = "  ─"

    info_lines = [
        f"<b>TAIEX {tf_label}（^TWII）</b>",
        f"資料截至：{ohlc.index[-1].strftime('%Y-%m-%d')}",
        f"收盤 {last_close:,.0f}　{chg:+,.0f}（{chg_pct:+.2f}%）",
        "──────────────────",
        f"Swing 高點：{recent_high:,.0f}",
        f"Swing 低點：{recent_low:,.0f}",
        f"50% 線性　：{fib50_lin:,.0f}",
        f"50% 半對數：{fib50_sl:,.0f}",
        "──────────────────",
        f"EW 波浪分：{wave_score}",
        "──────────────────",
        "<b>最新箭頭訊號</b>",
        arrow_line,
    ]
    if tz_lines:
        info_lines += ["──────────────────", "Fib 時間窗（XTAI）", *tz_lines]

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

    # ── 版面樣式 ────────────────────────────────────────────────────────
    axis_style = dict(
        gridcolor=GRID, gridwidth=1,
        color=TEXT, showline=True, linecolor="#30363d",
        tickfont=dict(color=TEXT, size=10),
    )
    rb = [dict(bounds=["sat", "mon"])] if use_rangebreaks else []

    fig.update_layout(
        title=dict(
            text=f"TAIEX {tf_label}分析（{ohlc.index[-1].strftime('%Y-%m-%d')}）",
            font=dict(size=15, color=TEXT), x=0.03,
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        autosize=True, height=700,
        margin=dict(l=50, r=10, t=50, b=40),
        xaxis =dict(**axis_style, rangeslider_visible=False, rangebreaks=rb),
        xaxis2=dict(**axis_style, rangebreaks=rb),
        yaxis =dict(**axis_style, title="點位", tickformat=","),
        yaxis2=dict(**axis_style, title="量",   tickformat=".2s"),
        legend=dict(
            orientation="h", x=0.01, y=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=10),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e2433", font_color=TEXT, font_size=11),
    )

    config = dict(responsive=True, scrollZoom=True,
                  displayModeBar=True, displaylogo=False)
    fig.write_html(output_path, include_plotlyjs=True, config=config, full_html=True)
    print(f"[{tf_label}] 已儲存：{output_path}（{os.path.getsize(output_path)//1024} KB）")


def main():
    daily = load_csv(CSV_PATH)
    print(f"資料範圍：{daily.index[0].date()} ~ {daily.index[-1].date()}（{len(daily)} 筆）")

    # 週線（週五收盤）/ 月線（月底收盤）
    try:
        weekly  = resample_ohlcv(daily, "W-FRI")
        monthly = resample_ohlcv(daily, "ME")
    except ValueError:
        weekly  = resample_ohlcv(daily, "W-FRI")
        monthly = resample_ohlcv(daily, "M")
    print(f"週線：{len(weekly)} 筆，月線：{len(monthly)} 筆")

    # ── 日線 ─────────────────────────────────────────────────────────
    build_chart(
        daily, "日線",
        os.path.join(OUT_DIR, "txf_analysis.html"),
        swing_days=2, ignore_thresh=200, peak_order=5,
        show_fib_time=True, use_rangebreaks=True,
    )

    # ── 週線 ─────────────────────────────────────────────────────────
    build_chart(
        weekly, "週線",
        os.path.join(OUT_DIR, "txf_analysis_weekly.html"),
        swing_days=2, ignore_thresh=500, peak_order=4,
        show_fib_time=False, use_rangebreaks=False,
    )

    # ── 月線 ─────────────────────────────────────────────────────────
    build_chart(
        monthly, "月線",
        os.path.join(OUT_DIR, "txf_analysis_monthly.html"),
        swing_days=1, ignore_thresh=1000, peak_order=2,
        show_fib_time=False, use_rangebreaks=False,
    )


if __name__ == "__main__":
    main()
