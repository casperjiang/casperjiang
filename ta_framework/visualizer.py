"""
visualizer.py — TAVisualizer: 台指期/大盤 K 線分析圖表模組
"""
from __future__ import annotations
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

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
FIB_TIME_LINE  = "rgba(250,204,21,0.6)"

_AXIS = dict(
    gridcolor=GRID, gridwidth=1,
    color=TEXT, showline=True,
    linecolor="#30363d",
    tickfont=dict(color=TEXT, size=10),
)


class TAVisualizer:
    """
    全功能 K 線分析圖表產生器。

    使用方式
    --------
    viz = TAVisualizer(title="TAIEX 日線分析")
    fig = viz.plot_full_analysis(
        daily, swing_result, wave_pattern,
        fib_levels, time_zones, signal,
        output_html="chart.html"
    )
    """

    def __init__(
        self,
        title:  str = "台指期 日線分析",
        width:  int = 1500,
        height: int = 900,
        embed_plotlyjs: bool = True,
    ):
        self.title          = title
        self.width          = width
        self.height         = height
        self.embed_plotlyjs = embed_plotlyjs

    # ----------------------------------------------------------------
    # 主要繪圖入口
    # ----------------------------------------------------------------

    def plot_full_analysis(
        self,
        daily:        pd.DataFrame,
        swing_result  = None,
        wave_pattern  = None,
        fib_levels:   Optional[dict] = None,
        time_zones:   Optional[list] = None,
        signal        = None,
        tf_states:    Optional[dict] = None,
        output_html:  Optional[str]  = None,
    ) -> go.Figure:

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.78, 0.22],
            vertical_spacing=0.02,
        )

        dates = daily.index
        close = daily["Close"]
        ma20  = close.rolling(20).mean()
        ma60  = close.rolling(60).mean()

        # ── Candlestick ──────────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=dates,
            open=daily["Open"], high=daily["High"],
            low=daily["Low"],   close=close,
            increasing_line_color=UP_COLOR,
            decreasing_line_color=DN_COLOR,
            increasing_fillcolor=UP_COLOR,
            decreasing_fillcolor=DN_COLOR,
            line_width=1, name="Price", showlegend=False,
        ), row=1, col=1)

        # ── MA ───────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(x=dates, y=ma20,
            line=dict(color=MA20_CLR, width=1.2),
            name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ma60,
            line=dict(color=MA60_CLR, width=1.2),
            name="MA60"), row=1, col=1)

        # ── Gann Swing ───────────────────────────────────────────────
        if swing_result is not None:
            self._add_swing(fig, swing_result, daily)

        # ── Fibonacci 水平線 ─────────────────────────────────────────
        if fib_levels:
            self._add_fib_levels(fig, fib_levels, dates, daily)

        # ── Elliott Wave ─────────────────────────────────────────────
        if wave_pattern is not None:
            self._add_wave(fig, wave_pattern)

        # ── Fibonacci 時間窗口 ───────────────────────────────────────
        if time_zones:
            self._add_time_zones(fig, time_zones, dates, daily)

        # ── Volume ───────────────────────────────────────────────────
        if "Volume" in daily.columns:
            vol_colors = [
                UP_COLOR if c >= o else DN_COLOR
                for c, o in zip(daily["Close"], daily["Open"])
            ]
            fig.add_trace(go.Bar(
                x=dates, y=daily["Volume"],
                marker_color=vol_colors, opacity=0.6,
                name="Volume", showlegend=False,
            ), row=2, col=1)

        # ── 資訊方塊 ─────────────────────────────────────────────────
        self._add_info_box(fig, daily, signal, tf_states, fib_levels)

        # ── 樣式 ─────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(text=self.title,
                       font=dict(size=16, color=TEXT), x=0.03),
            paper_bgcolor=BG, plot_bgcolor=BG,
            width=self.width, height=self.height,
            margin=dict(l=60, r=240, t=60, b=40),
            xaxis=dict(**_AXIS, rangeslider_visible=False,
                       rangebreaks=[dict(bounds=["sat","mon"])]),
            xaxis2=dict(**_AXIS,
                        rangebreaks=[dict(bounds=["sat","mon"])]),
            yaxis=dict(**_AXIS, title="點位", tickformat=","),
            yaxis2=dict(**_AXIS, title="成交量", tickformat=".2s"),
            legend=dict(orientation="h", x=0.01, y=1.02,
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(color=TEXT, size=10)),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1e2433", font_color=TEXT),
        )

        if output_html:
            fig.write_html(output_html,
                           include_plotlyjs=self.embed_plotlyjs)

        return fig

    # ----------------------------------------------------------------
    # 私有方法：各圖層
    # ----------------------------------------------------------------

    def _add_swing(self, fig, swing_result, daily):
        from swing_engine import SwingDir
        pivots = swing_result.pivots
        if not pivots:
            return

        price_range = daily["High"].max() - daily["Low"].min()
        offset      = price_range * 0.004

        swing_dates  = [p.date for p in pivots]
        swing_prices = [p.price for p in pivots]
        fig.add_trace(go.Scatter(
            x=swing_dates, y=swing_prices, mode="lines",
            line=dict(color=SWING_CLR, width=1, dash="dot"),
            name="Gann Swing", opacity=0.7,
        ), row=1, col=1)

        highs = [p for p in pivots if p.direction == SwingDir.UP]
        lows  = [p for p in pivots if p.direction == SwingDir.DOWN]

        fig.add_trace(go.Scatter(
            x=[p.date for p in highs],
            y=[p.price + offset for p in highs],
            mode="markers+text",
            marker=dict(symbol="triangle-down", color=DN_COLOR, size=8),
            text=[f"{p.price:,.0f}" for p in highs],
            textposition="top center",
            textfont=dict(size=8, color=DN_COLOR),
            name="Swing H", showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[p.date for p in lows],
            y=[p.price - offset for p in lows],
            mode="markers+text",
            marker=dict(symbol="triangle-up", color=UP_COLOR, size=8),
            text=[f"{p.price:,.0f}" for p in lows],
            textposition="bottom center",
            textfont=dict(size=8, color=UP_COLOR),
            name="Swing L", showlegend=False,
        ), row=1, col=1)

    def _add_fib_levels(self, fig, fib_levels, dates, daily):
        recent_high = fib_levels.get("recent_high", daily["High"].max())
        recent_low  = fib_levels.get("recent_low",  daily["Low"].min())
        key_ratios  = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0}

        x_start = dates[max(0, len(dates) - 80)]
        x_end   = dates[-1]

        for label_key, color, side in [
            ("linear",  FIB_LIN, "right"),
            ("semilog", FIB_SL,  "left"),
        ]:
            levels = fib_levels.get(label_key, {})
            for lbl, price in levels.items():
                ratio_str = lbl.replace(" SL","").replace("%","").strip()
                try:
                    ratio = float(ratio_str) / 100
                except ValueError:
                    continue
                if ratio not in key_ratios:
                    continue
                fig.add_shape(type="line",
                    x0=x_start, x1=x_end,
                    y0=price,   y1=price,
                    line=dict(color=color, width=1,
                              dash="dash" if label_key=="linear" else "dot"),
                    row=1, col=1,
                )
                xanchor = "left" if side == "right" else "right"
                xshift  = 4      if side == "right" else -4
                prefix  = "" if side == "right" else "SL "
                fig.add_annotation(
                    x=x_end if side=="right" else x_start,
                    y=price,
                    text=f"{prefix}{ratio:.1%} {price:,.0f}",
                    showarrow=False, xanchor=xanchor,
                    font=dict(size=8, color=color),
                    xshift=xshift, row=1, col=1,
                )

    def _add_wave(self, fig, wave_pattern):
        pts    = [wp for wp in wave_pattern.points if wp.date is not None]
        dates  = [wp.date  for wp in pts]
        prices = [wp.price for wp in pts]
        labels = [wp.label for wp in pts]
        score  = wave_pattern.score.total

        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode="lines+markers+text",
            line=dict(color=EW_CLR, width=2),
            marker=dict(color=EW_CLR, size=10,
                        line=dict(color="white", width=1)),
            text=labels,
            textposition="top center",
            textfont=dict(size=12, color=EW_CLR, family="Arial Black"),
            name=f"EW Impulse ({score})",
        ), row=1, col=1)

    def _add_time_zones(self, fig, time_zones, dates, daily):
        show_nums = {5, 8, 13, 21}
        price_top = daily["High"].max()
        for zone in time_zones:
            if zone.get("fib") not in show_nums:
                continue
            bar_idx = zone.get("bar", 0)
            if bar_idx >= len(dates):
                continue
            d0 = dates[max(0, bar_idx - 1)]
            d1 = dates[min(len(dates) - 1, bar_idx + 1)]
            fig.add_vrect(
                x0=d0, x1=d1,
                fillcolor=FIB_TIME_FILL, opacity=1.0,
                line_color=FIB_TIME_LINE, line_width=1,
                row=1, col=1,
            )
            fig.add_annotation(
                x=dates[min(bar_idx, len(dates)-1)],
                y=price_top * 0.998,
                text=f"T{zone['fib']}",
                showarrow=False,
                font=dict(size=9, color="#fde047"),
                row=1, col=1,
            )

    def _add_info_box(self, fig, daily, signal, tf_states, fib_levels):
        sig_map = {
            "strong_long":  "🟢 強力做多",
            "long":         "🔵 做多",
            "neutral":      "⚪ 觀望",
            "short":        "🔴 做空",
            "strong_short": "⛔ 強力做空",
        }
        last_close  = daily["Close"].iloc[-1]
        last_date   = daily.index[-1].strftime("%Y-%m-%d")
        prev_close  = daily["Close"].iloc[-2] if len(daily) > 1 else last_close
        chg         = last_close - prev_close
        chg_pct     = chg / prev_close * 100 if prev_close else 0

        sig_text    = sig_map.get(signal.signal.value, "─") if signal else "─"
        score_text  = f"{signal.score:+d}" if signal else "─"
        conf_text   = f"{signal.confidence:.0%}" if signal else "─"

        daily_state = tf_states.get("daily") if tf_states else None
        regime_text = daily_state.regime.value if daily_state else "─"

        recent_high = fib_levels.get("recent_high", "─") if fib_levels else "─"
        recent_low  = fib_levels.get("recent_low",  "─") if fib_levels else "─"

        lin50 = sl50 = "─"
        if fib_levels:
            lin50 = fib_levels.get("linear", {}).get("50.0%", "─")
            sl50  = fib_levels.get("semilog",{}).get("50.0% SL", "─")
            if isinstance(lin50, float): lin50 = f"{lin50:,.0f}"
            if isinstance(sl50,  float): sl50  = f"{sl50:,.0f}"

        dim_lines = []
        if signal and signal.dim_scores:
            for k, v in signal.dim_scores.items():
                dim_lines.append(f"  {k}: {v:+d}")

        lines = [
            f"<b>{self.title}</b>",
            f"最新日期：{last_date}",
            f"收盤：{last_close:,.0f}　{chg:+,.0f}（{chg_pct:+.2f}%）",
            "──────────────────",
            f"五維信號：{sig_text}",
            f"總得分：{score_text} / ±100",
            f"信心度：{conf_text}",
            "──────────────────",
            f"市場狀態：{regime_text}",
            "──────────────────",
            f"Fib 基準",
            f"  高：{recent_high if isinstance(recent_high,str) else f'{recent_high:,.0f}'}",
            f"  低：{recent_low  if isinstance(recent_low, str) else f'{recent_low:,.0f}'}",
            f"  50% 線性：{lin50}",
            f"  50% 半對數：{sl50}",
            "──────────────────",
            "各維得分",
            *dim_lines,
        ]

        fig.add_annotation(
            x=1.0, y=0.99,
            xref="paper", yref="paper",
            text="<br>".join(lines),
            showarrow=False, align="left",
            xanchor="right", yanchor="top",
            bgcolor="rgba(13,17,23,0.88)",
            bordercolor="#30363d",
            borderwidth=1, borderpad=8,
            font=dict(size=10, color=TEXT, family="monospace"),
        )
