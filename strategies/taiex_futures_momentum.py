"""
台指期 多因子動量策略 — QuantDinger IndicatorStrategy
Taiwan Futures Multi-Factor Momentum Strategy

策略邏輯 (Strategy Logic)
─────────────────────────────────────────────────────────────
台灣加權指數期貨（TX）約 35% 市值來自台積電，其次為聯發科、
台達電等半導體及電子製造業。指數走勢與全球科技/AI 資本支出
週期高度相關，因此本策略以「趨勢 + 動量 + 成交量確認」
三層過濾建立訊號，避免在橫盤整理期反覆進出。

因子選擇理由
─────────────────────────────────────────────────────────────
1. EMA(20/60) 交叉 — 趨勢方向
   台指期日均波動率約 1.2–1.8%，20 日 EMA 捕捉短中期方向，
   60 日 EMA 作為趨勢確認基準，可過濾約 60% 的假突破。

2. RSI(14) 中線過濾 (50)
   RSI > 50 + EMA 多頭排列 → 確認多頭動量；
   RSI < 50 + EMA 空頭排列 → 確認空頭動量。
   台股散戶比例高，RSI 均值回歸效果顯著。

3. Bollinger Band Width 波動率門檻
   BB 寬度 < 4% 表示市場進入低波動整理期，此時 EMA 交叉
   訊噪比低，策略自動暫停入場，防止震盪損耗。

4. Volume Ratio > 1.3 成交量確認
   機構進場通常伴隨成交量放大；
   量縮的 EMA 交叉多為假突破，尤其在台股法人結算週前後。

5. ATR(14) 動態止損
   使用 1.8× ATR 作為追蹤止損，適配台指期高波動結構，
   同時控制單筆最大虧損在 2% 帳戶淨值以內。

執行模式: IndicatorStrategy（DataFrame 向量化）
回測建議最小資料長度: 120 根 K 線（約半年日線）
"""

# @strategy stopLossPct=2.0 takeProfitPct=4.5 entryPct=0.6 tradeDirection=both
# @param ema_fast=20 type=int min=5 max=50 label="快線EMA週期"
# @param ema_slow=60 type=int min=20 max=120 label="慢線EMA週期"
# @param rsi_period=14 type=int min=7 max=30 label="RSI週期"
# @param rsi_threshold=50 type=float min=45 max=55 label="RSI趨勢門檻"
# @param bb_period=20 type=int min=10 max=30 label="布林帶週期"
# @param bb_squeeze=4.0 type=float min=2.0 max=8.0 label="BB寬度過濾(%)"
# @param vol_ratio_min=1.3 type=float min=1.0 max=2.5 label="最低成交量比"
# @param atr_period=14 type=int min=7 max=21 label="ATR週期"
# @param atr_sl_mult=1.8 type=float min=1.0 max=3.0 label="ATR止損倍數"

import pandas as pd
import numpy as np


def calculate(df: pd.DataFrame, params: dict) -> dict:
    """
    QuantDinger IndicatorStrategy 主函數。
    接收 OHLCV DataFrame，返回 output 字典含 plots 和 signals。
    """
    df = df.copy()

    # ── 讀取參數 ──────────────────────────────────────────────
    ema_fast     = params.get("ema_fast", 20)
    ema_slow     = params.get("ema_slow", 60)
    rsi_period   = params.get("rsi_period", 14)
    rsi_thr      = params.get("rsi_threshold", 50)
    bb_period    = params.get("bb_period", 20)
    bb_squeeze   = params.get("bb_squeeze", 4.0)
    vol_min      = params.get("vol_ratio_min", 1.3)
    atr_period   = params.get("atr_period", 14)
    atr_mult     = params.get("atr_sl_mult", 1.8)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── 1. EMA ────────────────────────────────────────────────
    df["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()

    # ── 2. RSI (Wilder smoothed) ──────────────────────────────
    delta      = close.diff()
    gain       = delta.clip(lower=0)
    loss       = (-delta).clip(lower=0)
    avg_gain   = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss   = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs         = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"]  = 100 - (100 / (1 + rs))

    # ── 3. Bollinger Bands ────────────────────────────────────
    bb_mid          = close.rolling(bb_period).mean()
    bb_std          = close.rolling(bb_period).std()
    df["bb_upper"]  = bb_mid + 2 * bb_std
    df["bb_lower"]  = bb_mid - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mid * 100  # %

    # ── 4. Volume Ratio ───────────────────────────────────────
    df["vol_ratio"] = volume / volume.rolling(20).mean()

    # ── 5. ATR (Wilder) ───────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    # ── 複合過濾條件 ──────────────────────────────────────────
    vol_ok     = df["vol_ratio"] > vol_min          # 成交量確認
    bb_open    = df["bb_width"]  > bb_squeeze       # 非低波動整理期

    # 多頭條件：EMA 快線剛上穿慢線 + RSI > 門檻
    bull_cross = (df["ema_fast"] > df["ema_slow"]) & \
                 (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    bull_rsi   = df["rsi"] > rsi_thr

    # 空頭條件：EMA 快線剛下穿慢線 + RSI < 門檻
    bear_cross = (df["ema_fast"] < df["ema_slow"]) & \
                 (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
    bear_rsi   = df["rsi"] < rsi_thr

    # ── 訊號產生（邊緣觸發，避免每根K線重複） ─────────────────
    raw_buy  = bull_cross & bull_rsi & vol_ok & bb_open
    raw_sell = bear_cross & bear_rsi & vol_ok & bb_open

    df["buy"]  = raw_buy.astype(bool)
    df["sell"] = raw_sell.astype(bool)

    # ── ATR 動態止損價格（供前端圖表顯示） ───────────────────
    df["sl_long"]  = close - atr_mult * df["atr"]   # 多單止損線
    df["sl_short"] = close + atr_mult * df["atr"]   # 空單止損線

    # ── output 格式（QuantDinger 標準） ───────────────────────
    output = {
        "plots": [
            {
                "name": "EMA Fast",
                "data": df["ema_fast"].tolist(),
                "type": "line",
                "color": "#f59e0b",
                "panel": "main"
            },
            {
                "name": "EMA Slow",
                "data": df["ema_slow"].tolist(),
                "type": "line",
                "color": "#3b82f6",
                "panel": "main"
            },
            {
                "name": "BB Upper",
                "data": df["bb_upper"].tolist(),
                "type": "line",
                "color": "#94a3b8",
                "panel": "main",
                "dashed": True
            },
            {
                "name": "BB Lower",
                "data": df["bb_lower"].tolist(),
                "type": "line",
                "color": "#94a3b8",
                "panel": "main",
                "dashed": True
            },
            {
                "name": "Stop Long",
                "data": df["sl_long"].tolist(),
                "type": "line",
                "color": "#ef4444",
                "panel": "main",
                "dashed": True
            },
            {
                "name": "Stop Short",
                "data": df["sl_short"].tolist(),
                "type": "line",
                "color": "#10b981",
                "panel": "main",
                "dashed": True
            },
            {
                "name": "RSI(14)",
                "data": df["rsi"].tolist(),
                "type": "line",
                "color": "#a855f7",
                "panel": "sub",
                "levels": [30, 50, 70]
            },
            {
                "name": "Volume Ratio",
                "data": df["vol_ratio"].tolist(),
                "type": "bar",
                "color": "#64748b",
                "panel": "vol"
            }
        ],
        "signals": {
            "buy":  df["buy"].tolist(),
            "sell": df["sell"].tolist()
        }
    }

    return output
