"""
update_taiwan_stock_chart.py — 台灣大盤/股票最新 K 線分析圖更新工具

資料來源優先順序：
  1. TWSE 官方 API（大盤 TAIEX：https://www.twse.com.tw/）
  2. yfinance（^TWII）

用法：
  python update_taiwan_stock_chart.py \\
      --symbol TAIEX \\
      --start  2025-04-01 \\
      --output-html TAIEX_latest_analysis.html \\
      --output-csv  TAIEX_latest_daily.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import requests

# 讓 ta_framework 模組可被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swing_engine     import SwingEngine, SwingDir
from fib_price_engine import FibPriceEngine
from fib_time_engine  import FibTimeEngine
from wave_engine      import WaveEngine
from regime_engine    import RegimeEngine
from decision_engine  import DecisionEngine
from visualizer       import TAVisualizer


# ================================================================
# 1. 資料下載
# ================================================================

TWSE_FMTQIK = "https://www.twse.com.tw/exchangeReport/FMTQIK"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.twse.com.tw/",
}


def _fetch_twse_month(year: int, month: int) -> pd.DataFrame | None:
    """
    呼叫 TWSE FMTQIK API 取得指定月份的大盤日資料。
    欄位：日期, 成交股數, 成交金額, 成交筆數, 發行量加權股價指數, 漲跌點數
    """
    date_str = f"{year}{month:02d}01"
    try:
        resp = requests.get(
            TWSE_FMTQIK,
            params={"response": "json", "date": date_str},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return None

    if data.get("stat") != "OK":
        return None

    fields = data.get("fields", [])
    rows   = data.get("data", [])
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=fields)
    return df


def fetch_taiex_twse(start: str, end: str) -> pd.DataFrame | None:
    """
    從 TWSE 官方 API 取得 TAIEX 日收盤資料（start ~ end）。
    回傳 DataFrame（Date, Open, High, Low, Close, Volume）；
    注意 TWSE FMTQIK 僅有收盤指數，Open/High/Low 以收盤代替。
    """
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)

    months = pd.date_range(
        start=start_dt.replace(day=1),
        end=end_dt,
        freq="MS",
    )

    all_rows = []
    for m in months:
        df = _fetch_twse_month(m.year, m.month)
        if df is not None:
            all_rows.append(df)
        time.sleep(0.4)   # 避免觸發 TWSE 速率限制

    if not all_rows:
        return None

    raw = pd.concat(all_rows, ignore_index=True)

    # 欄位清理
    raw.columns = raw.columns.str.strip()
    date_col  = [c for c in raw.columns if "日期" in c]
    close_col = [c for c in raw.columns if "加權" in c or "指數" in c]
    vol_col   = [c for c in raw.columns if "成交股數" in c or "成交量" in c]

    if not date_col or not close_col:
        return None

    date_col  = date_col[0]
    close_col = close_col[0]
    vol_col   = vol_col[0] if vol_col else None

    # 民國年 → 西元年
    def tw_date(s: str) -> pd.Timestamp | None:
        s = str(s).strip().replace(",", "")
        parts = s.split("/")
        if len(parts) == 3:
            try:
                y = int(parts[0]) + 1911
                return pd.Timestamp(f"{y}-{parts[1]}-{parts[2]}")
            except Exception:
                return None
        return None

    raw["Date"]  = raw[date_col].apply(tw_date)
    raw["Close"] = pd.to_numeric(
        raw[close_col].astype(str).str.replace(",",""), errors="coerce"
    )
    if vol_col:
        raw["Volume"] = pd.to_numeric(
            raw[vol_col].astype(str).str.replace(",",""), errors="coerce"
        )
    else:
        raw["Volume"] = 0

    raw = raw.dropna(subset=["Date","Close"])
    raw = raw[(raw["Date"] >= start_dt) & (raw["Date"] <= end_dt)]
    raw = raw.sort_values("Date").drop_duplicates("Date")
    raw = raw.set_index("Date")

    # TWSE FMTQIK 只有收盤，以收盤填充 OHLC（可後續以其他 API 補強）
    df_out = pd.DataFrame({
        "Open":   raw["Close"],
        "High":   raw["Close"],
        "Low":    raw["Close"],
        "Close":  raw["Close"],
        "Volume": raw["Volume"],
    })
    return df_out


def fetch_taiex_yfinance(start: str, end: str) -> pd.DataFrame | None:
    """
    透過 yfinance 取得 ^TWII（TAIEX）OHLCV。
    提供完整 Open/High/Low/Close。
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        ticker = yf.Ticker("^TWII")
        df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
        return df
    except Exception:
        return None


def fetch_data(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """
    主資料取得函數：依優先順序嘗試各來源。
    回傳 (DataFrame, source_name)；若全部失敗則 raise RuntimeError。
    """
    upper = symbol.upper()
    is_taiex = upper in ("TAIEX", "^TWII", "TWII", "加權指數")

    errors = []

    if is_taiex:
        print("  [資料] 嘗試 TWSE 官方 API…")
        df = fetch_taiex_twse(start, end)
        if df is not None and not df.empty:
            return df, "TWSE 官方 API"
        errors.append("TWSE API：無資料或連線失敗")

    print("  [資料] 嘗試 yfinance (^TWII)…")
    yf_symbol = "^TWII" if is_taiex else symbol
    df = fetch_taiex_yfinance(yf_symbol, end if end else "")
    if df is not None and not df.empty:
        # 過濾起始日
        df = df[df.index >= pd.Timestamp(start)]
        if not df.empty:
            return df, "yfinance (^TWII)"
    errors.append("yfinance：無資料或連線失敗")

    raise RuntimeError(
        "所有資料來源均失敗，無法取得資料。\n" +
        "\n".join(f"  • {e}" for e in errors) +
        "\n請確認網路連線或手動提供 CSV 檔案。"
    )


# ================================================================
# 2. 分析流程
# ================================================================

def run_analysis(df: pd.DataFrame, title: str) -> dict:
    """執行五維分析，回傳各引擎結果。"""
    swing_eng  = SwingEngine(swing_days=2, inside_down=True, ignore_threshold=100)
    fib_eng    = FibPriceEngine()
    time_eng   = FibTimeEngine()
    wave_eng   = WaveEngine(peak_order=5, fib_tol=0.10)
    regime_eng = RegimeEngine()
    dec_eng    = DecisionEngine(swing_days=2, peak_order=5, fib_tol=0.10)

    swing_result = swing_eng.calculate_swings(df)
    pivots_df    = swing_eng.pivots_to_dataframe(swing_result.pivots)
    signal       = dec_eng.analyze(df)
    tf_states    = regime_eng.classify_multi_tf(df)

    # Fib 水準
    from swing_engine import SwingDir as SD
    highs = [p for p in swing_result.pivots if p.direction == SD.UP]
    lows  = [p for p in swing_result.pivots if p.direction == SD.DOWN]
    recent_high = highs[-1].price if highs else df["High"].max()
    recent_low  = lows[-1].price  if lows  else df["Low"].min()

    lin_fibs = fib_eng.calc_retracement(recent_high, recent_low)
    sl_fibs  = fib_eng.calc_semilog_retracement(recent_high, recent_low)

    fib_levels = {
        "recent_high": recent_high,
        "recent_low":  recent_low,
        "linear":  {lv.label: lv.price for lv in lin_fibs},
        "semilog": {lv.label: lv.price for lv in sl_fibs},
    }

    # Wave
    wave_pivots = wave_eng.detect_pivots(df, order=5)
    up_patterns = wave_eng.find_impulse_waves(wave_pivots, direction="up")
    best_wave   = up_patterns[0] if up_patterns else None

    # Time zones
    last_pivot = swing_result.pivots[-1] if swing_result.pivots else None
    time_zones = []
    if last_pivot:
        raw_zones = time_eng.fibonacci_time_zones(
            last_pivot.index, dates=df.index, max_zones=10
        )
        time_zones = [
            {"fib": z.fib_number, "bar": z.bar_index,
             "date": str(z.date_approx)[:10] if z.date_approx else None}
            for z in raw_zones
        ]

    return dict(
        swing_result=swing_result,
        best_wave=best_wave,
        fib_levels=fib_levels,
        time_zones=time_zones,
        signal=signal,
        tf_states=tf_states,
    )


# ================================================================
# 3. CLI 主程式
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="台灣大盤/股票 K 線分析圖更新工具"
    )
    parser.add_argument("--symbol",      default="TAIEX",
                        help="代號（TAIEX / ^TWII / 股票代號）")
    parser.add_argument("--start",       default="2025-01-01",
                        help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end",         default=None,
                        help="結束日期 YYYY-MM-DD（預設今天）")
    parser.add_argument("--output-html", default="TAIEX_latest_analysis.html",
                        dest="output_html")
    parser.add_argument("--output-csv",  default="TAIEX_latest_daily.csv",
                        dest="output_csv")
    args = parser.parse_args()

    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(out_dir, args.output_html)
    csv_path  = os.path.join(out_dir, args.output_csv)

    print(f"\n{'='*55}")
    print(f"  台灣大盤分析圖更新")
    print(f"  Symbol : {args.symbol}")
    print(f"  Period : {args.start} → {end_date}")
    print(f"{'='*55}\n")

    # ── 取得資料 ────────────────────────────────────────────────────
    try:
        df, source = fetch_data(args.symbol, args.start, end_date)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    latest_date  = df.index[-1].strftime("%Y-%m-%d")
    latest_close = df["Close"].iloc[-1]
    print(f"\n  [成功] 來源：{source}")
    print(f"  資料筆數  ：{len(df)} 筆")
    print(f"  最新日期  ：{latest_date}")
    print(f"  最新收盤  ：{latest_close:,.2f}")

    # ── 儲存 CSV ────────────────────────────────────────────────────
    df.to_csv(csv_path)
    print(f"\n  [CSV] {csv_path}")

    # ── 執行分析 ────────────────────────────────────────────────────
    print("\n  執行五維分析…")
    results = run_analysis(df, title=f"{args.symbol} 日線分析｜五維決策")
    sig     = results["signal"]

    sig_map = {
        "strong_long":  "🟢 強力做多",
        "long":         "🔵 做多",
        "neutral":      "⚪ 觀望",
        "short":        "🔴 做空",
        "strong_short": "⛔ 強力做空",
    }
    print(f"  五維信號  ：{sig_map.get(sig.signal.value, sig.signal.value)}")
    print(f"  總得分    ：{sig.score:+d}")

    # ── 產生圖表 ────────────────────────────────────────────────────
    print(f"\n  產生 K 線分析圖…")
    viz = TAVisualizer(
        title=f"{args.symbol} 日線分析｜五維決策框架",
        embed_plotlyjs=True,
    )
    viz.plot_full_analysis(
        daily        = df,
        swing_result = results["swing_result"],
        wave_pattern = results["best_wave"],
        fib_levels   = results["fib_levels"],
        time_zones   = results["time_zones"],
        signal       = results["signal"],
        tf_states    = results["tf_states"],
        output_html  = html_path,
    )
    print(f"  [HTML] {html_path}")

    print(f"\n{'='*55}")
    print(f"  完成！")
    print(f"  最新資料日期  : {latest_date}")
    print(f"  最新收盤      : {latest_close:,.2f}")
    print(f"  CSV 路徑      : {csv_path}")
    print(f"  HTML 路徑     : {html_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
