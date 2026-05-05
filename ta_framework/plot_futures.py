"""
plot_futures.py — 台指期貨夜盤 K 線分析（日線 / 60分 / 15分）
資料來源：Excel（TXFPM1 格式，三個工作表：day / 60min / 15min）
執行：python plot_futures.py [excel_path]
輸出：txf_futures_day.html / txf_futures_60min.html / txf_futures_15min.html
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from plot_analysis import build_chart

EXCEL_PATH = (
    sys.argv[1] if len(sys.argv) > 1 else
    "/root/.claude/uploads/b1864915-4e54-4115-aea1-93d08cc1e722/59d7d0b0-____.xlsx"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_sheet(path: str, sheet: str) -> pd.DataFrame:
    """
    載入 TXFPM1 Excel 工作表。
    Row 0：metadata（TXFPM1 台指(全)近 ...）
    Row 1：欄位標題（時間 開盤 最高 最低 收盤 成交量 成交額）
    Row 2+：K 線資料
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=2)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main():
    print(f"讀取：{EXCEL_PATH}")

    day_df = load_sheet(EXCEL_PATH, "day")
    m60_df = load_sheet(EXCEL_PATH, "60min")
    m15_df = load_sheet(EXCEL_PATH, "15min")

    print(f"日線 ：{len(day_df):>4} 筆  {day_df.index[0].date()} ~ {day_df.index[-1].date()}")
    print(f"60分 ：{len(m60_df):>4} 筆  {m60_df.index[0]} ~ {m60_df.index[-1]}")
    print(f"15分 ：{len(m15_df):>4} 筆  {m15_df.index[0]} ~ {m15_df.index[-1]}")

    RB_DAILY = [dict(bounds=["sat", "mon"])]

    # 60分：早盤空白 05:01–09:44，午後空白 13:46–15:59
    RB_60MIN = [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[5.01, 9.74],  pattern="hour"),
        dict(bounds=[13.76, 15.99], pattern="hour"),
    ]

    # 15分：早盤空白 05:01–08:59，午後空白 13:46–15:14
    RB_15MIN = [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[5.01, 8.99],  pattern="hour"),
        dict(bounds=[13.76, 15.24], pattern="hour"),
    ]

    # ── 日線（9 年資料，完整分析）────────────────────────────────────
    build_chart(
        day_df, "日線（期貨）",
        os.path.join(OUT_DIR, "txf_futures_day.html"),
        swing_days=2, ignore_thresh=200, peak_order=5,
        show_fib_time=True, rangebreaks=RB_DAILY,
    )

    # ── 60 分線（日內，排除非交易時段）──────────────────────────────
    build_chart(
        m60_df, "60分線（期貨）",
        os.path.join(OUT_DIR, "txf_futures_60min.html"),
        swing_days=2, ignore_thresh=100, peak_order=5,
        show_fib_time=False, rangebreaks=RB_60MIN,
    )

    # ── 15 分線（日內，排除非交易時段）──────────────────────────────
    build_chart(
        m15_df, "15分線（期貨）",
        os.path.join(OUT_DIR, "txf_futures_15min.html"),
        swing_days=2, ignore_thresh=50, peak_order=8,
        show_fib_time=False, rangebreaks=RB_15MIN,
    )


if __name__ == "__main__":
    main()
