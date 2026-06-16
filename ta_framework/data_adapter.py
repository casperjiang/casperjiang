"""
data_adapter.py — Taiwan Futures (TXFPM1) Data Loader
Loads OHLCV data from local Excel files exported from trading platforms.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path


EXPECTED_COLS = {
    "時間": "Date",
    "開盤": "Open",
    "最高": "High",
    "最低": "Low",
    "收盤": "Close",
    "成交量": "Volume",
    "成交額": "Turnover",
}


def load_txf_excel(
    filepath: str | Path,
    sheet: str = "day",
    set_index: bool = True,
) -> pd.DataFrame:
    """
    Load Taiwan Futures OHLCV data from Excel.
    Supports: day / 60min / 15min sheets.

    Parameters
    ----------
    filepath : path to the .xlsx file
    sheet    : sheet name ('day', '60min', '15min')
    set_index: if True, set Date as DatetimeIndex

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    df = pd.read_excel(filepath, sheet_name=sheet, header=1)
    df = df.rename(columns=EXPECTED_COLS)

    # Drop rows without a valid date
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])

    # Keep only OHLCV
    keep = [c for c in ["Date","Open","High","Low","Close","Volume","Turnover"]
            if c in df.columns]
    df = df[keep]

    if set_index:
        df = df.set_index("Date").sort_index()

    return df


def load_all_timeframes(filepath: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all available timeframes from the Excel file.
    Returns dict keyed by sheet name.
    """
    import openpyxl
    wb = openpyxl.load_workbook(filepath, read_only=True)
    sheets = wb.sheetnames
    wb.close()

    result = {}
    for sheet in sheets:
        try:
            result[sheet] = load_txf_excel(filepath, sheet=sheet)
        except Exception as e:
            print(f"Warning: could not load sheet '{sheet}': {e}")
    return result
