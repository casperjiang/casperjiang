# 技術分析研究框架：六大 Repo 深度研究報告

> **研究目標**：評估並比較六個開源技術分析框架，以建立「台指期五維操盤決策模型」基礎架構。  
> **核心需求**：Elliott Wave、半對數 Fibonacci 價格回測/延伸、Fibonacci 時間週期、Gann Swing、Pivot 偵測。  
> **研究日期**：2026-05-03

---

## 目錄

1. [Repo 個別深度分析](#一repo-個別深度分析)
   - 1.1 python-taew
   - 1.2 elliot-waves-auto
   - 1.3 fibonacci_ml
   - 1.4 Stock-market
   - 1.5 gann-swing
   - 1.6 pricechartingtool
2. [六大 Repo 功能矩陣對比](#二六大-repo-功能矩陣對比)
3. [綜合評分表](#三綜合評分表)
4. [整合架構設計](#四整合架構設計)
5. [模組實作建議](#五模組實作建議)
6. [結論與下一步行動](#六結論與下一步行動)

---

## 一、Repo 個別深度分析

---

### 1.1 python-taew

**來源**：`https://github.com/DrEdwardPCB/python-taew`  
**本地路徑**：`/home/user/ta_research/python-taew/`  
**語言**：Python 3.7+  
**核心檔案**：`taew/ew.py`（991 行）

#### 檔案結構

```
python-taew/
├── readme.md
├── setup.py
└── taew/
    ├── __init__.py
    └── ew.py          ← 全部核心邏輯
```

#### 核心演算法

**Fibonacci 驗證函數**（`ew.py:8-75`）

```python
# Wave 2：回撤 Wave 1 的 14.6%–85.4%
def wave2_fibonacci_check(wave1, wave2):
    ratio = abs(wave2) / abs(wave1)
    return any(abs(ratio - r) < 0.005 for r in [0.146,0.236,0.382,0.5,0.618,0.764,0.854])

# Wave 3：延伸 Wave 1 的 1.236x–4.236x
def wave3_fibonacci_check(wave1, wave3):
    ratio = abs(wave3) / abs(wave1)
    return any(abs(ratio - r) < 0.01 for r in [1.236,1.382,1.5,1.618,2.0,2.618,3.618,4.236])
```

**四種波浪標記方法**

| 方法 | 說明 | 時間週期約束 |
|------|------|------------|
| `Alternative_ElliottWave_label_upward()` | 上升波（含 Fib 時間週期） | ✓ 有 |
| `Alternative_ElliottWave_label_downward()` | 下降波（含 Fib 時間週期） | ✓ 有 |
| `Traditional_ElliottWave_label_upward()` | 上升波（僅 Fib 價格） | ✗ 無 |
| `Traditional_ElliottWave_label_downward()` | 下降波（僅 Fib 價格） | ✗ 無 |

**時間週期實作**（`ew.py:159,180,211,236`）
- Wave 3 時間長度 = Wave 1 × 0.4011x 至 1.6989x
- Wave 5 時間長度 = Wave 1 × 0.4011x 至 1.6989x

#### 功能清單

| 功能 | 狀態 | 備註 |
|------|------|------|
| Elliott Wave 標記（1-2-3-4-5） | ✅ 完整 | 上下雙向均有 |
| 衝擊波（Impulse）偵測 | ✅ 完整 | 5 波架構 |
| 修正波（ABC Correction） | ⚠️ 隱含 | 無明顯 ABC 標記 |
| Fibonacci 回撤 | ✅ 7 個水準 | 14.6%–85.4% |
| Fibonacci 延伸 | ✅ 8 個倍數 | 1.236x–4.236x |
| Fibonacci 時間週期 | ✅ 有實作 | Alternative 模式 |
| 半對數價格計算 | ❌ 無 | — |
| Gann Swing | ❌ 無 | — |
| Pivot 偵測 | ❌ 無 | 使用局部極值 |
| 回測框架 | ❌ 無 | — |
| 圖表視覺化 | ❌ 無 | — |

#### 代碼品質觀察

**優點**：
- 可透過 `pip install taew` 直接安裝
- 零外部依賴（僅 numpy + pandas）
- 數學邏輯嚴謹，Fibonacci 比值驗證清晰
- Alternative 方法包含時間週期約束（業界少見）

**缺點**：
- 大量 `print()` debug 輸出未清理
- 變數命名混亂（`x`, `z`, `b`, `v`, `j`）
- 無錯誤處理，輸入格式需自行保證
- 輸出為 list of dict，缺乏結構化物件
- Practical 預測方法（wave 3/4/5 預測）實作但無文件

#### 評分

| 項目 | 分數 |
|------|------|
| 可用性 | **7/10** |
| 程式碼品質 | **5/10** |
| 適合改造台指期 | **是**（波浪核心可直接用） |
| 最大缺點 | 無 ABC 修正波明確標記、無視覺化 |
| **最適合抽取模組** | `wave2/3/4/5_fibonacci_check()` + `Alternative_ElliottWave_label_upward/downward()` |

---

### 1.2 elliot-waves-auto

**來源**：`https://github.com/ESJavadex/elliot-waves-auto`  
**本地路徑**：`/home/user/ta_research/elliot-waves-auto/`  
**語言**：Python 3.10+  
**核心檔案**：6 個模組 + Flask App（合計 ~7,561 行）

#### 檔案結構

```
elliot-waves-auto/
├── app_v5_automated.py      ← Flask Web 應用（4,129 行）
├── improved_elliott.py      ← 多準則評分系統（510 行）
├── pattern_recognition.py   ← EW 規則驗證（575 行）
├── fibonacci_analysis.py    ← Fibonacci 計算（495 行）
├── trade_signals.py         ← 交易信號生成（418 行）
├── super_strategy.py        ← 整合策略（1,153 行）
├── requirements.txt
├── Dockerfile
└── templates/               ← Web 界面
```

#### 核心演算法

**Elliott Wave 嚴格規則驗證**（`pattern_recognition.py:36-209`）

```python
def identify_impulse_wave(waves):
    # Rule 1: Wave 2 永遠不回撤超過 Wave 1 的 100%
    if wave2_retrace > 1.0: return False
    # Rule 2: Wave 3 永遠不是最短波
    if wave3 < wave1 and wave3 < wave5: return False
    # Rule 3: Wave 4 不進入 Wave 1 的價格區間
    if wave4_low < wave1_high: return False
```

**多準則計分系統**（`improved_elliott.py:101-335`）

```
嚴格規則（各 100 分）：Rule1 + Rule2 + Rule3
指引準則（各 15 分）：Wave3最長、Wave5 Fib相關、交替原則
技術指標（5-30 分）：量能、RSI 背離、趨勢通道
```

**Fibonacci 群集分析**（`fibonacci_analysis.py:66-124`）
- 計算多個波段的回撤/延伸後，找出密集重疊的支撐/阻力區

**交易信號生成**（`trade_signals.py:25-150`）
- 入場點、停損、三個停利目標（1:1, 1.618:1, 2.618:1 RR）
- 停損預設 = 1.5 × ATR

#### 功能清單

| 功能 | 狀態 | 備註 |
|------|------|------|
| Elliott Wave 標記（1-5） | ✅ 完整 | 含三規則嚴格驗證 |
| 衝擊波（Impulse） | ✅ 完整 | — |
| 修正波 ABC | ✅ 完整 | Zigzag/Flat/Triangle |
| Fibonacci 回撤 | ✅ 6 水準 | 23.6%–88.6% |
| Fibonacci 延伸 | ✅ 8 倍數 | 1.0x–4.236x |
| Fibonacci 時間週期 | ❌ 無 | — |
| 半對數價格計算 | ❌ 無 | — |
| Gann Swing | ❌ 無 | — |
| Pivot 偵測 | ✅ 有 | scipy.signal.find_peaks |
| 回測框架 | ✅ 有 | run_backtest_simulation |
| 圖表視覺化 | ✅ 完整 | Plotly 互動圖表 |
| RSI 背離偵測 | ✅ 有 | W3 vs W5 |
| 交易信號 | ✅ 完整 | SL/TP/RR |

#### 代碼品質觀察

**優點**：
- 最完整的 EW 實作，含三規則驗證 + 指引準則
- Plotly 視覺化 + Flask Web 界面
- Docker 部署支援
- 有回測功能
- RSI 背離確認（對台指期高度有用）

**缺點**：
- 缺少 Fibonacci 時間週期
- 程式碼分散，fix_yfinance_ratelimit.py 是空白檔
- Flask Web App 難以嵌入自定義流水線
- 依賴 yfinance 取資料（台指期需另外接資料源）
- 部分 tolerance 硬編碼（10%/3%），難以調校

#### 評分

| 項目 | 分數 |
|------|------|
| 可用性 | **8/10** |
| 程式碼品質 | **7/10** |
| 適合改造台指期 | **是**（需替換資料源，抽取核心模組） |
| 最大缺點 | 無 Fib 時間週期；與資料源耦合太強 |
| **最適合抽取模組** | `pattern_recognition.py` + `fibonacci_analysis.py` + `trade_signals.py` |

---

### 1.3 fibonacci_ml

**來源**：`https://github.com/faraway1nspace/fibonacci_ml`  
**本地路徑**：`/home/user/ta_research/fibonacci_ml/`  
**語言**：Python 3.6+  
**核心檔案**：`core.py`（945 行）、`fib_utils.py`（432 行）

#### 檔案結構

```
fibonacci_ml/
├── readme.md
├── core.py                          ← 主框架（945 行）
├── fib_utils.py                     ← Fibonacci 工具（432 行）
├── variables.py                     ← 設定（48 行）
├── subjective_drawdown_finder.py    ← ML 回撤偵測（328 行）
├── demo_fibonacci_ml.py             ← 示範腳本（21 行）
└── subjective_drawdown_models/
    ├── subjective_drawdown_model1.pkl
    └── subjective_drawdown_model2.pkl
```

#### 核心演算法

**三重記憶架構**（`core.py:528-734`）

```
Memory 1（短期）：最近一次回撤的 Fibonacci 水準
Memory 2（中期）：前一次回撤的 Fibonacci 水準
Memory 3（長期）：跨越數十年的主要回撤（5 個加權準則）
```

**Fibonacci 水準**（`variables.py`）
```python
FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786,
              1, 1.618, 2.618, 4.236, 6.854, 11.09, 17.944]
```
注意：包含超長期延伸（11.09x, 17.944x），適合長線週期研究。

**半對數 Fibonacci 特徵工程**（`core.py:836-842`）

```python
# 將 Fibonacci 水準轉為對數空間的特徵
fib_lev_d = np.log(fib_level + epsilon)
# 價格在 Fibonacci 區間內的相對位置 [0, 1]
box01_d = (price - lower_fib) / (upper_fib - lower_fib)
```

**ML 自動優化回撤門檻**（`subjective_drawdown_finder.py:97-263`）
- 使用 `DecisionTreeRegressor` 決定「什麼幅度才算顯著回撤」
- 兩個預訓練 pickle 模型（model1: 初始門檻預測; model2: 密度精煉）

#### 功能清單

| 功能 | 狀態 | 備註 |
|------|------|------|
| Elliott Wave 標記 | ❌ 無 | 只有回撤，無波浪計數 |
| Fibonacci 回撤 | ✅ 完整 | 自動識別 + 13 個水準 |
| Fibonacci 延伸 | ✅ 完整 | 1.618x–17.944x |
| Fibonacci 時間週期 | ⚠️ 隱含 | 追蹤回撤時長，非主動時間週期 |
| 半對數特徵 | ✅ 完整 | log transform + box01 位置 |
| Gann Swing | ❌ 無 | — |
| Pivot 偵測 | ✅ 有 | 累積高低點追蹤 |
| 回測框架 | ⚠️ 部分 | ML 特徵驗證，非交易回測 |
| 圖表視覺化 | ✅ 有 | matplotlib |
| ML 特徵工程 | ✅ 完整 | 最獨特功能 |

#### 代碼品質觀察

**優點**：
- 三重記憶架構設計獨特，適合多週期分析（日線/週線/月線同步）
- 嚴格的無未來偏見（lookahead-free）設計
- 半對數轉換概念正確，對台指期長線分析有用
- 有預訓練模型，可直接使用

**缺點**：
- 無 setup.py / requirements.txt，安裝麻煩
- Pickle 模型版本依賴 scikit-learn 版本（風險）
- 某些函數使用未定義的 `ticker` 變數（疑似 bug）
- 無 Elliott Wave，需要自行搭配

#### 評分

| 項目 | 分數 |
|------|------|
| 可用性 | **5/10** |
| 程式碼品質 | **7/10** |
| 適合改造台指期 | **部分**（半對數 + 多記憶架構可借鑒） |
| 最大缺點 | 無 EW 波浪計數；安裝困難；無時間週期主動分析 |
| **最適合抽取模組** | 三重記憶架構設計理念 + `fib_utils.py` 的 Fib 計算邏輯 |

---

### 1.4 Stock-market

**來源**：`https://github.com/Louisli0515/Stock-market`  
**本地路徑**：`/home/user/ta_research/Stock-market/`  
**語言**：Python（純數學計算）  
**核心檔案**：`wave.py`（77 行）、`high_low.py`（16 行）

#### 檔案結構

```
Stock-market/
├── README.md      ← 詳細理論說明（最有價值部分）
├── wave.py        ← 波浪價格計算（77 行）
└── high_low.py    ← 半對數幾何平均價格（16 行）
```

#### 核心演算法

**半對數幾何平均價格計算**（`high_low.py:4-12`）

```python
def high_low(x, y):
    p1 = x**(0.125) * y**(0.875)   # 12.5/87.5 Fibonacci 加權
    p3 = x**(0.382) * y**(0.618)   # 38.2/61.8 黃金比例加權
    p4 = x**(0.5)   * y**(0.5)     # 50/50 幾何中點
    p5 = x**(0.618) * y**(0.382)   # 61.8/38.2 加權
    p7 = x**(0.875) * y**(0.125)   # 87.5/12.5 加權
    return p1, p3, p4, p5, p7
```

**半對數波浪延伸公式**（`wave.py:3-32`）

```python
def wave_negcor(x, y):   # 修正波（下跌）
    nec4  = p1 * ((1 - abs(pe))**0.5)     # 50% 回撤對應水準
    nec5  = p1 * ((1 - abs(pe))**0.618)   # 61.8% 回撤
    nec12 = p1 * ((1 - abs(pe))**1.618)   # 161.8% 延伸

def wave_poscor(x, y):   # 衝擊波（上漲）
    poc4  = p1 * ((1 + pe)**0.618)        # 61.8% 延伸
    poc11 = p1 * ((1 + pe)**1.618)        # 161.8%
    poc13 = p1 * ((1 + pe)**2.618)        # 261.8%
```

> **重要發現**：這是六個 repo 中**唯一真正實作半對數 Fibonacci 價格計算**的框架。指數加權幾何平均的概念（`x^a * y^(1-a)`）在半對數座標軸上等同線性插值，這正是台指期長線分析需要的核心數學。

#### 功能清單

| 功能 | 狀態 | 備註 |
|------|------|------|
| 半對數 Fibonacci 計算 | ✅ **唯一完整實作** | 指數加權幾何平均 |
| Fibonacci 回撤（半對數） | ✅ 完整 | 38.2%, 50%, 61.8% |
| Fibonacci 延伸（半對數） | ✅ 完整 | 161.8%, 261.8% |
| Elliott Wave 標記 | ⚠️ 有公式、無自動化 | 需手動輸入 |
| ABC 修正波 | ⚠️ 有公式、無自動化 | — |
| Gann Swing | ❌ 無 | — |
| Pivot 偵測 | ❌ 無 | — |
| 回測 | ❌ 無 | — |
| 視覺化 | ❌ 無 | — |

#### 代碼品質觀察

**優點**：
- 半對數幾何平均概念在學術上完全正確
- README 理論說明詳盡（Elliott Wave 規則、三角形型態等）
- 零外部依賴

**缺點**：
- **嚴重 Bug**：`wave.py` 函數內使用了未在函數作用域定義的 `p1`, `pe` 變數，會立即 `NameError`
- 完全手動輸入，無自動化
- 僅 93 行，缺乏完整框架
- 未完成（README 提到 Bear 市場但未實作）

#### 評分

| 項目 | 分數 |
|------|------|
| 可用性 | **2/10** |
| 程式碼品質 | **2/10** |
| 適合改造台指期 | **部分**（半對數數學公式有參考價值） |
| 最大缺點 | 程式碼有 Bug 無法執行；完全手動 |
| **最適合抽取模組** | `high_low.py` 的半對數幾何平均公式（修正 bug 後） |

---

### 1.5 gann-swing

**來源**：`https://github.com/monch1962/gann-swing`  
**本地路徑**：`/home/user/ta_research/gann-swing/`  
**語言**：Python 3.6+  
**核心檔案**：`gannswing.py`（184 行）

#### 檔案結構

```
gann-swing/
├── README.md
├── setup.py
├── requirements.txt
├── gannswing.py       ← 核心 Gann swing 邏輯（184 行）
├── __init__.py
└── tests/
    ├── test_day_types.py
    ├── test_swings.py
    ├── test_calculate_parameters.py
    └── data/optuma.csv
```

#### 核心演算法

**K 線類型分類**（`gannswing.py:72-122`）

```python
def _up_day(self, bar):
    return bar['High'] > self.prev_bar['High'] and bar['Low'] >= self.prev_bar['Low']

def _down_day(self, bar):
    return bar['Low'] < self.prev_bar['Low'] and bar['High'] <= self.prev_bar['High']

def _inside_day(self, bar):
    return bar['High'] <= self.prev_bar['High'] and bar['Low'] >= self.prev_bar['Low']

def _outside_day(self, bar):
    return bar['High'] > self.prev_bar['High'] and bar['Low'] < self.prev_bar['Low']
```

**Gann Swing 計算**（`gannswing.py:62-70`）

```python
def calculate_swings(self, swing_days=1, inside_down=False,
                     ignore_threshold=0, use_close_of_outside_bar=False):
    # swing_days=1: 1-bar reversal (最敏感)
    # swing_days=2: 2-bar reversal (標準 Gann swing)
    # inside_down=True: 內縮日視為向下
```

**Tick Size 自動計算**（`gannswing.py:160-181`）
- 分析最近 20 根 K 線，自動找出最小價格間隔

#### 功能清單

| 功能 | 狀態 | 備註 |
|------|------|------|
| Gann Swing 偵測 | ✅ **完整** | N-bar reversal 可設定 |
| K 線類型分類 | ✅ 完整 | Up/Down/Inside/Outside day |
| Pivot 偵測 | ✅ 有 | Swing 轉折點即為 Pivot |
| Fibonacci | ❌ 無 | — |
| Elliott Wave | ❌ 無 | — |
| 半對數計算 | ❌ 無 | — |
| 回測 | ❌ 無 | — |
| 視覺化 | ⚠️ 部分 | Plotly OHLC（波浪覆蓋未實作） |
| 單元測試 | ✅ 有 | 7 個測試檔案 |

#### 代碼品質觀察

**優點**：
- 六個 repo 中**唯一有完整單元測試**的
- 介面清晰，輸入為標準 pandas DataFrame（OHLC）
- `swing_days` 參數可調整靈敏度（台指期可設 2-bar 或 3-bar）
- Inside day / Outside day 處理邏輯符合 Gann 原著

**缺點**：
- 多個進階功能標記為 TODO（`inside_down`, `ignore_threshold`）
- 未發布至 PyPI（需本地 setup.py 安裝）
- 不包含 Fibonacci 或 EW

#### 評分

| 項目 | 分數 |
|------|------|
| 可用性 | **8/10** |
| 程式碼品質 | **8/10** |
| 適合改造台指期 | **是**（直接可用的 Swing 模組） |
| 最大缺點 | 僅 Swing 功能；多項進階功能未完成 |
| **最適合抽取模組** | 整個 `gannswing.py`，特別是 `_up_day/_down_day/_inside_day/_outside_day()` |

---

### 1.6 pricechartingtool

**來源**：`https://github.com/rluu/pricechartingtool`  
**本地路徑**：`/home/user/ta_research/pricechartingtool/`  
**語言**：Python 3.4+ + PyQt5  
**規模**：284,847 行 / 131 個 Python 檔案

#### 檔案結構（精選）

```
pricechartingtool/
├── src/
│   ├── main.py                  ← Qt5 主程式（5,652 行）
│   ├── pricebarchart.py         ← 圖表引擎（55,540 行）
│   ├── data_objects.py          ← 49 種 TA 物件（23,199 行）
│   ├── ephemeris.py             ← 星曆計算（天文週期）
│   ├── lookbackmultiple_calc.py ← 週期搜尋（98K bytes）
│   └── mastercharts.py          ← Gann Square of 9（TODO）
└── misc/
    ├── SwingFileScripts/        ← Swing pivot 提取腳本
    ├── SqOf9SpiralChartScripts/ ← 九方格計算
    └── EphemerisGeneration/     ← 天文星曆生成
```

#### 核心演算法

**49 種技術分析物件**（`data_objects.py`）

```
PriceBarChartGannFanArtifact      ← Gann 扇形線
PriceBarChartFibFanArtifact       ← Fibonacci 扇形線
PriceBarChartTimeRetracementArtifact  ← 時間回撤
PriceBarChartPriceRetracementArtifact ← 價格回撤
PriceBarChartPriceTimeVectorArtifact  ← 價格時間向量
PriceBarChartOctaveFanArtifact        ← 八度扇形線
（10 種 Vedic Dasa 時間週期物件）
```

**Fibonacci 比值**（`data_objects.py:964-1040`）
```python
# 包含黃金比例的冪次
(1/phi)^4, (1/phi)^3, (1/phi)^2, (1/phi)^1
sqrt(1/phi), (1/phi)^(1/3)
1.0
0.5 + (1/phi), phi^(1/3), sqrt(phi), phi, phi^2, phi^3, phi^4
```

**Swing Pivot 系統**（`misc/SwingFileScripts/swing.py`）
```
H / HH / HHH    ← 不同級別的高點
L / LL / LLL    ← 不同級別的低點
```

**天文週期分析**（`lookbackmultiple_calc.py`）
- 計算行星（木星、土星等）未來合相日期
- 分散式平行運算支援（AWS EC2 腳本）
- 精度 2 秒

#### 功能清單

| 功能 | 狀態 | 備註 |
|------|------|------|
| Gann Swing | ✅ 有 | misc/SwingFileScripts |
| Gann Fan | ✅ 完整 | GUI 繪製工具 |
| Gann Square of 9 | ⚠️ 框架 | mastercharts.py 有框架但 TODO |
| Fibonacci 回撤/延伸 | ✅ 完整 | 15+ 比值 |
| Fibonacci 時間週期 | ✅ 有 | 天文星曆型（非傳統 Fib 數列型） |
| Elliott Wave | ❌ 無 | — |
| 半對數計算 | ❌ 無 | 線性座標 |
| Pivot 偵測 | ✅ 有 | H/HH/HHH 多級系統 |
| 回測 | ❌ 無 | 純分析工具 |
| 圖表視覺化 | ✅ 完整 | PyQt5 全功能桌面應用 |
| 天文週期分析 | ✅ 完整 | 獨特功能（行星合相） |

#### 代碼品質觀察

**優點**：
- 最龐大最完整的 TA 工具集（2010 年起持續開發）
- 49 種獨特 TA 物件，涵蓋 Gann + Fibonacci + 天文
- 多級 Pivot 系統（H/HH/HHH）設計精良
- 天文週期分析獨樹一幟

**缺點**：
- **已停止開發**（最後更新 2015 年）
- 安裝複雜（需編譯 pyswisseph C 擴展）
- 55,540 行的圖表引擎難以拆解
- 無 Elliott Wave
- 無 Semi-log 縮放
- 無交易回測

#### 評分

| 項目 | 分數 |
|------|------|
| 可用性 | **3/10** |
| 程式碼品質 | **6/10** |
| 適合改造台指期 | **部分**（Pivot 系統可借鑒；主程式難以移植） |
| 最大缺點 | 停止開發；安裝門檻高；PyQt5 耦合太深 |
| **最適合抽取模組** | `misc/SwingFileScripts/swing.py` + H/HH/HHH Pivot 分級概念 |

---

## 二、六大 Repo 功能矩陣對比

| 功能 | python-taew | elliot-waves-auto | fibonacci_ml | Stock-market | gann-swing | pricechartingtool |
|------|:-----------:|:-----------------:|:------------:|:------------:|:----------:|:-----------------:|
| **EW 1-2-3-4-5 標記** | ✅ | ✅ | ❌ | ⚠️ | ❌ | ❌ |
| **EW 衝擊波（Impulse）** | ✅ | ✅ | ❌ | ⚠️ | ❌ | ❌ |
| **EW 修正波 ABC** | ⚠️ | ✅ | ❌ | ⚠️ | ❌ | ❌ |
| **EW Zigzag/Flat/Triangle** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **EW 三規則驗證** | ⚠️ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Fibonacci 回撤（線性）** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Fibonacci 延伸（線性）** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Fibonacci 時間週期** | ✅ | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| **半對數 Fibonacci** | ❌ | ❌ | ⚠️ | ✅ | ❌ | ❌ |
| **Gann Swing** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Gann Fan** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Pivot 偵測** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **多級 Pivot（H/HH/HHH）** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **回測框架** | ❌ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| **交易信號生成** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **圖表視覺化** | ❌ | ✅ | ✅ | ❌ | ⚠️ | ✅ |
| **RSI/量能指標** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **單元測試** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **可 pip 安裝** | ✅ | ✅ | ❌ | ❌ | ⚠️ | ❌ |

> ✅ = 完整實作　⚠️ = 部分/隱含　❌ = 無

---

## 三、綜合評分表

| Repo | 可用性 | 程式碼品質 | 適合台指期 | 最大缺點 | 最適合抽取模組 |
|------|:------:|:----------:|:----------:|----------|----------------|
| **python-taew** | 7/10 | 5/10 | ✅ 是 | 無視覺化；無 ABC 修正波明確標記 | `Alternative_ElliottWave_label_*()` + Fib 驗證函數 |
| **elliot-waves-auto** | 8/10 | 7/10 | ✅ 是（需換資料源） | 無 Fib 時間週期；與 yfinance 耦合 | `pattern_recognition.py` + `fibonacci_analysis.py` |
| **fibonacci_ml** | 5/10 | 7/10 | ⚠️ 部分 | 無 EW；安裝繁瑣 | 三重記憶架構設計概念 + Fib 計算邏輯 |
| **Stock-market** | 2/10 | 2/10 | ⚠️ 部分 | 程式碼 Bug；完全手動 | `high_low.py` 半對數幾何平均公式 |
| **gann-swing** | 8/10 | 8/10 | ✅ 是 | 僅 Swing；多項 TODO 未完成 | 整個 `gannswing.py` |
| **pricechartingtool** | 3/10 | 6/10 | ⚠️ 部分 | 停止開發；安裝複雜；PyQt5 耦合 | Swing 分級概念 + Fib 比值清單 |

---

## 四、整合架構設計

基於以上分析，提出以下**台指期五維操盤決策框架**架構：

```
ta_framework/
├── wave_engine.py        ← Elliott Wave 波浪引擎
├── fib_price_engine.py   ← Fibonacci 價格回撤/延伸引擎（含半對數）
├── fib_time_engine.py    ← Fibonacci 時間週期引擎
├── swing_engine.py       ← Gann Swing + Pivot 偵測引擎
├── regime_engine.py      ← 市場狀態判斷引擎
├── decision_engine.py    ← 五維整合決策引擎
├── data_adapter.py       ← 台指期資料接口
├── visualizer.py         ← 圖表視覺化
└── backtest.py           ← 回測框架
```

### 各模組的資料來源與設計重點

---

#### `wave_engine.py` — Elliott Wave 波浪引擎

**主要資料來源**：
- `python-taew/taew/ew.py` — 借用 Fibonacci 驗證函數 + Alternative 時間週期邏輯
- `elliot-waves-auto/pattern_recognition.py` — 借用三規則嚴格驗證
- `elliot-waves-auto/improved_elliott.py` — 借用多準則計分系統

**核心功能**：
```python
class WaveEngine:
    def detect_pivots(ohlc: pd.DataFrame, order: int = 5) -> pd.DataFrame
    def label_impulse_waves(pivots: pd.DataFrame) -> List[WavePattern]
    def label_corrective_waves(pivots: pd.DataFrame) -> List[WavePattern]
    def score_wave_quality(pattern: WavePattern) -> WaveScore
    def predict_next_target(pattern: WavePattern, wave_num: int) -> PriceTarget
```

**關鍵設計決策**：
- 採用 `scipy.signal.find_peaks` 做初步 pivot 偵測（elliot-waves-auto 方案）
- 使用 python-taew 的 Fibonacci 比值驗證（含 0.4011x–1.6989x 時間約束）
- 使用 elliot-waves-auto 的三規則硬驗證（Rule1/2/3 各 100 分）
- 輸出結構化 `WavePattern` 物件，含信心度分數

---

#### `fib_price_engine.py` — Fibonacci 價格引擎

**主要資料來源**：
- `elliot-waves-auto/fibonacci_analysis.py` — 回撤/延伸/群集計算
- `Stock-market/high_low.py` — **半對數幾何平均公式**（修正 bug 後）
- `fibonacci_ml/fib_utils.py` — 累積高低點追蹤邏輯

**核心功能**：
```python
class FibPriceEngine:
    # 線性 Fibonacci
    def calc_retracement(high: float, low: float) -> Dict[str, float]
    def calc_extension(wave1: float, wave2_end: float) -> Dict[str, float]
    def find_price_clusters(levels: List[float], tolerance: float = 0.003) -> List[Cluster]
    
    # 半對數 Fibonacci（核心差異化功能）
    def calc_semilog_retracement(high: float, low: float) -> Dict[str, float]
    def calc_semilog_extension(p1: float, p2: float, p3: float) -> Dict[str, float]
    def geometric_mean_price(x: float, y: float, ratio: float) -> float
```

**半對數計算公式**（來自 Stock-market/high_low.py 修正版）：
```python
def geometric_mean_price(self, x: float, y: float, ratio: float) -> float:
    """在半對數空間中計算兩價格的 Fibonacci 插值"""
    return x ** ratio * y ** (1 - ratio)

def calc_semilog_retracement(self, high: float, low: float) -> dict:
    levels = [0.0, 0.146, 0.236, 0.382, 0.5, 0.618, 0.764, 0.854, 1.0]
    return {f"{r:.1%}": self.geometric_mean_price(low, high, r) for r in levels}
```

---

#### `fib_time_engine.py` — Fibonacci 時間週期引擎

**主要資料來源**：
- `python-taew/taew/ew.py:159,180,211,236` — Wave 時間比例約束
- `fibonacci_ml/core.py:297-301` — 時間長度特徵追蹤
- `pricechartingtool/lookbackmultiple_calc.py` — 週期搜尋概念（不使用天文計算）

**核心功能**：
```python
class FibTimeEngine:
    # 傳統 Fibonacci 時間週期
    def project_time_targets(wave_start: int, wave_end: int) -> List[TimeTarget]
    def calc_time_ratios(wave1_bars: int) -> Dict[str, int]
    
    # Fibonacci 時間序列（1,1,2,3,5,8,13,21,34,55...)
    def fibonacci_time_zones(start_bar: int) -> List[int]
    
    # 波浪時間約束驗證
    def validate_wave_timing(wave_bars: int, ref_wave_bars: int,
                              min_ratio: float = 0.4011,
                              max_ratio: float = 1.6989) -> bool
```

**時間比例邏輯**（來自 python-taew Alternative 方法）：
```python
FIB_TIME_RATIOS = {
    'wave2_min': 0.4011, 'wave2_max': 1.6989,   # Wave2 時間 = Wave1 × 0.4011–1.6989
    'wave3_min': 1.0,    'wave3_max': 4.236,     # Wave3 通常較長
    'wave4_min': 0.4011, 'wave4_max': 1.6989,    # Wave4 與 Wave2 交替
    'wave5_min': 0.4011, 'wave5_max': 1.6989,    # Wave5 時間約束
}
```

---

#### `swing_engine.py` — Gann Swing + Pivot 引擎

**主要資料來源**：
- `gann-swing/gannswing.py` — **直接使用**（最高品質的單一功能模組）
- `pricechartingtool/misc/SwingFileScripts/swing.py` — H/HH/HHH 多級概念

**核心功能**：
```python
class SwingEngine:
    # Gann Swing 偵測（來自 gannswing.py）
    def classify_bar(bar: pd.Series, prev_bar: pd.Series) -> BarType
    def calculate_swings(ohlc: pd.DataFrame, swing_days: int = 2) -> pd.DataFrame
    
    # 多級 Pivot 系統（來自 pricechartingtool 概念）
    def detect_pivots(ohlc: pd.DataFrame, lookback: int = 5) -> pd.DataFrame
    def classify_pivot_strength(pivots: pd.DataFrame) -> pd.DataFrame
    # 輸出：H（普通高點）/ HH（重要高點）/ HHH（關鍵高點）
    
    # 台指期專用
    def detect_structure_break(swings: pd.DataFrame) -> List[StructureBreak]
    def is_higher_high(swings: pd.DataFrame) -> bool
    def is_lower_low(swings: pd.DataFrame) -> bool
```

**K 線分類**（直接抽取自 gannswing.py）：
```python
class BarType(Enum):
    UP_DAY      = "up"       # 高點更高 AND 低點不更低
    DOWN_DAY    = "down"     # 低點更低 AND 高點不更高
    INSIDE_DAY  = "inside"   # 高低均在前根範圍內
    OUTSIDE_DAY = "outside"  # 高更高 AND 低更低
```

---

#### `regime_engine.py` — 市場狀態判斷引擎

**主要資料來源**：
- `elliot-waves-auto/super_strategy.py` — 整合策略邏輯借鑒
- `fibonacci_ml/core.py` — 三重記憶（多週期）架構

**核心功能**：
```python
class RegimeEngine:
    # 多週期狀態（參考 fibonacci_ml 三重記憶概念）
    def get_trend_regime(ohlc_daily: pd.DataFrame,
                         ohlc_weekly: pd.DataFrame,
                         ohlc_monthly: pd.DataFrame) -> Regime
    
    # 波浪計數階段
    def identify_wave_position(wave_pattern: WavePattern) -> WavePosition
    # 輸出：IMPULSE_WAVE_1, IMPULSE_WAVE_3, IMPULSE_WAVE_5,
    #       CORRECTION_A, CORRECTION_B, CORRECTION_C
    
    # Fibonacci 支撐/阻力強度
    def evaluate_fib_confluence(price: float, fib_levels: List[float],
                                swing_pivots: pd.DataFrame) -> ConfluenceScore
```

**市場狀態分類**：
```python
class Regime(Enum):
    TRENDING_UP       = "trending_up"      # 上升衝擊波段
    TRENDING_DOWN     = "trending_down"    # 下降衝擊波段
    CORRECTING_UP     = "correcting_up"    # 上升修正中
    CORRECTING_DOWN   = "correcting_down"  # 下降修正中
    RANGING           = "ranging"          # 盤整
    BREAKOUT_PENDING  = "breakout_pending" # 準備突破
```

---

#### `decision_engine.py` — 五維整合決策引擎

**整合所有模組的最終決策層**：

```python
class DecisionEngine:
    """
    五維操盤決策模型：
    維度1 - 波浪位置（WaveEngine）
    維度2 - Fibonacci 價格水準（FibPriceEngine）
    維度3 - Fibonacci 時間週期（FibTimeEngine）
    維度4 - Gann Swing 方向（SwingEngine）
    維度5 - 市場狀態（RegimeEngine）
    """
    
    def generate_signal(self, ohlc: pd.DataFrame) -> TradingSignal:
        wave_pos    = self.wave_engine.label_impulse_waves(...)
        fib_price   = self.fib_price_engine.calc_semilog_retracement(...)
        fib_time    = self.fib_time_engine.project_time_targets(...)
        swing_dir   = self.swing_engine.calculate_swings(...)
        regime      = self.regime_engine.get_trend_regime(...)
        
        return self._synthesize(wave_pos, fib_price, fib_time, swing_dir, regime)
    
    def _synthesize(self, *dims) -> TradingSignal:
        """
        得分加總邏輯（參考 elliot-waves-auto 計分概念）：
        - 五維同向：強烈做多/做空信號
        - 四維同向：正常信號
        - 三維以下：觀望
        """
```

---

## 五、模組實作建議

### 5.1 立即可用（直接抽取）

| 來源 | 抽取內容 | 放入模組 | 修改量 |
|------|---------|---------|--------|
| `gann-swing/gannswing.py` | 全部 | `swing_engine.py` | 微小（介面統一） |
| `python-taew/taew/ew.py:8-75` | `wave2/3/4/5_fibonacci_check()` | `wave_engine.py` | 無 |
| `elliot-waves-auto/pattern_recognition.py:36-209` | `identify_impulse_wave()` | `wave_engine.py` | 中等（移除 yfinance 依賴） |
| `elliot-waves-auto/fibonacci_analysis.py` | 全部 | `fib_price_engine.py` | 微小 |

### 5.2 需要修復後使用

| 來源 | 問題 | 修復方式 |
|------|------|---------|
| `Stock-market/high_low.py` | `p1/pe` 未定義 | 將函數參數化後移植 |
| `python-taew/ew.py:159-236` | 時間約束硬編碼 | 參數化 + 加入 `fib_time_engine.py` |
| `fibonacci_ml/subjective_drawdown_finder.py` | Pickle 版本依賴 | 重新訓練或轉為 joblib |

### 5.3 借鑒概念（重新實作）

| 概念 | 來源 | 重新實作原因 |
|------|------|------------|
| 三重記憶多週期架構 | `fibonacci_ml/core.py` | 原版耦合度高，需為台指期重設週期參數 |
| H/HH/HHH 多級 Pivot | `pricechartingtool` | 原版是 GUI 工具，需提取為純 Python 函數 |
| 多準則波浪計分 | `elliot-waves-auto/improved_elliott.py` | 分數參數需為台指期調校 |
| 天文週期分析 | `pricechartingtool/lookbackmultiple_calc.py` | 替換為傳統 Fibonacci 時間序列 |

### 5.4 台指期專用調校建議

**資料源替換**：
```python
# 替換 yfinance 為台指期資料
# 建議使用 fubon API / 永豐 API / 台灣期交所 Open Data
class TaiwanFuturesAdapter(DataAdapter):
    def get_ohlc(self, contract: str, interval: str) -> pd.DataFrame:
        ...
```

**Fibonacci 比值調校**：
- 台指期日線：標準 Fibonacci（38.2%, 50%, 61.8%）
- 台指期週線：強調 50% 回撤（均值回歸特性較強）
- 台指期月線：使用半對數 Fibonacci（長線價格空間等比）

**Swing 參數建議**：
```python
# 台指期建議配置
swing_engine = SwingEngine(
    swing_days=2,         # 2-bar 反轉（過濾雜訊）
    inside_down=True,     # 內縮日視為弱勢（台指期特性）
    ignore_threshold=50,  # 忽略 50 點以內的小波動
)
```

---

## 六、結論與下一步行動

### 整體結論

六個 repo 中**沒有任何一個**能直接滿足台指期五維操盤模型的全部需求，但透過精心整合可以建立完整框架：

| 需求 | 最佳來源 | 可用度 |
|------|---------|--------|
| Elliott Wave 核心 | python-taew + elliot-waves-auto | 高 |
| Fibonacci 價格（線性） | elliot-waves-auto/fibonacci_analysis.py | 高 |
| Fibonacci 價格（半對數） | Stock-market/high_low.py（修復後） | 中 |
| Fibonacci 時間週期 | python-taew Alternative 模式 | 中 |
| Gann Swing + K 線分類 | gann-swing/gannswing.py | 高 |
| 多級 Pivot | pricechartingtool（概念借鑒） | 低（需重實作） |
| 回測框架 | elliot-waves-auto（需拆解） | 中 |
| 視覺化 | elliot-waves-auto + Plotly | 高 |

### 建議開發優先序

**Phase 1（基礎）**：
1. 建立 `data_adapter.py`（台指期資料介面）
2. 整合 `swing_engine.py`（直接使用 gann-swing）
3. 整合 `fib_price_engine.py`（elliot-waves-auto fibonacci_analysis.py + Stock-market half-log 修復）

**Phase 2（波浪）**：
4. 建立 `wave_engine.py`（python-taew Fib 驗證 + elliot-waves-auto 規則）
5. 建立 `fib_time_engine.py`（python-taew Alternative 時間邏輯參數化）

**Phase 3（整合）**：
6. 建立 `regime_engine.py`（多週期狀態判斷）
7. 建立 `decision_engine.py`（五維信號整合）
8. 建立 `visualizer.py`（Plotly 互動圖表）
9. 建立 `backtest.py`（回測框架）

### 最終建議

> **核心策略**：以 `gann-swing` 為結構骨架（Swing 偵測），以 `elliot-waves-auto` 為波浪驗證引擎（規則計分），以 `python-taew` 補充時間週期約束，以 `Stock-market/high_low.py` 修復版本提供半對數 Fibonacci 定價，再加入三重週期記憶（日/週/月）完成五維決策架構。

這個組合能覆蓋所有核心需求，且三個主要 repo（gann-swing, elliot-waves-auto, python-taew）均可安裝並有可抽取的模組，開發風險較低。

---

*報告產生：2026-05-03*  
*分析涵蓋行數：~293,000 行原始碼*  
*研究分支：`claude/research-ta-frameworks-nuQ9i`*
