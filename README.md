# 50 Startups：多元線性迴歸（CRISP-DM）

用 scikit-learn 解 Kaggle「50 Startups」多元線性迴歸問題：從 R&D、Administration、Marketing 三項支出與所在州別（State），預測新創公司的年度利潤（Profit）。整個專案嚴格依照 **CRISP-DM** 六個階段進行，並提供一份靜態報告與一個可在瀏覽器即時重算的互動式迴歸工具。

## Demo

| 頁面 | 說明 | 連結 |
|---|---|---|
| CRISP-DM 專案紀錄 | 靜態報告，逐階段呈現 EDA、特徵選擇、評估與部署結果 | `https://<your-project>.vercel.app/` |
| 50 Startups 迴歸實驗室 | 互動工具：勾選特徵，即時重算係數、p 值、診斷圖、假設檢定與預測 | `https://<your-project>.vercel.app/app` |

> 部署到 Vercel 後（見下方〈部署〉），把 `<your-project>` 換成實際網址。

本機預覽：

```bash
python -m http.server 8000
# 開啟 http://localhost:8000/ 與 http://localhost:8000/app.html
```

## 結果摘要

| 項目 | 結果 |
|---|---|
| 最終模型 | `LinearRegression`，只用 **R&D Spend** 一個特徵 |
| 迴歸方程式（以全部 50 筆重新訓練） | Profit ≈ 49,033 + 0.854 × R&D Spend |
| Test R² | **0.9265**（全部 4 個特徵的模型為 0.8987） |
| Test RMSE ／ MAE | $7,714 ／ $6,077 |
| MAPE | 9.14% |
| 5-fold CV R² | 0.937 ± 0.037 |
| 成功標準 | Test R² ≥ 0.90、CV R² ≥ 0.85、MAPE ≤ 10%：**全部通過** |

主要發現：

- **R&D 支出是利潤的主要驅動因素**：與 Profit 的相關係數 r = 0.973，F-score 676，Random Forest 重要度約 93%。研發每多投入 $1，利潤平均多約 $0.85。
- **Administration 與 State 幾乎沒有預測力**：Lasso 會把兩個州別的係數壓到 0，加入它們反而讓測試誤差變大。
- **少即是多**：只用 R&D 的模型在測試集上優於使用全部特徵的模型。
- **限制**：最終模型的 RESET（線性）與 Jarque-Bera（常態）檢定未通過，主要受一筆離群值影響（第 49 筆，Profit $14,681）。樣本只有 50 筆，用於實際決策前應蒐集更多資料。

## CRISP-DM 流程

| 階段 | 內容 |
|---|---|
| 1 Business Understanding | 商業目標、現況評估（資源、限制、風險）、資料探勘目標、事先訂定的成功標準、專案計畫 |
| 2 Data Understanding | 下載資料並比對 SHA-256、敘述統計、相關係數、EDA 圖、缺值／重複／離群值檢查 |
| 3 Data Preparation | 先切分 80/20，再以 `ColumnTransformer` 做 OneHotEncoder（drop first，避開虛擬變數陷阱）與 StandardScaler；特徵選擇分析（Pearson、SelectKBest F-score、RFE、Lasso、Random Forest） |
| 4 Modeling | VIF 共線性檢查；4 個特徵的全部 15 種組合以 5-fold CV 比較，依 1-SD 精簡原則選模型；比較 5 種特徵選擇演算法 |
| 5 Evaluation | 測試集指標、殘差診斷、對照成功標準、流程審查、決定下一步 |
| 6 Deployment | 通過評估才以全部資料重新訓練，並用 joblib 儲存；監控計畫與最終報告 |

### 避免資料洩漏

- 所有前處理都包在 `Pipeline` 裡，只在訓練集上 fit。
- 特徵選擇只用訓練集的交叉驗證，測試集只用來回報最終結果。
- 常見範例會直接用測試集 RMSE 挑特徵，本專案刻意不這樣做，所以 forward selection 第 2 步選到的特徵可能和那些範例不同。

### 5 種特徵選擇演算法

| 檔名前綴 | 演算法 |
|---|---|
| `lasso_l1_ranking` | Lasso 路徑上特徵進入的順序 |
| `forward_sfs` | `SequentialFeatureSelector`（forward） |
| `rfe` | Recursive Feature Elimination |
| `elasticnet_ranking` | ElasticNet 路徑上特徵進入的順序（l1_ratio = 0.5） |
| `exhaustive_search` | 每個特徵數下的最佳子集 |

每種演算法都只看訓練集，以 5-fold CV RMSE 決定特徵，並記錄用 1 到 5 個特徵時的測試集 RMSE 與 R²。

## 互動式迴歸實驗室（app.html）

所有計算都在瀏覽器裡完成，不需要後端：

- **模型設定**：勾選特徵、一鍵套用「最終模型／CV 最佳／全部特徵」、切換顯著水準 α 與擬合資料（訓練集或全部）
- **係數表**：B、SE、t、p 值、信賴區間、標準化 β、VIF，以及 ANOVA 表
- **評估**：擬合 R²／Adj R²／F、5-fold CV R²、測試集 R²／RMSE／MAE／MAPE，並對照成功標準
- **診斷圖**：實際 vs 預測、殘差 vs 預測值（標出 Cook's D 影響點）、標準化 β、殘差分布
- **假設檢定**：Ramsey RESET（線性）、Breusch-Pagan（同質變異）、Jarque-Bera（常態）、Durbin-Watson、VIF、Cook's D
- **特徵選擇**：5 種演算法比較圖、15 種特徵組合排行榜（點一下即套用）
- **部署**：利潤預測器（95% 預測區間、外插警示）與 APA 7 報告段落

`regression.js` 的所有輸出都已和 numpy／scipy／scikit-learn 逐項比對（最大相對誤差約 1e-12）。網頁使用和 `main.py` 相同的訓練／測試切分與 CV 分組（存在 `model_data.js`），所以數字與 Python 一致。

## 專案結構

```
main.py                              CRISP-DM 六階段主程式
feature_progression_analysis.py      5 種特徵選擇演算法 × 1～5 個特徵
regression.js                        瀏覽器端 OLS 統計引擎
index.html                           靜態報告
app.html                             互動式迴歸實驗室
model_data.json / model_data.js      main.py 產生的網頁資料
50_Startups.csv                      資料集（沒有時 main.py 會自動下載並驗證）
requirements.txt                     Python 套件版本
vercel.json                          Vercel 靜態網站設定

{method}_progression.csv             各演算法逐步結果
{method}_plot.png                    各演算法的 RMSE／R² 對特徵數圖
all_5_algorithms_comparison_plot.png 5 種演算法的 Test RMSE 比較
feature_selection_comparison.csv/png 各演算法最佳子集比較
crispdm_output/                      各階段圖表、部署模型（.joblib）、final_report.json
```

## 執行

需要 Python 3.10 以上。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`main.py` 會依序跑完六個階段，並產生上面所有的輸出檔與網頁資料。只想重跑特徵選擇比較：

```bash
python feature_progression_analysis.py
```

## 部署

網頁都是靜態檔案，不需要建置步驟。部署到 Vercel：

```bash
npx vercel
```

第一次執行會要求登入並連結專案。完成後把上方 Demo 表格裡的網址換成實際網址。

## 致謝與參考資料

- **資料集**：[50 Startups](https://www.kaggle.com/code/akash17/50-startups-linearregression)（Kaggle）。程式從 [krishnaik06/Multiple-Linear-Regression](https://github.com/krishnaik06/Multiple-Linear-Regression) 的公開副本下載，並比對 SHA-256 確認與其他獨立副本一致。
- **互動頁設計參考**：[StatsUnlock — Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)。互動頁的資訊架構（摘要卡片、完整係數表、ANOVA、四張診斷圖、假設檢定徽章、結果解讀與 APA 報告段落）參考了這個網站。
- **方法論**：Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
- **套件**：[scikit-learn](https://scikit-learn.org/)、[pandas](https://pandas.pydata.org/)、[NumPy](https://numpy.org/)、[SciPy](https://scipy.org/)、[Matplotlib](https://matplotlib.org/)、[joblib](https://joblib.readthedocs.io/)、[Requests](https://requests.readthedocs.io/)
- **字型**：[Google Fonts](https://fonts.google.com/)：Archivo、Noto Sans TC、JetBrains Mono
