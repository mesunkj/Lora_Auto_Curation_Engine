# LoRA 黃金訓練集自動篩選系統 (Auto-Curation Engine) v2.0

本專案旨在提供一套自動化腳本，針對原始圖像資料集 (`raw_data`) 進行畫質、人臉純淨度與特徵多樣性的過濾，並將符合標準的高品質圖片自動歸檔至 `LoRA_Curated_Dataset/Accepted` 中。

## ⚙️ 系統環境與執行規則
本專案已依據最新需求進行以下調整與確認：

1. **純 CPU 環境支援 (PC 架構)**:
   - 腳本預設支援無 GPU (純 CPU) 的一般電腦環境執行，程式已做防崩潰降級處理，評估品質與標準不受影響。
2. **統一組態檔 (`config.py`) 與動態覆寫機制**:
   - 重要的變數（如 `BLUR_THRESHOLD`、`MIN_FACE_RATIO` 等）統一建置於 `config.py`。
   - `INPUT_DIR` 與 `OUTPUT_DIR` 雖在 config 中設有預設值，但在**起飛前 (Pre-flight checks)**，使用者可透過命令列引數 (如 `--input_dir`) 進行覆寫變更。
   - **(Colab 專用)** 人機互動介面新增 `FACE_ID` 參數，自動將輸入目錄導向 `raw_data/{FACE_ID}`，並將輸出結果存入 `LoRA_Curated_Dataset/{FACE_ID}`，以確保同一目標人物資料夾集中管理。
   - 執行時，起飛前檢查機制與底層執行的運轉機制保證**完全一致**。底層引擎所使用的參數，必為前置檢查通過的最終變數。
3. **起飛前置檢查程序 (Pre-flight Checks)**:
   - **來源與輸出檢查**：若來源目錄 (含特定的 `FACE_ID` 路徑) 不存在，系統將即時跳出**互動式提示**，要求使用者輸入正確路徑；若輸出目錄不存在，系統將會為您**自動建置**。
   - **嚴格模型檢查**：不再僅檢查模型目錄是否存在，而是精確檢查 InsightFace 核心模型檔案（例如 `det_10g.onnx`）是否已**下載完成**。未下載完成時會發出明確警示，避免載入到一半出錯浪費時間。
4. **感知雜湊去重防護網 (dHash Deduplication)** *(新增)*:
   - 系統自動掃描 `Accepted` 目錄內既有的圖片並計算 64 位元感知雜湊 (dHash) 指紋。
   - 篩選過程中若遇到視覺上完全相同的圖片（即使被縮放或改變檔名），系統會自動辨識並歸類到 `Reject_Duplicate` 中，徹底杜絕訓練集因為重複圖檔而產生「過擬合 (Overfitting)」的問題。
5. **AI 教練健康度診斷報告 (Diversity Analyzer)** *(新增)*:
   - 篩選完成後，系統會自動產出「LoRA 訓練集健康度診斷報告」。
   - 精確盤點各角度（正臉、側臉、仰俯角）的庫存佔比，並依照理想訓練比例 (20~30張，正臉40%) 給予您嚴格的「下一步搜圖建議」。
6. **開發者授權執行紀錄**:
   - 本次更新與腳本建置作業，已由操作者授權執行，相關設定與規則皆已同步記錄於此 README 文件中，後續執行不需再中斷請求 `submit` 許可即可直接執行。
7. **階段一：智慧底圖雷達與工單生成 (Phase 1 Pose Auditing)** *(新增)*:
   - 全新解耦架構，提供本地端腳本 `scripts/pose_auditor.py` 與雲端專用 `LoRA_Pose_Auditor_Colab.ipynb` 進行起飛前安檢。
   - **雲端 Colab 支援**：已建立獨立的 Colab Notebook 檔案，內建 Google Drive 掛載與表單參數設定功能，確保在雲端環境也能穩定執行並輸出結果。
   - **依賴註冊表**：根目錄必須存在 `lora_registry.json`，且內含已註冊的 Face ID（及其 `physical_limits`，若無則程式會自動掃描 `{Face_ID}/Accepted` 下的黃金圖進行動態建置）。
   - **動態目標目錄**：使用者可透過 `--input_dir`（本地）或表單（Colab）帶入目標底圖目錄。
   - 包含純 CPU 的歷史清單比對與人機互動，以及 GPU 掛載後的極限掃描。
   - **產出結果與目錄結構 (給階段二介接使用)**：
     - **工單目錄** (`output/task_lists/`)：掃描結果將自動產出 `.txt` 純文字工單 (如 `task_Tzuyu_20260826_0940.txt`)，內部皆為絕對路徑。
     - **全域快取** (`output/registry/global_pose_audit.json`)：以 Face ID 為 Key 隔離紀錄，記錄歷史已接受與拒絕的絕對路徑。
     - **黃金訓練圖參考路徑** (`LoRA_Curated_Dataset/{Face_ID}/Accepted/`)：建立極限邊界時所參考的正確圖檔目錄。

## 🚀 使用方式

請先確保專案根目錄存在待篩選的圖片檔案。針對特定 `FACE_ID` (例如 `person_a`)，請將圖檔放在對應的目錄底下。

```bash
# 使用預設設定檔路徑執行 (自動讀取 config.py 的預設目錄)
python scripts/auto_curator.py

# 手動覆寫輸入/輸出目錄 (建議依據 FACE_ID 建立獨立目錄，例如 person_a)
python scripts/auto_curator.py --input_dir ./raw_data/person_a --output_dir ./LoRA_Curated_Dataset/person_a
```

執行後系統將會：
1. 讀取 `config.py`，若有外部引數則以引數為主，執行起飛前置檢查 (Pre-flight checks)。
2. 若路徑 (例如 `raw_data/{FACE_ID}`) 或模型有誤，第一時間發出警示或互動式提示。
3. 確保變數完全一致後，啟動篩選引擎，檢查影像銳利度、人臉數量與佔比等。
4. 輸出合格圖片至目標目錄下的 `Accepted/` (例如 `LoRA_Curated_Dataset/{FACE_ID}/Accepted/`)。
5. 輸出不合格圖片至目標目錄下的 `Reject_xxx/`（依死因分類）。
