# LoRA 黃金訓練集自動篩選系統 (Auto-Curation Engine) v2.0

本專案旨在提供一套自動化腳本，針對原始圖像資料集 (`raw_data`) 進行畫質、人臉純淨度與特徵多樣性的過濾，並將符合標準的高品質圖片自動歸檔至 `LoRA_Curated_Dataset/Accepted` 中。

> [!IMPORTANT]
> **Agent 開發者請注意**：請務必優先閱讀並遵守本專案的 [Agent專案執行規範](Agent專案執行規範.md)。

---

## ✨ 核心特色與功能

1. **純 CPU 環境支援 (PC 架構)**
   - 腳本預設支援無 GPU (純 CPU) 的一般電腦環境執行，程式已做防崩潰降級處理，評估品質與標準不受影響。
2. **統一組態檔與動態覆寫機制**
   - 重要的變數（如 `BLUR_THRESHOLD`、`MIN_FACE_RATIO` 等）統一建置於 `config.py`。
   - `INPUT_DIR` 與 `OUTPUT_DIR` 可透過命令列引數 (如 `--input_dir`) 進行覆寫變更。
   - **(Colab 專用)** 支援 `FACE_ID` 參數，自動將輸入目錄導向 `raw_data/{FACE_ID}`，集中管理。
3. **起飛前置檢查程序 (Pre-flight Checks)**
   - **來源與輸出檢查**：若來源目錄不存在，系統將即時跳出**互動式提示**；若輸出目錄不存在，系統將自動建置。
   - **嚴格模型檢查**：精確檢查 InsightFace 核心模型檔案（例如 `det_10g.onnx`）是否已下載完成，避免中途崩潰。
4. **感知雜湊去重防護網 (dHash Deduplication)**
   - 自動掃描既有圖片並計算 64 位元感知雜湊 (dHash) 指紋。
   - 自動辨識視覺上完全相同的圖片，歸類到 `Reject_Duplicate`，徹底杜絕訓練集產生「過擬合 (Overfitting)」。
5. **AI 教練健康度診斷報告 (Diversity Analyzer)**
   - 篩選完成後，系統會精確盤點各角度（正臉、側臉、仰俯角）的庫存佔比。
   - 依照理想訓練比例 (20~30張，正臉40%) 給予您嚴格的「下一步搜圖建議」。
6. **階段三：合成前置安檢與底圖精準篩選雷達 (Pre-Synthesis Pose Auditing)**
   - 提供本地端腳本 `scripts/pose_auditor.py` 與雲端專用 `LoRA_Pose_Auditor_Colab.ipynb` 進行起飛前底圖安檢。
   - **依賴註冊表**：依賴 `lora_registry.json` 與 `mapping.json`，精準檢查底圖臉孔是否符合黃金訓練基準的物理極限。
   - **產出工單目錄** (`output/task_lists/`)：掃描結果將自動產出 `.txt` 純文字工單，供下一階段（如 SD 量產引擎）介接使用。

---

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
2. 若路徑或模型有誤，第一時間發出警示或互動式提示。
3. 確保變數完全一致後，啟動篩選引擎，檢查影像銳利度、人臉數量與佔比等。
4. 輸出合格圖片至目標目錄下的 `Accepted/` (例如 `LoRA_Curated_Dataset/{FACE_ID}/Accepted/`)。
5. 輸出不合格圖片至目標目錄下的 `Reject_xxx/`（依死因分類）。
