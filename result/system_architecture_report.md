# 📊 LoRA 黃金訓練集自動篩選系統 (Auto-Curation Engine) v2.0 - 系統架構報告

這份報告詳細解構了 LoRA 自動篩選系統的底層架構、核心資料流以及功能模組。系統透過雙階段的解耦設計與防呆安檢機制，為您的 AI 模型提供最高品質、最純淨的黃金訓練集。

---

## 🏗️ 專案骨幹架構 (Project Skeleton)

系統採用高度模組化與專案級別的目錄隔離設計。不論在雲端或本地執行，所有產出與來源皆圍繞著特定的 `FACE_ID` 進行，確保多人物任務不會互相干擾。

```text
Lora_Auto_Curation_Engine/
├── raw_data/                       # 📂 原始未過濾圖片輸入區 (Input Directory)
│   └── {FACE_ID}/                  # 依目標人物 ID 建立的來源圖資料夾
├── LoRA_Curated_Dataset/           # 🎯 系統自動生成的最終黃金訓練集 (Output Directory)
│   └── {FACE_ID}/
│       ├── Accepted/               # ✅ 通過所有安檢的黃金高品質訓練圖
│       ├── Reject_Blurry/          # ❌ 因畫質過於模糊遭剔除
│       ├── Reject_NoFace/          # ❌ 找不到符合標準的人臉
│       ├── Reject_MultiFace/       # ❌ 畫面中存在多張人臉
│       ├── Reject_TooSmall/        # ❌ 人臉佔比過小
│       └── Reject_Duplicate/       # ❌ 經 dHash 判定為重複圖檔
├── scripts/                        # ⚙️ 核心執行腳本區
│   ├── auto_curator.py             # 主過濾引擎腳本 (Phase 2)
│   └── pose_auditor.py             # 智慧底圖雷達與工單生成腳本 (Phase 1)
├── output/                         # 🗃️ 暫存與報告輸出區
│   ├── task_lists/                 # 階段一產生的掃描結果工單 (.txt)
│   └── registry/                   # 全域快取與歷史紀錄 (global_pose_audit.json)
├── config.py                       # 🔧 系統全域參數與環境組態設定檔
├── lora_registry.json              # 📜 人物目標依賴註冊表 (Face ID 屬性與極限邊界管理)
├── README.md                       # 📖 專案說明與執行規則文件
├── LoRA_Auto_Curation_Colab.ipynb  # ☁️ 雲端 GPU 環境專用 Colab Notebook (主過濾引擎)
└── LoRA_Pose_Auditor_Colab.ipynb   # ☁️ 雲端 GPU 環境專用 Colab Notebook (底圖安檢)
```

---

## 🔄 核心資料流邏輯圖 (Data Flow Logic)

本系統的運作流程經過嚴格把關，從最前方的環境偵測，到多維度影像檢核，最後產生健康度診斷報告，形成一個完整的資料淨化循環。

```mermaid
graph TD
    %% 定義樣式
    classDef init fill:#f9f9fa,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef check fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef reject fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    classDef accept fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef report fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    A[使用者放入未處理圖片]:::init --> B(raw_data / FACE_ID)
    
    subgraph Phase 1: 智慧底圖雷達安檢 (Pose Auditing)
        B --> C{安檢腳本掃描與邊界比對}:::check
        C -->|產出掃描清單| D[產出工單 .txt & 更新全域快取]:::process
    end
    
    subgraph Phase 2: 自動過濾引擎 (Auto Curation)
        D --> E{起飛前檢查 Pre-flight Checks}:::check
        E -->|路徑或模型缺失| F[中斷 / 互動式提示補全]:::reject
        E -->|檢查通過| G{動態硬體資源偵測}:::check
        
        G -->|無 GPU| H[純 CPU 安全降級模式]:::process
        G -->|有 GPU| I[GPU 極速防崩潰模式]:::process
        
        H --> J[載入 InsightFace & Laplacian 偵測器]
        I --> J
        
        J --> K{1. 拉普拉斯銳利度檢測}:::check
        K -->|分數過低| L[Reject_Blurry]:::reject
        K -->|邊緣清晰| M{2. 幾何純淨度掃描}:::check
        
        M -->|找不到臉| N[Reject_NoFace]:::reject
        M -->|大於一張臉| O[Reject_MultiFace]:::reject
        M -->|人臉佔比 < 5%| P[Reject_TooSmall]:::reject
        M -->|及格| Q{3. dHash 感知雜湊去重}:::check
        
        Q -->|64位元指紋重複| R[Reject_Duplicate]:::reject
        Q -->|獨一無二| S{4. 3D 歐拉角多樣性分類}:::check
        
        S --> T[儲存至 Accepted / 黃金訓練集]:::accept
    end
    
    T --> U[產出 AI 教練健康度診斷報告]:::report
```

---

## 🌟 全系統功能摘要 (System Functions)

本引擎不單單是一個裁切工具，而是一個具備「防呆、自適應、高教準」的智慧篩選管線。以下為系統的六大核心功能模組：

### 1. 🛡️ 起飛前安檢機制 (Pre-flight Checks)
*   **功能描述**：在任何重度運算開始前，系統會主動檢查目標來源目錄 (`raw_data/{FACE_ID}`) 以及 InsightFace 的核心模型檔案 (`det_10g.onnx`) 是否已完全就緒。
*   **效益**：若發現缺失，不會產生生硬的報錯，而是透過「人機互動提示」要求使用者補齊參數，避免載入到一半出錯浪費寶貴時間。

### 2. 🧠 智慧環境偵測與無縫切換 (Environment Router)
*   **功能描述**：利用 ONNXRuntime 底層 API 自動探測當下環境資源。若在配備顯示卡的環境執行，將自動套用 2GB 記憶體安全限制並啟用 **GPU 極速掃描**；若在一般筆電執行，則自動無縫降級為 **CPU 運算模式**。
*   **效益**：保證跨平台 (PC / Colab) 皆可穩定執行且絕不崩潰，且 CPU 模式的篩選標準與 GPU 模式完全一致。

### 3. 🔍 多維度高教準影像過濾 (Multi-dimension Curation)
*   **畫質銳利度檢測**：利用 Laplacian 演算法計算影像二階導數變異數，無情剔除低解析度與模糊圖檔。
*   **幾何純淨度掃描**：強制規定「單圖單人」，若畫面中有多張臉，或主要人臉佔比低於系統下限 (`MIN_FACE_RATIO`)，將被直接剔除。
*   **3D 特徵多樣性分析**：透過提取人臉姿勢參數 (Pitch, Yaw, Roll)，精準辨識「正臉、側臉、仰俯角」，保障訓練集的多角度特徵涵蓋率。

### 4. 🧬 dHash 感知雜湊去重防護 (Deduplication)
*   **功能描述**：系統會自動掃描 `Accepted` 資料夾內既有的圖片，並為每一張圖片計算 64 位元的感知雜湊 (dHash) 指紋。
*   **效益**：即使圖片被縮放、改變檔名、甚至是稍微裁剪，只要視覺上高度相似，系統都會將其辨識為重複圖檔並歸入 `Reject_Duplicate`，徹底杜絕模型因重複資料而產生「過擬合 (Overfitting)」。

### 5. 📈 AI 教練健康度診斷報告 (Diversity Analyzer)
*   **功能描述**：在每一批次篩選完成後，系統會自動產出「LoRA 訓練集健康度診斷報告」。
*   **效益**：精確盤點各角度庫存佔比，並依照理想模型訓練比例 (例如正臉需佔 40%) 給出嚴格的「下一步搜圖建議」，作為使用者的搜圖指南針。

### 6. 🔗 階段解耦與雙端雲原生支援 (Phase Separation & Cloud Support)
*   **功能描述**：將繁重的運算流程優雅地解耦為「Phase 1 (底圖雷達安檢)」與「Phase 2 (主過濾引擎)」。
*   **效益**：兩階段皆提供本地 `.py` 腳本與雲端 `.ipynb` Notebook。Colab Notebooks 已內建 Google Drive 自動掛載與表單化介面，讓使用者即使沒有強大顯卡，也能享受雲端高算力帶來的高效能。
