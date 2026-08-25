

# **📄 LoRA 黃金訓練集自動篩選系統 (Auto-Curation Engine) v2.0 完整需求計畫書**

### **一、 系統目標與預期效益**

* **核心目標**：建構一套跨平台的自動化影像過濾腳本。輸入大量未經整理的候選圖片，系統將自動依據「畫質、人臉純淨度、構圖比例、角度多樣性」進行多維度審查，過濾劣質圖片。  
* **環境泛用性 (新增)**：具備智慧環境偵測能力，若於配備顯示卡的環境執行，將套用記憶體安全限制並啟用 GPU 極速掃描；若於一般筆電或純 CPU 伺服器執行，將自動無縫降級為 CPU 模式運算，確保程式不崩潰。  
* **專案級目錄收納 (新增)**：統一在專案根目錄下建立專屬的 LoRA\_Curated\_Dataset 資料夾，將合格與淘汰的圖片自動分類收納，便於後續直接打包進行模型訓練。

### **二、 核心評估與處理元件 (Components)**

> 1. **智慧環境偵測與模型調度器 (Environment & Provider Router)**  
   * **原理**：透過呼叫 ONNXRuntime 的底層 API，偵測系統是否具備 CUDAExecutionProvider。  
   * **判定**：自動決定要分配 GPU 記憶體資源，或是完全依賴 CPU 計算，並在終端機印出當下使用的硬體狀態。  
> 2. **I/O 與專案級分類路由元件**  
   * **原理**：在專案底下動態建立分類結構，包含 Accepted/ (合格區) 以及各類死因專屬的 Rejected/ (剔除區)。  
> 3. **拉普拉斯邊緣銳利度檢測器 (Laplacian Blur Detector)**  
   * **原理**：利用 OpenCV 計算影像二階導數的變異數。分數越低代表邊緣越平滑（模糊）。  
   * **判定**：低於設定閾值直接剔除，防堵低清畫質。  
> 4. **InsightFace 幾何純淨度掃描器 (Geometry Scanner)**  
   * **原理**：利用 det\_10g.onnx 模型進行高強度掃描。  
   * **判定**：人臉數量 \!= 1 或佔比低於 5% 直接剔除。  
> 5. **3D 歐拉角多樣性分類器 (Euler Angles Classifier)**  
   * **原理**：提取 pose 參數 (Pitch, Yaw, Roll)，將照片標籤化為「正臉、左/右側臉、仰/俯角」，確保特徵多樣性。

### **三、 系統實作程式碼 (Python)**

請將以下程式碼儲存為 **scripts/auto\_curator.py**。此版本已全面升級，完美支援跨硬體執行與自動化建檔：

Python  
import os  
import cv2  
import shutil  
import numpy as np  
import onnxruntime as ort

\# \=========================================================================  
\# ⚠️ 環境劫持防護網 (僅在有 GPU 的環境下會發揮作用，CPU 環境自動略過)  
\# \=========================================================================  
import ctypes  
try:  
    import site  
    site\_pkg \= site.getsitepackages()\[0\]  
    cuda\_runtime\_path \= os.path.join(site\_pkg, 'nvidia', 'cuda\_runtime', 'lib', 'libcudart.so.12')  
    if os.path.exists(cuda\_runtime\_path):  
        ctypes.CDLL(cuda\_runtime\_path, mode=ctypes.RTLD\_GLOBAL)  
except Exception:  
    pass

import insightface  
from insightface.app import FaceAnalysis

class LoRACurator:  
    def \_\_init\_\_(self, blur\_threshold=100.0, min\_face\_ratio=0.05):  
        self.blur\_threshold \= blur\_threshold  
        self.min\_face\_ratio \= min\_face\_ratio  
          
        print("\\n" \+ "="\*50)  
        print("🚀 \[Init\] 啟動黃金資料篩選引擎 (Auto-Curation Engine)")  
        print("="\*50)  
          
        \# \---------------------------------------------------------  
        \# 🌟 核心需求一：智慧環境偵測 (自動判斷 GPU / CPU)  
        \# \---------------------------------------------------------  
        available\_providers \= ort.get\_available\_providers()  
        if 'CUDAExecutionProvider' in available\_providers:  
            print("\>\> 🟢 \[環境偵測\] 偵測到 GPU (CUDA)，將啟動極速掃描模式。")  
            \# 套用嚴格的 2GB 記憶體限制，防止 OOM  
            cuda\_options \= {"gpu\_mem\_limit": int(2 \* 1024 \* 1024 \* 1024), "arena\_extend\_strategy": "kSameAsRequested"}  
            providers \= \[("CUDAExecutionProvider", cuda\_options), "CPUExecutionProvider"\]  
        else:  
            print("\>\> 🟡 \[環境偵測\] 未偵測到 GPU，系統自動切換為純 CPU 模式執行。")  
            print("\>\> 💡 (提示：CPU 模式較耗時，但評估品質與標準完全相同)")  
            providers \= \["CPUExecutionProvider"\]  
              
        self.app \= FaceAnalysis(name='buffalo\_l', providers=providers)  
        \# 訓練圖的偵測標準要嚴格 (det\_thresh=0.6)，抓不到的代表特徵不夠明顯，直接淘汰  
        self.app.prepare(ctx\_id=0, det\_size=(640, 640), det\_thresh=0.6)

    def measure\_blurriness(self, image):  
        """計算影像銳利度 (分數越高越清晰)"""  
        gray \= cv2.cvtColor(image, cv2.COLOR\_BGR2GRAY)  
        return cv2.Laplacian(gray, cv2.CV\_64F).var()

    def get\_pose\_label(self, pose):  
        """解析 3D 角度 (Pitch:上下, Yaw:左右, Roll:傾斜)"""  
        pitch, yaw, roll \= pose  
        if yaw \> 25: return "右側臉"  
        if yaw \< \-25: return "左側臉"  
        if pitch \> 20: return "俯角"  
        if pitch \< \-20: return "仰角"  
        return "標準正臉"

    def evaluate\_image(self, img\_path):  
        img \= cv2.imread(img\_path)  
        if img is None: return False, "Reject\_ReadError", None

        \# 1\. 模糊度測試  
        blur\_score \= self.measure\_blurriness(img)  
        if blur\_score \< self.blur\_threshold:  
            return False, f"Reject\_Blurry", f"(銳利度 {blur\_score:.1f} 不及格)"

        \# 2\. 人臉偵測  
        faces \= self.app.get(img)  
        if len(faces) \== 0: return False, "Reject\_NoFace", "(找不到符合標準的人臉)"  
        if len(faces) \> 1: return False, "Reject\_MultiFace", f"(偵測到 {len(faces)} 張臉)"

        face \= faces\[0\]  
          
        \# 3\. 人臉比例測試  
        h, w \= img.shape\[:2\]  
        box\_area \= (face.bbox\[2\] \- face.bbox\[0\]) \* (face.bbox\[3\] \- face.bbox\[1\])  
        face\_ratio \= box\_area / (h \* w)  
        if face\_ratio \< self.min\_face\_ratio:  
            return False, "Reject\_TooSmall", f"(臉部佔比 {face\_ratio\*100:.1f}% 太小)"

        \# 4\. 角度分類與數據匯出  
        pose\_label \= self.get\_pose\_label(face.pose)  
          
        return True, "Accepted", f"\[{pose\_label}\] 銳利度:{blur\_score:.0f}, 佔比:{face\_ratio\*100:.1f}%"

def run\_curation(input\_dir):  
    \# \---------------------------------------------------------  
    \# 🌟 核心需求二：在專案底下動態建置專屬的輸出目錄  
    \# \---------------------------------------------------------  
    \# 取得當前執行目錄 (專案根目錄)  
    project\_root \= os.getcwd()  
    output\_dir \= os.path.join(project\_root, "LoRA\_Curated\_Dataset")  
    accepted\_dir \= os.path.join(output\_dir, "Accepted")  
      
    \# 確保目錄存在，若已存在則不影響  
    os.makedirs(accepted\_dir, exist\_ok=True)

    if not os.path.exists(input\_dir):  
        print(f"❌ 錯誤：找不到來源資料夾 {input\_dir}，請確認路徑。")  
        return

    \# 初始化過濾器 (設定銳利度及格線為 150，要求較高)  
    curator \= LoRACurator(blur\_threshold=150.0)  
      
    print(f"\\n📂 開始掃描目錄: {input\_dir}")  
    print(f"📁 合格圖檔將儲存至: {accepted\_dir}")  
    print("-" \* 50)  
      
    stats \= {"Accepted": 0, "Rejected": 0}  
      
    for filename in os.listdir(input\_dir):  
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):   
            continue  
              
        filepath \= os.path.join(input\_dir, filename)  
        passed, status\_dir, info \= curator.evaluate\_image(filepath)  
          
        if passed:  
            print(f"✅ \[合格\] {filename} \-\> {info}")  
            shutil.copy(filepath, os.path.join(accepted\_dir, filename))  
            stats\["Accepted"\] \+= 1  
        else:  
            print(f"❌ \[剔除\] {filename} \-\> {info}")  
            \# 依據死因 (status\_dir) 建立專屬剔除資料夾  
            rej\_dir \= os.path.join(output\_dir, status\_dir)  
            os.makedirs(rej\_dir, exist\_ok=True)  
            shutil.copy(filepath, os.path.join(rej\_dir, filename))  
            stats\["Rejected"\] \+= 1

    print("\\n" \+ "="\*50)  
    print(f"📊 篩選完成報告")  
    print(f"總處理數: {stats\['Accepted'\] \+ stats\['Rejected'\]} 張")  
    print(f"✨ 黃金訓練圖 (合格): {stats\['Accepted'\]} 張 (已存入 {accepted\_dir})")  
    print(f"🗑️ 淘汰圖片 (不合格): {stats\['Rejected'\]} 張")  
    print("="\*50)

if \_\_name\_\_ \== "\_\_main\_\_":  
    \# 執行時請確保專案根目錄下有一個名為 'raw\_dataset' 的資料夾，裡面放滿候選圖片  
    run\_curation(input\_dir="./raw\_dataset")

只要把您用手機或網路抓來的大量圖片全部丟進專案底下的 raw\_dataset 資料夾，不管是在配備高階顯卡的桌機，還是普通的文書筆電上執行這支腳本，它都會幫您自動把最精華的臉部數據，整齊地歸檔到 LoRA\_Curated\_Dataset/Accepted/ 當中！