import os
import sys
import cv2
import shutil
import argparse
import numpy as np
import onnxruntime as ort

# =========================================================================
# 加入 config 設定檔
# =========================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def pre_flight_checks(args):
    """
    起飛前置檢查程序 (Pre-flight checks):
    確認變數與模型狀態，避免載入模型後才發現環境配置不對，浪費時間。
    """
    print("\n" + "="*50)
    print("🛫 [Pre-flight Check] 啟動前置檢查程序...")
    print("="*50)
    
    # 1. 檢查並允許互動式修正 INPUT_DIR
    input_dir = args.input_dir or config.INPUT_DIR
    while not os.path.exists(input_dir):
        print(f"❌ [錯誤] 找不到來源資料夾: {input_dir}")
        user_input = input("💡 請輸入有效的來源資料夾路徑 (或按 Ctrl+C 結束): ").strip()
        if user_input:
            input_dir = user_input
    print(f"✅ [檢查通過] 來源資料夾存在: {input_dir}")
    
    # 2. 檢查並建置 OUTPUT_DIR
    output_dir = args.output_dir or config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        print(f"⚠️ [提示] 輸出資料夾不存在: {output_dir}")
        print(">> 系統將自動為您建置此目錄。")
        os.makedirs(output_dir, exist_ok=True)
    print(f"✅ [檢查通過] 輸出資料夾已確認: {output_dir}")

    # 3. 檢查變數型態與合理性 (與底層執行保持一致)
    blur_threshold = config.BLUR_THRESHOLD
    min_face_ratio = config.MIN_FACE_RATIO
    det_thresh = config.DET_THRESH
    
    assert isinstance(blur_threshold, (int, float)), "BLUR_THRESHOLD 必須為數值"
    assert isinstance(min_face_ratio, (int, float)), "MIN_FACE_RATIO 必須為數值"
    assert isinstance(det_thresh, (int, float)), "DET_THRESH 必須為數值"
        
    # 4. 檢查模型是否已下載完成
    model_dir = config.INSIGHTFACE_MODEL_DIR
    # InsightFace buffalo_l 模型核心檔案之一為 det_10g.onnx，以此判斷是否下載完整
    required_model_file = os.path.join(model_dir, "det_10g.onnx")
    if not os.path.exists(required_model_file):
        print(f"⚠️ [警告] 找不到 InsightFace 模型核心檔案: {required_model_file}")
        print(">> 系統將在稍後初始化時自動嘗試下載模型。此過程可能需要一些時間，請耐心等候。")
    else:
        print(f"✅ [檢查通過] 偵測到模型已完整下載: {model_dir}")
        
    print("="*50)
    print("🚀 所有變數與環境檢查完畢，準備起飛！\n")
    
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "blur_threshold": blur_threshold,
        "min_face_ratio": min_face_ratio,
        "det_thresh": det_thresh,
        "model_name": config.MODEL_NAME
    }

# =========================================================================
# ⚠️ 環境劫持防護網 (僅在有 GPU 的環境下會發揮作用，CPU 環境自動略過)
# =========================================================================
import ctypes
try:
    import site
    site_pkg = site.getsitepackages()[0]
    cuda_runtime_path = os.path.join(site_pkg, 'nvidia', 'cuda_runtime', 'lib', 'libcudart.so.12')
    if os.path.exists(cuda_runtime_path):
        ctypes.CDLL(cuda_runtime_path, mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

import insightface
from insightface.app import FaceAnalysis

class LoRACurator:
    def __init__(self, run_config):
        # 確保底層執行的變數與前置檢查完全一致，不產生歧義
        self.blur_threshold = run_config["blur_threshold"]
        self.min_face_ratio = run_config["min_face_ratio"]
        self.det_thresh = run_config["det_thresh"]
        self.model_name = run_config["model_name"]
        
        print("\n" + "="*50)
        print("🤖 [Init] 啟動黃金資料篩選引擎 (Auto-Curation Engine)")
        print("="*50)
        
        # ---------------------------------------------------------
        # 🌟 智慧環境偵測 (專為 PC 無 GPU 環境優化)
        # ---------------------------------------------------------
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            print(">> 🟢 [環境偵測] 偵測到 GPU (CUDA)，將啟動極速掃描模式。")
            cuda_options = {"gpu_mem_limit": int(2 * 1024 * 1024 * 1024), "arena_extend_strategy": "kSameAsRequested"}
            providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        else:
            print(">> 🟡 [環境偵測] 未偵測到 GPU，系統自動切換為純 CPU 模式執行。")
            print(">> 💡 (提示：本次專案主要針對 PC 環境執行，CPU 模式評估品質與標準完全相同)")
            providers = ["CPUExecutionProvider"]
            
        self.app = FaceAnalysis(name=self.model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=self.det_thresh)

    def get_image_hash(self, image, hash_size=8):
        """產生感知雜湊指紋 (dHash)"""
        # 將圖片轉為灰階並縮小為 9x8
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        # 比較相鄰像素的明暗差異
        diff = resized[:, 1:] > resized[:, :-1]
        # 將 64 個布林值轉換為一個 16 進位字串指紋 (以整數表示)
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    def measure_blurriness(self, image):
        """計算影像銳利度 (分數越高越清晰)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def get_pose_label(self, pose):
        """解析 3D 角度 (Pitch:上下, Yaw:左右, Roll:傾斜)"""
        pitch, yaw, roll = pose
        if yaw > 25: return "右側臉"
        if yaw < -25: return "左側臉"
        if pitch > 20: return "俯角"
        if pitch < -20: return "仰角"
        return "標準正臉"

    def evaluate_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return False, "Reject_ReadError", None

        # 1. 模糊度測試
        blur_score = self.measure_blurriness(img)
        if blur_score < self.blur_threshold:
            return False, "Reject_Blurry", f"(銳利度 {blur_score:.1f} 不及格)"

        # 2. 人臉偵測
        faces = self.app.get(img)
        if len(faces) == 0: return False, "Reject_NoFace", "(找不到符合標準的人臉)"
        if len(faces) > 1: return False, "Reject_MultiFace", f"(偵測到 {len(faces)} 張臉)"

        face = faces[0]
        
        # 3. 人臉比例測試
        h, w = img.shape[:2]
        box_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        face_ratio = box_area / (h * w)
        if face_ratio < self.min_face_ratio:
            return False, "Reject_TooSmall", f"(臉部佔比 {face_ratio*100:.1f}% 太小)"

        # 4. 角度分類與數據匯出
        pose_label = self.get_pose_label(face.pose)
        
        return True, "Accepted", f"[{pose_label}] 銳利度:{blur_score:.0f}, 佔比:{face_ratio*100:.1f}%"

def run_curation(run_config):
    # ---------------------------------------------------------
    # 🌟 動態建置專屬的輸出目錄
    # ---------------------------------------------------------
    output_dir = run_config["output_dir"]
    accepted_dir = os.path.join(output_dir, "Accepted")
    
    os.makedirs(accepted_dir, exist_ok=True)

    input_dir = run_config["input_dir"]

    curator = LoRACurator(run_config)
    
    print(f"\n📂 開始掃描目錄: {input_dir}")
    print(f"📁 合格圖檔將儲存至: {accepted_dir}")
    print("-" * 50)
    
    stats = {"Accepted": 0, "Rejected": 0, "Duplicated": 0}
    
    # --- 新增：角度統計字典 ---
    pose_counts = {
        "標準正臉": 0,
        "左側臉": 0,
        "右側臉": 0,
        "仰角": 0,
        "俯角": 0
    }
    
    # 建立指紋資料庫，掃描 Accepted 資料夾裡已有的圖片
    existing_hashes = set()
    existing_accepted_count = 0
    print("🔍 正在建立已收錄圖片的防重指紋庫與分析歷史資料...")
    if os.path.exists(accepted_dir):
        for existing_file in os.listdir(accepted_dir):
            if existing_file.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                img_path = os.path.join(accepted_dir, existing_file)
                img = cv2.imread(img_path)
                if img is not None:
                    existing_hashes.add(curator.get_image_hash(img))
                    existing_accepted_count += 1
                    
                    # 重新辨識既有圖片的姿勢，確保總數統計正確
                    faces = curator.app.get(img)
                    if len(faces) > 0:
                        pose_label = curator.get_pose_label(faces[0].pose)
                        if pose_label in pose_counts:
                            pose_counts[pose_label] += 1
                            
    print(f"🔒 防護網啟動：已紀錄 {len(existing_hashes)} 張圖的指紋，並載入歷史姿勢數據。\n")
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')): 
            continue
            
        filepath = os.path.join(input_dir, filename)
        
        # --- 新增的去重防護網 ---
        img = cv2.imread(filepath)
        if img is not None:
            img_hash = curator.get_image_hash(img)
            if img_hash in existing_hashes:
                print(f"♻️ [剔除] {filename} -> (重複圖片，已收錄過)")
                rej_dir = os.path.join(output_dir, "Reject_Duplicate")
                os.makedirs(rej_dir, exist_ok=True)
                shutil.copy(filepath, os.path.join(rej_dir, filename))
                stats["Duplicated"] += 1
                continue # 直接跳過後續評估
        # ------------------------
        
        passed, status_dir, info = curator.evaluate_image(filepath)
        
        if passed:
            print(f"✅ [合格] {filename} -> {info}")
            shutil.copy(filepath, os.path.join(accepted_dir, filename))
            stats["Accepted"] += 1
            existing_hashes.add(img_hash)
            
            # --- 新增：從 info 字串中萃取角度並計數 ---
            for pose_type in pose_counts.keys():
                if pose_type in info:
                    pose_counts[pose_type] += 1
                    break
        else:
            print(f"❌ [剔除] {filename} -> {info}")
            rej_dir = os.path.join(output_dir, status_dir)
            os.makedirs(rej_dir, exist_ok=True)
            shutil.copy(filepath, os.path.join(rej_dir, filename))
            stats["Rejected"] += 1

    print("\n" + "="*50)
    print(f"📊 篩選完成報告")
    print(f"總處理數: {stats['Accepted'] + stats['Rejected'] + stats['Duplicated']} 張")
    print(f"✨ 黃金訓練圖 (合格): {stats['Accepted']} 張 (已存入 {accepted_dir})")
    print(f"🗑️ 淘汰圖片 (不合格): {stats['Rejected']} 張")
    print(f"♻️ 重複圖片 (已收錄): {stats['Duplicated']} 張")
    print("="*50)

    # === 訓練集健康檢查與搜圖建議 ===
    print("\n" + "="*50)
    print(f"🩺 LoRA 訓練集健康度診斷報告")
    print("="*50)
    
    total_accepted = stats["Accepted"] + existing_accepted_count
    if total_accepted == 0:
        print("目前尚無任何合格圖片，請開始搜集資料！")
        return

    print("【目前整體角度庫存佔比 (含歷史收錄)】")
    for pose, count in pose_counts.items():
        percent = (count / total_accepted) * 100
        print(f"  - {pose:5s}: {count:2d} 張 ({percent:.1f}%)")
        
    print("\n【💡 AI 教練搜圖建議】")
    
    # 完美 LoRA 訓練集的建議配置目標：
    # 總數約 20~30 張。正臉 40%, 左右側臉各 20%, 仰/俯角各 10%
    target_total = 25 
    
    if total_accepted < 15:
        print(f"⚠️ [張數不足] 目前僅有 {total_accepted} 張。建議總量需達到 {target_total} 張，請再搜集至少 {target_total - total_accepted} 張高品質照片。")
    elif total_accepted > 35:
        print(f"⚠️ [張數過多] 目前有 {total_accepted} 張。資料量太大可能導致訓練失焦，建議手動刪除一些特徵過於相近的照片。")
    else:
        print(f"✅ [張數達標] 目前 {total_accepted} 張，落在 15~35 張的最佳訓練區間！")

    # 針對角度分佈給予嚴格指引
    front_ratio = pose_counts["標準正臉"] / total_accepted
    side_ratio = (pose_counts["左側臉"] + pose_counts["右側臉"]) / total_accepted
    
    if front_ratio > 0.6:
        print(f"🚨 [嚴重失衡] 正臉比例高達 {front_ratio*100:.0f}%！未來模型側臉會崩壞。")
        print("   👉 下一步行動：請『停止』搜集正面照！請專門去搜尋目標人物的【側臉】與【轉頭】照片。")
        
    if pose_counts["左側臉"] == 0 or pose_counts["右側臉"] == 0:
        print("🚨 [結構缺失] 缺乏側臉輪廓！AI 無法建構 3D 深度。")
        if pose_counts["左側臉"] == 0: print("   👉 請去搜尋目標人物【看向畫面左邊】的照片。")
        if pose_counts["右側臉"] == 0: print("   👉 請去搜尋目標人物【看向畫面右邊】的照片。")

    if pose_counts["仰角"] == 0 and pose_counts["俯角"] == 0:
        print("⚠️ [多樣性不足] 缺乏高低視角。這會限制未來合成仰角或低頭底圖的能力。")
        print("   👉 下一步行動：可尋找微抬下巴，或鏡頭從上往下拍的日常照片。")

    if front_ratio <= 0.6 and side_ratio >= 0.2 and pose_counts["左側臉"] > 0 and pose_counts["右側臉"] > 0:
        print("🌟 [完美平衡] 您的資料集角度非常均衡！具備了煉製極品 LoRA 的潛力！")
        
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 自動篩選引擎 (Auto-Curation Engine)")
    parser.add_argument("--input_dir", type=str, default=None, help="覆寫來源圖片資料夾路徑")
    parser.add_argument("--output_dir", type=str, default=None, help="覆寫輸出資料夾路徑")
    args = parser.parse_args()
    
    # 執行前置檢查並獲取最終執行變數
    run_config = pre_flight_checks(args)
    
    # 開始執行
    run_curation(run_config)
