import os
import sys
import json
import time
import argparse
import datetime
import cv2
import numpy as np

# =========================================================================
# 加入 config 設定檔
# =========================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_face_ids_from_registry(registry_path):
    if not os.path.exists(registry_path):
        return []
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return list(data.keys())
    except Exception as e:
        print(f"❌ 讀取 {registry_path} 失敗: {e}")
        return []

def get_physical_limits(registry_path, face_id):
    if not os.path.exists(registry_path):
        return None
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if face_id in data and "physical_limits" in data[face_id]:
                return data[face_id]["physical_limits"]
    except Exception:
        pass
    return None

def save_physical_limits(registry_path, face_id, limits):
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if face_id in data:
            data[face_id]["physical_limits"] = limits
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"💾 動態計算之極限邊界已寫回註冊表。")
    except Exception as e:
        print(f"⚠️ 寫回註冊表失敗: {e}")

def pre_flight_checks(args):
    print("\n" + "="*50)
    print("🛫 [Pre-flight Check] 啟動底圖安檢雷達...")
    print("="*50)
    
    # 1. 確認 lora_registry.json
    registry_path = os.path.join(config.PROJECT_ROOT, "lora_registry.json")
    if not os.path.exists(registry_path):
        print(f"❌ [錯誤] 找不到註冊表: {registry_path}")
        print("💡 請確保專案根目錄下有 lora_registry.json")
        sys.exit(1)
        
    face_ids = get_face_ids_from_registry(registry_path)
    if not face_ids:
        print(f"❌ [錯誤] 註冊表內沒有任何 Face ID 或讀取失敗")
        sys.exit(1)
        
    # 互動式選擇 Face ID
    print("📋 已註冊的 Face ID 列表:")
    for i, fid in enumerate(face_ids, 1):
        print(f"  [{i}] {fid}")
        
    selected_fid = None
    while not selected_fid:
        try:
            choice = input(f"👉 請輸入目標 Face ID 編號 (1-{len(face_ids)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(face_ids):
                selected_fid = face_ids[idx]
            else:
                print("❌ 輸入錯誤，請輸入有效的編號。")
        except ValueError:
             print("❌ 輸入無效。")
    print(f"✅ 已選定 Face ID: {selected_fid}")
    
    # 2. 檢查並允許互動式修正 INPUT_DIR
    input_dir = args.input_dir or config.INPUT_DIR
    while not os.path.exists(input_dir):
        print(f"❌ [錯誤] 找不到來源資料夾: {input_dir}")
        user_input = input("💡 請輸入有效的來源資料夾路徑 (或按 Ctrl+C 結束): ").strip()
        if user_input:
            input_dir = user_input
    print(f"✅ [檢查通過] 目標底圖目錄存在: {input_dir}")
    
    # 3. 確保 Global Cache 與 Task List 目錄存在
    os.makedirs(os.path.dirname(config.GLOBAL_POSE_CACHE_FILE), exist_ok=True)
    os.makedirs(config.TASK_LIST_DIR, exist_ok=True)

    print("="*50)
    print("🚀 前置檢查完畢\n")
    
    return {
        "input_dir": input_dir,
        "face_id": selected_fid,
        "registry_path": registry_path
    }

def load_global_cache():
    if not os.path.exists(config.GLOBAL_POSE_CACHE_FILE):
        return {}
    try:
        with open(config.GLOBAL_POSE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_global_cache(cache_data):
    with open(config.GLOBAL_POSE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=4, ensure_ascii=False)

def build_dynamic_limits_if_needed(face_id, app):
    """如果 registry 裡沒有 limits，就掃描 Golden Images 建立"""
    # 黃金圖路徑依照回饋：在 OUTPUT_DIR 下的 {face_id}/Accepted
    golden_dir = os.path.join(config.OUTPUT_DIR, face_id, "Accepted")
    if not os.path.exists(golden_dir):
        print(f"❌ [錯誤] 找不到黃金訓練圖目錄來建立基準: {golden_dir}")
        sys.exit(1)
        
    print(f"🔍 正在掃描 {face_id} 的黃金訓練圖以建立絕對物理邊界...")
    
    aspect_ratios = []
    yaws = []
    pitches = []
    rolls = []
    
    for filename in os.listdir(golden_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')): continue
        filepath = os.path.join(golden_dir, filename)
        img = cv2.imread(filepath)
        if img is None: continue
        
        faces = app.get(img)
        if len(faces) == 0: continue
        
        # 只取面積最大的一張臉
        face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
        
        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]
        aspect_ratios.append(w / h)
        
        pitch, yaw, roll = face.pose
        pitches.append(pitch)
        yaws.append(yaw)
        rolls.append(roll)
        
    if not aspect_ratios:
         print(f"❌ [錯誤] 無法從黃金圖中提取特徵，請確認圖片有效性。")
         sys.exit(1)
         
    # 建立邊界，給予些許寬容度 (例如向外擴張 10%)
    limits = {
        "aspect_ratio": [min(aspect_ratios) * 0.9, max(aspect_ratios) * 1.1],
        "yaw": [min(yaws) - 10, max(yaws) + 10],
        "pitch": [min(pitches) - 10, max(pitches) + 10],
        "roll": [min(rolls) - 10, max(rolls) + 10]
    }
    
    print(f"✅ 動態邊界建立完成: {limits}")
    return limits

def run_auditor(run_config):
    input_dir = run_config["input_dir"]
    face_id = run_config["face_id"]
    registry_path = run_config["registry_path"]
    
    print(f"\n📂 開始盤點目標目錄: {input_dir}")
    
    # 1. 輕量盤點 (純 CPU 字串比對)
    cache = load_global_cache()
    if face_id not in cache:
        cache[face_id] = {"accept": [], "reject": []}
        
    face_cache = cache[face_id]
    accepted_paths = set(face_cache["accept"])
    rejected_paths = set(face_cache["reject"])
    
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    history_accept = []
    history_reject = []
    unknown = []
    
    # 標準化絕對路徑做比對
    for fpath in all_files:
        abs_path = os.path.abspath(fpath)
        if abs_path in accepted_paths:
            history_accept.append(abs_path)
        elif abs_path in rejected_paths:
            history_reject.append(abs_path)
        else:
            unknown.append(abs_path)
            
    print("\n📊 盤點戰情報告:")
    print(f"✅ 歷史已接受: {len(history_accept)} 張")
    print(f"❌ 歷史已拒絕: {len(history_reject)} 張")
    print(f"❓ 未知待檢定: {len(unknown)} 張")
    
    if len(unknown) == 0:
        print("\n🎉 無未知圖片需要檢定。")
        finalize_task(face_id, history_accept, [], cache)
        return

    # 2. 互動閥門
    choice = input(f"\n👉 是否準備啟動 GPU 載入模型，檢定這 {len(unknown)} 張未知圖片？(y/N): ")
    if choice.lower() != 'y':
        print("🛑 已取消模型檢定，程式結束。")
        return
        
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

    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    # 初始化 InsightFace 模型
    print("\n" + "="*50)
    print("🤖 載入 InsightFace 模型...")
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        print(">> 🟢 [環境偵測] 偵測到 GPU (CUDA)，將啟動極速掃描模式。")
        cuda_options = {"gpu_mem_limit": int(2 * 1024 * 1024 * 1024), "arena_extend_strategy": "kSameAsRequested"}
        providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
    else:
        print(">> 🟡 [環境偵測] 未偵測到 GPU，系統自動切換為純 CPU 模式執行。")
        providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(name=config.MODEL_NAME, providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=config.DET_THRESH)
    
    # 3. 讀取或建立物理極限
    limits = get_physical_limits(registry_path, face_id)
    if not limits:
        print("⚠️ 註冊表中無 physical_limits，嘗試動態掃描黃金訓練圖...")
        limits = build_dynamic_limits_if_needed(face_id, app)
        save_physical_limits(registry_path, face_id, limits)
        
    # 4. 未知底圖安檢執行
    print(f"\n🔍 開始物理極限精準掃描 {len(unknown)} 張未知圖片...")
    print(">> 為避免畫面洗版，詳細明細將寫入 Log 檔，此處僅顯示進度。")
    new_accept = []
    new_reject = []
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"audit_log_{face_id}_{timestamp}.txt"
    log_filepath = os.path.join(config.TASK_LIST_DIR, log_filename)
    
    with open(log_filepath, "w", encoding="utf-8") as log_f:
        log_f.write(f"=== {face_id} 底圖安檢詳細紀錄 ===\n\n")
        
        for idx, fpath in enumerate(unknown):
            img = cv2.imread(fpath)
            if img is None:
                new_reject.append(fpath)
                log_f.write(f"❌ [Reject] {fpath} -> 原因: 無法讀取圖片\n")
                continue
                
            faces = app.get(img)
            if len(faces) == 0:
                 new_reject.append(fpath)
                 log_f.write(f"❌ [Reject] {fpath} -> 原因: 找不到人臉\n")
                 continue
                 
            # 只取最明顯的人臉做檢查
            face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
            
            # 計算各項數值
            h, w_img = img.shape[:2]
            box_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            face_ratio = box_area / (h * w_img)
            
            face_w = face.bbox[2] - face.bbox[0]
            face_h = face.bbox[3] - face.bbox[1]
            ar = face_w / face_h
            pitch, yaw, roll = face.pose
            
            reasons = []
            
            if face_ratio < config.MIN_FACE_RATIO:
                reasons.append(f"佔比太小 {face_ratio*100:.1f}%")
                
            if not (limits["aspect_ratio"][0] <= ar <= limits["aspect_ratio"][1]):
                reasons.append(f"臉型比例不符 {ar:.2f}")
                
            if not (limits["yaw"][0] <= yaw <= limits["yaw"][1]):
                reasons.append(f"左右轉頭超標 Yaw:{yaw:.1f}")
                
            if not (limits["pitch"][0] <= pitch <= limits["pitch"][1]):
                reasons.append(f"上下仰俯超標 Pitch:{pitch:.1f}")
                
            if not (limits["roll"][0] <= roll <= limits["roll"][1]):
                reasons.append(f"歪頭超標 Roll:{roll:.1f}")
                
            if reasons:
                new_reject.append(fpath)
                log_f.write(f"❌ [Reject] {fpath}\n")
                log_f.write(f"   -> 綜合失敗原因: {', '.join(reasons)}\n")
            else:
                new_accept.append(fpath)
                log_f.write(f"✅ [Accept] {fpath}\n")
                log_f.write(f"   -> 數據: 佔比:{face_ratio*100:.1f}%, 比例:{ar:.2f}, Yaw:{yaw:.1f}, Pitch:{pitch:.1f}, Roll:{roll:.1f}\n")
                
            if idx % 10 == 0 or idx == len(unknown) - 1:
                print(f"⏳ 掃描進度: {idx+1}/{len(unknown)} ...", end='\r')
                
    print(f"\n✅ 掃描完成！單筆詳細紀錄已匯出至: {log_filepath}")
        
    # 5. 寫回快取與產出工單
    finalize_task(face_id, history_accept, new_accept, cache, new_reject)

def finalize_task(face_id, history_accept, new_accept, cache, new_reject=None):
    if new_reject is None: new_reject = []
    
    # 全域更新 (不污染其他 Face_ID)
    if new_accept:
        cache[face_id]["accept"].extend(new_accept)
    if new_reject:
        cache[face_id]["reject"].extend(new_reject)
        
    if new_accept or new_reject:
        save_global_cache(cache)
        print("\n💾 盤點紀錄已寫回 Global Cache。")
        
    # 產出工單
    all_task_accept = history_accept + new_accept
    
    if not all_task_accept:
        print("⚠️ 本次任務沒有任何合格底圖可產出工單。")
        return
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 修正時間格式 %Y%m%d
    task_filename = f"task_{face_id}_{timestamp}.txt"
    task_filepath = os.path.join(config.TASK_LIST_DIR, task_filename)
    
    with open(task_filepath, "w", encoding="utf-8") as f:
        for path in all_task_accept:
            f.write(f"{path}\n")
            
    print("="*50)
    print(f"🧾 工單產出完成!")
    print(f"檔案路徑: {task_filepath}")
    print(f"合格底圖數量: {len(all_task_accept)} 張")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="底圖防呆安檢雷達 (Phase 1)")
    parser.add_argument("--input_dir", type=str, default=None, help="覆寫來源圖片資料夾路徑")
    args = parser.parse_args()
    
    run_config = pre_flight_checks(args)
    run_auditor(run_config)
