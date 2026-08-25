import os

# --- Paths ---
PROJECT_ROOT = os.getcwd()
INPUT_DIR = os.path.join(PROJECT_ROOT, "raw_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "LoRA_Curated_Dataset")

# --- Model Variables ---
MODEL_NAME = "buffalo_l"
INSIGHTFACE_MODEL_DIR = os.path.expanduser(f"~/.insightface/models/{MODEL_NAME}")

# --- Curating Thresholds ---
BLUR_THRESHOLD = 150.0      # 拉普拉斯邊緣銳利度閾值 (Laplacian edge blur threshold)
MIN_FACE_RATIO = 0.05       # 最小人臉佔比 (Minimum face ratio)
DET_THRESH = 0.6            # 偵測信心閾值 (Detection threshold)
