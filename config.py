# =====================================================
# SoyGuard AI
# config.py
# =====================================================

import os

# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# =====================================================
# FOLDER
# =====================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

CROP_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "crops"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "reports"
)

# =====================================================
# MODEL
# =====================================================

DETECT_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "detect.pt"
)

CLASSIFY_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "classify.pt"
)

# =====================================================
# CREATE DIRECTORY
# =====================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)