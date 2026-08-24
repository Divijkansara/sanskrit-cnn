"""
Central configuration for the Sanskrit (Devanagari) character CNN project.

Every path in the project is derived from PROJECT_ROOT so the code runs
identically on any machine, from any working directory.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # untouched download
PROCESSED_DIR = DATA_DIR / "processed"     # preprocessed .npz splits

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"
SAMPLE_DIR = PROJECT_ROOT / "samples"      # a few loose PNGs for predict.py demos

for _d in (RAW_DIR, PROCESSED_DIR, FIGURE_DIR, METRIC_DIR, MODEL_DIR, SAMPLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# DHCD (Devanagari Handwritten Character Dataset), Acharya et al., SKIMA 2015.
# Mirrored by the original authors as a single .npz inside a public GitHub repo.
DATASET_REPO = "https://github.com/Prasanna1991/DHCD_Dataset.git"
DATASET_NPZ_RELPATH = "dataset/dataset.npz"
RAW_NPZ = RAW_DIR / "dhcd.npz"

IMAGE_SIZE = 32          # images are resized to IMAGE_SIZE x IMAGE_SIZE
NUM_CHANNELS = 1         # grayscale
NUM_CLASSES = 46         # 36 consonants + 10 digits

# Pixel statistics of the DHCD training split (computed in data_prep.py and
# rewritten there; these are the values that ship with the repo).
NORM_MEAN = 0.2400
NORM_STD = 0.3865

VAL_FRACTION = 0.10      # carved out of the official training split
SEED = 42

# --------------------------------------------------------------------------- #
# Training hyper-parameters
# --------------------------------------------------------------------------- #
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 2
EARLY_STOP_PATIENCE = 6

BEST_MODEL_PATH = MODEL_DIR / "sanskrit_cnn_best.pt"
HISTORY_PATH = METRIC_DIR / "training_history.csv"

# --------------------------------------------------------------------------- #
# Class names
# --------------------------------------------------------------------------- #
# Order verified by rendering one sample per integer label from the raw .npz.
# Raw labels are 1..46; we store them 0-indexed everywhere downstream.
CLASS_NAMES = [
    "ka",   "kha",  "ga",   "gha",  "nga",
    "cha",  "chha", "ja",   "jha",  "nya",
    "ta",   "tha",  "da",   "dha",  "na",       # retroflex (ट ठ ड ढ ण)
    "taa",  "thaa", "daa",  "dhaa", "naa",      # dental   (त थ द ध न)
    "pa",   "pha",  "ba",   "bha",  "ma",
    "ya",   "ra",   "la",   "va",
    "sha",  "shha", "sa",   "ha",
    "ksha", "tra",  "gya",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

DEVANAGARI = [
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह",
    "क्ष", "त्र", "ज्ञ",
    "०", "१", "२", "३", "४", "५", "६", "७", "८", "९",
]

assert len(CLASS_NAMES) == NUM_CLASSES, len(CLASS_NAMES)
assert len(DEVANAGARI) == NUM_CLASSES, len(DEVANAGARI)


def label_text(idx: int) -> str:
    """Human-readable label, e.g. 'ka (क)'."""
    return f"{CLASS_NAMES[idx]} ({DEVANAGARI[idx]})"
