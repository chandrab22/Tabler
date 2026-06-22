import os
from dotenv import dotenv_values, find_dotenv, load_dotenv

_DOTENV_PATH = find_dotenv(usecwd=True)
load_dotenv(_DOTENV_PATH or None)
_DOTENV_VALUES = dotenv_values(_DOTENV_PATH) if _DOTENV_PATH else {}


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_str(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = str(raw).strip()
    return raw if raw else default


def _pct(value):
    return float(value) / 100.0


def _same_bucket_rules(thresholds):
    return {
        cls_name: {"strong": threshold, "medium": threshold}
        for cls_name, threshold in thresholds.items()
    }

# --- Model Settings ---
PIPELINE_PROFILE = _env_str("PIPELINE_PROFILE", "local_45_46")
PIPELINE_PROFILE_DEFAULTS = {
    "api_45_46_optimal": {
        "general_id": "tabler/45",
        "general_confidence": 1,
        "column_id": "tabler/46",
        "column_confidence": 1,
    },
    "local_45_46": {
        "general_id": "tabler/45",
        "general_confidence": 1,
        "column_id": "tabler/46",
        "column_confidence": 1,
    },
}
_profile_defaults = PIPELINE_PROFILE_DEFAULTS.get(
    PIPELINE_PROFILE,
    PIPELINE_PROFILE_DEFAULTS["local_45_46"],
)

MODEL_GENERAL_ID = _env_str("MODEL_GENERAL_ID", _profile_defaults["general_id"])
MODEL_GENERAL_CONFIDENCE = _env_int(
    "MODEL_GENERAL_CONFIDENCE",
    _profile_defaults["general_confidence"],
)

MODEL_COLUMN_ID = _env_str("MODEL_COLUMN_ID", _profile_defaults["column_id"])
MODEL_COLUMN_CONFIDENCE = _env_int(
    "MODEL_COLUMN_CONFIDENCE",
    _profile_defaults["column_confidence"],
)

MODEL_GENERAL_SOURCE = "general"
MODEL_COLUMN_SOURCE = "column"

ENABLE_GENERAL_COLUMN_CLASSES = _env_flag("ENABLE_GENERAL_COLUMN_CLASSES", default=False)

MODEL_ALL_CLASSES_OVERLAP = 50
ROBOFLOW_INFER_COORD_SPACE = _env_str("ROBOFLOW_INFER_COORD_SPACE", "original").lower()
if ROBOFLOW_INFER_COORD_SPACE not in {"original", "preprocessed"}:
    raise ValueError(
        "ROBOFLOW_INFER_COORD_SPACE must be either 'original' or 'preprocessed'."
    )

# --- Extraction Thresholds ---
IOS_THRESH_NMS_DEFAULT = 0.7
IOS_THRESH_COLUMNS = 0.3
IOS_THRESH_ROWS = 0.75
IOS_THRESH_SPANS = 0.7
INTERSECTION_THRESH_OCR_GRID = 0.5
INTERSECTION_THRESH_SPLIT = 0.6
INTERSECTION_THRESH_SPAN_CELL = 0.8

MEDIAN_HEIGHT_DEFAULT = 35
HEADER_STACK_GAP_MARGIN_ROWS = 0.20
HEADER_BODY_TOUCH_TOL_PX = 3.0
HEADER_BODY_TOUCH_TOL_FRAC = 0.10
HEADER_BODY_SOFT_INTRUSION_FRAC = 0.18
HEADER_BODY_HARD_INTRUSION_HEADER_FRAC = 0.45
HEADER_BODY_HARD_INTRUSION_ROW_FRAC = 0.40
COLUMN_HEADER_MIN_SUPPORTED_WIDTH_FRAC = 0.60
COLUMN_HEADER_MIN_WIDTH_HEIGHT_MULT = 1.10
SPANNING_HEADER_MIN_SUPPORTED_WIDTH_FRAC = 0.60
SPANNING_HEADER_MIN_WIDTH_HEIGHT_MULT = 1.10
SPANNING_HEADER_SINGLE_COL_WIDTH_MULT = 1.10

# --- Class Names ---
CLASS_NAME_COLUMN = "column"
CLASS_NAME_ROW = "row"
CLASS_NAME_COLUMN_HEADER = "column_header"
CLASS_NAME_ROW_HEADER = "row_header"
CLASS_NAME_TABLE_HEADER = "table_header"
CLASS_NAME_SPLITTED_COLUMN = "splitted_column"
CLASS_NAME_SPANNING_ROW_CELL = "spanning_row_cell"
CLASS_NAME_SPANNING_COLUMN_CELL = "spanning_column_cell"
CLASS_NAME_SPANNING_COLUMN_HEADER = "spanning_column_header"
CLASS_NAME_TEXT = "text"

# Per-model class ownership used when merging the two Roboflow outputs.
MODEL_ALLOWED_CLASSES = {
    MODEL_GENERAL_SOURCE: {
        CLASS_NAME_ROW,
        CLASS_NAME_ROW_HEADER,
        CLASS_NAME_COLUMN_HEADER,
        CLASS_NAME_SPANNING_COLUMN_CELL,
        CLASS_NAME_SPANNING_COLUMN_HEADER,
        CLASS_NAME_SPANNING_ROW_CELL,
        CLASS_NAME_TABLE_HEADER,
        CLASS_NAME_TEXT,
    },
    MODEL_COLUMN_SOURCE: {
        CLASS_NAME_COLUMN,
        CLASS_NAME_SPLITTED_COLUMN,
    },
}

if ENABLE_GENERAL_COLUMN_CLASSES:
    MODEL_ALLOWED_CLASSES[MODEL_GENERAL_SOURCE].update(
        {
            CLASS_NAME_COLUMN,
            CLASS_NAME_SPLITTED_COLUMN,
        }
    )

_PROFILE_45_GENERAL_CLASS_THRESHOLDS = {
    CLASS_NAME_COLUMN: _pct(51),
    CLASS_NAME_COLUMN_HEADER: _pct(39),
    CLASS_NAME_ROW: _pct(41),
    CLASS_NAME_ROW_HEADER: _pct(48),
    CLASS_NAME_SPANNING_COLUMN_CELL: _pct(20),
    CLASS_NAME_SPANNING_COLUMN_HEADER: _pct(39),
    CLASS_NAME_SPANNING_ROW_CELL: _pct(20),
    CLASS_NAME_SPLITTED_COLUMN: _pct(15),
    CLASS_NAME_TABLE_HEADER: _pct(29),
    CLASS_NAME_TEXT: _pct(28),
}
_PROFILE_46_COLUMN_CLASS_THRESHOLDS = {
    CLASS_NAME_COLUMN: _pct(31),
    CLASS_NAME_SPLITTED_COLUMN: _pct(37),
}
_PROFILE_45_GENERAL_PREDICTION_RULES = _same_bucket_rules(
    _PROFILE_45_GENERAL_CLASS_THRESHOLDS
)
_PROFILE_46_COLUMN_PREDICTION_RULES = _same_bucket_rules(
    _PROFILE_46_COLUMN_CLASS_THRESHOLDS
)
_PROFILE_45_46_FALLBACK_PREDICTION_RULES = {}
_PROFILE_45_46_FALLBACK_PREDICTION_RULES.update(_PROFILE_46_COLUMN_PREDICTION_RULES)
_PROFILE_45_46_FALLBACK_PREDICTION_RULES.update(_PROFILE_45_GENERAL_PREDICTION_RULES)

if PIPELINE_PROFILE in {"api_45_46_optimal", "local_45_46"}:
    _general_class_thresholds = dict(_PROFILE_45_GENERAL_CLASS_THRESHOLDS)
    _column_class_thresholds = dict(_PROFILE_46_COLUMN_CLASS_THRESHOLDS)
    _general_prediction_rules = dict(_PROFILE_45_GENERAL_PREDICTION_RULES)
    _column_prediction_rules = dict(_PROFILE_46_COLUMN_PREDICTION_RULES)
    _fallback_prediction_rules = dict(_PROFILE_45_46_FALLBACK_PREDICTION_RULES)
else:
    _general_class_thresholds = dict(_PROFILE_45_GENERAL_CLASS_THRESHOLDS)
    _column_class_thresholds = dict(_PROFILE_46_COLUMN_CLASS_THRESHOLDS)
    _general_prediction_rules = dict(_PROFILE_45_GENERAL_PREDICTION_RULES)
    _column_prediction_rules = dict(_PROFILE_46_COLUMN_PREDICTION_RULES)
    _fallback_prediction_rules = dict(_PROFILE_45_46_FALLBACK_PREDICTION_RULES)

# Per-model minimum confidence used before geometry validation.
MODEL_CLASS_THRESHOLDS = {
    MODEL_GENERAL_SOURCE: _general_class_thresholds,
    MODEL_COLUMN_SOURCE: _column_class_thresholds,
}

if ENABLE_GENERAL_COLUMN_CLASSES:
    for class_name in (CLASS_NAME_COLUMN, CLASS_NAME_SPLITTED_COLUMN):
        MODEL_CLASS_THRESHOLDS[MODEL_GENERAL_SOURCE].setdefault(
            class_name,
            MODEL_CLASS_THRESHOLDS[MODEL_COLUMN_SOURCE][class_name],
        )

# Model-specific prediction confidence buckets used during post-processing.
# Strong detections are trusted directly; medium detections must be supported
# by OCR/geometry; weak detections are ignored.
MODEL_PREDICTION_CONFIDENCE_RULES = {
    MODEL_GENERAL_SOURCE: _general_prediction_rules,
    MODEL_COLUMN_SOURCE: _column_prediction_rules,
}

if ENABLE_GENERAL_COLUMN_CLASSES:
    for class_name in (CLASS_NAME_COLUMN, CLASS_NAME_SPLITTED_COLUMN):
        MODEL_PREDICTION_CONFIDENCE_RULES[MODEL_GENERAL_SOURCE].setdefault(
            class_name,
            MODEL_PREDICTION_CONFIDENCE_RULES[MODEL_COLUMN_SOURCE][class_name],
        )

# Fallback rules for predictions that do not carry source_model metadata.
PREDICTION_CONFIDENCE_RULES = dict(_fallback_prediction_rules)

# --- OCR Settings ---
OCR_PROFILE = _env_str("OCR_PROFILE", "baseline")
OCR_LANG = "en"
OCR_USE_ANGLE_CLS = _env_flag("OCR_USE_ANGLE_CLS", default=(OCR_PROFILE == "baseline"))
OCR_ENABLE_MKLDNN = False
OCR_USE_DOC_ORIENTATION = False
OCR_USE_DOC_UNWARPING = False
OCR_USE_TEXTLINE_ORIENTATION = OCR_USE_ANGLE_CLS
if OCR_PROFILE == "fast":
    OCR_TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
    OCR_TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
elif OCR_PROFILE == "baseline":
    OCR_TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
    OCR_TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
else:
    raise ValueError(f"Unsupported OCR_PROFILE: {OCR_PROFILE}")
OCR_CROP_MARGIN_X_RATIO = 0.04
OCR_CROP_MARGIN_Y_RATIO = 0.05
OCR_CROP_MARGIN_MIN = 40

# --- Directory Settings ---
INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
OUTPUT_PROFILE_DIR = os.path.join(OUTPUT_DIR, OCR_PROFILE)
PREPROCESSED_DIR = os.getenv("PREPROCESSED_DIR", "preprocessed_input")
KEEP_PREPROCESSED_IMAGES = _env_flag("KEEP_PREPROCESSED_IMAGES", default=False)
