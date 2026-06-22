import csv
import os
import re
import cv2
import numpy as np

from config import *


HEADER_ROW_TYPE = "__header_row__"
HEADER_GLOBAL_FALLBACK_INTERSECTION_THRESH = 0.15
ENABLE_ROW_SEPARATOR_SPLIT = os.getenv("ENABLE_ROW_SEPARATOR_SPLIT", "1").strip().lower() not in {"0", "false", "no", "off"}
ENABLE_COLUMN_SEPARATOR_SPLIT = os.getenv("ENABLE_COLUMN_SEPARATOR_SPLIT", "1").strip().lower() not in {"0", "false", "no", "off"}
ENABLE_SPLIT_COLUMN_SEPARATOR_RECOVERY = os.getenv("ENABLE_SPLIT_COLUMN_SEPARATOR_RECOVERY", "1").strip().lower() not in {"0", "false", "no", "off"}
ENABLE_HEADER_SPLIT_RECOVERY = os.getenv("ENABLE_HEADER_SPLIT_RECOVERY", "1").strip().lower() not in {"0", "false", "no", "off"}
ENABLE_LINE_BASED_SPAN_CELL_RECOVERY = os.getenv("ENABLE_LINE_BASED_SPAN_CELL_RECOVERY", "1").strip().lower() not in {"0", "false", "no", "off"}
STRUCTURE_LINE_DARK_PIXEL_MAX = int(os.getenv("STRUCTURE_LINE_DARK_PIXEL_MAX", "200"))


def normalize_class(cls):
    normalized = cls.replace(" ", "_").lower()
    aliases = {
        "rowheader": CLASS_NAME_ROW_HEADER,
        "tabler_header": CLASS_NAME_TABLE_HEADER,
    }
    return aliases.get(normalized, normalized)


def calculate_intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


def get_box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def o_to_box(ocr):
    opt = ocr["box"]
    # Handle list of lists (polygon) vs list of coords (box)
    if isinstance(opt[0], list):
        xs = [pt[0] for pt in opt]
        ys = [pt[1] for pt in opt]
        return [min(xs), min(ys), max(xs), max(ys)]
    return [
        min(pt[0] for pt in opt),
        min(pt[1] for pt in opt),
        max(pt[0] for pt in opt),
        max(pt[1] for pt in opt),
    ]


def get_box_center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def bbox_to_xyxy(pred):
    return [
        pred["x"] - pred["width"] / 2,
        pred["y"] - pred["height"] / 2,
        pred["x"] + pred["width"] / 2,
        pred["y"] + pred["height"] / 2,
    ]


def _nms_priority(pred):
    box = pred["box"]
    w = max(1.0, float(box[2] - box[0]))
    h = max(1.0, float(box[3] - box[1]))
    cls = pred["class"]
    conf = float(pred.get("confidence", 0.0))

    shape_bonus = 0.0
    if cls == CLASS_NAME_COLUMN:
        shape_bonus = _clamp01((h / w - 1.0) / 2.0)
    elif cls == CLASS_NAME_ROW:
        shape_bonus = _clamp01((w / h - 1.4) / 3.5)
    elif cls in {CLASS_NAME_COLUMN_HEADER, CLASS_NAME_SPANNING_COLUMN_HEADER, CLASS_NAME_TABLE_HEADER}:
        shape_bonus = _clamp01((w / h - 1.2) / 3.0)
    elif cls == CLASS_NAME_ROW_HEADER:
        shape_bonus = _clamp01((h / w - 0.8) / 1.5)
    elif cls == CLASS_NAME_SPLITTED_COLUMN:
        shape_bonus = _clamp01((h / w - 0.8) / 2.0)
    elif cls in {CLASS_NAME_SPANNING_ROW_CELL, CLASS_NAME_SPANNING_COLUMN_CELL}:
        shape_bonus = _clamp01(max(w / h, h / w) / 4.0)
    return conf + 0.05 * shape_bonus


def group_nms(boxes, ios_thresh=IOS_THRESH_NMS_DEFAULT):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (_nms_priority(b), b["confidence"]), reverse=True)
    keep = []
    for box in boxes:
        is_dupe = False
        for kept in keep:
            inter = calculate_intersection_area(box["box"], kept["box"])
            smaller_area = min(get_box_area(box["box"]), get_box_area(kept["box"]))
            if inter / (smaller_area + 1e-12) > ios_thresh:
                is_dupe = True
                break
        if not is_dupe:
            keep.append(box)
    return keep


def _prediction_bucket(pred):
    source_model = pred.get("source_model")
    rules = None
    if source_model in MODEL_PREDICTION_CONFIDENCE_RULES:
        rules = MODEL_PREDICTION_CONFIDENCE_RULES[source_model].get(pred["class"])
    if rules is None:
        rules = PREDICTION_CONFIDENCE_RULES.get(pred["class"], {"strong": 0.80, "medium": 0.55})
    conf = float(pred.get("confidence", 0.0))
    if conf >= rules["strong"]:
        return "strong"
    if conf >= rules["medium"]:
        return "medium"
    return "weak"


def _prediction_min_conf(cls, source_model=None):
    if source_model in MODEL_CLASS_THRESHOLDS:
        if cls in MODEL_CLASS_THRESHOLDS[source_model]:
            return MODEL_CLASS_THRESHOLDS[source_model][cls]
    rules = PREDICTION_CONFIDENCE_RULES.get(cls)
    if rules:
        return rules["medium"]
    return 0.0


def _prediction_source_allowed(cls, source_model=None):
    if source_model in MODEL_ALLOWED_CLASSES:
        return cls in MODEL_ALLOWED_CLASSES[source_model]
    return True


def _basic_geometry_ok(pred, table_box):
    box = pred["box"]
    cls = pred["class"]
    w = max(1.0, float(box[2] - box[0]))
    h = max(1.0, float(box[3] - box[1]))

    if table_box:
        table_w = max(1.0, float(table_box[2] - table_box[0]))
        table_h = max(1.0, float(table_box[3] - table_box[1]))
        cy = (box[1] + box[3]) / 2.0
        top_ratio = (cy - table_box[1]) / table_h
        left_ratio = ((box[0] + box[2]) / 2.0 - table_box[0]) / table_w
    else:
        table_w = w
        table_h = h
        top_ratio = 0.0
        left_ratio = 0.0

    if cls == CLASS_NAME_COLUMN:
        return True
    elif cls == CLASS_NAME_ROW:
        # Compact tables and taller measurement rows can still be valid rows.
        # Defer row rejection to support-scoring instead of a blunt aspect-ratio cut.
        return True
    elif cls == CLASS_NAME_COLUMN_HEADER:
        # Keep early geometry permissive. Header lateness is decided later using
        # detected body rows instead of brittle whole-table top_ratio checks.
        return True
    elif cls == CLASS_NAME_SPANNING_COLUMN_HEADER:
        # Keep early geometry permissive. Header lateness is decided later using
        # detected body rows instead of brittle whole-table top_ratio checks.
        return True
    elif cls == CLASS_NAME_ROW_HEADER:
        # Do not hard-reject wide/full-width row-header detections here.
        # Row-header validity is decided later by row-header specific checks.
        return True
    elif cls == CLASS_NAME_SPLITTED_COLUMN:
        return True
    elif cls == CLASS_NAME_SPANNING_COLUMN_CELL:
        return True
    elif cls == CLASS_NAME_SPANNING_ROW_CELL:
        return True
    return True


def _collect_structure_predictions(preds):
    relevant_classes = set()
    for classes in MODEL_ALLOWED_CLASSES.values():
        relevant_classes.update(classes)
    relevant_classes.discard(CLASS_NAME_TEXT)

    raw_preds = []
    for pred in preds:
        if not isinstance(pred, dict) or "class" not in pred:
            continue

        cls = normalize_class(pred["class"])
        if cls not in relevant_classes:
            continue

        source_model = pred.get("source_model")
        if not _prediction_source_allowed(cls, source_model):
            continue

        normalized = {
            "box": bbox_to_xyxy(pred),
            "class": cls,
            "confidence": float(pred.get("confidence", 0.0)),
            "source_model": source_model,
            "x": pred["x"],
            "y": pred["y"],
            "width": pred["width"],
            "height": pred["height"],
        }
        if normalized["confidence"] < _prediction_min_conf(cls, source_model):
            continue
        raw_preds.append(normalized)

    if not raw_preds:
        return []

    table_box = [
        min(pred["box"][0] for pred in raw_preds),
        min(pred["box"][1] for pred in raw_preds),
        max(pred["box"][2] for pred in raw_preds),
        max(pred["box"][3] for pred in raw_preds),
    ]

    return [pred for pred in raw_preds if _basic_geometry_ok(pred, table_box)]


def _collect_relaxed_detected_predictions(preds, cls, table_box, min_conf=0.15):
    recovered = []
    for pred in preds:
        if not isinstance(pred, dict) or "class" not in pred:
            continue

        normalized_cls = normalize_class(pred["class"])
        if normalized_cls != cls:
            continue

        source_model = pred.get("source_model")
        if not _prediction_source_allowed(cls, source_model):
            continue

        normalized = {
            "box": bbox_to_xyxy(pred),
            "class": normalized_cls,
            "confidence": float(pred.get("confidence", 0.0)),
            "source_model": source_model,
            "x": pred["x"],
            "y": pred["y"],
            "width": pred["width"],
            "height": pred["height"],
        }
        if normalized["confidence"] < min_conf:
            continue
        if not _basic_geometry_ok(normalized, table_box):
            continue
        recovered.append(normalized)

    return recovered


def _ocr_items_in_box(box, ocr_data, intersection_thresh=0.2):
    matches = []
    if not ocr_data:
        return matches

    for item in ocr_data:
        ibox = o_to_box(item)
        center = get_box_center(ibox)
        inter = calculate_intersection_area(box, ibox)
        item_area = get_box_area(ibox)
        if (
            box[0] <= center[0] <= box[2]
            and box[1] <= center[1] <= box[3]
        ) or (inter / (item_area + 1e-12)) >= intersection_thresh:
            matches.append(item)
    return matches


def _ocr_item_hits_exclusion_box(item_box, exclusion_box):
    center = get_box_center(item_box)
    inter = calculate_intersection_area(item_box, exclusion_box)
    item_area = get_box_area(item_box) + 1e-12
    if (
        exclusion_box[0] <= center[0] <= exclusion_box[2]
        and exclusion_box[1] <= center[1] <= exclusion_box[3]
    ):
        return True
    if (inter / item_area) >= 0.25:
        return True
    if (
        exclusion_box[0] <= center[0] <= exclusion_box[2]
        and item_box[1] <= (exclusion_box[3] - 2.0)
        and (inter / item_area) >= 0.12
    ):
        return True
    return False


def _same_class_ios(box_a, box_b):
    inter = calculate_intersection_area(box_a, box_b)
    denom = min(get_box_area(box_a), get_box_area(box_b)) + 1e-12
    return inter / denom


def _smaller_region_overlap_ratio(box_a, box_b):
    inter = calculate_intersection_area(box_a, box_b)
    denom = min(get_box_area(box_a), get_box_area(box_b)) + 1e-12
    return inter / denom


def _is_header_row_type(row_type):
    return row_type in (CLASS_NAME_COLUMN_HEADER, CLASS_NAME_SPANNING_COLUMN_HEADER, HEADER_ROW_TYPE)


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _count_anchor_hits(box, anchors, min_ratio=0.12):
    hits = 0
    for anchor in anchors or []:
        inter = calculate_intersection_area(box, anchor["box"])
        denom = min(get_box_area(box), get_box_area(anchor["box"])) + 1e-12
        if (inter / denom) >= min_ratio:
            hits += 1
    return hits


def _x_overlap_ratio(box_a, box_b):
    inter_w = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    denom = min(max(1.0, box_a[2] - box_a[0]), max(1.0, box_b[2] - box_b[0]))
    return inter_w / denom


def _y_overlap_ratio(box_a, box_b):
    inter_h = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    denom = min(max(1.0, box_a[3] - box_a[1]), max(1.0, box_b[3] - box_b[1]))
    return inter_h / denom


def _max_same_class_ios(box, kept_same_class):
    if not kept_same_class:
        return 0.0
    return max((_same_class_ios(box, kept["box"]) for kept in kept_same_class), default=0.0)


def _prediction_support_score(pred, cls, ocr_data, table_box, support_rows=None, support_cols=None, kept_same_class=None):
    kept_same_class = kept_same_class or []
    box = pred["box"]
    box_w = max(1.0, float(box[2] - box[0]))
    box_h = max(1.0, float(box[3] - box[1]))
    table_w = max(1.0, float(table_box[2] - table_box[0])) if table_box else box_w
    table_h = max(1.0, float(table_box[3] - table_box[1])) if table_box else box_h
    conf = float(pred.get("confidence", 0.0))

    ocr_items = _ocr_items_in_box(box, ocr_data, intersection_thresh=0.18)
    ocr_count = len(ocr_items)
    col_hits = _count_anchor_hits(box, support_cols, min_ratio=0.10)
    row_hits = _count_anchor_hits(box, support_rows, min_ratio=0.10)
    top_ratio = ((box[1] + box[3]) / 2 - table_box[1]) / table_h if table_box else 0.0
    same_ios = _max_same_class_ios(box, kept_same_class)
    conflict_penalty = _clamp01((same_ios - 0.35) / 0.45)
    inside_score = 1.0
    if table_box:
        inter = calculate_intersection_area(box, table_box)
        inside_score = _clamp01(inter / (get_box_area(box) + 1e-12))
    ocr_score = _clamp01(ocr_count / 2.0)

    min_score = 0.55
    score = 0.35 * conf + 0.15 * inside_score + 0.15 * ocr_score - 0.30 * conflict_penalty

    if cls == CLASS_NAME_COLUMN:
        vertical_cover = _clamp01(box_h / (0.60 * table_h))
        slender_score = _clamp01((box_h / box_w - 0.9) / 1.5)
        row_support = _clamp01(row_hits / 2.0)
        score = (
            0.35 * conf
            + 0.20 * vertical_cover
            + 0.15 * slender_score
            + 0.15 * row_support
            + 0.15 * ocr_score
            + 0.10 * inside_score
            - 0.30 * conflict_penalty
        )
        min_score = 0.54

    elif cls == CLASS_NAME_ROW:
        horizontal_cover = _clamp01(box_w / (0.55 * table_w))
        band_score = _clamp01((box_w / box_h - 1.4) / 3.5)
        col_support_count = sum(1 for col in support_cols or [] if _x_overlap_ratio(box, col["box"]) >= 0.10)
        col_support = _clamp01(col_support_count / 2.0)
        score = (
            0.35 * conf
            + 0.20 * horizontal_cover
            + 0.15 * band_score
            + 0.15 * col_support
            + 0.15 * ocr_score
            + 0.10 * inside_score
            - 0.30 * conflict_penalty
        )
        min_score = 0.53

    elif cls == CLASS_NAME_SPANNING_COLUMN_HEADER:
        overlaps_cols = sum(1 for col in support_cols or [] if _x_overlap_ratio(box, col["box"]) >= 0.20)
        top_score = _clamp01(1.0 - max(0.0, top_ratio - 0.35) / 0.35)
        span_score = _clamp01(overlaps_cols / 2.0)
        score = (
            0.35 * conf
            + 0.20 * top_score
            + 0.20 * span_score
            + 0.15 * ocr_score
            + 0.10 * inside_score
            - 0.25 * conflict_penalty
        )
        min_score = 0.53

    elif cls == CLASS_NAME_ROW_HEADER:
        first_col_score = 0.0
        if support_cols:
            first_col = min(support_cols, key=lambda col: col["box"][0])
            first_col_score = _x_overlap_ratio(box, first_col["box"])
        score = (
            0.35 * conf
            + 0.25 * _clamp01(first_col_score / 0.20)
            + 0.20 * ocr_score
            + 0.20 * inside_score
            - 0.25 * conflict_penalty
        )
        min_score = 0.54

    elif cls == CLASS_NAME_SPLITTED_COLUMN:
        parent_overlap = 0.0
        width_score = 0.0
        vertical_score = _clamp01(box_h / (0.35 * table_h))
        if support_cols:
            parent_hits = [
                col for col in support_cols
                if calculate_intersection_area(box, col["box"]) / (get_box_area(box) + 1e-12) >= INTERSECTION_THRESH_SPLIT
            ]
            if parent_hits:
                parent = max(parent_hits, key=lambda col: calculate_intersection_area(box, col["box"]))
                parent_overlap = calculate_intersection_area(box, parent["box"]) / (get_box_area(box) + 1e-12)
                parent_w = max(1.0, float(parent["box"][2] - parent["box"][0]))
                width_frac = box_w / parent_w
                if 0.08 <= width_frac <= 0.85:
                    width_score = 1.0
                else:
                    width_score = _clamp01(1.0 - min(abs(width_frac - 0.20), abs(width_frac - 0.80)))
        score = (
            0.30 * conf
            + 0.25 * _clamp01(parent_overlap)
            + 0.20 * width_score
            + 0.15 * vertical_score
            + 0.10 * ocr_score
            - 0.25 * conflict_penalty
        )
        min_score = 0.52

    elif cls in (CLASS_NAME_SPANNING_ROW_CELL, CLASS_NAME_SPANNING_COLUMN_CELL):
        # Support score for spanning cells has been replaced by deterministic checkers.
        # This code path should not be reached.
        score = 0.0
        min_score = 1.0

    return score, min_score, same_ios


def _is_supported_prediction(pred, cls, ocr_data, table_box, support_rows=None, support_cols=None, kept_same_class=None):
    score, min_score, same_ios = _prediction_support_score(
        pred,
        cls,
        ocr_data,
        table_box,
        support_rows=support_rows,
        support_cols=support_cols,
        kept_same_class=kept_same_class,
    )
    if same_ios >= IOS_THRESH_NMS_DEFAULT:
        return False
    return score >= min_score


def _best_parent_column_for_split(split, support_cols):
    if not split or not support_cols:
        return None

    split_box = split["box"]
    split_area = get_box_area(split_box) + 1e-12
    best_parent = None
    best_overlap = 0.0
    best_area = 0.0
    for col in support_cols:
        inter = calculate_intersection_area(split_box, col["box"])
        overlap = inter / split_area
        if overlap < INTERSECTION_THRESH_SPLIT:
            continue
        if overlap > best_overlap + 1e-9:
            best_parent = col
            best_overlap = overlap
            best_area = inter
            continue
        if abs(overlap - best_overlap) <= 1e-9 and inter > best_area:
            best_parent = col
            best_area = inter

    return best_parent


def _split_x_range_inside_parent(split_box, parent_box, min_inside_width_frac=0.80):
    if not split_box or not parent_box:
        return False
    split_w = max(1.0, float(split_box[2] - split_box[0]))
    split_cx = (float(split_box[0]) + float(split_box[2])) / 2.0
    if not (float(parent_box[0]) <= split_cx <= float(parent_box[2])):
        return False
    inside_w = max(0.0, min(float(split_box[2]), float(parent_box[2])) - max(float(split_box[0]), float(parent_box[0])))
    return (inside_w / split_w) >= min_inside_width_frac


def _split_body_row_overlap_score(split_box, support_rows):
    if not split_box or not support_rows:
        return 0.0

    split_h = max(1.0, float(split_box[3] - split_box[1]))
    total_inter_h = 0.0
    max_row_overlap = 0.0
    for row in support_rows:
        row_box = row.get("box")
        if not row_box:
            continue
        inter_h = max(0.0, min(split_box[3], row_box[3]) - max(split_box[1], row_box[1]))
        if inter_h <= 0.0:
            continue
        row_h = max(1.0, float(row_box[3] - row_box[1]))
        total_inter_h += inter_h
        max_row_overlap = max(max_row_overlap, min(inter_h / split_h, inter_h / row_h))

    return max(max_row_overlap, _clamp01(total_inter_h / split_h))


def _split_has_ocr_alignment(split_box, ocr_data):
    if not split_box or not ocr_data:
        return False

    for item in _ocr_items_in_box(split_box, ocr_data, intersection_thresh=0.05):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        item_box = o_to_box(item)
        cx, cy = get_box_center(item_box)
        if split_box[0] <= cx <= split_box[2] and split_box[1] <= cy <= split_box[3]:
            return True
        inter = calculate_intersection_area(split_box, item_box)
        if (inter / (get_box_area(item_box) + 1e-12)) >= 0.20:
            return True
    return False


def _ocr_repeated_subcolumn_pattern(parent_box, support_rows, ocr_data):
    if not parent_box or not support_rows or not ocr_data:
        return False

    parent_w = max(1.0, float(parent_box[2] - parent_box[0]))
    min_gap = max(18.0, 0.12 * parent_w)
    row_cluster_hits = 0
    all_centers = []

    for row in support_rows:
        row_box = row.get("box")
        if not row_box:
            continue
        row_centers = []
        for item in ocr_data:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            item_box = o_to_box(item)
            cx, cy = get_box_center(item_box)
            if not (parent_box[0] <= cx <= parent_box[2] and row_box[1] <= cy <= row_box[3]):
                continue
            item_w = max(1.0, float(item_box[2] - item_box[0]))
            if item_w > 0.80 * parent_w:
                continue
            row_centers.append(cx)
            all_centers.append(cx)

        if len(row_centers) < 2:
            continue
        row_centers.sort()
        if any((right - left) >= min_gap for left, right in zip(row_centers, row_centers[1:])):
            row_cluster_hits += 1

    if row_cluster_hits >= 2:
        return True

    all_centers = sorted(all_centers)
    return len(all_centers) >= 3 and any(
        (right - left) >= min_gap for left, right in zip(all_centers, all_centers[1:])
    )


def _split_has_neighbor_support(pred, candidates, parent_key):
    split_box = pred.get("box")
    if not split_box or parent_key is None:
        return False

    for other in candidates:
        if other is pred or other.get("_split_parent_key") != parent_key:
            continue
        other_box = other.get("box")
        if not other_box:
            continue
        if _y_overlap_ratio(split_box, other_box) < 0.12:
            continue
        if _smaller_region_overlap_ratio(split_box, other_box) > 0.65:
            continue
        return True
    return False


def _split_candidate_passes_low_conf_rules(pred, candidates, support_rows, support_cols, ocr_data):
    if pred.get("class") != CLASS_NAME_SPLITTED_COLUMN:
        return True

    if float(pred.get("confidence", 0.0)) < 0.15:
        return False

    parent = _best_parent_column_for_split(pred, support_cols)
    split_box = pred["box"]
    parent_box = parent["box"] if parent is not None else None
    if parent is not None:
        if not _split_x_range_inside_parent(split_box, parent_box):
            return False
        if not _is_valid_split_candidate(pred, parent):
            return False

    if _prediction_bucket(pred) == "strong":
        return True

    if parent is None or _prediction_bucket(parent) != "strong":
        return False

    if _split_body_row_overlap_score(split_box, support_rows) < 0.25:
        return False
    if not _split_has_ocr_alignment(split_box, ocr_data):
        return False

    parent_key = pred.get("_split_parent_key") or _box_anchor_key(parent_box)
    if _split_has_neighbor_support(pred, candidates, parent_key):
        return True
    return _ocr_repeated_subcolumn_pattern(parent_box, support_rows, ocr_data)


def _filter_split_candidates_by_low_conf_rules(preds, support_rows, support_cols, ocr_data):
    if not preds:
        return []
    return [
        pred
        for pred in preds
        if _split_candidate_passes_low_conf_rules(
            pred,
            preds,
            support_rows,
            support_cols,
            ocr_data,
        )
    ]


def _row_ocr_band_hits(row_box, ocr_data):
    hits = 0
    for item in ocr_data or []:
        text_val = str(item.get("text", "")).strip()
        if not text_val:
            continue
        ibox = o_to_box(item)
        if not _item_matches_row_band(ibox, row_box, allow_assigned_edge_slack=True):
            continue
        hits += 1
    return hits


def _row_candidate_passes_grid_check(pred, support_cols, ocr_data):
    if pred.get("class") != CLASS_NAME_ROW or not support_cols:
        return True

    row_box = pred["box"]
    row_w = max(1.0, float(row_box[2] - row_box[0]))
    col_left = min(float(col["box"][0]) for col in support_cols)
    col_right = max(float(col["box"][2]) for col in support_cols)
    col_span_w = max(1.0, col_right - col_left)
    inter_w = max(0.0, min(float(row_box[2]), col_right) - max(float(row_box[0]), col_left))
    column_span_cover = inter_w / col_span_w
    row_inside_col_span = inter_w / row_w
    col_hits = sum(1 for col in support_cols if _x_overlap_ratio(row_box, col["box"]) >= 0.10)
    ocr_hits = _row_ocr_band_hits(row_box, ocr_data)

    if col_hits >= 2:
        return True
    if col_hits >= 1 and column_span_cover >= 0.55 and row_inside_col_span >= 0.45:
        return True
    if ocr_hits >= 2 and column_span_cover >= 0.40 and row_inside_col_span >= 0.35:
        return True
    if ocr_hits >= 1 and column_span_cover >= 0.50 and row_inside_col_span >= 0.45:
        return True
    return False


def _filter_row_candidates_by_grid_check(preds, support_cols, ocr_data):
    if not preds or not support_cols:
        return preds
    return [
        pred for pred in preds
        if _row_candidate_passes_grid_check(pred, support_cols, ocr_data)
    ]


def _column_header_candidate_passes_checker(pred, support_rows, support_cols, table_box):
    if _prediction_bucket(pred) == "weak":
        return False
    if not support_cols:
        return False

    box = pred["box"]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    header_h = max(1.0, float(box[3] - box[1]))

    aligns_with_col = False
    for col in support_cols:
        col_box = col["box"]
        if col_box[0] <= cx <= col_box[2] or _x_overlap_ratio(box, col_box) >= 0.20:
            aligns_with_col = True
            break
    if not aligns_with_col:
        return False

    if table_box:
        table_h = max(1.0, float(table_box[3] - table_box[1]))
        top_ratio = (cy - table_box[1]) / table_h
        if top_ratio > 0.45 and not support_rows:
            return False

    if support_rows:
        row_boxes = [row["box"] for row in support_rows]
        first_body_top = min(row_box[1] for row_box in row_boxes)
        row_heights = [max(1.0, row_box[3] - row_box[1]) for row_box in row_boxes]
        median_row_h = float(np.median(row_heights)) if row_heights else header_h
        near_tol = max(3.0, 0.35 * median_row_h)

        for row_box in row_boxes:
            if row_box[1] <= cy <= row_box[3]:
                return False

        if min(box[3], cy) > first_body_top + near_tol:
            return False

    return True


def _spanning_column_header_candidate_passes_checker(pred, support_cols, table_box):
    if _prediction_bucket(pred) == "weak":
        return False
    if not support_cols:
        return False

    box = pred["box"]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    aligns_with_col = False
    for col in support_cols:
        col_box = col["box"]
        if col_box[0] <= cx <= col_box[2] or _x_overlap_ratio(box, col_box) >= 0.20:
            aligns_with_col = True
            break
    if not aligns_with_col:
        return False

    if table_box:
        table_h = max(1.0, float(table_box[3] - table_box[1]))
        top_ratio = (cy - table_box[1]) / table_h
        if top_ratio > 0.45:
            return False

    return True


def _filter_column_header_candidates_by_checkers(preds, support_rows, support_cols, table_box):
    if not preds:
        return []
    return [
        pred for pred in preds
        if _column_header_candidate_passes_checker(pred, support_rows, support_cols, table_box)
    ]


def _filter_spanning_column_header_candidates_by_checkers(preds, support_cols, table_box):
    if not preds:
        return []
    return [
        pred for pred in preds
        if _spanning_column_header_candidate_passes_checker(pred, support_cols, table_box)
    ]


def _spanning_row_cell_candidate_passes_checker(pred, support_rows, support_cols, ocr_data):
    if _prediction_bucket(pred) == "weak":
        return False
    if not support_rows or not support_cols:
        return False

    box = pred["box"]
    box_w = max(1.0, float(box[2] - box[0]))

    # Rule 3: OCR text must be present inside the spanning cell box
    has_ocr = False
    for item in ocr_data or []:
        ibox = item.get("box") or []
        if len(ibox) == 4:
            ixs = [p[0] for p in ibox]
            iys = [p[1] for p in ibox]
            ix1, iy1, ix2, iy2 = min(ixs), min(iys), max(ixs), max(iys)
            if calculate_intersection_area(box, [ix1, iy1, ix2, iy2]) / (get_box_area([ix1, iy1, ix2, iy2]) + 1e-12) >= 0.40:
                has_ocr = True
                break
    if not has_ocr:
        return False

    # Rule 2: Must vertically span more than one row
    # Let's count how many rows intersect with the spanning cell Y-range
    row_overlaps = 0
    for row in support_rows:
        row_box = row["box"]
        inter_y = max(0.0, min(box[3], row_box[3]) - max(box[1], row_box[1]))
        row_h = max(1.0, float(row_box[3] - row_box[1]))
        if inter_y / row_h >= 0.35:
            row_overlaps += 1
    if row_overlaps < 2:
        return False

    # Rule 1: 80% Horizontal Overlap/Area Dominance
    # We must find at least one column (either standard column or split sub-column) 
    # that accounts for >= 80% of the spanning cell's width/overlap.
    # First check if the parent columns are splitted columns.
    splitted_cols = [col for col in support_cols if col.get("class") == CLASS_NAME_SPLITTED_COLUMN]
    target_grid = splitted_cols if splitted_cols else [col for col in support_cols if col.get("class") == CLASS_NAME_COLUMN]
    
    dominance_ok = False
    for col in target_grid:
        cbox = col["box"]
        inter_x = max(0.0, min(box[2], cbox[2]) - max(box[0], cbox[0]))
        if inter_x / box_w >= 0.80:
            dominance_ok = True
            break
    if not dominance_ok:
        return False

    return True


def _filter_spanning_row_cell_candidates_by_checkers(preds, support_rows, support_cols, ocr_data):
    if not preds:
        return []
    return [
        pred for pred in preds
        if _spanning_row_cell_candidate_passes_checker(pred, support_rows, support_cols, ocr_data)
    ]



def _spanning_column_cell_candidate_passes_checker(pred, support_rows, support_cols, ocr_data):
    if _prediction_bucket(pred) == "weak":
        return False
    if not support_rows or not support_cols:
        return False

    box = pred["box"]
    box_h = max(1.0, float(box[3] - box[1]))

    # Rule 2: OCR text must be present inside the spanning cell box
    has_ocr = False
    for item in ocr_data or []:
        ibox = item.get("box") or []
        if len(ibox) == 4:
            ixs = [p[0] for p in ibox]
            iys = [p[1] for p in ibox]
            ix1, iy1, ix2, iy2 = min(ixs), min(iys), max(ixs), max(iys)
            if calculate_intersection_area(box, [ix1, iy1, ix2, iy2]) / (get_box_area([ix1, iy1, ix2, iy2]) + 1e-12) >= 0.40:
                has_ocr = True
                break
    if not has_ocr:
        return False

    # Rule 3: Must be part of more than 1 columns
    col_overlaps = 0
    for col in support_cols:
        col_box = col["box"]
        inter_x = max(0.0, min(box[2], col_box[2]) - max(box[0], col_box[0]))
        col_w = max(1.0, float(col_box[2] - col_box[0]))
        if inter_x / col_w >= 0.35:
            col_overlaps += 1
    if col_overlaps < 2:
        return False

    # Rule 1: min 80 percent part of the single row
    row_dominance_ok = False
    for row in support_rows:
        row_box = row["box"]
        inter_y = max(0.0, min(box[3], row_box[3]) - max(box[1], row_box[1]))
        if inter_y / box_h >= 0.80:
            row_dominance_ok = True
            break
    if not row_dominance_ok:
        return False

    return True


def _filter_spanning_column_cell_candidates_by_checkers(preds, support_rows, support_cols, ocr_data):
    if not preds:
        return []
    return [
        pred for pred in preds
        if _spanning_column_cell_candidate_passes_checker(pred, support_rows, support_cols, ocr_data)
    ]


def _line_y_crosses_ocr_text_in_band(y, x1, x2, ocr_data, inner_frac=0.30):
    if not ocr_data:
        return False

    band_w = max(1.0, float(x2 - x1))
    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        inter_w = max(0.0, min(float(x2), float(ibox[2])) - max(float(x1), float(ibox[0])))
        if inter_w / band_w < 0.15:
            continue
        box_h = max(1.0, float(ibox[3]) - float(ibox[1]))
        edge_margin = min(max(2.0, inner_frac * box_h), max(2.0, 0.45 * box_h))
        if float(ibox[1]) + edge_margin <= float(y) <= float(ibox[3]) - edge_margin:
            return True
    return False



def _horizontal_separator_has_continuous_run(structure_gray, box, y, min_run_frac=0.72):
    if structure_gray is None or not box:
        return False

    img_h, img_w = structure_gray.shape[:2]
    x1 = max(0, int(round(box[0])))
    x2 = min(img_w, int(round(box[2])))
    yy = int(round(float(y)))
    if x2 <= x1 or yy < 0 or yy >= img_h:
        return False

    y1 = max(0, yy - 1)
    y2 = min(img_h, yy + 2)
    band = structure_gray[y1:y2, x1:x2]
    if band.size == 0:
        return False

    dark_cols = np.any(band <= STRUCTURE_LINE_DARK_PIXEL_MAX, axis=0)
    longest = 0
    run = 0
    for is_dark in dark_cols:
        if bool(is_dark):
            run += 1
            longest = max(longest, run)
        else:
            run = 0

    width = max(1, x2 - x1)
    return (longest / width) >= min_run_frac
def _segment_has_ocr_text(segment_box, ocr_data):
    if not segment_box or not ocr_data:
        return False
    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        text_area = get_box_area(ibox) + 1e-12
        inter_area = calculate_intersection_area(ibox, segment_box)
        if inter_area / text_area >= 0.55:
            return True
    return False


def _row_boundary_has_horizontal_separator(structure_gray, y, final_cols, min_col_frac=0.80, row_h=None):
    if (
        structure_gray is None
        or structure_gray.ndim != 2
        or not final_cols
        or y is None
    ):
        return False

    img_h, img_w = structure_gray.shape[:2]
    y0 = int(round(float(y)))
    if row_h is None:
        window = 2
    else:
        window = max(2, int(round(min(8.0, max(2.0, float(row_h) * 0.12)))))
    candidate_ys = range(max(0, y0 - window), min(img_h - 1, y0 + window) + 1)
    needed = max(1, int(np.ceil(float(len(final_cols)) * float(min_col_frac))))

    for yy in candidate_ys:
        hit_count = 0
        for col in final_cols:
            col_box = col.get("box") if isinstance(col, dict) else None
            if not col_box:
                continue
            cx1 = max(0, int(round(col_box[0])) + 2)
            cx2 = min(img_w, int(round(col_box[2])) - 2)
            if cx2 - cx1 < 16:
                continue
            col_strip = structure_gray[yy:yy + 1, cx1:cx2]
            if col_strip.size == 0:
                continue
            col_dark = col_strip[0] <= STRUCTURE_LINE_DARK_PIXEL_MAX
            if (
                float(np.mean(col_dark)) >= 0.55
                and _longest_true_run_ratio(col_dark) >= 0.88
            ):
                hit_count += 1
        if hit_count >= needed:
            return True
    return False


def _row_boundary_has_nearby_horizontal_separator(
    structure_gray,
    y,
    final_cols,
    row_box,
    median_row_h,
    min_col_frac=0.80,
):
    if y is None:
        return False

    row_h = max(1.0, float(row_box[3] - row_box[1])) if row_box else 1.0
    # Boundary separators may sit just outside/inside an imperfect detected row.
    # Use a small window derived from this row's own height, not a case-specific
    # fixed pixel rule.
    search_margin = max(2, int(round(min(8.0, max(2.0, row_h * 0.12)))))
    for yy in range(int(round(float(y))) - search_margin, int(round(float(y))) + search_margin + 1):
        if _row_boundary_has_horizontal_separator(
            structure_gray,
            yy,
            final_cols,
            min_col_frac=min_col_frac,
            row_h=row_h,
        ):
            return True
    return False


def _find_internal_row_separator_ys(structure_gray, row_box, final_cols, median_row_h, ocr_data=None):
    if (
        not ENABLE_ROW_SEPARATOR_SPLIT
        or structure_gray is None
        or structure_gray.ndim != 2
        or not row_box
        or not final_cols
    ):
        return []

    img_h, img_w = structure_gray.shape[:2]
    row_y1 = max(0, int(round(row_box[1])))
    row_y2 = min(img_h, int(round(row_box[3])))
    row_h = max(1.0, float(row_y2 - row_y1))
    baseline = max(1.0, float(median_row_h or row_h))

    x1 = max(0, int(round(min(col["box"][0] for col in final_cols))) + 2)
    x2 = min(img_w, int(round(max(col["box"][2] for col in final_cols))) - 2)
    if x2 - x1 < 24:
        return []

    edge_margin = max(2, int(round(min(row_h * 0.18, max(4.0, 0.45 * baseline)))))
    edge_margin = min(edge_margin, max(1, int(round(row_h * 0.30))))
    scan_y1 = row_y1 + edge_margin
    scan_y2 = row_y2 - edge_margin
    if scan_y2 <= scan_y1:
        return []

    candidate_ys = []
    for y in range(scan_y1, scan_y2 + 1):
        line_cols = []
        for col in final_cols:
            col_box = col.get("box") if isinstance(col, dict) else None
            if not col_box:
                continue
            cx1 = max(0, int(round(col_box[0])) + 2)
            cx2 = min(img_w, int(round(col_box[2])) - 2)
            if cx2 - cx1 < 16:
                continue
            col_strip = structure_gray[y:y + 1, cx1:cx2]
            if col_strip.size == 0:
                continue
            col_dark = col_strip[0] <= STRUCTURE_LINE_DARK_PIXEL_MAX
            if (
                float(np.mean(col_dark)) >= 0.55
                and _longest_true_run_ratio(col_dark) >= 0.88
            ):
                line_cols.append((cx1, cx2))
        if not line_cols:
            continue
        if any(_line_y_crosses_ocr_text_in_band(y, lx1, lx2, ocr_data) for lx1, lx2 in line_cols):
            continue
        # A real row boundary can be visible in only one column when other
        # columns are row-spanned; keep all local line-quality checks above.
        needed = 1
        if len(line_cols) >= needed:
            candidate_ys.append(y)

    if not candidate_ys:
        return []

    groups = []
    current = [candidate_ys[0]]
    for y in candidate_ys[1:]:
        if y - current[-1] <= 2:
            current.append(y)
        else:
            groups.append(current)
            current = [y]
    groups.append(current)

    if not groups:
        return []

    separators = sorted(float(sum(group) / len(group)) for group in groups)
    if not separators:
        return []
    return separators


def _split_body_rows_at_internal_separators(body_rows, final_cols, structure_gray, ocr_data=None):
    if not ENABLE_ROW_SEPARATOR_SPLIT or not body_rows or not final_cols or structure_gray is None:
        return body_rows

    body_heights = [
        max(1.0, float(row["box"][3] - row["box"][1]))
        for row in body_rows
        if row.get("type") == CLASS_NAME_ROW
    ]
    median_row_h = 0.0
    if body_heights:
        sorted_heights = sorted(body_heights)
        median_all = float(np.median(sorted_heights))
        lower_half = [height for height in sorted_heights if height <= median_all]
        median_row_h = float(np.median(lower_half or sorted_heights))
    split_rows = []
    for row in body_rows:
        if row.get("type") != CLASS_NAME_ROW:
            split_rows.append(row)
            continue

        separators = _find_internal_row_separator_ys(structure_gray, row["box"], final_cols, median_row_h, ocr_data)
        if not separators:
            split_rows.append(row)
            continue

        row_box = row["box"]
        boundaries = [float(row_box[1])] + separators + [float(row_box[3])]
        valid_segments = True
        for seg_y1, seg_y2 in zip(boundaries, boundaries[1:]):
            segment_box = [row_box[0], seg_y1, row_box[2], seg_y2]
            if not _segment_has_ocr_text(segment_box, ocr_data):
                valid_segments = False
                break
        if not valid_segments:
            split_rows.append(row)
            continue

        for idx, (seg_y1, seg_y2) in enumerate(zip(boundaries, boundaries[1:])):
            clone = dict(row)
            clone["box"] = [row_box[0], seg_y1, row_box[2], seg_y2]
            clone["y"] = (seg_y1 + seg_y2) / 2.0
            clone["source"] = "separator_split_row"
            clone["split_parent_box"] = list(row_box)
            clone["split_segment_index"] = idx
            clone["split_segment_count"] = len(boundaries) - 1
            split_rows.append(clone)

    return sorted(split_rows, key=lambda row: (row["y"], 0 if row["type"] == CLASS_NAME_ROW_HEADER else 1))


def _y_in_any_band(y, bands, pad=0.0):
    if not bands:
        return False
    y = float(y)
    for band in bands:
        if not band:
            continue
        if float(band[1]) - pad <= y <= float(band[3]) + pad:
            return True
    return False


def _find_internal_column_separator_xs(structure_gray, col_box, table_box, median_col_w, ignore_bands=None):
    if (
        not ENABLE_COLUMN_SEPARATOR_SPLIT
        or structure_gray is None
        or structure_gray.ndim != 2
        or not col_box
        or not table_box
    ):
        return []

    img_h, img_w = structure_gray.shape[:2]
    col_x1 = max(0, int(round(col_box[0])))
    col_x2 = min(img_w, int(round(col_box[2])))
    col_w = max(1.0, float(col_x2 - col_x1))
    baseline = max(1.0, float(median_col_w or col_w))
    if col_w < max(24.0, 1.55 * baseline):
        return []

    y1 = max(0, int(round(table_box[1])) + 2)
    y2 = min(img_h, int(round(table_box[3])) - 2)
    if y2 - y1 < 24:
        return []

    min_segment_w = max(8.0, min(24.0, 0.34 * baseline))
    edge_margin = max(5, int(round(min(col_w * 0.18, max(8.0, 0.45 * baseline)))))
    scan_x1 = col_x1 + edge_margin
    scan_x2 = col_x2 - edge_margin
    if scan_x2 <= scan_x1:
        return []

    active_y = [
        yy
        for yy in range(y1, y2)
        if not _y_in_any_band(yy, ignore_bands, pad=2.0)
    ]
    if len(active_y) < 24:
        return []

    candidate_xs = []
    for x in range(scan_x1, scan_x2 + 1):
        strip = structure_gray[active_y, x:x + 1]
        if strip.size == 0:
            continue
        dark = strip[:, 0] <= STRUCTURE_LINE_DARK_PIXEL_MAX
        if _longest_true_run_ratio(dark) >= 0.70:
            candidate_xs.append(x)

    if not candidate_xs:
        return []

    groups = []
    current = [candidate_xs[0]]
    for x in candidate_xs[1:]:
        if x - current[-1] <= 2:
            current.append(x)
        else:
            groups.append(current)
            current = [x]
    groups.append(current)

    separators = [float(sum(group) / len(group)) for group in groups]
    separators = sorted(x for x in separators if (col_x1 + min_segment_w) <= x <= (col_x2 - min_segment_w))
    if not separators:
        return []

    boundaries = [float(col_x1)] + separators + [float(col_x2)]
    if any((boundaries[idx + 1] - boundaries[idx]) < min_segment_w for idx in range(len(boundaries) - 1)):
        return []
    return separators


def _split_columns_at_internal_separators(cols_raw, table_box, structure_gray, ignore_bands=None):
    if not ENABLE_COLUMN_SEPARATOR_SPLIT or not cols_raw or not table_box or structure_gray is None:
        return cols_raw

    col_widths = [
        max(1.0, float(col["box"][2] - col["box"][0]))
        for col in cols_raw
        if col.get("class", CLASS_NAME_COLUMN) == CLASS_NAME_COLUMN
    ]
    median_col_w = 0.0
    if col_widths:
        sorted_widths = sorted(col_widths)
        median_all = float(np.median(sorted_widths))
        lower_half = [width for width in sorted_widths if width <= median_all]
        median_col_w = float(np.median(lower_half or sorted_widths))
        
    split_cols = []
    for col in cols_raw:
        if col.get("class", CLASS_NAME_COLUMN) != CLASS_NAME_COLUMN:
            split_cols.append(col)
            continue

        separators = _find_internal_column_separator_xs(
            structure_gray,
            col["box"],
            table_box,
            median_col_w,
            ignore_bands=ignore_bands,
        )
        if not separators:
            split_cols.append(col)
            continue

        col_box = col["box"]
        boundaries = [float(col_box[0])] + separators + [float(col_box[2])]
        for idx, (seg_x1, seg_x2) in enumerate(zip(boundaries, boundaries[1:])):
            clone = dict(col)
            clone["box"] = [seg_x1, col_box[1], seg_x2, col_box[3]]
            clone["x"] = (seg_x1 + seg_x2) / 2.0
            clone["width"] = seg_x2 - seg_x1
            clone["source"] = "separator_split_col"
            clone["split_parent_box"] = list(col_box)
            clone["split_segment_index"] = idx
            clone["split_segment_count"] = len(boundaries) - 1
            split_cols.append(clone)

    return sorted(split_cols, key=lambda col: col["x"])


def _line_x_crosses_ocr_text(x, y1, y2, ocr_data):
    if not ocr_data:
        return False

    band_h = max(1.0, float(y2 - y1))
    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        inter_h = max(0.0, min(float(y2), float(ibox[3])) - max(float(y1), float(ibox[1])))
        if inter_h / band_h < 0.35:
            continue
        # Text strokes can look like short vertical rules. Only reject when
        # the candidate line is clearly inside the OCR text box, not just near
        # its outer edge.
        if float(ibox[0]) + 2.0 <= float(x) <= float(ibox[2]) - 2.0:
            return True
    return False


def _row_band_has_parent_column_ocr(parent_box, y1, y2, ocr_data):
    if not parent_box or not ocr_data:
        return False

    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        cx, cy = get_box_center(ibox)
        if parent_box[0] <= cx <= parent_box[2] and y1 <= cy <= y2:
            return True
        inter = calculate_intersection_area([parent_box[0], y1, parent_box[2], y2], ibox)
        if inter / (get_box_area(ibox) + 1e-12) >= 0.20:
            return True
    return False


def _find_split_column_separator_xs_for_row(structure_gray, col_box, row_box, ocr_data):
    if (
        not ENABLE_SPLIT_COLUMN_SEPARATOR_RECOVERY
        or structure_gray is None
        or structure_gray.ndim != 2
        or not col_box
        or not row_box
    ):
        return []

    img_h, img_w = structure_gray.shape[:2]
    col_x1 = max(0, int(round(col_box[0])))
    col_x2 = min(img_w, int(round(col_box[2])))
    row_y1 = max(0, int(round(max(float(row_box[1]), float(col_box[1])))))
    row_y2 = min(img_h, int(round(min(float(row_box[3]), float(col_box[3])))))
    col_w = max(1.0, float(col_x2 - col_x1))
    row_h = max(1.0, float(row_y2 - row_y1))

    if col_w < 36.0 or row_h < 10.0:
        return []
    if not _row_band_has_parent_column_ocr(col_box, row_y1, row_y2, ocr_data):
        return []

    min_segment_w = max(10.0, min(32.0, 0.12 * col_w))
    edge_margin = max(4, int(round(min(14.0, max(6.0, 0.06 * col_w)))))
    scan_x1 = col_x1 + edge_margin
    scan_x2 = col_x2 - edge_margin
    if scan_x2 <= scan_x1:
        return []

    candidate_xs = []
    for x in range(scan_x1, scan_x2 + 1):
        if _line_x_crosses_ocr_text(x, row_y1, row_y2, ocr_data):
            continue
        strip = structure_gray[row_y1:row_y2, x:x + 1]
        if strip.size == 0:
            continue
        dark = strip[:, 0] <= STRUCTURE_LINE_DARK_PIXEL_MAX
        dark_ratio = float(np.mean(dark))
        if dark_ratio < 0.42:
            continue
        if _longest_true_run_ratio(dark) < 0.62:
            continue
        candidate_xs.append(x)

    if not candidate_xs:
        return []

    groups = []
    current = [candidate_xs[0]]
    for x in candidate_xs[1:]:
        if x - current[-1] <= 2:
            current.append(x)
        else:
            groups.append(current)
            current = [x]
    groups.append(current)

    separators = [float(sum(group) / len(group)) for group in groups]
    separators = sorted(x for x in separators if (col_x1 + min_segment_w) <= x <= (col_x2 - min_segment_w))
    if not separators:
        return []

    boundaries = [float(col_x1)] + separators + [float(col_x2)]
    if any((boundaries[idx + 1] - boundaries[idx]) < min_segment_w for idx in range(len(boundaries) - 1)):
        return []
    return separators


def _recover_split_columns_from_separator_lines(cols_raw, rows_raw, structure_gray, ocr_data, existing_splits=None):
    if (
        not ENABLE_SPLIT_COLUMN_SEPARATOR_RECOVERY
        or not cols_raw
        or not rows_raw
        or structure_gray is None
    ):
        return []

    recovered = []
    for col in cols_raw:
        if col.get("class", CLASS_NAME_COLUMN) != CLASS_NAME_COLUMN:
            continue
        col_box = col.get("box")
        if not col_box:
            continue

        existing_in_col = [
            split for split in existing_splits or []
            if calculate_intersection_area(split["box"], col_box) / (get_box_area(split["box"]) + 1e-12) > INTERSECTION_THRESH_SPLIT
            and _is_valid_split_candidate(split, col)
        ]
        existing_count = _infer_split_count(existing_in_col, col_box) if existing_in_col else 0

        row_segments = []
        max_line_split_count = 0
        for row in rows_raw:
            row_box = row.get("box")
            if not row_box:
                continue
            inter_h = max(0.0, min(float(row_box[3]), float(col_box[3])) - max(float(row_box[1]), float(col_box[1])))
            row_h = max(1.0, float(row_box[3] - row_box[1]))
            if inter_h / row_h < 0.55:
                continue

            separators = _find_split_column_separator_xs_for_row(structure_gray, col_box, row_box, ocr_data)
            split_count = len(separators) + 1 if separators else 0
            if split_count < 2:
                continue
            boundaries = [float(col_box[0])] + list(separators) + [float(col_box[2])]
            if existing_count < 2 and _split_row_segment_ocr_support(boundaries, row_box, ocr_data) < 2:
                continue
            max_line_split_count = max(max_line_split_count, split_count)
            row_segments.append((row_box, separators, split_count))

        if max_line_split_count < 2 or max_line_split_count <= existing_count:
            continue

        for row_box, separators, split_count in row_segments:
            if split_count != max_line_split_count:
                continue
            seg_y1 = max(float(row_box[1]), float(col_box[1]))
            seg_y2 = min(float(row_box[3]), float(col_box[3]))
            boundaries = [float(col_box[0])] + list(separators) + [float(col_box[2])]
            for idx, (seg_x1, seg_x2) in enumerate(zip(boundaries, boundaries[1:])):
                if seg_x2 <= seg_x1:
                    continue
                recovered.append(
                    {
                        "class": CLASS_NAME_SPLITTED_COLUMN,
                        "box": [seg_x1, seg_y1, seg_x2, seg_y2],
                        "confidence": 0.92,
                        "source_model": "synthetic_separator_split",
                        "source": "separator_split_column_line",
                        "x": (seg_x1 + seg_x2) / 2.0,
                        "y": (seg_y1 + seg_y2) / 2.0,
                        "width": seg_x2 - seg_x1,
                        "height": seg_y2 - seg_y1,
                        "split_parent_box": list(col_box),
                        "split_segment_index": idx,
                        "split_segment_count": split_count,
                    }
                )

    return recovered


def _split_row_segment_ocr_support(boundaries, row_box, ocr_data):
    if not boundaries or len(boundaries) < 3 or not row_box or not ocr_data:
        return 0

    row_y1, row_y2 = float(row_box[1]), float(row_box[3])
    occupied_segments = set()
    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        cx, cy = get_box_center(ibox)
        if not (row_y1 <= cy <= row_y2):
            continue
        for idx, (seg_x1, seg_x2) in enumerate(zip(boundaries, boundaries[1:])):
            if float(seg_x1) <= cx <= float(seg_x2):
                occupied_segments.add(idx)
                break

    return len(occupied_segments)


def _annotate_split_candidates_for_parent_nms(preds, support_cols):
    if not preds or not support_cols:
        return list(preds or [])

    annotated = []
    for pred in preds:
        clone = dict(pred)
        parent = _best_parent_column_for_split(clone, support_cols)
        if parent is not None:
            parent_box = list(parent["box"])
            clone["_split_parent_key"] = _box_anchor_key(parent_box)
            clone["_split_parent_box"] = parent_box
            clone["_split_geom_valid"] = _is_valid_split_candidate(clone, parent)
            parent_w = max(1.0, float(parent_box[2] - parent_box[0]))
            split_w = max(1.0, float(clone["box"][2] - clone["box"][0]))
            width_frac = split_w / parent_w
            clone["_split_width_frac"] = width_frac
        annotated.append(clone)

    return annotated


def _split_nms_sort_key(pred):
    valid_bonus = 1 if pred.get("_split_geom_valid") else 0
    width_frac = float(pred.get("_split_width_frac", 1.0))
    width_pref = 0.0
    if pred.get("_split_parent_key") is not None:
        # Prefer child-like splits over full-width parent-shaped duplicates.
        width_pref = _clamp01((0.92 - width_frac) / 0.84)
    return (
        valid_bonus,
        width_pref,
        _nms_priority(pred),
        float(pred.get("confidence", 0.0)),
    )


def _group_nms_by_parent(preds, ios_thresh):
    if not preds:
        return []

    grouped = {}
    passthrough = []
    for pred in preds:
        parent_key = pred.get("_split_parent_key")
        if parent_key is None:
            passthrough.append(pred)
            continue
        grouped.setdefault(parent_key, []).append(pred)

    keep = []
    for parent_key in sorted(grouped):
        boxes = sorted(grouped[parent_key], key=_split_nms_sort_key, reverse=True)
        parent_keep = []
        for box in boxes:
            is_dupe = False
            for kept in parent_keep:
                inter = calculate_intersection_area(box["box"], kept["box"])
                smaller_area = min(get_box_area(box["box"]), get_box_area(kept["box"]))
                if inter / (smaller_area + 1e-12) > ios_thresh:
                    is_dupe = True
                    break
            if not is_dupe:
                parent_keep.append(box)
        keep.extend(parent_keep)

    if passthrough:
        keep.extend(group_nms(passthrough, ios_thresh))
    return keep


def _compatible_kept_predictions(pred, kept_same_class, cls):
    if cls != CLASS_NAME_SPLITTED_COLUMN:
        return kept_same_class

    parent_key = pred.get("_split_parent_key")
    if parent_key is None:
        return kept_same_class

    return [
        kept
        for kept in kept_same_class or []
        if kept.get("_split_parent_key") == parent_key
    ]


def _strip_split_candidate_helpers(preds):
    cleaned = []
    for pred in preds or []:
        clone = dict(pred)
        clone.pop("_split_parent_key", None)
        clone.pop("_split_parent_box", None)
        clone.pop("_split_geom_valid", None)
        clone.pop("_split_width_frac", None)
        cleaned.append(clone)
    return cleaned


def _select_predictions_by_support(preds, cls, ocr_data, table_box, support_rows=None, support_cols=None, ios_thresh=IOS_THRESH_NMS_DEFAULT):
    if not preds:
        return []

    if cls == CLASS_NAME_ROW and support_cols:
        preds = _filter_row_candidates_by_grid_check(preds, support_cols, ocr_data)
        if not preds:
            return []
    if cls == CLASS_NAME_COLUMN_HEADER:
        preds = _filter_column_header_candidates_by_checkers(
            preds,
            support_rows,
            support_cols,
            table_box,
        )
        if not preds:
            return []
        keep = group_nms(preds, ios_thresh)
        return sorted(keep, key=lambda item: item["confidence"], reverse=True)
    if cls == CLASS_NAME_SPANNING_COLUMN_HEADER:
        preds = _filter_spanning_column_header_candidates_by_checkers(
            preds,
            support_cols,
            table_box,
        )
        if not preds:
            return []
        keep = group_nms(preds, ios_thresh)
        return sorted(keep, key=lambda item: item["confidence"], reverse=True)
    if cls == CLASS_NAME_SPANNING_ROW_CELL:
        preds = _filter_spanning_row_cell_candidates_by_checkers(
            preds,
            support_rows,
            support_cols,
            ocr_data,
        )
        if not preds:
            return []
        keep = group_nms(preds, ios_thresh)
        return sorted(keep, key=lambda item: item["confidence"], reverse=True)
    if cls == CLASS_NAME_SPANNING_COLUMN_CELL:
        preds = _filter_spanning_column_cell_candidates_by_checkers(
            preds,
            support_rows,
            support_cols,
            ocr_data,
        )
        if not preds:
            return []
        keep = group_nms(preds, ios_thresh)
        return sorted(keep, key=lambda item: item["confidence"], reverse=True)

    parent_aware_split_nms = cls == CLASS_NAME_SPLITTED_COLUMN and bool(support_cols)
    if parent_aware_split_nms:
        preds = _annotate_split_candidates_for_parent_nms(preds, support_cols)
        preds = _filter_split_candidates_by_low_conf_rules(
            preds,
            support_rows,
            support_cols,
            ocr_data,
        )
        keep = _group_nms_by_parent(preds, ios_thresh)
        keep = _strip_split_candidate_helpers(keep)
        return sorted(keep, key=lambda item: item["confidence"], reverse=True)

    strong = [pred for pred in preds if _prediction_bucket(pred) == "strong"]
    medium = [pred for pred in preds if _prediction_bucket(pred) == "medium"]
    weak = [pred for pred in preds if _prediction_bucket(pred) == "weak"]

    keep = group_nms(strong, ios_thresh)
    for pred in sorted(medium, key=lambda item: item["confidence"], reverse=True):
        kept_same_class = _compatible_kept_predictions(pred, keep, cls)
        if _is_supported_prediction(
            pred,
            cls,
            ocr_data,
            table_box,
            support_rows=support_rows,
            support_cols=support_cols,
            kept_same_class=kept_same_class,
        ):
            keep.append(pred)

    # High-confidence seeds can pull in a few weaker but structurally aligned
    # detections when they fit the emerging table grid.
    for pred in sorted(weak, key=lambda item: item["confidence"], reverse=True):
        kept_same_class = _compatible_kept_predictions(pred, keep, cls)
        score, min_score, same_ios = _prediction_support_score(
            pred,
            cls,
            ocr_data,
            table_box,
            support_rows=support_rows,
            support_cols=support_cols,
            kept_same_class=kept_same_class,
        )
        if same_ios >= ios_thresh:
            continue
        if score >= max(min_score + 0.08, 0.62):
            keep.append(pred)

    keep = group_nms(keep, ios_thresh)
    return sorted(keep, key=lambda item: item["confidence"], reverse=True)


def _sort_ocr_items_reading_order(items):
    if not items:
        return []

    heights = [max(1.0, float(o_to_box(item)[3] - o_to_box(item)[1])) for item in items]
    line_gap = max(8.0, 0.55 * float(np.median(heights))) if heights else 10.0

    ordered = sorted(items, key=lambda item: (get_box_center(o_to_box(item))[1], o_to_box(item)[0]))
    lines = []
    for item in ordered:
        cy = get_box_center(o_to_box(item))[1]
        if lines and abs(cy - lines[-1]["cy"]) <= line_gap:
            lines[-1]["items"].append(item)
            count = len(lines[-1]["items"])
            lines[-1]["cy"] = ((lines[-1]["cy"] * (count - 1)) + cy) / count
        else:
            lines.append({"cy": cy, "items": [item]})

    result = []
    for line in lines:
        result.extend(sorted(line["items"], key=lambda item: o_to_box(item)[0]))
    return result


def get_ocr_text_in_box(box, ocr_data, intersection_thresh=INTERSECTION_THRESH_OCR_GRID):
    text_items = []
    for ocr_item in ocr_data:
        o_box = o_to_box(ocr_item)
        o_center = get_box_center(o_box)

        is_center_in = (
            box[0] <= o_center[0] <= box[2]
            and box[1] <= o_center[1] <= box[3]
        )

        inter = calculate_intersection_area(box, o_box)
        area = get_box_area(o_box)
        is_intersecting = (inter / (area + 1e-12)) > intersection_thresh

        if is_center_in or is_intersecting:
            text_items.append(ocr_item)

    if not text_items:
        return ""

    text_items = _drop_ocr_fragments_covered_by_larger_boxes(text_items)
    text_items = _dedupe_near_duplicate_text_items(text_items)
    text_items = _sort_ocr_items_reading_order(text_items)
    return " ".join(item["text"] for item in text_items).strip()


def _get_header_text_in_box_dedup(box, ocr_data, intersection_thresh=INTERSECTION_THRESH_OCR_GRID, center_only=False):
    text_items = []
    for ocr_item in ocr_data or []:
        o_box = o_to_box(ocr_item)
        o_center = get_box_center(o_box)
        is_center_in = (
            box[0] <= o_center[0] <= box[2]
            and box[1] <= o_center[1] <= box[3]
        )
        if center_only:
            if is_center_in:
                text_items.append(ocr_item)
            continue

        inter = calculate_intersection_area(box, o_box)
        area = get_box_area(o_box)
        is_intersecting = (inter / (area + 1e-12)) > intersection_thresh
        if is_center_in or is_intersecting:
            text_items.append(ocr_item)

    if not text_items:
        return ""

    text_items = _drop_ocr_fragments_covered_by_larger_boxes(text_items)
    text_items = _dedupe_near_duplicate_text_items(text_items)
    text_items = _sort_ocr_items_reading_order(text_items)
    return " ".join(item["text"] for item in text_items).strip()



def get_text_center_only_in_box(target_box, ocr_items):
    found = []
    for item in ocr_items:
        ibox = o_to_box(item)
        cx = (ibox[0] + ibox[2]) / 2
        cy = (ibox[1] + ibox[3]) / 2
        if (
            target_box[0] <= cx <= target_box[2]
            and target_box[1] <= cy <= target_box[3]
        ):
            found.append(item)

    if not found:
        return ""

    found = _sort_ocr_items_reading_order(found)
    return " ".join(item["text"] for item in found).strip()

def get_text_center_in_box(target_box, ocr_items):
    found = []
    for item in ocr_items:
        ibox = o_to_box(item)

        item_center_x = (ibox[0] + ibox[2]) / 2
        item_center_y = (ibox[1] + ibox[3]) / 2

        if (
            target_box[0] <= item_center_x <= target_box[2]
            and target_box[1] <= item_center_y <= target_box[3]
        ):
            found.append(item)
            continue

        inter = calculate_intersection_area(target_box, ibox)
        if (inter / (get_box_area(ibox) + 1e-12)) > 0.45:
            found.append(item)

    found = _sort_ocr_items_reading_order(found)
    return " ".join(item["text"] for item in found).strip()


def _row_band_intersection_height(item_box, row_box):
    return max(0.0, min(item_box[3], row_box[3]) - max(item_box[1], row_box[1]))


def _item_matches_row_band(item_box, row_box, allow_assigned_edge_slack=False):
    inter_h = _row_band_intersection_height(item_box, row_box)
    item_h = max(1.0, float(item_box[3] - item_box[1]))
    row_h = max(1.0, float(row_box[3] - row_box[1]))
    cy = (item_box[1] + item_box[3]) / 2
    if row_box[1] <= cy <= row_box[3]:
        return True
    if inter_h > 0:
        if (inter_h / item_h) >= 0.08:
            return True
        if (inter_h / row_h) >= 0.08:
            return True
        return False

    edge_tol = max(2.0, 0.08 * row_h)
    if allow_assigned_edge_slack:
        edge_tol = max(edge_tol, min(8.0, 0.18 * max(row_h, item_h)))
    if abs(item_box[1] - row_box[3]) <= edge_tol:
        return True
    if abs(item_box[3] - row_box[1]) <= edge_tol:
        return True
    return False


def _build_row_text_map(row_box, final_cols, ocr_data):
    if not row_box or not final_cols or not ocr_data:
        return None

    assigned = {i: [] for i in range(len(final_cols))}
    for item in ocr_data:
        ibox = o_to_box(item)
        if not _item_matches_row_band(ibox, row_box, allow_assigned_edge_slack=True):
            continue
        best_idx = _preferred_col_idx_for_item(ibox, final_cols)
        if best_idx is None:
            continue
        assigned[best_idx].append(item)

    row_map = {}
    for idx, items in assigned.items():
        if not items:
            row_map[idx] = ""
            continue
        items = _dedupe_near_duplicate_text_items(items)
        items = _sort_ocr_items_reading_order(items)
        row_map[idx] = " ".join(item["text"] for item in items).strip()
    return row_map


def _build_row_header_text_map(final_cols, row_ocr_items):
    if not final_cols or not row_ocr_items:
        return {}

    items = _dedupe_near_duplicate_text_items(row_ocr_items)
    items = _sort_ocr_items_reading_order(items)
    if len(items) < 2:
        return {}

    first_col_box = _col_assignment_box(final_cols[0]) or final_cols[0]["box"]
    first_col_left = float(first_col_box[0])
    first_col_width = max(1.0, float(first_col_box[2] - first_col_box[0]))
    left_start_slack = max(6.0, min(18.0, 0.10 * first_col_width))

    grouped = {}
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        # Row-header text that starts from the left edge belongs to the first
        # column, even when the OCR box is long and its center/overlap reaches
        # later columns. Other row-header text boxes still use normal geometry.
        if float(ibox[0]) <= first_col_left + left_start_slack:
            col_idx = 0
        else:
            col_idx = _preferred_col_idx_for_item(ibox, final_cols)
        if col_idx is None:
            continue
        grouped.setdefault(col_idx, []).append(item)

    if len(grouped) < 2:
        return {}

    row_header_map = {}
    for col_idx, group_items in grouped.items():
        group_items = _dedupe_near_duplicate_text_items(group_items)
        group_items = _sort_ocr_items_reading_order(group_items)
        text = " ".join(
            str(item.get("text", "")).strip()
            for item in group_items
            if str(item.get("text", "")).strip()
        ).strip()
        if text:
            row_header_map[col_idx] = text

    return row_header_map

def _dedupe_near_duplicate_text_items(items):
    if not items:
        return []

    kept = []
    for item in _sort_ocr_items_reading_order(items):
        text = str(item.get("text", "")).strip()
        text_key = _label_key(text)
        if not text_key:
            kept.append(item)
            continue

        box = o_to_box(item)
        cx, cy = get_box_center(box)
        box_scale = max(6.0, 0.18 * max(float(box[2] - box[0]), float(box[3] - box[1]), 1.0))
        is_dup = False
        for kept_item in kept:
            kept_text_key = _label_key(str(kept_item.get("text", "")).strip())
            if kept_text_key != text_key:
                continue
            kept_box = o_to_box(kept_item)
            kcx, kcy = get_box_center(kept_box)
            if _same_class_ios(box, kept_box) >= 0.65:
                is_dup = True
                break
            if abs(cx - kcx) <= box_scale and abs(cy - kcy) <= max(4.0, 0.75 * box_scale):
                is_dup = True
                break
        if not is_dup:
            kept.append(item)

    return kept


def _drop_ocr_fragments_covered_by_larger_boxes(items, min_inside=0.78):
    if not items:
        return []

    candidates = list(items)
    drop_ids = set()
    for idx, item in enumerate(candidates):
        if idx in drop_ids:
            continue
        text = str(item.get("text", "")).strip()
        text_key = _label_key(text)
        if not text_key:
            continue
        box = o_to_box(item)
        area = get_box_area(box) + 1e-12

        for other_idx, other in enumerate(candidates):
            if idx == other_idx:
                continue
            other_text = str(other.get("text", "")).strip()
            other_key = _label_key(other_text)
            if not other_key:
                continue
            other_box = o_to_box(other)
            other_area = get_box_area(other_box) + 1e-12
            if len(other_key) <= len(text_key):
                continue
            if other_area <= area:
                continue

            inside = calculate_intersection_area(box, other_box) / area
            if inside < min_inside:
                continue

            # Keep the larger OCR phrase when a smaller OCR fragment is
            # geometrically inside it. This handles PaddleOCR outputs like
            # "fosc = 8 MHz" plus "fosc" plus "= 8 MHz".
            if len(other_key) >= len(text_key) + 2:
                drop_ids.add(idx)
                break

            if text_key in other_key:
                drop_ids.add(idx)
                break

            text_tokens = set(re.findall(r"[a-z0-9]+", text_key))
            other_tokens = set(re.findall(r"[a-z0-9]+", other_key))
            if text_tokens and len(text_tokens & other_tokens) / len(text_tokens) >= 0.70:
                drop_ids.add(idx)
                break

    return [item for idx, item in enumerate(candidates) if idx not in drop_ids]


def _best_col_support_for_item(item_box, final_cols):
    if not item_box or not final_cols:
        return False, 0.0

    def _scan(use_assign_box):
        cx, _ = get_box_center(item_box)
        item_w = max(1.0, float(item_box[2] - item_box[0]))
        best_cover = 0.0
        center_inside = False
        for col in final_cols:
            col_box = _col_assignment_box(col) if use_assign_box else col["box"]
            if col_box[0] <= cx <= col_box[2]:
                center_inside = True
            inter_w = max(0.0, min(item_box[2], col_box[2]) - max(item_box[0], col_box[0]))
            if inter_w <= 0.0:
                continue
            col_w = max(1.0, float(col_box[2] - col_box[0]))
            best_cover = max(best_cover, inter_w / item_w, inter_w / col_w)
        return center_inside, best_cover

    center_inside, best_cover = _scan(use_assign_box=True)
    if center_inside or best_cover > 0.0:
        return center_inside, best_cover
    return _scan(use_assign_box=False)


def _col_assignment_box(col):
    if not col:
        return None
    assign_box = col.get("assign_box")
    if assign_box:
        return assign_box
    return col.get("box")


def _center_matching_col_candidates(item_box, final_cols, use_assign_box=True):
    if not item_box or not final_cols:
        return []

    cx, _ = get_box_center(item_box)
    matches = []
    last_idx = len(final_cols) - 1
    for idx, col in enumerate(final_cols):
        col_box = _col_assignment_box(col) if use_assign_box else col["box"]
        if not col_box:
            continue
        left = float(col_box[0])
        right = float(col_box[2])
        center_inside = (left <= cx <= right) if idx == last_idx else (left <= cx < right)
        if not center_inside:
            continue
        center_x = (left + right) / 2.0
        matches.append(
            {
                "idx": idx,
                "center_dx": abs(cx - center_x),
                "width": max(1.0, right - left),
            }
        )

    matches.sort(key=lambda entry: (entry["center_dx"], entry["width"]))
    return matches


def _column_support_candidates(item_box, final_cols, allow_soft=False):
    if not item_box or not final_cols:
        return []

    def _scan(use_assign_box):
        cx, _ = get_box_center(item_box)
        item_w = max(1.0, float(item_box[2] - item_box[0]))
        center_matches = _center_matching_col_candidates(
            item_box,
            final_cols,
            use_assign_box=use_assign_box,
        )
        center_match_idxs = {entry["idx"] for entry in center_matches}
        candidates = []
        for idx, col in enumerate(final_cols):
            col_box = _col_assignment_box(col) if use_assign_box else col["box"]
            if center_match_idxs and idx not in center_match_idxs:
                continue
            col_w = max(1.0, float(col_box[2] - col_box[0]))
            inter_w = max(0.0, min(item_box[2], col_box[2]) - max(item_box[0], col_box[0]))
            item_overlap = inter_w / item_w
            col_overlap = inter_w / col_w
            center_inside = col_box[0] <= cx <= col_box[2]
            score = max(item_overlap, col_overlap) + (1.0 if center_inside else 0.0)

            strong = center_inside or item_overlap >= 0.18 or col_overlap >= 0.35
            soft = inter_w > 0.0 and item_overlap >= 0.10 and col_overlap >= 0.18
            if not strong and not (allow_soft and soft):
                continue

                candidates.append(
                    {
                        "idx": idx,
                        "center_inside": center_inside,
                        "center_dx": abs(cx - ((col_box[0] + col_box[2]) / 2.0)),
                        "item_overlap": item_overlap,
                        "col_overlap": col_overlap,
                        "score": score,
                    }
                )

        if center_match_idxs:
            candidates.sort(
                key=lambda entry: (
                    entry["center_dx"],
                    -entry["item_overlap"],
                    -entry["col_overlap"],
                )
            )
        else:
            candidates.sort(
                key=lambda entry: (
                    entry["score"],
                    entry["item_overlap"],
                    entry["col_overlap"],
                ),
                reverse=True,
            )
        return candidates

    candidates = _scan(use_assign_box=True)
    if candidates:
        return candidates
    return _scan(use_assign_box=False)


def _supported_col_idx_for_item(item_box, final_cols, allow_soft=False):
    center_matches = _center_matching_col_candidates(item_box, final_cols, use_assign_box=True)
    if center_matches:
        return center_matches[0]["idx"]
    center_matches = _center_matching_col_candidates(item_box, final_cols, use_assign_box=False)
    if center_matches:
        return center_matches[0]["idx"]

    candidates = _column_support_candidates(item_box, final_cols, allow_soft=allow_soft)
    if not candidates:
        return None
    return candidates[0]["idx"]


def _item_is_assignable_table_text(
    item_box,
    final_rows,
    row_y_ranges,
    row_centers,
    final_cols,
    first_assignable_row_top,
    last_assignable_row_bottom,
    avg_row_h,
):
    if not item_box or not final_rows or not row_y_ranges or not final_cols:
        return False

    ix1, iy1, ix2, iy2 = item_box
    _, cy = get_box_center(item_box)
    item_h = max(1.0, float(iy2 - iy1))
    table_x1 = min(col["box"][0] for col in final_cols)
    table_x2 = max(col["box"][2] for col in final_cols)

    # Ignore OCR that sits fully outside the table width.
    if ix2 <= (table_x1 - 12.0) or ix1 >= (table_x2 + 12.0):
        return False

    center_in_row = False
    best_row_cover = 0.0
    for r_idx, (ry1, ry2) in enumerate(row_y_ranges):
        if ry1 <= cy <= ry2:
            center_in_row = True
        inter_h = max(0.0, min(iy2, ry2) - max(iy1, ry1))
        if inter_h > 0.0:
            row_h = max(1.0, float(ry2 - ry1))
            best_row_cover = max(best_row_cover, min(inter_h / item_h, inter_h / row_h))

    center_in_col, best_col_cover = _best_col_support_for_item(item_box, final_cols)
    vertical_margin = max(8.0, 0.45 * avg_row_h)

    # Drop likely footer/title/outside notes that do not really attach to any row band.
    if cy < (first_assignable_row_top - vertical_margin) and best_row_cover < 0.18:
        return False
    if cy > (last_assignable_row_bottom + vertical_margin) and best_row_cover < 0.18:
        return False

    row_supported = (
        center_in_row
        or best_row_cover >= 0.18
    )
    if not row_supported:
        return False

    if center_in_col or best_col_cover >= 0.35:
        return True

    # Keep wide left-band section/row-header text if it still strongly tracks a real row.
    first_col_box = final_cols[0]["box"]
    inter_w = max(0.0, min(ix2, first_col_box[2]) - max(ix1, first_col_box[0]))
    first_col_cover = inter_w / max(1.0, float(ix2 - ix1))
    return first_col_cover >= 0.20 and best_row_cover >= 0.35


def _split_text_by_x_boundaries(text, item_box, boundary_xs):
    text = str(text or "").strip()
    if not text or not item_box or not boundary_xs:
        return None

    gap_positions = [idx for idx, ch in enumerate(text) if ch.isspace()]
    if not gap_positions:
        return None

    item_w = max(1.0, float(item_box[2] - item_box[0]))
    split_points = []
    prev_idx = 0
    max_jump = max(8, int(0.22 * len(text)))

    for boundary_x in boundary_xs:
        frac = (float(boundary_x) - item_box[0]) / item_w
        frac = min(0.92, max(0.08, frac))
        target_idx = int(round(frac * len(text)))
        candidates = [
            idx for idx in gap_positions
            if (prev_idx + 1) <= idx < (len(text) - 1)
        ]
        if not candidates:
            return None
        best_idx = min(candidates, key=lambda idx: abs(idx - target_idx))
        if abs(best_idx - target_idx) > max_jump:
            return None
        split_points.append(best_idx)
        prev_idx = best_idx + 1

    pieces = []
    start = 0
    for split_idx in split_points:
        part = text[start:split_idx].strip()
        if not part:
            return None
        pieces.append(part)
        start = split_idx + 1
    tail = text[start:].strip()
    if not tail:
        return None
    pieces.append(tail)
    return pieces


def _split_item_text_for_adjacent_cols(text, item_box, col_boxes):
    text = str(text or "").strip()
    if not text or not item_box or len(col_boxes) < 2:
        return None

    tokens = text.split()
    if len(tokens) == len(col_boxes):
        # Fast path for compact numeric/text pairs like "1 5" where OCR merged
        # two adjacent cells into one wide line.
        if all(_is_numericish_text(tok) or _alpha_frac(tok) >= 0.25 for tok in tokens):
            return tokens

    boundary_xs = []
    for idx in range(len(col_boxes) - 1):
        left_box = col_boxes[idx]
        right_box = col_boxes[idx + 1]
        left_edge = min(item_box[2], left_box[2])
        right_edge = max(item_box[0], right_box[0])
        if right_edge <= left_edge:
            return None
        boundary_xs.append((left_edge + right_edge) / 2.0)

    return _split_text_by_x_boundaries(text, item_box, boundary_xs)


def _repair_wide_body_item_spill(matrix, final_rows, final_cols, row_assignments, span_row_overrides=None, span_col_overrides=None):
    if not matrix or not final_rows or not final_cols or not row_assignments:
        return

    blocked_cells = set((span_row_overrides or {}).keys()) | set((span_col_overrides or {}).keys())
    for r_idx, row_meta in enumerate(final_rows):
        if row_meta["type"] != CLASS_NAME_ROW or r_idx >= len(matrix):
            continue

        row = matrix[r_idx]
        if _row_looks_like_compact_grid(row):
            continue

        # Only repair dense table rows where one wide OCR line is clearly hiding
        # a missing adjacent cell; sparse descriptive rows are left untouched.
        dense_cols = sum(1 for cell in row if str(cell).strip())
        if dense_cols < max(3, len(final_cols) - 1):
            continue

        for item in row_assignments[r_idx]:
            item_text = str(item.get("text", "")).strip()
            if not item_text:
                continue

            ibox = o_to_box(item)
            item_w = max(1.0, float(ibox[2] - ibox[0]))
            covered = []
            for c_idx, col in enumerate(final_cols):
                col_box = col["box"]
                inter_w = max(0.0, min(ibox[2], col_box[2]) - max(ibox[0], col_box[0]))
                if inter_w <= 0.0:
                    continue
                col_w = max(1.0, float(col_box[2] - col_box[0]))
                if (inter_w / item_w) >= 0.18 and (inter_w / col_w) >= 0.22:
                    covered.append(c_idx)

            if len(covered) != 2 or covered[1] != covered[0] + 1:
                continue
            if any((r_idx, c_idx) in blocked_cells for c_idx in covered):
                continue

            left_idx, right_idx = covered
            owner_idx = _preferred_col_idx_for_item(ibox, final_cols)
            if owner_idx not in covered:
                continue

            owner_text = str(row[owner_idx]).strip()
            other_idx = right_idx if owner_idx == left_idx else left_idx
            other_text = str(row[other_idx]).strip()
            if not owner_text:
                continue
            dup_fill = (
                other_text
                and _label_key(other_text) == _label_key(item_text)
                and _label_key(owner_text) == _label_key(item_text)
            )
            if other_text and not dup_fill:
                continue
            if _label_key(owner_text) != _label_key(item_text):
                continue
            if item_w < 1.35 * min(
                max(1.0, float(final_cols[left_idx]["box"][2] - final_cols[left_idx]["box"][0])),
                max(1.0, float(final_cols[right_idx]["box"][2] - final_cols[right_idx]["box"][0])),
            ):
                continue

            parts = _split_item_text_for_adjacent_cols(
                item_text,
                ibox,
                [final_cols[left_idx]["box"], final_cols[right_idx]["box"]],
            )
            if not parts or len(parts) != 2:
                continue

            if min(len(_label_key(part)) for part in parts) < 2:
                continue

            row[left_idx] = parts[0]
            row[right_idx] = parts[1]


def _suppress_sparse_fragment_row_spill(matrix, final_rows, final_cols, row_assignments, blocked_cells=None):
    if not matrix or not final_rows or not final_cols or not row_assignments:
        return
    blocked_cells = blocked_cells or set()

    for r_idx, row_meta in enumerate(final_rows):
        if row_meta["type"] != CLASS_NAME_ROW or r_idx >= len(matrix):
            continue

        row = matrix[r_idx]
        if not row or not str(row[0]).strip():
            continue
        if not _looks_like_wrapped_label_fragment(row[0]):
            continue
        if not any(str(cell).strip() for cell in row[1:]):
            continue

        assigned_cols = set()
        for item in row_assignments[r_idx]:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            col_idx = _preferred_col_idx_for_item(o_to_box(item), final_cols)
            if col_idx is not None:
                assigned_cols.add(col_idx)

        if assigned_cols != {0}:
            continue

        for c_idx in range(1, len(row)):
            if (r_idx, c_idx) in blocked_cells:
                continue
            row[c_idx] = ""


def _suppress_blank_leading_duplicate_bridge_rows(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) < 3:
        return matrix, final_rows

    total_cols = max((len(row) for row in matrix), default=0)
    min_leading_blank = max(2, min(3, total_cols // 3))
    kept_matrix = []
    kept_rows = []

    for r_idx, row in enumerate(matrix):
        row_meta = final_rows[r_idx] if r_idx < len(final_rows) else {}
        if row_meta.get("type") != CLASS_NAME_ROW:
            kept_matrix.append(row)
            kept_rows.append(row_meta)
            continue

        prev_idx = next(
            (
                idx
                for idx in range(r_idx - 1, -1, -1)
                if idx < len(final_rows) and final_rows[idx].get("type") == CLASS_NAME_ROW
            ),
            None,
        )
        next_idx = next(
            (
                idx
                for idx in range(r_idx + 1, len(matrix))
                if idx < len(final_rows) and final_rows[idx].get("type") == CLASS_NAME_ROW
            ),
            None,
        )
        if prev_idx is None and next_idx is None:
            kept_matrix.append(row)
            kept_rows.append(row_meta)
            continue

        normalized = [str(cell).strip() for cell in row]
        nonempty_indices = [idx for idx, cell in enumerate(normalized) if cell]
        if len(nonempty_indices) < 2:
            kept_matrix.append(row)
            kept_rows.append(row_meta)
            continue

        first_nonempty = nonempty_indices[0]

        if first_nonempty < min_leading_blank:
            kept_matrix.append(row)
            kept_rows.append(row_meta)
            continue

        def _duplicates_neighbor(neighbor_idx):
            if neighbor_idx is None:
                return False
            neighbor = [str(cell).strip() for cell in matrix[neighbor_idx]]
            if not any(neighbor[idx].strip() for idx in range(min(first_nonempty, len(neighbor)))):
                return False
            duplicate_hits = 0
            for idx in nonempty_indices:
                if idx < len(neighbor) and normalized[idx] and normalized[idx] == neighbor[idx].strip():
                    duplicate_hits += 1
            return duplicate_hits >= max(2, int(0.80 * len(nonempty_indices)))

        if _duplicates_neighbor(prev_idx) or _duplicates_neighbor(next_idx):
            continue

        kept_matrix.append(row)
        kept_rows.append(row_meta)

    return kept_matrix, kept_rows


def _propagate_split_parent_headers(matrix, final_rows, final_cols):
    if not matrix or not final_rows or not final_cols:
        return

    split_groups = {}
    for idx, col in enumerate(final_cols):
        if not col.get("is_split"):
            continue
        parent_key = _box_anchor_key(col.get("parent_box", col["box"]))
        if parent_key is None:
            continue
        split_groups.setdefault(parent_key, []).append(idx)

    split_groups = {
        parent_key: sorted(indices)
        for parent_key, indices in split_groups.items()
        if len(indices) >= 2
    }
    if not split_groups:
        return

    for r_idx, row_meta in enumerate(final_rows):
        if r_idx >= len(matrix) or not _is_header_row_type(row_meta["type"]):
            continue

        row = matrix[r_idx]
        for indices in split_groups.values():
            texts = []
            for c_idx in indices:
                if c_idx >= len(row):
                    continue
                cell_text = str(row[c_idx]).strip()
                if cell_text:
                    texts.append(cell_text)

            unique_texts = []
            seen_keys = set()
            for text in texts:
                key = _label_key(text)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_texts.append(text)

            if len(unique_texts) != 1:
                continue

            if all(str(row[c_idx]).strip() for c_idx in indices if c_idx < len(row)):
                continue

            parent_text = unique_texts[0]
            for c_idx in indices:
                if c_idx < len(row) and not str(row[c_idx]).strip():
                    row[c_idx] = parent_text


def _merge_header_fragments_across_missing_row_separator(matrix, final_rows, final_cols, structure_gray):
    if (
        not matrix
        or not final_rows
        or not final_cols
        or structure_gray is None
        or getattr(structure_gray, "ndim", None) != 2
    ):
        return

    header_indices = [
        idx
        for idx, row in enumerate(final_rows[: len(matrix)])
        if _is_header_row_type(row.get("type"))
    ]
    if len(header_indices) < 2:
        return

    for upper_idx, lower_idx in zip(header_indices, header_indices[1:]):
        if lower_idx != upper_idx + 1:
            continue
        upper_row = final_rows[upper_idx]
        lower_row = final_rows[lower_idx]

        separator_by_col = []
        for col in final_cols:
            separator_by_col.append(
                _owner_column_has_row_separator(
                    structure_gray,
                    col["box"],
                    upper_row["box"],
                    lower_row["box"],
                )
            )
        if not any(separator_by_col):
            continue

        for c_idx, col in enumerate(final_cols):
            if c_idx >= len(matrix[upper_idx]) or c_idx >= len(matrix[lower_idx]):
                continue
            upper_text = str(matrix[upper_idx][c_idx]).strip()
            lower_text = str(matrix[lower_idx][c_idx]).strip()
            if not upper_text or not lower_text:
                continue
            if any(
                item.get("class") == CLASS_NAME_SPANNING_COLUMN_HEADER
                and _smaller_region_overlap_ratio(item.get("box"), col["box"]) >= 0.20
                for item in upper_row.get("items", [])
            ):
                continue
            if separator_by_col[c_idx]:
                continue

            neighbor_has_separator = any(
                separator_by_col[n_idx]
                for n_idx in (c_idx - 1, c_idx + 1)
                if 0 <= n_idx < len(separator_by_col)
            )
            if not neighbor_has_separator:
                continue

            upper_key = _label_key(upper_text)
            lower_key = _label_key(lower_text)
            if not upper_key or not lower_key:
                continue
            if upper_key == lower_key:
                merged_text = upper_text
            elif lower_key in upper_key:
                merged_text = upper_text
            elif upper_key in lower_key:
                merged_text = lower_text
            else:
                merged_text = f"{upper_text} {lower_text}".strip()

            matrix[upper_idx][c_idx] = ""
            matrix[lower_idx][c_idx] = merged_text


def _row_items_overlapping_split_group(row_items, group_x1, group_x2):
    group_items = []
    for item in row_items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        inter_w = max(0.0, min(ibox[2], group_x2) - max(ibox[0], group_x1))
        if inter_w <= 0.0:
            continue
        group_items.append(item)
    return group_items


def _split_child_idx_for_item_start(item_box, final_cols, indices):
    if not item_box or not final_cols or not indices:
        return None

    item_left = float(item_box[0])
    matches = []
    for pos, c_idx in enumerate(indices):
        col_box = _col_assignment_box(final_cols[c_idx]) or final_cols[c_idx]["box"]
        left = float(col_box[0])
        right = float(col_box[2])
        col_w = max(1.0, right - left)
        # Roboflow/OCR boxes can be off by a few pixels; keep the slack small so
        # the rule only changes clear split-child starts, not normal columns.
        slack = max(2.0, min(6.0, 0.04 * col_w))
        if (left - slack) <= item_left <= (left + slack):
            matches.append((abs(item_left - left), pos, c_idx))

    if not matches:
        return None
    matches.sort(key=lambda entry: (entry[0], entry[1]))
    return matches[0][2]


def _should_assign_left_aligned_unsplit_group_item(
    item_box,
    group_items,
    final_cols,
    indices,
    target_col_idx,
    current_item=None,
):
    if (
        not item_box
        or not final_cols
        or not indices
        or len(indices) < 2
        or target_col_idx == indices[0]
    ):
        return False

    # If another item in this split group already has its center inside the first
    # sub-column, the split is genuine — don't force the current item there too
    # (that would pile both items into col 0 and leave col 1 empty).
    # However if the first sub-column is unclaimed, still apply the left-edge
    # check below so that a spanning-cell text whose center drifted into col 1
    # (due to its wide OCR box) is correctly pulled back to col 0.
    if len(group_items) > 1:
        first_col_box = (
            _col_assignment_box(final_cols[indices[0]])
            or final_cols[indices[0]]["box"]
        )
        first_x1 = float(first_col_box[0])
        first_x2 = float(first_col_box[2])
        for other in group_items:
            if other is current_item:
                continue
            other_box = o_to_box(other) if isinstance(other, dict) else other
            ocx = (float(other_box[0]) + float(other_box[2])) / 2.0
            if first_x1 <= ocx <= first_x2:
                # Another item genuinely belongs to col 0 — respect center-based
                # assignment for the current item.
                return False

    first_col = final_cols[indices[0]]
    first_box = _col_assignment_box(first_col) or first_col["box"]
    group_x1 = min((_col_assignment_box(final_cols[c_idx]) or final_cols[c_idx]["box"])[0] for c_idx in indices)
    group_x2 = max((_col_assignment_box(final_cols[c_idx]) or final_cols[c_idx]["box"])[2] for c_idx in indices)

    group_w = max(1.0, float(group_x2 - group_x1))
    first_w = max(1.0, float(first_box[2] - first_box[0]))

    left_slack = max(8.0, min(22.0, 0.12 * first_w, 0.035 * group_w))
    return float(item_box[0]) <= float(group_x1) + left_slack


def _reassign_split_group_text_by_center(matrix, final_rows, final_cols, row_assignments, blocked_cells=None):
    if not matrix or not final_rows or not final_cols or not row_assignments:
        return

    blocked_cells = blocked_cells or set()
    split_groups = {}
    for idx, col in enumerate(final_cols):
        if not col.get("is_split"):
            continue
        parent_key = _box_anchor_key(col.get("parent_box", col["box"]))
        if parent_key is None:
            continue
        split_groups.setdefault(parent_key, []).append(idx)

    split_groups = {
        parent_key: sorted(indices)
        for parent_key, indices in split_groups.items()
        if len(indices) >= 2
    }
    if not split_groups:
        return

    for r_idx, row_meta in enumerate(final_rows):
        if row_meta["type"] != CLASS_NAME_ROW or r_idx >= len(matrix) or r_idx >= len(row_assignments):
            continue

        row_items = row_assignments[r_idx]
        if not row_items:
            continue

        row = matrix[r_idx]
        for indices in split_groups.values():
            if any((r_idx, c_idx) in blocked_cells for c_idx in indices):
                continue

            group_x1 = min(final_cols[c_idx]["box"][0] for c_idx in indices)
            group_x2 = max(final_cols[c_idx]["box"][2] for c_idx in indices)
            items_by_col = {c_idx: [] for c_idx in indices}
            group_items = _row_items_overlapping_split_group(row_items, group_x1, group_x2)

            for item in row_items:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                ibox = o_to_box(item)
                inter_w = max(0.0, min(ibox[2], group_x2) - max(ibox[0], group_x1))
                if inter_w <= 0.0:
                    continue

                target_col_idx = _split_child_idx_for_item_start(ibox, final_cols, indices)
                if target_col_idx is None:
                    cx, _ = get_box_center(ibox)
                    last_pos = len(indices) - 1
                    for pos, c_idx in enumerate(indices):
                        col_box = _col_assignment_box(final_cols[c_idx]) or final_cols[c_idx]["box"]
                        left = float(col_box[0])
                        right = float(col_box[2])
                        center_inside = (left <= cx <= right) if pos == last_pos else (left <= cx < right)
                        if center_inside:
                            target_col_idx = c_idx
                            break

                if target_col_idx is None:
                    continue
                if _should_assign_left_aligned_unsplit_group_item(
                    ibox,
                    group_items,
                    final_cols,
                    indices,
                    target_col_idx,
                    current_item=item,
                ):
                    target_col_idx = indices[0]
                items_by_col[target_col_idx].append(item)

            if not any(items_by_col[c_idx] for c_idx in indices):
                continue

            for c_idx in indices:
                items = items_by_col[c_idx]
                if not items:
                    row[c_idx] = ""
                    continue
                items = _dedupe_near_duplicate_text_items(items)
                items = _sort_ocr_items_reading_order(items)
                row[c_idx] = " ".join(
                    str(item.get("text", "")).strip()
                    for item in items
                    if str(item.get("text", "")).strip()
                ).strip()


def _preferred_col_idx_for_item(item_box, final_cols):
    if not item_box or not final_cols:
        return None

    supported_idx = _supported_col_idx_for_item(item_box, final_cols, allow_soft=False)
    if supported_idx is not None:
        return supported_idx

    def _fallback_idx(use_assign_box):
        cx, _ = get_box_center(item_box)
        col_boxes = [(_col_assignment_box(col) if use_assign_box else col["box"]) for col in final_cols]
        col_centers = [(col_box[0] + col_box[2]) / 2 for col_box in col_boxes]
        candidate_idxs = [i for i, col_box in enumerate(col_boxes) if col_box[0] <= cx <= col_box[2]]
        if candidate_idxs:
            return min(candidate_idxs, key=lambda i: abs(cx - col_centers[i]))

        item_w = max(1.0, float(item_box[2] - item_box[0]))
        overlap_scores = []
        for idx, col_box in enumerate(col_boxes):
            inter_w = max(0.0, min(item_box[2], col_box[2]) - max(item_box[0], col_box[0]))
            overlap_scores.append((inter_w / item_w, inter_w, idx))

        best_overlap = max(overlap_scores, default=(0.0, 0.0, None))
        if best_overlap[0] >= 0.35:
            return best_overlap[2]
        return None

    fallback_idx = _fallback_idx(use_assign_box=True)
    if fallback_idx is not None:
        return fallback_idx
    fallback_idx = _fallback_idx(use_assign_box=False)
    if fallback_idx is not None:
        return fallback_idx

    return None


def _cluster_row_items_by_y(row_items, gap_thresh):
    if not row_items:
        return []

    ordered = sorted(row_items, key=lambda item: get_box_center(o_to_box(item))[1])
    clusters = [[ordered[0]]]
    prev_cy = get_box_center(o_to_box(ordered[0]))[1]
    for item in ordered[1:]:
        cy = get_box_center(o_to_box(item))[1]
        if (cy - prev_cy) > gap_thresh:
            clusters.append([item])
        else:
            clusters[-1].append(item)
        prev_cy = cy
    return clusters


def _row_cell_raw_items(logic_cell_box, cell_box, final_cols, target_col_idx, ocr_data):
    items = []
    target_col = final_cols[target_col_idx] if 0 <= target_col_idx < len(final_cols) else None
    for item in _ocr_items_in_box(logic_cell_box, ocr_data, intersection_thresh=0.05):
        ibox = o_to_box(item)
        if target_col and target_col.get("is_split"):
            center_matches = _center_matching_col_candidates(ibox, final_cols, use_assign_box=True)
            if not center_matches:
                center_matches = _center_matching_col_candidates(ibox, final_cols, use_assign_box=False)
            if center_matches:
                if center_matches[0]["idx"] == target_col_idx:
                    items.append(item)
                continue

        preferred_idx = _preferred_col_idx_for_item(ibox, final_cols)
        if preferred_idx != target_col_idx:
            continue
        cx = get_box_center(ibox)[0]
        if cell_box[0] <= cx <= cell_box[2]:
            items.append(item)
            continue
        inter_w = max(0.0, min(ibox[2], cell_box[2]) - max(ibox[0], cell_box[0]))
        if inter_w / (max(1.0, ibox[2] - ibox[0])) >= 0.50:
            items.append(item)
    return items


def _row_has_compact_split_support(row_items, final_cols, cluster_centers, baseline):
    if not row_items or not final_cols or len(cluster_centers) != 2:
        return False

    support_items = []
    for item in row_items:
        col_idx = _preferred_col_idx_for_item(o_to_box(item), final_cols)
        if col_idx is None or col_idx == 0:
            continue
        support_items.append(item)

    if len(support_items) < 2:
        return False

    support_clusters = _cluster_row_items_by_y(support_items, gap_thresh=max(10.0, 0.24 * baseline))
    if len(support_clusters) != 2:
        return False

    support_centers = []
    covered_cols = set()
    for cluster in support_clusters:
        cluster_ys = []
        cluster_cols = set()
        for item in cluster:
            ibox = o_to_box(item)
            cluster_ys.append(get_box_center(ibox)[1])
            col_idx = _preferred_col_idx_for_item(ibox, final_cols)
            if col_idx is not None:
                cluster_cols.add(col_idx)
                covered_cols.add(col_idx)
        if not cluster_ys or not cluster_cols:
            return False
        support_centers.append(sum(cluster_ys) / len(cluster_ys))

    if len(covered_cols) < 2:
        return False

    tolerance = max(12.0, 0.35 * baseline)
    return all(abs(support_centers[idx] - cluster_centers[idx]) <= tolerance for idx in range(2))


def get_ocr_items_in_box(box, ocr_data, intersection_thresh=INTERSECTION_THRESH_OCR_GRID):
    items = []
    for ocr_item in ocr_data:
        o_box = o_to_box(ocr_item)
        o_center = get_box_center(o_box)
        is_center_in = box[0] <= o_center[0] <= box[2] and box[1] <= o_center[1] <= box[3]
        inter = calculate_intersection_area(box, o_box)
        area = get_box_area(o_box)
        is_intersecting = (inter / (area + 1e-12)) > intersection_thresh
        if is_center_in or is_intersecting:
            items.append(ocr_item)

    return _sort_ocr_items_reading_order(items)


def _find_best_row_item_for_col(row_items, target_box):
    best_item = None
    best_score = 0.0
    for item in row_items:
        inter = calculate_intersection_area(item["box"], target_box)
        item_area = get_box_area(item["box"])
        score = inter / (item_area + 1e-12)
        if score > best_score:
            best_score = score
            best_item = item
    if best_score > 0.35:
        return best_item
    return None


def _find_covering_header_item_for_col(row_items, target_box):
    best_item = None
    best_score = 0.0
    for item in row_items:
        item_box = item["box"]
        x_cover = _x_overlap_ratio(item_box, target_box)
        if x_cover < 0.85:
            continue
        inter = calculate_intersection_area(item_box, target_box)
        target_area = get_box_area(target_box)
        item_area = get_box_area(item_box)
        # Prefer headers that substantially cover the target column, but
        # mildly favor the narrower/more specific candidate if several fit.
        score = (inter / (target_area + 1e-12)) + 0.15 * (inter / (item_area + 1e-12))
        if score > best_score:
            best_score = score
            best_item = item
    if best_score > 0.55:
        return best_item
    return None


def _extract_box_text_prefer_row(
    box,
    row_ocr_data,
    ocr_data,
    first_col=False,
):
    if not box:
        return ""

    if first_col:
        text = _get_header_text_in_box_dedup(box, row_ocr_data, center_only=True).strip()
        if text:
            return text
        text = _get_header_text_in_box_dedup(box, row_ocr_data, intersection_thresh=0.55).strip()
        if text:
            return text
        text = _get_header_text_in_box_dedup(box, ocr_data, center_only=True).strip()
        if text:
            return text
        text = _get_header_text_in_box_dedup(box, ocr_data, intersection_thresh=0.55).strip()
        return text.strip()

    text = _get_header_text_in_box_dedup(box, row_ocr_data, intersection_thresh=0.05).strip()
    if not text:
        text = _get_header_text_in_box_dedup(box, row_ocr_data, center_only=True).strip()
    if not text:
        text = _get_header_text_in_box_dedup(
            box,
            ocr_data,
            intersection_thresh=HEADER_GLOBAL_FALLBACK_INTERSECTION_THRESH,
        ).strip()
    if not text:
        text = _get_header_text_in_box_dedup(box, ocr_data, center_only=True).strip()
    return text.strip()


def _find_split_parent_header_item(header_items, final_cols, col, logic_cell_box):
    if not header_items or not final_cols or not col.get("is_split"):
        return None

    parent_box = col.get("parent_box", col["box"])
    parent_key = _box_anchor_key(parent_box)
    sibling_cols = [
        sibling
        for sibling in final_cols
        if sibling.get("is_split")
        and _box_anchor_key(sibling.get("parent_box", sibling["box"])) == parent_key
    ]
    if len(sibling_cols) < 2:
        return None

    parent_header_box = [
        parent_box[0],
        logic_cell_box[1],
        parent_box[2],
        logic_cell_box[3],
    ]

    best_item = None
    best_score = 0.0
    for item in header_items:
        item_box = item["box"]
        item_area = get_box_area(item_box) + 1e-12
        supported_children = 0
        for sibling in sibling_cols:
            sibling_box = _col_assignment_box(sibling) or sibling["box"]
            scx = (sibling_box[0] + sibling_box[2]) / 2.0
            inter = calculate_intersection_area(item_box, sibling_box)
            if (
                sibling_box[0] <= scx <= sibling_box[2]
                and item_box[0] <= scx <= item_box[2]
            ) or _x_overlap_ratio(item_box, sibling_box) >= 0.18 or (inter / item_area) >= 0.18:
                supported_children += 1

        if supported_children < 2:
            continue

        inter = calculate_intersection_area(item_box, parent_header_box)
        parent_area = get_box_area(parent_header_box) + 1e-12
        score = (
            float(supported_children)
            + (inter / parent_area)
            + 0.25 * _x_overlap_ratio(item_box, parent_header_box)
        )
        if score > best_score:
            best_score = score
            best_item = item

    return best_item


def _should_skip_header_item_for_propagation(final_rows, r_idx, row, item, final_cols):
    if item.get("class") != CLASS_NAME_COLUMN_HEADER or not final_cols:
        return False

    table_left = final_cols[0]["box"][0]
    table_right = final_cols[-1]["box"][2]
    table_w = max(1.0, float(final_cols[-1]["box"][2] - final_cols[0]["box"][0]))
    item_box = item["box"]
    item_w = max(1.0, float(item_box[2] - item_box[0]))
    item_h = max(1.0, float(item_box[3] - item_box[1]))
    if item_w < 0.45 * table_w:
        return False
    if (
        (item_box[0] - table_left) > (0.10 * table_w)
        and (table_right - item_box[2]) > (0.10 * table_w)
    ):
        return False

    narrow_siblings = 0
    for other in row.get("items", []):
        if other is item or other.get("class") != CLASS_NAME_COLUMN_HEADER:
            continue
        other_w = max(1.0, float(other["box"][2] - other["box"][0]))
        if other_w <= 0.25 * table_w:
            narrow_siblings += 1
    if narrow_siblings < 2:
        return False

    for next_row in final_rows[r_idx + 1:r_idx + 3]:
        if next_row["type"] != CLASS_NAME_ROW_HEADER:
            continue
        gap = next_row["box"][1] - item_box[3]
        if gap > max(12.0, 0.25 * item_h):
            continue
        if _x_overlap_ratio(item_box, next_row["box"]) >= 0.60:
            return True

    return False


def _header_boxes_near_same(box_a, box_b, tol=4.0):
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(box_a, box_b))


def _lower_header_children(parent, header_children):
    parent_box = parent["box"]
    parent_center_y = (parent_box[1] + parent_box[3]) / 2.0
    parent_h = max(1.0, float(parent_box[3] - parent_box[1]))

    children = []
    for child in header_children:
        child_box = child["box"]
        if child is parent or _header_boxes_near_same(parent_box, child_box):
            continue
        if child_box[1] <= (parent_center_y + 0.05 * parent_h):
            continue
        if _x_overlap_ratio(parent_box, child_box) < 0.45:
            continue
        children.append(child)
    return children


def _filter_valid_spanning_column_headers(span_col_headers, col_headers):
    if not span_col_headers:
        return []

    header_children = list(col_headers) + list(span_col_headers)
    span_ids = {id(span) for span in span_col_headers}
    memo = {}
    visiting = set()

    def _is_valid_span(span):
        key = id(span)
        if key in memo:
            return memo[key]
        if key in visiting:
            return False
        visiting.add(key)

        span_box = span["box"]
        children = _lower_header_children(span, header_children)
        if not children:
            visiting.remove(key)
            memo[key] = False
            return False

        valid_children = []
        for child in children:
            if id(child) in span_ids:
                if _is_valid_span(child):
                    valid_children.append(child)
                continue
            valid_children.append(child)

        if not valid_children:
            visiting.remove(key)
            memo[key] = False
            return False

        # Preserve real multi-level span chains even when a parent has only one
        # lower spanning child. Only suppress the classic duplicate case where a
        # spanning header and a normal column header are effectively the same band.
        if len(valid_children) == 1:
            child = valid_children[0]
            child_box = child["box"]
            same_width_band = (
                _x_overlap_ratio(span_box, child_box) >= 0.85
                and _x_overlap_ratio(child_box, span_box) >= 0.85
            )
            vertical_overlap = max(
                0.0,
                min(float(span_box[3]), float(child_box[3])) - max(float(span_box[1]), float(child_box[1])),
            )
            min_h = max(
                1.0,
                min(
                    float(span_box[3] - span_box[1]),
                    float(child_box[3] - child_box[1]),
                ),
            )
            overlapping_same_band = (vertical_overlap / min_h) >= 0.60
            if child.get("class") == CLASS_NAME_COLUMN_HEADER and same_width_band and overlapping_same_band:
                visiting.remove(key)
                memo[key] = False
                return False

        visiting.remove(key)
        memo[key] = True
        return True

    return [span for span in span_col_headers if _is_valid_span(span)]


def _headers_are_same_band_duplicates(header_a, header_b):
    box_a = header_a.get("box")
    box_b = header_b.get("box")
    if not box_a or not box_b:
        return False

    same_x = _x_overlap_ratio(box_a, box_b) >= 0.88 and _x_overlap_ratio(box_b, box_a) >= 0.88
    same_y = _y_overlap_ratio(box_a, box_b) >= 0.72 and _y_overlap_ratio(box_b, box_a) >= 0.72
    if not (same_x and same_y):
        return False

    # Allow small Roboflow jitter, but do not collapse genuinely wider parent headers.
    width_delta = abs(float(box_a[2] - box_a[0]) - float(box_b[2] - box_b[0]))
    height_delta = abs(float(box_a[3] - box_a[1]) - float(box_b[3] - box_b[1]))
    max_w = max(1.0, float(box_a[2] - box_a[0]), float(box_b[2] - box_b[0]))
    max_h = max(1.0, float(box_a[3] - box_a[1]), float(box_b[3] - box_b[1]))
    return width_delta <= max(8.0, 0.08 * max_w) and height_delta <= max(8.0, 0.15 * max_h)


def _same_header_band_members(reference_box, headers):
    if not reference_box:
        return []

    members = []
    for header in headers or []:
        box = header.get("box")
        if not box:
            continue
        if _y_overlap_ratio(reference_box, box) >= 0.55 or _y_overlap_ratio(box, reference_box) >= 0.55:
            members.append(header)
    return members


def _resolve_same_band_header_class_conflicts(col_headers, span_col_headers):
    if not col_headers or not span_col_headers:
        return col_headers, span_col_headers

    all_headers = list(col_headers) + list(span_col_headers)
    drop_col_ids = set()
    drop_span_ids = set()

    for span in span_col_headers:
        if id(span) in drop_span_ids:
            continue
        span_box = span.get("box")
        if not span_box:
            continue

        duplicate_cols = [
            col for col in col_headers
            if id(col) not in drop_col_ids and _headers_are_same_band_duplicates(col, span)
        ]
        if not duplicate_cols:
            continue

        # Same y-band context decides the class. This prevents one OCR/header box
        # from becoming two stacked rows only because Roboflow emitted both classes.
        band_members = _same_header_band_members(span_box, all_headers)
        col_support = sum(
            float(item.get("confidence", 0.0))
            for item in band_members
            if item.get("class") == CLASS_NAME_COLUMN_HEADER and id(item) not in drop_col_ids
        )
        span_support = sum(
            float(item.get("confidence", 0.0))
            for item in band_members
            if item.get("class") == CLASS_NAME_SPANNING_COLUMN_HEADER and id(item) not in drop_span_ids
        )

        best_col = max(duplicate_cols, key=lambda item: float(item.get("confidence", 0.0)))
        col_conf = float(best_col.get("confidence", 0.0))
        span_conf = float(span.get("confidence", 0.0))

        if span_support > (col_support * 1.10):
            keep_span = True
        elif col_support > (span_support * 1.10):
            keep_span = False
        else:
            keep_span = span_conf > col_conf

        if keep_span:
            for col in duplicate_cols:
                drop_col_ids.add(id(col))
        else:
            drop_span_ids.add(id(span))

    resolved_cols = [col for col in col_headers if id(col) not in drop_col_ids]
    resolved_spans = [span for span in span_col_headers if id(span) not in drop_span_ids]
    return resolved_cols, resolved_spans

def _clone_header_segment(header, seg_box, source):
    clone = dict(header)
    clone["box"] = [float(seg_box[0]), float(seg_box[1]), float(seg_box[2]), float(seg_box[3])]
    clone["x"] = (clone["box"][0] + clone["box"][2]) / 2.0
    clone["y"] = (clone["box"][1] + clone["box"][3]) / 2.0
    clone["width"] = max(0.0, clone["box"][2] - clone["box"][0])
    clone["height"] = max(0.0, clone["box"][3] - clone["box"][1])
    clone["source"] = source
    clone["split_parent_box"] = list(header["box"])
    return clone


def _header_ocr_items(header_box, ocr_data):
    if not header_box or not ocr_data:
        return []

    header_h = max(1.0, float(header_box[3] - header_box[1]))
    items = []
    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        inter_h = max(0.0, min(float(header_box[3]), float(ibox[3])) - max(float(header_box[1]), float(ibox[1])))
        inter_w = max(0.0, min(float(header_box[2]), float(ibox[2])) - max(float(header_box[0]), float(ibox[0])))
        if inter_h / header_h < 0.35 or inter_w <= 0:
            continue
        item_area = get_box_area(ibox) + 1e-12
        inter = calculate_intersection_area(header_box, ibox)
        cx, cy = get_box_center(ibox)
        if inter / item_area >= 0.20 or (header_box[0] <= cx <= header_box[2] and header_box[1] <= cy <= header_box[3]):
            items.append(item)
    return _sort_ocr_items_reading_order(items)


def _header_segment_has_ocr(seg_box, ocr_items):
    if not seg_box or not ocr_items:
        return False

    seg_h = max(1.0, float(seg_box[3] - seg_box[1]))
    for item in ocr_items:
        ibox = o_to_box(item)
        cx, cy = get_box_center(ibox)
        if seg_box[0] <= cx <= seg_box[2] and seg_box[1] <= cy <= seg_box[3]:
            return True
        inter_h = max(0.0, min(float(seg_box[3]), float(ibox[3])) - max(float(seg_box[1]), float(ibox[1])))
        inter_w = max(0.0, min(float(seg_box[2]), float(ibox[2])) - max(float(seg_box[0]), float(ibox[0])))
        if inter_h / seg_h >= 0.35 and inter_w / (max(1.0, float(ibox[2] - ibox[0]))) >= 0.45:
            return True
    return False


def _find_header_separator_xs(header_box, ocr_items, structure_gray):
    if (
        not ENABLE_HEADER_SPLIT_RECOVERY
        or structure_gray is None
        or structure_gray.ndim != 2
        or not header_box
    ):
        return []

    img_h, img_w = structure_gray.shape[:2]
    hx1 = max(0, int(round(header_box[0])))
    hx2 = min(img_w, int(round(header_box[2])))
    hy1 = max(0, int(round(header_box[1])))
    hy2 = min(img_h, int(round(header_box[3])))
    header_w = max(1.0, float(hx2 - hx1))
    header_h = max(1.0, float(hy2 - hy1))
    if header_w < 36.0 or header_h < 8.0:
        return []

    min_segment_w = max(16.0, min(34.0, 0.10 * header_w))
    scan_x1 = int(round(hx1 + min_segment_w))
    scan_x2 = int(round(hx2 - min_segment_w))
    if scan_x2 <= scan_x1:
        return []

    candidate_xs = []
    for x in range(scan_x1, scan_x2 + 1):
        if _line_x_crosses_ocr_text(x, hy1, hy2, ocr_items):
            continue
        strip = structure_gray[hy1:hy2, x:x + 1]
        if strip.size == 0:
            continue
        dark = strip[:, 0] <= STRUCTURE_LINE_DARK_PIXEL_MAX
        if float(np.mean(dark)) < 0.42:
            continue
        if _longest_true_run_ratio(dark) < 0.80:
            continue
        candidate_xs.append(x)

    if not candidate_xs:
        return []

    groups = []
    current = [candidate_xs[0]]
    for x in candidate_xs[1:]:
        if x - current[-1] <= 2:
            current.append(x)
        else:
            groups.append(current)
            current = [x]
    groups.append(current)

    return [float(sum(group) / len(group)) for group in groups]


def _split_header_by_separators(header, ocr_items, structure_gray):
    header_box = header["box"]
    separators = _find_header_separator_xs(header_box, ocr_items, structure_gray)
    if not separators:
        return []

    boundaries = [float(header_box[0])] + separators + [float(header_box[2])]
    segments = []
    for idx, (seg_x1, seg_x2) in enumerate(zip(boundaries, boundaries[1:])):
        if seg_x2 - seg_x1 < 12.0:
            continue
        seg_box = [seg_x1, header_box[1], seg_x2, header_box[3]]
        if not _header_segment_has_ocr(seg_box, ocr_items):
            continue
        clone = _clone_header_segment(header, seg_box, "header_separator_split")
        clone["split_segment_index"] = idx
        clone["split_segment_count"] = len(boundaries) - 1
        segments.append(clone)

    return segments if len(segments) >= 2 else []


def _header_ocr_supported_col_idx(item_box, support_cols):
    if not item_box or not support_cols:
        return None

    item_cx, _ = get_box_center(item_box)
    item_w = max(1.0, float(item_box[2] - item_box[0]))
    best_idx = None
    best_score = 0.0
    for idx, col in enumerate(support_cols):
        col_box = col["box"]
        overlap = max(0.0, min(float(item_box[2]), float(col_box[2])) - max(float(item_box[0]), float(col_box[0])))
        overlap_frac = overlap / item_w
        center_hit = 1.0 if col_box[0] <= item_cx <= col_box[2] else 0.0
        score = max(overlap_frac, center_hit)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None
    if best_score >= 0.45:
        return best_idx
    return None


def _split_header_by_ocr_columns(header, ocr_items, support_cols):
    if (
        not ENABLE_HEADER_SPLIT_RECOVERY
        or header.get("class") != CLASS_NAME_COLUMN_HEADER
        or not ocr_items
        or not support_cols
    ):
        return []

    header_box = header["box"]
    header_w = max(1.0, float(header_box[2] - header_box[0]))
    ordered = sorted(ocr_items, key=lambda item: o_to_box(item)[0])
    groups = []
    for item in ordered:
        ibox = o_to_box(item)
        col_idx = _header_ocr_supported_col_idx(ibox, support_cols)
        if col_idx is None:
            continue
        if groups and groups[-1]["col_idx"] == col_idx and ibox[0] - groups[-1]["box"][2] < max(10.0, 0.025 * header_w):
            gbox = groups[-1]["box"]
            groups[-1]["items"].append(item)
            groups[-1]["box"] = [
                min(gbox[0], ibox[0]),
                min(gbox[1], ibox[1]),
                max(gbox[2], ibox[2]),
                max(gbox[3], ibox[3]),
            ]
            continue
        groups.append({"col_idx": col_idx, "items": [item], "box": list(ibox)})

    if len(groups) < 2:
        return []

    distinct_cols = []
    for group in groups:
        if group["col_idx"] not in distinct_cols:
            distinct_cols.append(group["col_idx"])
    if len(distinct_cols) < 2:
        return []

    col_to_boxes = {}
    for group in groups:
        col_to_boxes.setdefault(group["col_idx"], []).append(group["box"])

    segments = []
    previous_text_right = None
    for col_idx in sorted(col_to_boxes):
        boxes = col_to_boxes[col_idx]
        text_left = min(box[0] for box in boxes)
        text_right = max(box[2] for box in boxes)
        if previous_text_right is not None and text_left - previous_text_right < max(10.0, 0.025 * header_w):
            return []
        previous_text_right = text_right

        col_box = support_cols[col_idx]["box"]
        seg_x1 = max(float(header_box[0]), float(col_box[0]))
        seg_x2 = min(float(header_box[2]), float(col_box[2]))
        if seg_x2 - seg_x1 < 12.0:
            continue
        seg_box = [seg_x1, header_box[1], seg_x2, header_box[3]]
        if _header_segment_has_ocr(seg_box, ocr_items):
            clone = _clone_header_segment(header, seg_box, "header_ocr_column_split")
            clone["split_column_index"] = col_idx
            segments.append(clone)

    return segments if len(segments) >= 2 else []


def _split_header_predictions_by_recovery(header_items, support_cols, ocr_data, structure_gray=None, allow_ocr=False):
    if not ENABLE_HEADER_SPLIT_RECOVERY or not header_items:
        return header_items

    recovered = []
    for header in header_items:
        ocr_items = _header_ocr_items(header["box"], ocr_data)
        separator_segments = _split_header_by_separators(header, ocr_items, structure_gray)
        if separator_segments:
            recovered.extend(separator_segments)
            continue
        if allow_ocr:
            ocr_segments = _split_header_by_ocr_columns(header, ocr_items, support_cols)
            if ocr_segments:
                recovered.extend(ocr_segments)
                continue
        recovered.append(header)

    return sorted(group_nms(recovered, IOS_THRESH_NMS_DEFAULT), key=lambda item: (item["box"][1], item["box"][0]))


def _header_row_definition_supported_cols(header_def, final_cols):
    if not header_def or not final_cols:
        return set()

    supported = set()
    members = header_def.get("items") or [{"box": header_def["box"]}]
    for member in members:
        mbox = member["box"]
        mcx, _ = get_box_center(mbox)
        member_area = get_box_area(mbox) + 1e-12
        for idx, col in enumerate(final_cols):
            cbox = col["box"]
            inter = calculate_intersection_area(mbox, cbox)
            if (
                cbox[0] <= mcx <= cbox[2]
                or _x_overlap_ratio(mbox, cbox) >= 0.18
                or inter / member_area >= 0.18
            ):
                supported.add(idx)
    return supported


def _is_proper_multi_header_band(header_def, final_cols):
    if not header_def:
        return False
    items = header_def.get("items") or []
    if len(items) >= 1:
        return True
    return len(_header_row_definition_supported_cols(header_def, final_cols)) >= 1


def _header_row_clear_gap(prev_box, cur_box):
    if not prev_box or not cur_box:
        return False
    prev_h = max(1.0, float(prev_box[3] - prev_box[1]))
    cur_h = max(1.0, float(cur_box[3] - cur_box[1]))
    gap = float(cur_box[1] - prev_box[3])
    return gap > max(18.0, 0.30 * max(prev_h, cur_h))


def _header_body_intrusion_level(header_box, rows_raw):
    if not header_box or not rows_raw:
        return "none"

    body_boxes = [row["box"] for row in rows_raw if row.get("box")]
    if not body_boxes:
        return "none"

    header_h = max(1.0, float(header_box[3] - header_box[1]))
    header_cy = (header_box[1] + header_box[3]) / 2.0
    first_body_top = min(float(box[1]) for box in body_boxes)

    if header_box[3] <= first_body_top + max(HEADER_BODY_TOUCH_TOL_PX, HEADER_BODY_TOUCH_TOL_FRAC * header_h):
        return "none"

    soft_hit = False
    for row_box in body_boxes:
        inter_h = max(0.0, min(header_box[3], row_box[3]) - max(header_box[1], row_box[1]))
        if inter_h <= 0:
            continue

        row_h = max(1.0, float(row_box[3] - row_box[1]))
        header_overlap = inter_h / header_h
        row_overlap = inter_h / row_h
        center_inside_row = row_box[1] <= header_cy <= row_box[3]

        if (
            center_inside_row
            or header_overlap >= HEADER_BODY_HARD_INTRUSION_HEADER_FRAC
            or row_overlap >= HEADER_BODY_HARD_INTRUSION_ROW_FRAC
        ):
            return "hard"

        if (
            header_overlap >= HEADER_BODY_SOFT_INTRUSION_FRAC
            or row_overlap >= HEADER_BODY_SOFT_INTRUSION_FRAC
        ):
            soft_hit = True

    return "soft" if soft_hit else "none"


def _header_supported_span_width(final_cols, supported_indices):
    if not final_cols or not supported_indices:
        return 0.0

    boxes = [
        final_cols[idx]["box"]
        for idx in sorted(set(supported_indices))
        if 0 <= idx < len(final_cols)
    ]
    if not boxes:
        return 0.0

    left = min(float(box[0]) for box in boxes)
    right = max(float(box[2]) for box in boxes)
    return max(1.0, right - left)


def _header_has_enough_supported_width(header_row, final_cols, supported_indices):
    if not header_row or not final_cols or not supported_indices:
        return False

    header_box = header_row["box"]
    header_w = max(1.0, float(header_box[2] - header_box[0]))
    header_h = max(1.0, float(header_box[3] - header_box[1]))
    supported_span_w = _header_supported_span_width(final_cols, supported_indices)
    if supported_span_w <= 0.0:
        return False

    header_class = header_row.get("header_class")
    if header_class == CLASS_NAME_COLUMN_HEADER:
        return header_w >= max(
            COLUMN_HEADER_MIN_SUPPORTED_WIDTH_FRAC * supported_span_w,
            COLUMN_HEADER_MIN_WIDTH_HEIGHT_MULT * header_h,
        )

    if header_class == CLASS_NAME_SPANNING_COLUMN_HEADER:
        if len(supported_indices) == 1:
            if _is_proper_multi_header_band(header_row, final_cols):
                return True
            return header_w >= max(
                SPANNING_HEADER_SINGLE_COL_WIDTH_MULT * supported_span_w,
                SPANNING_HEADER_MIN_WIDTH_HEIGHT_MULT * header_h,
            )
        return header_w >= max(
            SPANNING_HEADER_MIN_SUPPORTED_WIDTH_FRAC * supported_span_w,
            SPANNING_HEADER_MIN_WIDTH_HEIGHT_MULT * header_h,
        )

    return True


def _filter_late_isolated_header_rows(header_rows_definitions, final_cols, rows_raw=None):
    if not header_rows_definitions:
        return []

    ordered = sorted(header_rows_definitions, key=lambda row: row["y"])
    kept = [ordered[0]]
    last_kept_box = ordered[0]["box"]
    body_heights = [
        max(1.0, float(row["box"][3] - row["box"][1]))
        for row in (rows_raw or [])
        if row.get("box")
    ]
    median_body_h = float(np.median(body_heights)) if body_heights else float(MEDIAN_HEIGHT_DEFAULT)
    for header_row in ordered[1:]:
        clear_gap = _header_row_clear_gap(last_kept_box, header_row["box"])
        supported_indices = sorted(_header_row_definition_supported_cols(header_row, final_cols))
        supported_cols = len(supported_indices)
        proper_multi_band = _is_proper_multi_header_band(header_row, final_cols)
        intrusion = _header_body_intrusion_level(header_row["box"], rows_raw)
        width_ok = _header_has_enough_supported_width(header_row, final_cols, supported_indices)
        header_stack_bottom = _top_header_stack_bottom(kept) if kept else None
        late_vs_stack = (
            header_stack_bottom is not None
            and float(header_row["box"][1]) >= (
                header_stack_bottom + max(4.0, HEADER_STACK_GAP_MARGIN_ROWS * median_body_h)
            )
        )

        if header_row.get("header_class") == CLASS_NAME_COLUMN_HEADER:
            if rows_raw:
                if intrusion == "hard":
                    continue

                if intrusion == "soft" and late_vs_stack and supported_cols == 0:
                    continue

                if (
                    late_vs_stack
                    and supported_cols > 0
                    and not width_ok
                    and not proper_multi_band
                ):
                    continue

                if late_vs_stack and supported_cols == 0 and not proper_multi_band:
                    continue
            elif clear_gap and not proper_multi_band:
                continue
        elif header_row.get("header_class") == CLASS_NAME_SPANNING_COLUMN_HEADER:
            if rows_raw:
                if intrusion == "hard":
                    continue

                if intrusion == "soft" and late_vs_stack and supported_cols == 0:
                    continue

                if (
                    late_vs_stack
                    and supported_cols > 0
                    and not width_ok
                    and not proper_multi_band
                ):
                    continue

                if late_vs_stack and supported_cols == 0 and not proper_multi_band:
                    continue
        kept.append(header_row)
        last_kept_box = header_row["box"]
    return kept


def _top_header_stack_bottom(header_rows_definitions):
    if not header_rows_definitions:
        return None

    ordered = sorted(header_rows_definitions, key=lambda row: row["y"])
    stack_bottom = float(ordered[0]["box"][3])
    prev_box = ordered[0]["box"]
    for header_row in ordered[1:]:
        if _header_row_clear_gap(prev_box, header_row["box"]):
            break
        stack_bottom = max(stack_bottom, float(header_row["box"][3]))
        prev_box = header_row["box"]
    return stack_bottom


def _build_split_bands(valid_splits, y_gap=40):
    if not valid_splits:
        return []

    by_y = sorted(valid_splits, key=lambda split: (split["box"][1], (split["box"][1] + split["box"][3]) / 2))
    bands = []
    for split in by_y:
        split_box = split["box"]
        cy = (split["box"][1] + split["box"][3]) / 2
        split_h = max(1.0, float(split_box[3] - split_box[1]))
        matched_band = None
        for band in bands:
            band_h = max(1.0, float(band["y2"] - band["y1"]))
            overlap_h = max(0.0, min(split_box[3], band["y2"]) - max(split_box[1], band["y1"]))
            overlap_ratio = overlap_h / (min(split_h, band_h) + 1e-12)
            center_close = abs(cy - band["cy"]) <= y_gap
            if overlap_ratio >= 0.12 or (overlap_h > 0 and center_close):
                matched_band = band
                break

        if matched_band is not None:
            matched_band["items"].append(split)
            matched_band["y1"] = min(matched_band["y1"], split_box[1])
            matched_band["y2"] = max(matched_band["y2"], split_box[3])
            count = len(matched_band["items"])
            matched_band["cy"] = ((matched_band["cy"] * (count - 1)) + cy) / count
            continue

        if bands and abs(cy - bands[-1]["cy"]) <= y_gap:
            bands[-1]["items"].append(split)
            bands[-1]["y1"] = min(bands[-1]["y1"], split_box[1])
            bands[-1]["y2"] = max(bands[-1]["y2"], split_box[3])
            count = len(bands[-1]["items"])
            bands[-1]["cy"] = ((bands[-1]["cy"] * (count - 1)) + cy) / count
        else:
            bands.append({"cy": cy, "y1": split_box[1], "y2": split_box[3], "items": [split]})
    return bands


def _split_cluster_representative(cluster):
    if not cluster:
        return None
    return min(
        cluster,
        key=lambda split: (
            max(1.0, float(split["box"][2] - split["box"][0])),
            -float(split.get("confidence", 0.0)),
        ),
    )


def _cluster_split_items(split_items, min_gap):
    if not split_items:
        return []

    sorted_items = sorted(split_items, key=lambda split: (split["box"][0] + split["box"][2]) / 2)
    clusters = []
    for split in sorted_items:
        center = (split["box"][0] + split["box"][2]) / 2
        if not clusters:
            clusters.append([split])
            continue

        representative = _split_cluster_representative(clusters[-1])
        representative_box = representative["box"] if representative else clusters[-1][-1]["box"]
        representative_center = (representative_box[0] + representative_box[2]) / 2
        x_overlap = _x_overlap_ratio(split["box"], representative_box)
        if x_overlap >= 0.65:
            clusters[-1].append(split)
        elif center - representative_center > min_gap:
            clusters.append([split])
        else:
            clusters[-1].append(split)
    return clusters


def _infer_split_count(valid_splits, col_box=None):
    if not valid_splits:
        return 0

    min_gap = 30.0
    if col_box is not None:
        col_w = max(1.0, float(col_box[2] - col_box[0]))
        min_gap = max(24.0, 0.12 * col_w)

    best = 0
    for band in _build_split_bands(valid_splits):
        clusters = _cluster_split_items(band["items"], min_gap)
        best = max(best, len(clusters))

    return best


def _is_valid_split_candidate(split, col):
    if not split or not col:
        return False

    col_box = col["box"]
    col_w = max(1.0, float(col_box[2] - col_box[0]))
    col_h = max(1.0, float(col_box[3] - col_box[1]))
    split_box = split["box"]
    split_w = max(1.0, float(split_box[2] - split_box[0]))
    split_h = max(1.0, float(split_box[3] - split_box[1]))

    width_frac = split_w / col_w
    height_frac = split_h / col_h

    if split_w < max(10.0, 0.035 * col_w):
        return False
    if width_frac > 0.92:
        return False
    return True


def _split_boundaries_from_centers(valid_splits, col_box, split_count):
    if split_count < 2 or not valid_splits:
        return None

    col_w = max(1.0, float(col_box[2] - col_box[0]))
    min_gap = max(24.0, 0.12 * col_w)
    split_bands = _build_split_bands(valid_splits)
    reference_bands = []
    for band in split_bands:
        clusters = _cluster_split_items(band["items"], min_gap)
        if len(clusters) == split_count:
            reference_bands.append(clusters)

    avg_bounds = []
    if reference_bands:
        # Use the densest detected split pattern so the whole parent column is
        # subdivided by the maximum required number of sub-columns.
        for cluster_idx in range(split_count):
            cluster_items = []
            for band_clusters in reference_bands:
                representative = _split_cluster_representative(band_clusters[cluster_idx])
                if representative is not None:
                    cluster_items.append(representative)
            if not cluster_items:
                return None
            avg_xmin = sum(s["box"][0] for s in cluster_items) / len(cluster_items)
            avg_xmax = sum(s["box"][2] for s in cluster_items) / len(cluster_items)
            avg_bounds.append((avg_xmin, avg_xmax))
    else:
        valid_splits_sorted = sorted(valid_splits, key=lambda s: (s["box"][0] + s["box"][2]) / 2)
        clusters = _cluster_split_items(valid_splits_sorted, min_gap)

        while len(clusters) > split_count:
            best_idx = None
            best_gap = None
            for idx in range(len(clusters) - 1):
                left_center = sum((s["box"][0] + s["box"][2]) / 2 for s in clusters[idx]) / len(clusters[idx])
                right_center = sum((s["box"][0] + s["box"][2]) / 2 for s in clusters[idx + 1]) / len(clusters[idx + 1])
                gap = abs(right_center - left_center)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_idx = idx
            if best_idx is None:
                break
            clusters[best_idx].extend(clusters[best_idx + 1])
            del clusters[best_idx + 1]

        if len(clusters) != split_count:
            return None

        for cluster in clusters:
            representative = _split_cluster_representative(cluster)
            if representative is None:
                return None
            avg_xmin = representative["box"][0]
            avg_xmax = representative["box"][2]
            avg_bounds.append((avg_xmin, avg_xmax))

    boundaries = [col_box[0]]
    for i in range(len(avg_bounds) - 1):
        left_xmax = min(col_box[2], max(col_box[0], avg_bounds[i][1]))
        right_xmin = min(col_box[2], max(col_box[0], avg_bounds[i + 1][0]))
        # Midpoint between right edge of left column and left edge of right column
        midpoint = (left_xmax + right_xmin) / 2
        if midpoint <= boundaries[-1]:
            return None
        boundaries.append(midpoint)
    boundaries.append(col_box[2])
    return boundaries


def _prediction_anchor_key(pred):
    if not pred:
        return None

    box = pred.get("box") or [0.0, 0.0, 0.0, 0.0]
    return (
        pred.get("class"),
        round(float(box[0]), 3),
        round(float(box[1]), 3),
        round(float(box[2]), 3),
        round(float(box[3]), 3),
        round(float(pred.get("confidence", 0.0)), 6),
    )


def _box_anchor_key(box):
    if not box:
        return None
    return tuple(round(float(v), 3) for v in box)


def _build_parent_split_plans(cols_raw, splits):
    plans = {}
    if not cols_raw:
        return plans

    for col in cols_raw:
        col_box = col["box"]
        parent_key = _box_anchor_key(col_box)
        splits_in_col = [
            split
            for split in splits
            if calculate_intersection_area(split["box"], col_box)
            / (get_box_area(split["box"]) + 1e-12)
            > INTERSECTION_THRESH_SPLIT
        ]

        valid_splits = []
        for split in splits_in_col:
            if _is_valid_split_candidate(split, col):
                valid_splits.append(split)

        split_count = _infer_split_count(valid_splits, col_box) if valid_splits else 0
        boundaries = None
        if split_count >= 2:
            boundaries = _split_boundaries_from_centers(valid_splits, col_box, split_count)

        plans[parent_key] = {
            "parent_box": list(col_box),
            "split_count": split_count,
            "boundaries": boundaries,
        }

    return plans


def _split_index_from_boundaries(span_box, boundaries):
    if not span_box or not boundaries or len(boundaries) < 2:
        return None

    span_cx = (span_box[0] + span_box[2]) / 2
    for idx in range(len(boundaries) - 1):
        left = boundaries[idx]
        right = boundaries[idx + 1]
        if left <= span_cx <= right:
            return idx

    if span_cx < boundaries[0]:
        return 0
    if span_cx > boundaries[-1]:
        return len(boundaries) - 2
    return None


def _anchor_row_spans_to_detected_structure(span_preds, cols_raw, splits):
    anchors = {}
    if not span_preds or not cols_raw:
        return anchors

    split_plans = _build_parent_split_plans(cols_raw, splits)

    for span in span_preds:
        anchor_key = _prediction_anchor_key(span)
        if anchor_key is None:
            continue

        parent_col_idx = _preferred_col_idx_for_item(span["box"], cols_raw)
        if parent_col_idx is None:
            candidates = _column_support_candidates(span["box"], cols_raw, allow_soft=True)
            if candidates:
                parent_col_idx = candidates[0]["idx"]
        if parent_col_idx is None:
            continue

        parent_col = cols_raw[parent_col_idx]
        parent_box = parent_col["box"]
        parent_key = _box_anchor_key(parent_box)
        plan = split_plans.get(parent_key, {})
        anchor = {"parent_box": list(parent_box)}

        boundaries = plan.get("boundaries")
        split_count = int(plan.get("split_count", 0) or 0)
        if split_count >= 2 and boundaries and len(boundaries) == split_count + 1:
            split_index = _split_index_from_boundaries(span["box"], boundaries)
            if split_index is not None:
                anchor["split_index"] = int(split_index)
                anchor["split_count"] = split_count

        anchors[anchor_key] = anchor

    return anchors


def _resolve_preanchored_row_span_cols(span, final_cols, row_span_anchors):
    if not row_span_anchors or not span or not final_cols:
        return []

    anchor = row_span_anchors.get(_prediction_anchor_key(span))
    if not anchor:
        return []

    parent_key = _box_anchor_key(anchor.get("parent_box"))
    if parent_key is None:
        return []

    matched = []
    split_index = anchor.get("split_index")
    split_count = anchor.get("split_count")

    for idx, col in enumerate(final_cols):
        if split_index is not None:
            if not col.get("is_split"):
                continue
            if _box_anchor_key(col.get("parent_box", col["box"])) != parent_key:
                continue
            if col.get("split_index") != split_index:
                continue
            if split_count is not None and col.get("total_splits") != split_count:
                continue
            matched.append(idx)
            continue

        if col.get("is_split"):
            continue
        if _box_anchor_key(col["box"]) == parent_key:
            matched.append(idx)

    return matched


def _adjust_row_span_owner_by_text(span_box, owner_col_indices, final_cols, ocr_data):
    if (
        not span_box
        or not owner_col_indices
        or not final_cols
        or not ocr_data
    ):
        return owner_col_indices

    valid_owner_indices = [
        idx for idx in owner_col_indices
        if isinstance(idx, int) and 0 <= idx < len(final_cols)
    ]
    if not valid_owner_indices:
        return owner_col_indices

    split_owner_indices = [
        idx for idx in valid_owner_indices
        if final_cols[idx].get("is_split")
    ]
    if not split_owner_indices:
        return owner_col_indices

    owner_idx = split_owner_indices[0]
    owner_col = final_cols[owner_idx]
    parent_key = _box_anchor_key(owner_col.get("parent_box", owner_col["box"]))
    if parent_key is None:
        return owner_col_indices

    sibling_indices = [
        idx for idx, col in enumerate(final_cols)
        if col.get("is_split")
        and _box_anchor_key(col.get("parent_box", col["box"])) == parent_key
    ]
    sibling_indices = sorted(sibling_indices)
    if len(sibling_indices) < 2:
        return owner_col_indices

    span_w = max(1.0, float(span_box[2] - span_box[0]))
    covered = []
    for idx in sibling_indices:
        col_box = final_cols[idx]["box"]
        col_w = max(1.0, float(col_box[2] - col_box[0]))
        inter_w = max(0.0, min(span_box[2], col_box[2]) - max(span_box[0], col_box[0]))
        if (inter_w / col_w) >= 0.35 or (inter_w / span_w) >= 0.18:
            covered.append(idx)

    if len(covered) < 2:
        return owner_col_indices

    center_hits = {idx: 0 for idx in covered}
    for item in _ocr_items_in_box(span_box, ocr_data, intersection_thresh=0.05):
        item_box = o_to_box(item)
        inter_area = calculate_intersection_area(item_box, span_box)
        if inter_area <= 0.0:
            continue

        item_x1, _, item_x2, _ = item_box
        item_cx = (item_x1 + item_x2) / 2.0
        for idx in covered:
            col_box = final_cols[idx]["box"]
            col_x1, _, col_x2, _ = col_box
            if col_x1 <= item_cx <= col_x2:
                center_hits[idx] += 1
                break

    best_count = max(center_hits.values())
    if best_count <= 0:
        return owner_col_indices

    best_indices = [idx for idx, count in center_hits.items() if count == best_count]
    if len(best_indices) != 1:
        return owner_col_indices
    return [best_indices[0]]


def _cluster_items_by_x_center(items, gap_thresh):
    if not items:
        return []

    ordered = sorted(items, key=lambda item: item["x"])
    clusters = [[ordered[0]]]
    for item in ordered[1:]:
        if (item["x"] - clusters[-1][-1]["x"]) > gap_thresh:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def _mapping_row_lead_key(text):
    compact = "".join(ch for ch in str(text or "").strip() if not ch.isspace())
    if not compact:
        return ""

    compact = compact.replace("Â", "")
    compact = compact.replace("Ω", "Ω")
    compact = compact.replace("μ", "µ")
    return _label_key(compact) or compact.upper()


def _cluster_split_text(text):
    parts = []
    for piece in re.split(r"\s+", str(text or "").strip()):
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return " ".join(parts)


def _looks_like_body_split_cluster(text):
    text = _cluster_split_text(text)
    if not text:
        return False

    compact = _mapping_row_lead_key(text)
    tokens = text.split()
    if _is_numericish_text(text) or _is_simple_unit_text(text) or _looks_like_compact_grid_text(text):
        return True
    if len(tokens) > 2 or len(compact) > 14:
        return False

    alpha = _alpha_frac(text)
    digit = _digit_frac(text)
    if alpha >= 0.72 and digit < 0.12:
        return False
    if any(len(token) > 10 for token in tokens):
        return False
    return alpha <= 0.60 or digit >= 0.20


def _infer_two_way_boundary_from_ocr(col_box, final_rows, ocr_data, allow_descriptive_pair=False):
    if not col_box or not final_rows or not ocr_data:
        return None

    col_w = max(1.0, float(col_box[2] - col_box[0]))
    gap_thresh = max(24.0, 0.18 * col_w)
    min_cluster_gap = max(42.0, 0.22 * col_w)
    cluster_tol = max(18.0, 0.08 * col_w)
    pair_support = []

    for row in final_rows:
        if row["type"] != CLASS_NAME_ROW:
            continue

        row_box = row["box"]
        row_items = []
        for item in ocr_data:
            text_val = str(item.get("text", "")).strip()
            if not text_val:
                continue

            ibox = o_to_box(item)
            if not _item_matches_row_band(ibox, row_box, allow_assigned_edge_slack=True):
                continue

            item_w = max(1.0, float(ibox[2] - ibox[0]))
            inter_w = max(0.0, min(ibox[2], col_box[2]) - max(ibox[0], col_box[0]))
            cx = get_box_center(ibox)[0]
            if not (col_box[0] <= cx <= col_box[2]) and (inter_w / item_w) < 0.35:
                continue

            # Very wide phrases usually represent one logical spanning label
            # inside the column region, not evidence of a missing sub-column.
            if item_w > 0.72 * col_w:
                continue
            if len(text_val) > 42 and _alpha_frac(text_val) >= 0.2:
                continue

            row_items.append({
                "item": item,
                "x": cx,
                "box": ibox,
            })

        if len(row_items) < 2:
            continue

        clusters = _cluster_items_by_x_center(row_items, gap_thresh)
        if len(clusters) != 2:
            continue

        cluster_centers = []
        cluster_widths = []
        cluster_texts = []
        for cluster in clusters:
            xs = [entry["x"] for entry in cluster]
            x1 = min(entry["box"][0] for entry in cluster)
            x2 = max(entry["box"][2] for entry in cluster)
            cluster_centers.append(sum(xs) / len(xs))
            cluster_widths.append(max(1.0, float(x2 - x1)))
            cluster_texts.append(
                " ".join(str(entry["item"].get("text", "")).strip() for entry in cluster).strip()
            )

        if (cluster_centers[1] - cluster_centers[0]) < min_cluster_gap:
            continue
        if any(width > 0.60 * col_w for width in cluster_widths):
            continue
        if not all(_looks_like_body_split_cluster(text) for text in cluster_texts):
            continue

        pair_support.append({
            "left": cluster_centers[0],
            "right": cluster_centers[1],
            "type": row["type"],
            "texts": cluster_texts,
        })

    body_support = [entry for entry in pair_support if entry["type"] == CLASS_NAME_ROW]
    if not body_support:
        return None
    if len(body_support) < 3:
        return None
    if len(pair_support) < 2:
        return None

    left_centers = [entry["left"] for entry in pair_support]
    right_centers = [entry["right"] for entry in pair_support]
    if np.std(left_centers) > cluster_tol or np.std(right_centers) > cluster_tol:
        return None

    boundary = (float(np.median(left_centers)) + float(np.median(right_centers))) / 2.0
    min_seg_w = max(40.0, 0.18 * col_w)
    if boundary <= (col_box[0] + min_seg_w) or boundary >= (col_box[2] - min_seg_w):
        return None

    return boundary


def _promote_orphan_split_columns(cols_raw, splits, rows_raw, header_items, table_box):
    if not cols_raw or not splits:
        return cols_raw

    augmented = list(cols_raw)
    table_w = max(1.0, float(table_box[2] - table_box[0])) if table_box else 1.0
    table_h = max(1.0, float(table_box[3] - table_box[1])) if table_box else 1.0

    for split in splits:
        split_box = split["box"]
        split_area = get_box_area(split_box) + 1e-12
        split_w = max(1.0, float(split_box[2] - split_box[0]))
        split_h = max(1.0, float(split_box[3] - split_box[1]))

        if split_h < max(28.0, 0.28 * table_h):
            continue
        if split_w < max(12.0, 0.03 * table_w):
            continue

        overlap_with_cols = max(
            (calculate_intersection_area(split_box, col["box"]) / split_area)
            for col in augmented
        )
        if overlap_with_cols >= 0.55:
            continue

        row_hits = sum(
            1 for row in rows_raw or []
            if _y_overlap_ratio(split_box, row["box"]) >= 0.18
        )
        header_hits = sum(
            1 for item in header_items or []
            if _x_overlap_ratio(split_box, item["box"]) >= 0.18
        )
        if row_hits < 2 and header_hits == 0:
            continue

        cx = (split_box[0] + split_box[2]) / 2
        left = max(
            (col for col in augmented if ((col["box"][0] + col["box"][2]) / 2) < cx),
            key=lambda col: col["box"][2],
            default=None,
        )
        right = min(
            (col for col in augmented if ((col["box"][0] + col["box"][2]) / 2) > cx),
            key=lambda col: col["box"][0],
            default=None,
        )
        if left is not None and right is not None:
            if split_box[0] < (left["box"][2] - 0.05 * table_w) and split_box[2] > (right["box"][0] + 0.05 * table_w):
                continue

        augmented.append(
            {
                **split,
                "class": CLASS_NAME_COLUMN,
                "x": (split_box[0] + split_box[2]) / 2,
                "y": (split_box[1] + split_box[3]) / 2,
                "width": split_w,
                "height": split_h,
                "from_orphan_split": True,
            }
        )

    if len(augmented) == len(cols_raw):
        return cols_raw

    augmented = merge_cols_by_x(augmented, x_threshold=10)
    augmented = group_nms(augmented, IOS_THRESH_COLUMNS)
    return sorted(augmented, key=lambda col: col["box"][0])


def _first_col_label_cluster_centers(ocr_items, first_col_box):
    x1 = first_col_box[0] - 10
    x2 = first_col_box[2] + 10
    label_items = []
    for item in ocr_items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        box = o_to_box(item)
        cx, cy = get_box_center(box)
        if x1 <= cx <= x2 and (_alpha_frac(text) >= 0.25 or ("[" in text and "]" in text)):
            label_items.append({"y": cy, "text": text})

    label_items.sort(key=lambda item: item["y"])
    clusters = []
    for item in label_items:
        if clusters and abs(item["y"] - clusters[-1][-1]["y"]) < 14:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    centers = []
    for cluster in clusters:
        cluster_text = " ".join(item["text"] for item in cluster)
        if _alpha_frac(cluster_text) < 0.2 and not ("[" in cluster_text and "]" in cluster_text):
            continue
        centers.append(sum(item["y"] for item in cluster) / len(cluster))
    return centers


def _looks_like_first_col_anchor(text):
    compact = "".join(ch for ch in str(text or "").lower() if not ch.isspace())
    if not compact:
        return False
    if compact.startswith("0x"):
        return True
    if "[" in text and "]" in text:
        return True
    if _alpha_frac(text) >= 0.18:
        return True
    return len(compact) <= 8 and any(ch.isalpha() for ch in compact)


def _looks_like_numeric_first_col_anchor(text):
    compact = "".join(ch for ch in str(text or "") if not ch.isspace())
    return compact.isdigit() and 1 <= len(compact) <= 3


def _first_col_anchor_clusters(ocr_items, first_col_box, y1=None, y2=None):
    if not ocr_items or not first_col_box:
        return []

    x1 = first_col_box[0] - 12
    x2 = first_col_box[2] + 12
    label_items = []
    for item in ocr_items:
        text_val = str(item.get("text", "")).strip()
        if not text_val:
            continue
        if not (_looks_like_first_col_anchor(text_val) or _looks_like_numeric_first_col_anchor(text_val)):
            continue

        box = o_to_box(item)
        cx, cy = get_box_center(box)
        if not (x1 <= cx <= x2):
            item_w = max(1.0, float(box[2] - box[0]))
            inter_w = max(0.0, min(box[2], x2) - max(box[0], x1))
            if (inter_w / item_w) < 0.28:
                continue
        if y1 is not None and cy < y1:
            continue
        if y2 is not None and cy > y2:
            continue
        label_items.append({"y": cy, "text": text_val, "box": box, "item": item})

    has_bitfield_like_anchor = any(
        (("[" in item["text"] and "]" in item["text"]) or str(item["text"]).strip().lower().startswith("0x"))
        for item in label_items
    )

    label_items.sort(key=lambda item: item["y"])
    clusters = []
    for item in label_items:
        if clusters and abs(item["y"] - clusters[-1][-1]["y"]) < 16:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    anchors = []
    for cluster in clusters:
        cluster_text = " ".join(item["text"] for item in cluster).strip()
        if not _looks_like_first_col_anchor(cluster_text):
            if not (has_bitfield_like_anchor and _looks_like_numeric_first_col_anchor(cluster_text)):
                continue
        cluster_x1 = min(item["box"][0] for item in cluster)
        cluster_y1 = min(item["box"][1] for item in cluster)
        cluster_x2 = max(item["box"][2] for item in cluster)
        cluster_y2 = max(item["box"][3] for item in cluster)
        anchors.append({
            "y": sum(item["y"] for item in cluster) / len(cluster),
            "text": cluster_text,
            "top": cluster_y1,
            "bottom": cluster_y2,
            "box": [cluster_x1, cluster_y1, cluster_x2, cluster_y2],
            "items": [item["item"] for item in cluster],
        })
    return anchors

def _left_label_cluster_centers(ocr_items, col_boxes, max_cols=1):
    if not col_boxes:
        return []
    first_col_box = col_boxes[0]
    return [anchor["y"] for anchor in _first_col_anchor_clusters(ocr_items, first_col_box)]



def _build_anchor_guided_body_maps(final_rows, final_cols, ocr_data):
    if not final_rows or not final_cols or not ocr_data:
        return {}

    if any(col.get("is_split") for col in final_cols):
        return {}
    body_rows = [(idx, row) for idx, row in enumerate(final_rows) if row["type"] == CLASS_NAME_ROW]
    if len(body_rows) < 2:
        return {}

    body_types = (CLASS_NAME_ROW, CLASS_NAME_ROW_HEADER)
    header_bottom = max((row["box"][3] for row in final_rows if row["type"] not in body_types), default=None)
    body_bottom = max(row["box"][3] for _, row in body_rows)
    heights = [max(1.0, float(row["box"][3] - row["box"][1])) for _, row in body_rows]
    median_h = float(np.median(heights)) if heights else float(MEDIAN_HEIGHT_DEFAULT)
    tolerance = max(14.0, 0.25 * median_h)

    first_col_box = final_cols[0]["box"]
    y1 = body_rows[0][1]["box"][1] - tolerance
    if header_bottom is not None:
        y1 = max(y1, header_bottom - tolerance)
    anchors = _first_col_anchor_clusters(ocr_data, first_col_box, y1=max(0.0, y1), y2=body_bottom + 6)
    anchors = _first_col_anchor_clusters(ocr_data, first_col_box, y1=max(0.0, y1), y2=body_bottom + 6)
    if len(anchors) < max(2, len(body_rows) - 1):
        return {}
    if len(anchors) > len(body_rows) + max(3, len(body_rows) // 2):
        return {}

    assignments = []
    anchor_cursor = 0
    for pos, (r_idx, row) in enumerate(body_rows):
        row_bottom = row["box"][3]
        chosen = None
        chosen_idx = None
        fallback = None
        fallback_idx = None
        for a_idx in range(anchor_cursor, len(anchors)):
            anchor = anchors[a_idx]
            if anchor["top"] > row_bottom + tolerance:
                break
            if fallback is None:
                fallback = anchor
                fallback_idx = a_idx
            if anchor["bottom"] >= row["box"][1] - tolerance or anchor["y"] >= row["box"][1] - tolerance:
                chosen = anchor
                chosen_idx = a_idx
                break
        if chosen is None and fallback is not None:
            chosen = fallback
            chosen_idx = fallback_idx
        if chosen is None:
            continue
        assignments.append((r_idx, chosen))
        anchor_cursor = chosen_idx + 1

    if len(assignments) < max(2, len(body_rows) // 2):
        return {}

    row_maps = {}
    for pos, (r_idx, anchor) in enumerate(assignments):
        next_anchor = assignments[pos + 1][1] if pos + 1 < len(assignments) else None
        search_top = anchor["top"]
        if next_anchor is not None:
            search_bottom = max(search_top + 4, next_anchor["top"] - 1)
        else:
            search_bottom = max(final_rows[r_idx]["box"][3], anchor["bottom"] + (1.25 * median_h))

        row_map = {0: anchor["text"].strip()}
        for c_idx in range(1, len(final_cols)):
            col_box = final_cols[c_idx]["box"]
            search_box = [col_box[0], search_top, col_box[2], search_bottom]
            text = get_text_center_only_in_box(search_box, ocr_data)
            if not text:
                text = get_ocr_text_in_box(search_box, ocr_data, intersection_thresh=0.05)
            row_map[c_idx] = text.strip()
        row_maps[r_idx] = row_map

    return row_maps


def _label_alignment_bonus(ocr_items, row_boxes, col_boxes):
    # Disabled permanently: a small number of aligned first-column labels can
    # over-trust an otherwise weak row set. Keep the helper as a no-op so any
    # existing callers remain stable.
    return 0.0

def _row_header_alignment_bonus(ocr_items, row_boxes, row_types, col_boxes):
    if not row_boxes or not row_types or not col_boxes or not ocr_items:
        return 0.0

    first_col = col_boxes[0]
    total = 0
    hit = 0
    for box, rtype in zip(row_boxes, row_types):
        if rtype != CLASS_NAME_ROW_HEADER:
            continue
        total += 1
        target = [first_col[0], box[1], first_col[2], box[3]]
        found = False
        for item in ocr_items:
            text = str(item.get('text', '')).strip()
            if not text:
                continue
            if _alpha_frac(text) < 0.2 and '[' not in text and ']' not in text:
                continue
            ibox = o_to_box(item)
            cx = (ibox[0] + ibox[2]) / 2
            cy = (ibox[1] + ibox[3]) / 2
            if target[0] <= cx <= target[2] and target[1] <= cy <= target[3]:
                found = True
                break
        if found:
            hit += 1

    if total == 0:
        return 0.0
    return 1.5 * (hit / total)



def merge_rows_by_y(rows, y_threshold=5):
    if not rows:
        return []

    rows = sorted(rows, key=lambda row: row["y"])
    merged = []
    current = rows[0]

    for next_row in rows[1:]:
        y1_max = max(current["box"][1], next_row["box"][1])
        y2_min = min(current["box"][3], next_row["box"][3])
        overlap_h = max(0, y2_min - y1_max)
        min_h = min(current["height"], next_row["height"])
        overlap_ratio = overlap_h / (min_h + 1e-6)

        # Only merge true duplicate rows with heavy vertical overlap.
        # Do not merge merely because y-centers are within a few pixels.
        if overlap_ratio > 0.8:
            x1 = min(current["box"][0], next_row["box"][0])
            y1 = min(current["box"][1], next_row["box"][1])
            x2 = max(current["box"][2], next_row["box"][2])
            y2 = max(current["box"][3], next_row["box"][3])
            current = {
                "box": [x1, y1, x2, y2],
                "class": current["class"],
                "confidence": max(current["confidence"], next_row["confidence"]),
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "items": current.get("items", [current]) + next_row.get("items", [next_row])
            }
        else:
            merged.append(current)
            current = next_row

    merged.append(current)
    return merged


def merge_cols_by_x(cols, x_threshold=10):
    if not cols:
        return []

    cols = sorted(cols, key=lambda col: col["x"])
    merged = []
    current = cols[0]

    for next_col in cols[1:]:
        x_diff = abs(next_col["x"] - current["x"])
        x1_max = max(current["box"][0], next_col["box"][0])
        x2_min = min(current["box"][2], next_col["box"][2])
        overlap_w = max(0, x2_min - x1_max)
        min_w = min(current["width"], next_col["width"])
        overlap_ratio = overlap_w / (min_w + 1e-6)

        if overlap_ratio > 0.55 or (overlap_ratio > 0.30 and x_diff < x_threshold):
            x1 = min(current["box"][0], next_col["box"][0])
            y1 = min(current["box"][1], next_col["box"][1])
            x2 = max(current["box"][2], next_col["box"][2])
            y2 = max(current["box"][3], next_col["box"][3])
            current = {
                "box": [x1, y1, x2, y2],
                "class": current["class"],
                "confidence": max(current["confidence"], next_col["confidence"]),
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        else:
            merged.append(current)
            current = next_col

    merged.append(current)
    return merged


def _group_header_items(header_items):
    if not header_items:
        return []

    ordered = sorted(
        header_items,
        key=lambda item: (
            str(item.get("class", "")),
            ((item["box"][1] + item["box"][3]) / 2),
            item["box"][0],
        ),
    )
    groups = []
    for item in ordered:
        ibox = item["box"]
        icy = (ibox[1] + ibox[3]) / 2
        ih = max(1.0, float(ibox[3] - ibox[1]))
        matched_group = None
        for group in groups:
            if group.get("class") != item.get("class"):
                continue
            gbox = group["box"]
            gcy = group["y"]
            gh = max(1.0, float(gbox[3] - gbox[1]))
            inter_h = max(0.0, min(ibox[3], gbox[3]) - max(ibox[1], gbox[1]))
            overlap_ratio = inter_h / min(ih, gh)
            close_in_y = abs(icy - gcy) <= max(10.0, 0.40 * max(ih, gh))
            if overlap_ratio >= 0.35 or close_in_y:
                matched_group = group
                break

        if matched_group is None:
            groups.append(
                {
                    "items": [item],
                    "box": list(ibox),
                    "y": icy,
                    "class": item.get("class"),
                }
            )
            continue

        matched_group["items"].append(item)
        matched_group["box"] = [
            min(matched_group["box"][0], ibox[0]),
            min(matched_group["box"][1], ibox[1]),
            max(matched_group["box"][2], ibox[2]),
            max(matched_group["box"][3], ibox[3]),
        ]
        matched_group["y"] = sum(
            ((member["box"][1] + member["box"][3]) / 2) for member in matched_group["items"]
        ) / len(matched_group["items"])

    groups.sort(key=lambda group: group["y"])
    return groups


def _cluster_header_items_same_class(header_items):
    if not header_items:
        return []

    ordered = sorted(
        header_items,
        key=lambda item: (
            str(item.get("class", "")),
            ((item["box"][1] + item["box"][3]) / 2),
            item["box"][0],
        ),
    )
    groups = []
    for item in ordered:
        ibox = item["box"]
        icy = (ibox[1] + ibox[3]) / 2
        ih = max(1.0, float(ibox[3] - ibox[1]))
        matched_group = None
        for group in groups:
            if group.get("class") != item.get("class"):
                continue
            gbox = group["box"]
            gh = max(1.0, float(gbox[3] - gbox[1]))
            gcy = group["y"]
            inter_h = max(0.0, min(ibox[3], gbox[3]) - max(ibox[1], gbox[1]))
            overlap_ratio = inter_h / min(ih, gh)
            center_gap = abs(icy - gcy)
            if overlap_ratio >= 0.45 or center_gap <= max(8.0, 0.20 * min(ih, gh)):
                matched_group = group
                break

        if matched_group is None:
            groups.append(
                {
                    "items": [item],
                    "box": list(ibox),
                    "y": icy,
                    "class": item.get("class"),
                }
            )
            continue

        matched_group["items"].append(item)
        matched_group["box"] = [
            min(matched_group["box"][0], ibox[0]),
            min(matched_group["box"][1], ibox[1]),
            max(matched_group["box"][2], ibox[2]),
            max(matched_group["box"][3], ibox[3]),
        ]
        matched_group["y"] = sum(
            ((member["box"][1] + member["box"][3]) / 2) for member in matched_group["items"]
        ) / len(matched_group["items"])

    groups.sort(key=lambda group: group["y"])
    return groups


def _ocr_key(item):
    box = o_to_box(item)
    return (
        str(item.get("text", "")).strip(),
        round(float(box[0]), 1),
        round(float(box[1]), 1),
        round(float(box[2]), 1),
        round(float(box[3]), 1),
    )


def _dedupe_ocr_items_by_box(items):
    unique = []
    seen = set()
    for item in items or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        key = _ocr_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _augment_ocr_with_header_crop(ocr_data, structure_image_path, header_items, min_col_x, max_col_x):
    """Run one tight header OCR crop so missed top header text can join the same y-band."""
    if not structure_image_path or not header_items:
        return ocr_data

    try:
        image = cv2.imread(structure_image_path)
        if image is None:
            return ocr_data
        img_h, img_w = image.shape[:2]
        y1 = max(0, int(round(min(item["box"][1] for item in header_items) - 10)))
        y2 = min(img_h, int(round(max(item["box"][3] for item in header_items) + 10)))
        x1 = max(0, int(round(min_col_x - 6)))
        x2 = min(img_w, int(round(max_col_x + 6)))
        if x2 <= x1 or y2 <= y1:
            return ocr_data

        from ocr_reader import read_text

        extra_items = read_text(structure_image_path, crop_box=[x1, y1, x2, y2])
    except Exception:
        return ocr_data

    if not extra_items:
        return ocr_data

    merged = list(ocr_data or [])
    for extra in extra_items:
        text = str(extra.get("text", "")).strip()
        if not text:
            continue
        ebox = o_to_box(extra)
        ecx, ecy = get_box_center(ebox)
        duplicate = False
        for existing in merged:
            if str(existing.get("text", "")).strip() != text:
                continue
            xbox = o_to_box(existing)
            xcx, xcy = get_box_center(xbox)
            if abs(ecx - xcx) <= 24.0 and abs(ecy - xcy) <= 14.0:
                duplicate = True
                break
            inter = calculate_intersection_area(ebox, xbox)
            if inter / (min(get_box_area(ebox), get_box_area(xbox)) + 1e-12) >= 0.65:
                duplicate = True
                break
        if not duplicate:
            merged.append(extra)

    return merged


def _build_y_band_header_row_definitions(col_headers, span_col_headers, ocr_data, min_col_x, max_col_x, rows_raw=None):
    header_items = list(col_headers or []) + list(span_col_headers or [])
    if not header_items:
        return []

    separator_rows = _build_separator_expanded_header_row_definitions(
        header_items,
        min_col_x,
        max_col_x,
    )
    if separator_rows:
        return separator_rows

    # Detection-driven header rows only. The older same-y-band OCR rule merged
    # column_header and spanning_column_header items into one visual row when
    # their OCR text shared a y-band; that caused unrelated header levels to
    # collapse together. Keep header rows separated by detected item class/box.
    fallback_groups = _cluster_header_items_same_class(header_items)
    return [
        {
            "header_class": HEADER_ROW_TYPE,
            "items": sorted(group["items"], key=lambda member: member["box"][0]),
            "box": [min_col_x, group["box"][1], max_col_x, group["box"][3]],
            "y": group["y"],
        }
        for group in fallback_groups
    ]


def _cluster_header_y_boundaries(values, tolerance=4.0):
    if not values:
        return []
    clusters = []
    for value in sorted(float(v) for v in values):
        if not clusters or abs(value - clusters[-1][-1]) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _header_band_index_for_box(item_box, bands):
    if not item_box or not bands:
        return None

    item_h = max(1.0, float(item_box[3] - item_box[1]))
    overlaps = []
    material_hits = []
    for idx, (band_y1, band_y2) in enumerate(bands):
        inter_h = max(0.0, min(float(item_box[3]), band_y2) - max(float(item_box[1]), band_y1))
        if inter_h <= 0.0:
            continue
        band_h = max(1.0, band_y2 - band_y1)
        score = inter_h / min(item_h, band_h)
        overlaps.append((inter_h, score, idx))
        if score >= 0.20:
            material_hits.append(idx)

    if not overlaps:
        return None
    if len(material_hits) >= 2:
        bottom_y = float(item_box[3])
        for idx, (band_y1, band_y2) in enumerate(bands):
            if band_y1 - 2.0 <= bottom_y <= band_y2 + 2.0:
                return idx
    overlaps.sort(reverse=True)
    return overlaps[0][2]


def _build_separator_expanded_header_row_definitions(header_items, min_col_x, max_col_x):
    expanded_items = [
        item
        for item in header_items
        if item.get("separator_expanded_header_cell")
    ]
    if not expanded_items:
        return []

    y_values = []
    for item in header_items:
        box = item.get("box")
        if not box:
            continue
        if item.get("separator_expanded_header_cell") or item.get("source") == "missing_header_slot_geometry":
            y_values.extend([box[1], box[3]])
    boundaries = _cluster_header_y_boundaries(y_values)
    if len(boundaries) < 3:
        return []

    bands = [
        (boundaries[idx], boundaries[idx + 1])
        for idx in range(len(boundaries) - 1)
        if boundaries[idx + 1] - boundaries[idx] >= 6.0
    ]
    if len(bands) < 2:
        return []

    groups = [
        {"items": [], "box": [min_col_x, band_y1, max_col_x, band_y2]}
        for band_y1, band_y2 in bands
    ]
    for item in header_items:
        band_idx = _header_band_index_for_box(item.get("box"), bands)
        if band_idx is None:
            continue
        groups[band_idx]["items"].append(item)

    rows = []
    for group in groups:
        if not group["items"]:
            continue
        rows.append(
            {
                "header_class": HEADER_ROW_TYPE,
                "items": sorted(group["items"], key=lambda member: member["box"][0]),
                "box": group["box"],
                "y": (group["box"][1] + group["box"][3]) / 2.0,
            }
        )
    return rows


def _header_segment_row_span_count(segment_box, header_groups):
    if not segment_box or not header_groups:
        return 0

    count = 0
    for group in header_groups:
        row_box = group["box"]
        inter_h = max(0.0, min(segment_box[3], row_box[3]) - max(segment_box[1], row_box[1]))
        row_h = max(1.0, float(row_box[3] - row_box[1]))
        seg_h = max(1.0, float(segment_box[3] - segment_box[1]))
        if inter_h / row_h >= 0.20 or inter_h / seg_h >= 0.20:
            count += 1
    return count


def _header_item_covers_slot_segment(item, segment_box):
    if not item or not segment_box:
        return False

    item_box = item.get("box")
    if not item_box:
        return False

    inter_w = max(0.0, min(item_box[2], segment_box[2]) - max(item_box[0], segment_box[0]))
    inter_h = max(0.0, min(item_box[3], segment_box[3]) - max(item_box[1], segment_box[1]))
    if inter_w <= 0.0 or inter_h <= 0.0:
        return False

    item_w = max(1.0, float(item_box[2] - item_box[0]))
    item_h = max(1.0, float(item_box[3] - item_box[1]))
    seg_w = max(1.0, float(segment_box[2] - segment_box[0]))
    seg_h = max(1.0, float(segment_box[3] - segment_box[1]))
    return (
        inter_w / min(item_w, seg_w) >= 0.35
        and inter_h / min(item_h, seg_h) >= 0.35
    )


def _header_segment_context_class(segment_box, recovered_box, header_groups, final_cols):
    if not segment_box or not recovered_box or not header_groups:
        return CLASS_NAME_COLUMN_HEADER

    best_group = None
    best_overlap = 0.0
    seg_h = max(1.0, float(segment_box[3] - segment_box[1]))
    for group in header_groups:
        group_box = group.get("box")
        if not group_box:
            continue
        inter_h = max(0.0, min(float(segment_box[3]), float(group_box[3])) - max(float(segment_box[1]), float(group_box[1])))
        score = inter_h / seg_h
        if score > best_overlap:
            best_overlap = score
            best_group = group

    if best_group is None or best_overlap < 0.20:
        return CLASS_NAME_COLUMN_HEADER

    same_band_items = best_group.get("items", [])
    has_wide_spanning_context = any(
        item.get("class") == CLASS_NAME_SPANNING_COLUMN_HEADER
        for item in same_band_items
    )
    if has_wide_spanning_context:
        return CLASS_NAME_SPANNING_COLUMN_HEADER

    return CLASS_NAME_COLUMN_HEADER


def _header_slot_ocr_items(slot_box, ocr_data):
    if not slot_box or not ocr_data:
        return []

    items = []
    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        cx, cy = get_box_center(ibox)
        if slot_box[0] <= cx <= slot_box[2] and slot_box[1] <= cy <= slot_box[3]:
            items.append(item)
            continue
        inter = calculate_intersection_area(slot_box, ibox)
        if inter / (get_box_area(ibox) + 1e-12) >= 0.12:
            items.append(item)
    return _sort_ocr_items_reading_order(items)


def _header_separator_y_crosses_ocr_text(y, slot_box, ocr_data):
    for item in ocr_data or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        cx, _cy = get_box_center(ibox)
        if not (slot_box[0] <= cx <= slot_box[2]):
            continue
        if float(ibox[1]) + 2.0 <= float(y) <= float(ibox[3]) - 2.0:
            return True
    return False


def _header_horizontal_separator_has_continuous_run(structure_gray, slot_box, y, min_run_frac=0.72):
    return _horizontal_separator_has_continuous_run(structure_gray, slot_box, y, min_run_frac=min_run_frac)


def _header_slot_horizontal_separator_ys(structure_gray, slot_box, ocr_data=None):
    if structure_gray is None or not slot_box:
        return []

    lines = _gap_separator_lines(structure_gray, slot_box, "horizontal")
    y1, y2 = float(slot_box[1]), float(slot_box[3])
    margin = max(5.0, 0.06 * max(1.0, y2 - y1))
    return sorted(
        float(y)
        for y, _ratio in lines
        if (y1 + margin) < float(y) < (y2 - margin)
        and not _header_separator_y_crosses_ocr_text(float(y), slot_box, ocr_data)
        and _header_horizontal_separator_has_continuous_run(structure_gray, slot_box, float(y))
    )


def _strong_header_separator_xs(structure_gray, header_band_box, min_ratio=0.80):
    if structure_gray is None or not header_band_box:
        return []
    return sorted(
        float(x)
        for x, ratio in _gap_separator_lines(structure_gray, header_band_box, "vertical")
        if ratio >= min_ratio
    )


def _expand_header_segment_to_separator_cell(segment_box, final_cols, structure_gray, min_col_x, max_col_x):
    if not segment_box or not final_cols or structure_gray is None:
        return list(segment_box), False

    band_box = [min_col_x, segment_box[1], max_col_x, segment_box[3]]
    separator_xs = _strong_header_separator_xs(structure_gray, band_box)
    boundaries = sorted(
        {
            float(min_col_x),
            float(max_col_x),
            *[
                x
                for x in separator_xs
                if float(min_col_x) - 2.0 <= x <= float(max_col_x) + 2.0
            ],
        }
    )
    if len(boundaries) < 2:
        return list(segment_box), False

    center_x = (float(segment_box[0]) + float(segment_box[2])) / 2.0
    expanded = None
    for left, right in zip(boundaries, boundaries[1:]):
        if left - 2.0 <= center_x <= right + 2.0 and right - left >= 5.0:
            expanded = [left, segment_box[1], right, segment_box[3]]
            break
    if expanded is None:
        return list(segment_box), False

    original_w = max(1.0, float(segment_box[2] - segment_box[0]))
    expanded_w = max(1.0, float(expanded[2] - expanded[0]))
    col_hits = _header_col_indices_for_box(expanded, final_cols, min_overlap=0.45)
    changed = expanded_w >= original_w * 1.35 and len(col_hits) >= 2
    return expanded, changed


def _recover_missing_header_slots_by_geometry(col_headers, span_col_headers, final_cols, ocr_data, structure_gray):
    if not final_cols or not ocr_data:
        return []

    header_items = list(col_headers or []) + list(span_col_headers or [])
    if not header_items:
        return []

    header_groups = _cluster_header_items_same_class(header_items)
    if not header_groups:
        return []

    stack_y1 = min(float(group["box"][1]) for group in header_groups)
    stack_y2 = max(float(group["box"][3]) for group in header_groups)
    if stack_y2 <= stack_y1:
        return []

    recovered = []
    for col_idx, col in enumerate(final_cols):
        col_box = col["box"]
        slot_box = [float(col_box[0]), stack_y1, float(col_box[2]), stack_y2]
        slot_items = _header_slot_ocr_items(slot_box, ocr_data)
        if not slot_items:
            continue

        separator_ys = _header_slot_horizontal_separator_ys(structure_gray, slot_box, ocr_data)
        boundaries = [stack_y1] + separator_ys + [stack_y2]
        for seg_y1, seg_y2 in zip(boundaries, boundaries[1:]):
            if seg_y2 - seg_y1 < 6.0:
                continue
            segment_box = [slot_box[0], seg_y1, slot_box[2], seg_y2]
            if any(_header_item_covers_slot_segment(item, segment_box) for item in header_items):
                continue
            segment_items = _header_slot_ocr_items(segment_box, ocr_data)
            if not segment_items:
                continue
            if _header_segment_row_span_count(segment_box, header_groups) < 1:
                continue
            recovered_box, separator_expanded = _expand_header_segment_to_separator_cell(
                segment_box,
                final_cols,
                structure_gray,
                final_cols[0]["box"][0],
                final_cols[-1]["box"][2],
            )
            recovered_class = _header_segment_context_class(
                segment_box,
                recovered_box,
                header_groups,
                final_cols,
            )
            recovered.append(
                {
                    "class": recovered_class,
                    "box": recovered_box,
                    "confidence": 0.94,
                    "source_model": "synthetic_recovery",
                    "source": "missing_header_slot_geometry",
                    "x": (recovered_box[0] + recovered_box[2]) / 2.0,
                    "y": (recovered_box[1] + recovered_box[3]) / 2.0,
                    "width": max(1.0, recovered_box[2] - recovered_box[0]),
                    "height": max(1.0, recovered_box[3] - recovered_box[1]),
                    "recovered_header_slot_col": col_idx,
                    "separator_expanded_header_cell": bool(separator_expanded),
                }
            )

    if not recovered:
        return []
    return sorted(group_nms(recovered, IOS_THRESH_NMS_DEFAULT), key=lambda item: (item["box"][1], item["box"][0]))


def _header_col_indices_for_box(box, final_cols, min_overlap=0.18):
    if not box or not final_cols:
        return []

    indices = []
    for idx, col in enumerate(final_cols):
        cbox = col["box"]
        inter_w = max(0.0, min(float(box[2]), float(cbox[2])) - max(float(box[0]), float(cbox[0])))
        if inter_w <= 0.0:
            continue
        col_w = max(1.0, float(cbox[2] - cbox[0]))
        box_w = max(1.0, float(box[2] - box[0]))
        if inter_w / col_w >= min_overlap or inter_w / box_w >= 0.10:
            indices.append(idx)
            continue
        cx = (float(cbox[0]) + float(cbox[2])) / 2.0
        if float(box[0]) <= cx <= float(box[2]):
            indices.append(idx)
    return indices


def _header_child_columns_in_lower_band(col_headers, span_box, col_indices, min_lower_gap=4.0):
    if not col_headers or not span_box or not col_indices:
        return set()

    span_mid_y = (float(span_box[1]) + float(span_box[3])) / 2.0
    hits = set()
    for item in col_headers:
        ibox = item["box"]
        if float(ibox[1]) <= span_mid_y + min_lower_gap:
            continue
        for col_idx in col_indices:
            # The caller passes final-column indices, so use the item x-range
            # against each candidate column via the index-specific final col.
            hits.add(col_idx)
    return hits


def _header_item_owns_col_in_band(item, col_box, band_box):
    if not item or not col_box or not band_box:
        return False

    ibox = item["box"]
    inter_w = max(0.0, min(float(ibox[2]), float(col_box[2])) - max(float(ibox[0]), float(col_box[0])))
    inter_h = max(0.0, min(float(ibox[3]), float(band_box[3])) - max(float(ibox[1]), float(band_box[1])))
    if inter_w <= 0.0 or inter_h <= 0.0:
        return False
    col_w = max(1.0, float(col_box[2] - col_box[0]))
    band_h = max(1.0, float(band_box[3] - band_box[1]))
    return inter_w / col_w >= 0.40 and inter_h / band_h >= 0.35


def _header_boundary_has_vertical_separator(structure_gray, band_box, left_col_box, right_col_box):
    if structure_gray is None or not band_box or not left_col_box or not right_col_box:
        return False

    x = (float(left_col_box[2]) + float(right_col_box[0])) / 2.0
    probe = [
        x - 2.5,
        float(band_box[1]) + 1.0,
        x + 2.5,
        float(band_box[3]) - 1.0,
    ]
    if probe[3] <= probe[1]:
        return False
    return bool(_gap_separator_lines(structure_gray, probe, "vertical"))


def _recover_missing_spanning_header_slots_by_geometry(col_headers, span_col_headers, final_cols, ocr_data, structure_gray):
    if not col_headers or not span_col_headers or not final_cols or not ocr_data:
        return [], list(span_col_headers or [])

    header_items = list(col_headers or []) + list(span_col_headers or [])
    header_groups = _cluster_header_items_same_class(header_items)
    if len(header_groups) < 2:
        return [], list(span_col_headers or [])

    stack_y1 = min(float(group["box"][1]) for group in header_groups)
    stack_y2 = max(float(group["box"][3]) for group in header_groups)
    if stack_y2 <= stack_y1:
        return [], list(span_col_headers or [])

    top_group = min(header_groups, key=lambda group: group["box"][1])
    top_band = [min(col["box"][0] for col in final_cols), top_group["box"][1], max(col["box"][2] for col in final_cols), top_group["box"][3]]
    recovered = []
    updated_spans = []

    # First extend real spanning headers when their top-band cell continues
    # into adjacent lower child headers and no top-level separator blocks it.
    for span in span_col_headers:
        span_box = list(map(float, span["box"]))
        covered = _header_col_indices_for_box(span_box, final_cols, min_overlap=0.35)
        if not covered:
            updated_spans.append(span)
            continue

        left_idx, right_idx = min(covered), max(covered)
        band_box = [span_box[0], span_box[1], span_box[2], span_box[3]]

        while right_idx + 1 < len(final_cols):
            next_col = final_cols[right_idx + 1]
            # A tall header owning the top band means the next column is a
            # separate header cell, not part of this spanning header.
            if any(
                item is not span and _header_item_owns_col_in_band(item, next_col["box"], band_box)
                for item in col_headers
            ):
                break
            if _header_boundary_has_vertical_separator(structure_gray, band_box, final_cols[right_idx]["box"], next_col["box"]):
                break
            lower_children = [
                child for child in col_headers
                if float(child["box"][1]) > float(band_box[3]) - 1.0
                and _x_overlap_ratio(child["box"], next_col["box"]) >= 0.35
            ]
            if not lower_children:
                break
            right_idx += 1

        new_box = [
            min(span_box[0], float(final_cols[left_idx]["box"][0])),
            span_box[1],
            max(span_box[2], float(final_cols[right_idx]["box"][2])),
            span_box[3],
        ]
        if new_box != span_box:
            clone = dict(span)
            clone["box"] = new_box
            clone["x"] = (new_box[0] + new_box[2]) / 2.0
            clone["y"] = (new_box[1] + new_box[3]) / 2.0
            clone["width"] = max(1.0, new_box[2] - new_box[0])
            clone["height"] = max(1.0, new_box[3] - new_box[1])
            clone["source"] = "spanning_header_geometry_extend"
            updated_spans.append(clone)
        else:
            updated_spans.append(span)

    # Then synthesize missing top-level spans from OCR text that lies in an
    # empty top header band above multiple lower child header columns.
    for item in _sort_ocr_items_reading_order(ocr_data):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        icy = (float(ibox[1]) + float(ibox[3])) / 2.0
        if not (float(top_band[1]) - 3.0 <= icy <= float(top_band[3]) + 8.0):
            continue
        cols_hit = _header_col_indices_for_box(ibox, final_cols, min_overlap=0.10)
        if len(cols_hit) < 2:
            continue
        first_idx, last_idx = min(cols_hit), max(cols_hit)
        span_box = [
            float(final_cols[first_idx]["box"][0]),
            float(top_band[1]),
            float(final_cols[last_idx]["box"][2]),
            float(top_band[3]),
        ]
        if any(_header_item_covers_slot_segment(existing, span_box) for existing in updated_spans):
            continue
        if any(_header_item_covers_slot_segment(existing, span_box) for existing in col_headers):
            continue

        child_hits = set()
        for child in col_headers:
            child_box = child["box"]
            if float(child_box[1]) <= float(span_box[3]) - 1.0:
                continue
            for col_idx in range(first_idx, last_idx + 1):
                if _x_overlap_ratio(child_box, final_cols[col_idx]["box"]) >= 0.35:
                    child_hits.add(col_idx)
        if len(child_hits) < 2:
            continue

        recovered.append(
            {
                "class": CLASS_NAME_SPANNING_COLUMN_HEADER,
                "box": span_box,
                "confidence": 0.94,
                "source_model": "synthetic_recovery",
                "source": "missing_spanning_header_geometry",
                "x": (span_box[0] + span_box[2]) / 2.0,
                "y": (span_box[1] + span_box[3]) / 2.0,
                "width": max(1.0, span_box[2] - span_box[0]),
                "height": max(1.0, span_box[3] - span_box[1]),
            }
        )

    combined_spans = sorted(group_nms(updated_spans + recovered, IOS_THRESH_NMS_DEFAULT), key=lambda item: (item["box"][1], item["box"][0]))
    return recovered, combined_spans


def _header_item_conflicts_with_expanded_span(candidate, span_item, expanded_box):
    if not candidate or candidate is span_item or not expanded_box:
        return False
    candidate_box = candidate.get("box")
    span_box = span_item.get("box") if span_item else None
    if not candidate_box or not span_box:
        return False

    inter_expanded = calculate_intersection_area(candidate_box, expanded_box)
    if inter_expanded <= 0.0:
        return False
    inter_original = calculate_intersection_area(candidate_box, span_box)
    candidate_area = get_box_area(candidate_box) + 1e-12
    return (inter_expanded / candidate_area) >= 0.20 and (inter_original / candidate_area) < 0.20


def _ocr_conflicts_with_expanded_header_span(ocr_data, span_box, expanded_box):
    if not ocr_data or not span_box or not expanded_box:
        return False

    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        cx, cy = get_box_center(ibox)
        if not (expanded_box[0] <= cx <= expanded_box[2] and expanded_box[1] <= cy <= expanded_box[3]):
            continue
        if span_box[0] <= cx <= span_box[2] and span_box[1] <= cy <= span_box[3]:
            continue
        # A real neighboring header has its own text inside the expansion
        # region. If found, do not stretch this spanning header into it.
        return True
    return False


def _expand_spanning_headers_to_separator_boundaries(span_col_headers, col_headers, final_cols, ocr_data, structure_gray):
    if not span_col_headers or not final_cols or structure_gray is None:
        return span_col_headers

    min_col_x = min(float(col["box"][0]) for col in final_cols)
    max_col_x = max(float(col["box"][2]) for col in final_cols)
    header_items = list(col_headers or []) + list(span_col_headers or [])
    expanded_spans = []

    for span in span_col_headers:
        span_box = list(map(float, span["box"]))
        band_box = [min_col_x, span_box[1], max_col_x, span_box[3]]
        separator_xs = _strong_header_separator_xs(structure_gray, band_box)
        boundaries = sorted(
            {
                min_col_x,
                max_col_x,
                *[
                    float(x)
                    for x in separator_xs
                    if min_col_x - 2.0 <= float(x) <= max_col_x + 2.0
                ],
            }
        )
        if len(boundaries) < 2:
            expanded_spans.append(span)
            continue

        center_x = (span_box[0] + span_box[2]) / 2.0
        target_box = None
        for left, right in zip(boundaries, boundaries[1:]):
            if right - left < 8.0:
                continue
            if left - 2.0 <= center_x <= right + 2.0:
                target_box = [left, span_box[1], right, span_box[3]]
                break
        if target_box is None:
            expanded_spans.append(span)
            continue

        target_cols = _header_col_indices_for_box(target_box, final_cols, min_overlap=0.45)
        if len(target_cols) < 2:
            expanded_spans.append(span)
            continue

        span_w = max(1.0, span_box[2] - span_box[0])
        target_w = max(1.0, target_box[2] - target_box[0])
        if target_w < span_w * 1.08:
            expanded_spans.append(span)
            continue

        if any(
            _header_item_conflicts_with_expanded_span(candidate, span, target_box)
            for candidate in header_items
        ):
            expanded_spans.append(span)
            continue
        if _ocr_conflicts_with_expanded_header_span(ocr_data, span_box, target_box):
            expanded_spans.append(span)
            continue

        clone = dict(span)
        clone["box"] = target_box
        clone["x"] = (target_box[0] + target_box[2]) / 2.0
        clone["y"] = (target_box[1] + target_box[3]) / 2.0
        clone["width"] = max(1.0, target_box[2] - target_box[0])
        clone["height"] = max(1.0, target_box[3] - target_box[1])
        clone["source"] = "spanning_header_separator_boundary_expand"
        expanded_spans.append(clone)

    return sorted(group_nms(expanded_spans, IOS_THRESH_NMS_DEFAULT), key=lambda item: (item["box"][1], item["box"][0]))


def _recover_detected_column_headers(raw_header_preds, cols_raw, rows_raw):
    if not raw_header_preds or not cols_raw:
        return []

    first_body_top = min((row["box"][1] for row in rows_raw), default=None)
    recovered = []
    for pred in sorted(raw_header_preds, key=lambda item: item.get("confidence", 0.0), reverse=True):
        box = pred["box"]
        box_h = max(1.0, float(box[3] - box[1]))
        box_cy = (box[1] + box[3]) / 2

        # Stay detection-driven: only recover header boxes that clearly sit
        # above the first detected body row band.
        if first_body_top is not None:
            if box_cy >= (first_body_top + max(4.0, 0.12 * box_h)):
                continue

        overlap_hits = [
            col for col in cols_raw
            if _x_overlap_ratio(box, col["box"]) >= 0.18
            or calculate_intersection_area(box, col["box"]) / (get_box_area(box) + 1e-12) >= 0.18
        ]
        if not overlap_hits:
            continue

        recovered.append(pred)

    if not recovered:
        return []

    return group_nms(recovered, IOS_THRESH_NMS_DEFAULT)


def _merge_header_row_groups(header_groups):
    if not header_groups:
        return []

    ordered = sorted(header_groups, key=lambda group: group["y"])
    merged = [ordered[0]]
    for group in ordered[1:]:
        current = merged[-1]
        cbox = current["box"]
        gbox = group["box"]
        ch = max(1.0, float(cbox[3] - cbox[1]))
        gh = max(1.0, float(gbox[3] - gbox[1]))
        height_ratio = max(ch, gh) / (min(ch, gh) + 1e-12)
        inter_h = max(0.0, min(cbox[3], gbox[3]) - max(cbox[1], gbox[1]))
        overlap_ratio = inter_h / (min(ch, gh) + 1e-12)
        center_gap = abs(current["y"] - group["y"])
        if height_ratio <= 1.6 and (
            overlap_ratio >= 0.55 or center_gap <= max(4.0, 0.12 * min(ch, gh))
        ):
            current["items"].extend(group["items"])
            current["items"].sort(key=lambda item: item["x"])
            current["box"] = [
                min(current["box"][0], group["box"][0]),
                min(current["box"][1], group["box"][1]),
                max(current["box"][2], group["box"][2]),
                max(current["box"][3], group["box"][3]),
            ]
            current["y"] = sum(
                ((item["box"][1] + item["box"][3]) / 2) for item in current["items"]
            ) / len(current["items"])
            continue
        merged.append(group)

    return merged


def _sort_header_row_groups(header_groups, same_band_tol=10.0):
    if not header_groups:
        return []

    ordered = sorted(header_groups, key=lambda group: group["y"])
    refined = []
    idx = 0
    while idx < len(ordered):
        band = [ordered[idx]]
        base_y = ordered[idx]["y"]
        idx += 1
        while idx < len(ordered) and abs(ordered[idx]["y"] - base_y) <= same_band_tol:
            band.append(ordered[idx])
            idx += 1
        band.sort(key=lambda group: (len(group.get("items", [])), group["y"]))
        refined.extend(band)
    return refined


def _build_span_cell_overrides(
    final_rows,
    final_cols,
    span_preds,
    ocr_data,
    span_axis,
    row_span_anchors=None,
    structure_gray=None,
):
    overrides = {}
    if not final_rows or not final_cols or not span_preds or not ocr_data:
        return overrides

    body_row_indices = [
        idx for idx, row in enumerate(final_rows)
        if row["type"] == CLASS_NAME_ROW
    ]
    if not body_row_indices:
        return overrides

    def _logical_col_key(c_idx):
        if c_idx is None or c_idx < 0 or c_idx >= len(final_cols):
            return None
        col = final_cols[c_idx]
        if col.get("is_split"):
            return ("split_parent", _box_anchor_key(col.get("parent_box", col["box"])))
        return ("column", _box_anchor_key(col["box"]))

    for span in span_preds:
        span_box = span["box"]
        covered_cells = []
        span_cx = (span_box[0] + span_box[2]) / 2
        span_w = max(1.0, float(span_box[2] - span_box[0]))
        span_h = max(1.0, float(span_box[3] - span_box[1]))

        owner_col_indices = None
        anchored_owner = False
        if span_axis == "row":
            owner_col_indices = _resolve_preanchored_row_span_cols(span, final_cols, row_span_anchors)
            anchored_owner = bool(owner_col_indices)
            if not owner_col_indices:
                preferred_col_idx = _preferred_col_idx_for_item(span_box, final_cols)
                if preferred_col_idx is not None:
                    owner_col_indices = [preferred_col_idx]
            owner_col_indices = _adjust_row_span_owner_by_text(
                span_box,
                owner_col_indices,
                final_cols,
                ocr_data,
            )
            if not owner_col_indices:
                continue
            if len(owner_col_indices) != 1:
                continue

        for r_idx in body_row_indices:
            row_box = final_rows[r_idx]["box"]
            row_h = max(1.0, float(row_box[3] - row_box[1]))
            row_inter_h = _row_band_intersection_height(span_box, row_box)
            row_cover = row_inter_h / row_h
            span_row_cover = row_inter_h / span_h
            row_center_inside = row_box[1] <= span["y"] <= row_box[3]
            if row_cover <= 0.0 and not row_center_inside:
                continue

            for c_idx, col in enumerate(final_cols):
                if owner_col_indices is not None and c_idx not in owner_col_indices:
                    continue
                col_box = col["box"]
                col_w = max(1.0, float(col_box[2] - col_box[0]))
                inter_w = max(0.0, min(span_box[2], col_box[2]) - max(span_box[0], col_box[0]))
                col_cover = inter_w / col_w
                col_center_inside = col_box[0] <= span_cx <= col_box[2]
                if col_cover <= 0.0 and not col_center_inside:
                    continue

                cell_box = [col_box[0], row_box[1], col_box[2], row_box[3]]
                # Span detections are often tighter than the full logical cell.
                # Use overlap against the smaller region so tight Roboflow boxes
                # inside split sub-columns still satisfy the 80% copy rule.
                cell_cover = _smaller_region_overlap_ratio(span_box, cell_box)

                if span_axis == "row":
                    # Row-spanning cells span vertically across multiple rows
                    # but usually belong to a single logical column. Roboflow
                    # boxes are often text-tight, especially in split columns,
                    # so anchor them to the supported owner column and require
                    # strong row support rather than full-width cell coverage.
                    span_item_cover = inter_w / span_w
                    if anchored_owner:
                        keep = row_cover >= 0.50 or row_center_inside
                    else:
                        keep = (
                            (row_cover >= 0.50 or row_center_inside)
                            and col_center_inside
                        )
                else:
                    keep = (
                        col_cover >= 0.50
                        and cell_cover >= INTERSECTION_THRESH_SPAN_CELL
                        and (row_cover >= 0.50 or row_center_inside)
                    )

                if keep:
                    covered_cells.append((r_idx, c_idx, cell_box))

        if not covered_cells:
            continue

        if span_axis == "col":
            logical_col_keys = {
                key for key in (_logical_col_key(c_idx) for _, c_idx, _ in covered_cells)
                if key is not None
            }
            if len(logical_col_keys) < 2:
                # A spanning-column-cell must cross logical columns. If it only
                # sits inside one detected column, including one split-parent
                # column, let normal OCR center assignment decide the cell.
                continue

        covered_cell_groups = [covered_cells]
        trimmed_span_text_box = None
        if span_axis == "row" and owner_col_indices and structure_gray is not None:
            covered_cell_groups = _split_row_span_cells_at_owner_separators(
                covered_cells,
                final_rows,
                final_cols[owner_col_indices[0]]["box"],
                structure_gray,
            )
            if not covered_cell_groups:
                continue
        elif span_axis == "col" and structure_gray is not None:
            covered_cell_groups = _split_col_span_cells_at_owner_separators(
                covered_cells,
                final_rows,
                final_cols,
                structure_gray,
            )
            if not covered_cell_groups:
                continue

        for span_cells in covered_cell_groups:
            if not span_cells:
                continue

            span_text_box = trimmed_span_text_box if trimmed_span_text_box is not None else span_box
            if span_axis in {"row", "col"} and len(covered_cell_groups) > 1:
                # When a visual separator divides an overlarge span detection,
                # each side becomes an independent logical span. Read text from
                # that segment only, so neighboring segment text does not leak
                # into unrelated rows or columns.
                span_text_box = [
                    min(cell_box[0] for _, _, cell_box in span_cells),
                    min(cell_box[1] for _, _, cell_box in span_cells),
                    max(cell_box[2] for _, _, cell_box in span_cells),
                    max(cell_box[3] for _, _, cell_box in span_cells),
                ]

            span_text = get_ocr_text_in_box(span_text_box, ocr_data).strip()
            if not span_text:
                span_text = get_text_center_in_box(span_text_box, ocr_data).strip()
            if not span_text:
                union_box_cells = [
                    min(cell_box[0] for _, _, cell_box in span_cells),
                    min(cell_box[1] for _, _, cell_box in span_cells),
                    max(cell_box[2] for _, _, cell_box in span_cells),
                    max(cell_box[3] for _, _, cell_box in span_cells),
                ]
                span_text = get_ocr_text_in_box(union_box_cells, ocr_data, intersection_thresh=0.05).strip()
                if not span_text:
                    span_text = get_text_center_in_box(union_box_cells, ocr_data).strip()
            if not span_text:
                continue

            for r_idx, c_idx, _ in span_cells:
                overrides[(r_idx, c_idx)] = span_text

    return overrides



def _owner_column_has_row_separator(structure_gray, owner_col_box, upper_row_box, lower_row_box):
    if structure_gray is None or structure_gray.ndim != 2:
        return False

    img_h, img_w = structure_gray.shape[:2]
    x1 = max(0, int(round(owner_col_box[0])) + 2)
    x2 = min(img_w, int(round(owner_col_box[2])) - 2)
    if x2 - x1 < 12:
        return False

    min_row_h = min(
        max(1.0, float(upper_row_box[3] - upper_row_box[1])),
        max(1.0, float(lower_row_box[3] - lower_row_box[1])),
    )
    search_half_h = max(2, min(6, int(round(0.16 * min_row_h))))

    # Normally the row boundary is halfway between the adjacent row edges. Some
    # model row boxes overlap heavily, so the midpoint can be pulled away from
    # the actual drawn separator. In that case also probe both row-edge
    # positions; this still requires a real continuous grid line in this owner
    # column before returning True.
    candidate_ys = [
        (float(upper_row_box[3]) + float(lower_row_box[1])) / 2.0,
        float(upper_row_box[3]),
        float(lower_row_box[1]),
    ]
    seen_ys = set()
    for candidate_y in candidate_ys:
        boundary_y = int(round(candidate_y))
        if boundary_y in seen_ys:
            continue
        seen_ys.add(boundary_y)
        y1 = max(0, boundary_y - search_half_h)
        y2 = min(img_h, boundary_y + search_half_h + 1)
        if y2 <= y1:
            continue

        strip = structure_gray[y1:y2, x1:x2]
        if strip.size == 0:
            continue

        dark_rows = strip <= STRUCTURE_LINE_DARK_PIXEL_MAX
        for dark_row in dark_rows:
            if float(np.mean(dark_row)) < 0.48:
                continue
            # A table rule is a long continuous stroke. Dense wrapped label text
            # can have many dark pixels near a row boundary, but it remains broken
            # into short character runs and must not split a valid merged cell.
            if _longest_true_run_ratio(dark_row) >= 0.80:
                return True
    return False


def _longest_true_run_ratio(mask_row):
    row_width = int(mask_row.size)
    if row_width <= 0:
        return 0.0

    longest = 0
    current = 0
    for value in mask_row:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / float(row_width)


def _split_row_span_cells_at_owner_separators(
    covered_cells,
    final_rows,
    owner_col_box,
    structure_gray,
):
    if len(covered_cells) < 2:
        return [covered_cells] if covered_cells else []

    ordered_cells = sorted(covered_cells, key=lambda item: item[0])
    segments = [[ordered_cells[0]]]
    for previous, current in zip(ordered_cells, ordered_cells[1:]):
        prev_row_box = final_rows[previous[0]]["box"]
        current_row_box = final_rows[current[0]]["box"]
        if _owner_column_has_row_separator(
            structure_gray,
            owner_col_box,
            prev_row_box,
            current_row_box,
        ):
            segments.append([current])
        else:
            segments[-1].append(current)

    return [segment for segment in segments if segment]


def _trim_row_span_cells_at_owner_separators(
    covered_cells,
    final_rows,
    owner_col_box,
    span_y,
    structure_gray,
):
    segments = _split_row_span_cells_at_owner_separators(
        covered_cells,
        final_rows,
        owner_col_box,
        structure_gray,
    )
    if len(segments) <= 1:
        return covered_cells if segments else []

    anchor_row_idx = min(
        (cell[0] for segment in segments for cell in segment),
        key=lambda row_idx: abs(final_rows[row_idx]["y"] - span_y),
    )
    for segment in segments:
        if any(row_idx == anchor_row_idx for row_idx, _, _ in segment):
            return segment
    return covered_cells


def _owner_row_has_col_separator(structure_gray, owner_row_box, left_col_box, right_col_box):
    if structure_gray is None or structure_gray.ndim != 2:
        return False

    img_h, img_w = structure_gray.shape[:2]
    y1 = max(0, int(round(owner_row_box[1])) + 2)
    y2 = min(img_h, int(round(owner_row_box[3])) - 2)
    if y2 - y1 < 12:
        return False

    boundary_x = int(round((float(left_col_box[2]) + float(right_col_box[0])) / 2.0))
    min_col_w = min(
        max(1.0, float(left_col_box[2] - left_col_box[0])),
        max(1.0, float(right_col_box[2] - right_col_box[0])),
    )
    search_half_w = max(2, min(6, int(round(0.16 * min_col_w))))
    x1 = max(0, boundary_x - search_half_w)
    x2 = min(img_w, boundary_x + search_half_w + 1)
    if x2 <= x1:
        return False

    strip = structure_gray[y1:y2, x1:x2]
    if strip.size == 0:
        return False

    dark_cols = (strip <= STRUCTURE_LINE_DARK_PIXEL_MAX).T
    for dark_col in dark_cols:
        if float(np.mean(dark_col)) < 0.48:
            continue
        # A table rule is a long continuous stroke. Vertical glyph strokes can
        # be dark near a column boundary, but they stay much shorter than a
        # border cutting through an owner row band.
        if _longest_true_run_ratio(dark_col) >= 0.80:
            return True
    return False


def _effective_owner_row_has_col_separator(structure_gray, owner_row, left_col_box, right_col_box):
    if owner_row and owner_row.get("type") == CLASS_NAME_ROW_HEADER:
        return True
    return _owner_row_has_col_separator(
        structure_gray,
        owner_row["box"] if isinstance(owner_row, dict) else owner_row,
        left_col_box,
        right_col_box,
    )


def _trim_col_span_cells_at_owner_separators(
    covered_cells,
    final_rows,
    final_cols,
    span_x,
    structure_gray,
):
    segments = _split_col_span_cells_at_owner_separators(
        covered_cells,
        final_rows,
        final_cols,
        structure_gray,
    )
    if len(segments) <= 1:
        return covered_cells if segments else []

    anchor_col_idx = min(
        (cell[1] for segment in segments for cell in segment),
        key=lambda col_idx: abs(
            ((final_cols[col_idx]["box"][0] + final_cols[col_idx]["box"][2]) / 2) - span_x
        ),
    )
    for segment in segments:
        if any(col_idx == anchor_col_idx for _, col_idx, _ in segment):
            return segment
    return covered_cells


def _split_col_span_cells_at_owner_separators(
    covered_cells,
    final_rows,
    final_cols,
    structure_gray,
):
    if len(covered_cells) < 2:
        return [covered_cells] if covered_cells else []

    by_row = {}
    for cell in covered_cells:
        by_row.setdefault(cell[0], []).append(cell)

    all_segments = []
    any_split = False
    for r_idx, row_cells in sorted(by_row.items()):
        if len(row_cells) < 2:
            all_segments.append(row_cells)
            continue

        ordered_cells = sorted(row_cells, key=lambda item: item[1])
        row_segments = [[ordered_cells[0]]]
        for previous, current in zip(ordered_cells, ordered_cells[1:]):
            prev_col_box = final_cols[previous[1]]["box"]
            current_col_box = final_cols[current[1]]["box"]
            if _effective_owner_row_has_col_separator(
                structure_gray,
                final_rows[r_idx],
                prev_col_box,
                current_col_box,
            ):
                row_segments.append([current])
            else:
                row_segments[-1].append(current)

        if len(row_segments) > 1:
            any_split = True
        all_segments.extend(row_segments)

    if not any_split:
        return [covered_cells]
    return [segment for segment in all_segments if segment]


def _line_recovery_support_threshold(peer_count):
    if peer_count <= 0:
        return 1
    if peer_count <= 2:
        return 1
    return max(2, int(np.ceil(0.45 * peer_count)))


def _line_recovery_span_text(box, ocr_data):
    text = get_ocr_text_in_box(box, ocr_data, intersection_thresh=0.05).strip()
    if not text:
        text = get_text_center_in_box(box, ocr_data).strip()
    return " ".join(text.split())


def _line_recovery_text_owned_by_column(span_box, col_box, ocr_data):
    for item in ocr_data or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        item_area = get_box_area(ibox)
        if item_area <= 0:
            continue
        if calculate_intersection_area(span_box, ibox) / (item_area + 1e-12) < 0.05:
            continue
        item_w = max(1.0, float(ibox[2] - ibox[0]))
        item_cx = (ibox[0] + ibox[2]) / 2.0
        inter_x = max(0.0, min(ibox[2], col_box[2]) - max(ibox[0], col_box[0]))
        if (col_box[0] - 2.0) <= item_cx <= (col_box[2] + 2.0):
            return True
        if inter_x / item_w >= 0.70:
            return True
    return False


def _line_recovery_cell_texts(row_indices, col_indices, final_rows, final_cols, ocr_data):
    texts = []
    for r_idx in row_indices:
        row_box = final_rows[r_idx]["box"]
        for c_idx in col_indices:
            col_box = final_cols[c_idx]["box"]
            cell_box = [col_box[0], row_box[1], col_box[2], row_box[3]]
            text = _line_recovery_span_text(cell_box, ocr_data)
            if text:
                texts.append(text)
    return texts


def _line_recovery_has_conflicting_cell_texts(row_indices, col_indices, final_rows, final_cols, ocr_data):
    texts = _line_recovery_cell_texts(row_indices, col_indices, final_rows, final_cols, ocr_data)
    if len(texts) <= 1:
        return False
    normalized = {
        re.sub(r"\s+", " ", text).strip().lower()
        for text in texts
        if text.strip()
    }
    normalized = {text for text in normalized if text}
    if len(normalized) <= 1:
        return False
    longest = max(normalized, key=len)
    # Wrapped OCR from one merged cell can appear as "full phrase" in the
    # merged box and as shorter continuation fragments in individual row/cell
    # slices. Those fragments are not conflicting independent values.
    if all(text == longest or text in longest for text in normalized):
        return False
    # Repeated same text is harmless; different per-cell values means this is
    # likely a normal grid region, not a merged/spanning cell.
    return len(normalized) > 1


def _logical_col_key_for_span_recovery(col):
    if col.get("is_split"):
        return ("split_parent", _box_anchor_key(col.get("parent_box", col["box"])))
    return ("column", _box_anchor_key(col["box"]))


def _recovered_span_prediction(cls, box, text, confidence=0.24):
    x1, y1, x2, y2 = [float(v) for v in box]
    return {
        "x": (x1 + x2) / 2.0,
        "y": (y1 + y2) / 2.0,
        "width": max(1.0, x2 - x1),
        "height": max(1.0, y2 - y1),
        "confidence": confidence,
        "class": cls,
        "class_id": -1,
        "detection_id": "synthetic_line_span_recovery",
        "source_model": "synthetic_recovery",
        "box": [x1, y1, x2, y2],
        "recovered_text": text,
    }


def _recover_spanning_row_cells_from_grid_lines(final_rows, final_cols, ocr_data, structure_gray):
    if (
        not ENABLE_LINE_BASED_SPAN_CELL_RECOVERY
        or structure_gray is None
        or not final_rows
        or len(final_cols) < 2
    ):
        return []

    body_row_indices = [
        idx for idx, row in enumerate(final_rows)
        if row.get("type") == CLASS_NAME_ROW
    ]
    if len(body_row_indices) < 2:
        return []

    boundary_visibility = {}
    supported_boundaries = set()
    for b_idx in range(len(body_row_indices) - 1):
        upper = final_rows[body_row_indices[b_idx]]
        lower = final_rows[body_row_indices[b_idx + 1]]
        visible_count = 0
        for c_idx, col in enumerate(final_cols):
            visible = _owner_column_has_row_separator(
                structure_gray,
                col["box"],
                upper["box"],
                lower["box"],
            )
            boundary_visibility[(b_idx, c_idx)] = visible
            if visible:
                visible_count += 1
        if visible_count >= 1:
            supported_boundaries.add(b_idx)

    recovered = []

    def flush_segment(c_idx, start_boundary, end_boundary):
        if start_boundary is None or end_boundary is None:
            return
        row_indices = body_row_indices[start_boundary:end_boundary + 2]
        if len(row_indices) < 2:
            return
        col_box = final_cols[c_idx]["box"]
        span_box = [
            col_box[0],
            min(final_rows[r_idx]["box"][1] for r_idx in row_indices),
            col_box[2],
            max(final_rows[r_idx]["box"][3] for r_idx in row_indices),
        ]
        text = _line_recovery_span_text(span_box, ocr_data)
        if not text:
            return
        if not _line_recovery_text_owned_by_column(span_box, col_box, ocr_data):
            return
        recovered.append(_recovered_span_prediction(CLASS_NAME_SPANNING_ROW_CELL, span_box, text))

    for c_idx, _ in enumerate(final_cols):
        run_start = None
        run_end = None
        for b_idx in range(len(body_row_indices) - 1):
            visible_here = boundary_visibility.get((b_idx, c_idx), False)
            peer_visible = sum(
                1 for other_idx in range(len(final_cols))
                if other_idx != c_idx and boundary_visibility.get((b_idx, other_idx), False)
            )
            missing_supported = (
                not visible_here
                and peer_visible >= 1
            )
            if missing_supported:
                if run_start is None:
                    run_start = b_idx
                run_end = b_idx
            else:
                flush_segment(c_idx, run_start, run_end)
                run_start = None
                run_end = None
        flush_segment(c_idx, run_start, run_end)

    return recovered


def _all_adjacent_column_boundaries_have_separator(structure_gray, final_rows, final_cols, body_row_indices):
    """Require every predicted adjacent column boundary to have real line evidence somewhere."""
    if structure_gray is None or len(final_cols) < 2 or not body_row_indices:
        return False

    for c_boundary in range(len(final_cols) - 1):
        left_col = final_cols[c_boundary]
        right_col = final_cols[c_boundary + 1]
        boundary_seen = False
        for r_idx in body_row_indices:
            if _owner_row_has_col_separator(
                structure_gray,
                final_rows[r_idx]["box"],
                left_col["box"],
                right_col["box"],
            ):
                boundary_seen = True
                break
        if not boundary_seen:
            return False
    return True


def _recover_spanning_column_cells_from_grid_lines(final_rows, final_cols, ocr_data, structure_gray):
    if (
        not ENABLE_LINE_BASED_SPAN_CELL_RECOVERY
        or structure_gray is None
        or not final_rows
        or len(final_cols) < 2
    ):
        return []

    body_row_indices = [
        idx for idx, row in enumerate(final_rows)
        if row.get("type") == CLASS_NAME_ROW
    ]
    if len(body_row_indices) < 2:
        return []

    # Column-span recovery is only safe for true vertical-grid tables. Every
    # adjacent predicted column boundary must show a separator somewhere in the
    # body area; otherwise missing lines can just be a non-grid layout.
    if not _all_adjacent_column_boundaries_have_separator(
        structure_gray,
        final_rows,
        final_cols,
        body_row_indices,
    ):
        return []

    boundary_count = len(final_cols) - 1
    boundary_visibility = {}
    supported_boundaries = set()
    for c_boundary in range(boundary_count):
        left_col = final_cols[c_boundary]
        right_col = final_cols[c_boundary + 1]
        visible_count = 0
        for r_idx in body_row_indices:
            visible = _owner_row_has_col_separator(
                structure_gray,
                final_rows[r_idx]["box"],
                left_col["box"],
                right_col["box"],
            )
            boundary_visibility[(r_idx, c_boundary)] = visible
            if visible:
                visible_count += 1
        if visible_count >= 1:
            supported_boundaries.add(c_boundary)

    if len(supported_boundaries) < boundary_count:
        return []

    recovered = []

    def flush_segment(r_idx, start_boundary, end_boundary):
        if start_boundary is None or end_boundary is None:
            return
        col_indices = list(range(start_boundary, end_boundary + 2))
        if len(col_indices) < 2:
            return
        logical_keys = {
            _logical_col_key_for_span_recovery(final_cols[c_idx])
            for c_idx in col_indices
        }
        if len(logical_keys) < 2:
            return
        row_box = final_rows[r_idx]["box"]
        span_box = [
            min(final_cols[c_idx]["box"][0] for c_idx in col_indices),
            row_box[1],
            max(final_cols[c_idx]["box"][2] for c_idx in col_indices),
            row_box[3],
        ]
        text = _line_recovery_span_text(span_box, ocr_data)
        if not text:
            return
        if _line_recovery_has_conflicting_cell_texts([r_idx], col_indices, final_rows, final_cols, ocr_data):
            return
        recovered.append(_recovered_span_prediction(CLASS_NAME_SPANNING_COLUMN_CELL, span_box, text))

    for r_idx in body_row_indices:
        run_start = None
        run_end = None
        for c_boundary in range(boundary_count):
            visible_here = boundary_visibility.get((r_idx, c_boundary), False)
            peer_visible = sum(
                1 for other_r_idx in body_row_indices
                if other_r_idx != r_idx and boundary_visibility.get((other_r_idx, c_boundary), False)
            )
            missing_supported = (
                c_boundary in supported_boundaries
                and not visible_here
                and peer_visible >= 1
            )
            if missing_supported:
                if run_start is None:
                    run_start = c_boundary
                run_end = c_boundary
            else:
                flush_segment(r_idx, run_start, run_end)
                run_start = None
                run_end = None
        flush_segment(r_idx, run_start, run_end)

    return recovered


def _recover_spanning_both_cells_from_grid_lines(final_rows, final_cols, ocr_data, structure_gray):
    if (
        not ENABLE_LINE_BASED_SPAN_CELL_RECOVERY
        or structure_gray is None
        or not final_rows
        or len(final_cols) < 2
        or not ocr_data
    ):
        return {}

    body_row_indices = [
        idx for idx, row in enumerate(final_rows)
        if row.get("type") == CLASS_NAME_ROW
    ]
    if len(body_row_indices) < 2:
        return {}

    # Mixed row+column span recovery also needs a reliable vertical column grid
    # before treating missing vertical separators as horizontal cell spans.
    if not _all_adjacent_column_boundaries_have_separator(
        structure_gray,
        final_rows,
        final_cols,
        body_row_indices,
    ):
        return {}

    def mixed_row_boundary_supported(visibility):
        # A mixed span may hide separators across most of a boundary. One visible
        # separator segment outside the missing region is enough grid evidence;
        # OCR/conflict/rectangle gates below decide whether the missing region is real.
        return any(visibility)

    def mixed_col_boundary_supported(visibility):
        # Same rule in the horizontal direction: visible grid evidence outside
        # the merged region is enough to test a missing vertical separator.
        return any(visibility)

    # Horizontal row-boundary evidence, evaluated per column region.
    row_boundary_visibility = {}
    supported_row_boundaries = set()
    for b_pos in range(len(body_row_indices) - 1):
        upper = final_rows[body_row_indices[b_pos]]
        lower = final_rows[body_row_indices[b_pos + 1]]
        visibility = []
        for c_idx, col in enumerate(final_cols):
            visible = _owner_column_has_row_separator(
                structure_gray,
                col["box"],
                upper["box"],
                lower["box"],
            )
            row_boundary_visibility[(b_pos, c_idx)] = visible
            visibility.append(visible)
        if mixed_row_boundary_supported(visibility):
            supported_row_boundaries.add(b_pos)

    # Vertical column-boundary evidence, evaluated per body row region.
    col_boundary_count = len(final_cols) - 1
    col_boundary_visibility = {}
    supported_col_boundaries = set()
    for c_boundary in range(col_boundary_count):
        left_col = final_cols[c_boundary]
        right_col = final_cols[c_boundary + 1]
        visibility = []
        for r_pos, r_idx in enumerate(body_row_indices):
            visible = _owner_row_has_col_separator(
                structure_gray,
                final_rows[r_idx]["box"],
                left_col["box"],
                right_col["box"],
            )
            col_boundary_visibility[(r_pos, c_boundary)] = visible
            visibility.append(visible)
        if mixed_col_boundary_supported(visibility):
            supported_col_boundaries.add(c_boundary)

    if (
        len(supported_row_boundaries) < len(body_row_indices) - 1
        or len(supported_col_boundaries) < col_boundary_count
    ):
        return {}

    missing_row_edges = set()
    for b_pos in range(len(body_row_indices) - 1):
        if b_pos not in supported_row_boundaries:
            continue
        for c_idx in range(len(final_cols)):
            visible_here = row_boundary_visibility.get((b_pos, c_idx), False)
            peer_visible = sum(
                1 for other_c_idx in range(len(final_cols))
                if other_c_idx != c_idx and row_boundary_visibility.get((b_pos, other_c_idx), False)
            )
            anchor_visible = any(
                row_boundary_visibility.get((b_pos, anchor_idx), False)
                for anchor_idx in range(min(2, len(final_cols)))
                if anchor_idx != c_idx
            )
            if not visible_here and (peer_visible >= 1 or anchor_visible):
                missing_row_edges.add((b_pos, c_idx))

    missing_col_edges = set()
    for c_boundary in range(col_boundary_count):
        if c_boundary not in supported_col_boundaries:
            continue
        for r_pos, _ in enumerate(body_row_indices):
            visible_here = col_boundary_visibility.get((r_pos, c_boundary), False)
            peer_visible = sum(
                1 for other_r_pos in range(len(body_row_indices))
                if other_r_pos != r_pos and col_boundary_visibility.get((other_r_pos, c_boundary), False)
            )
            anchor_visible = any(
                col_boundary_visibility.get((anchor_r_pos, c_boundary), False)
                for anchor_r_pos in range(min(3, len(body_row_indices)))
                if anchor_r_pos != r_pos
            )
            if not visible_here and (peer_visible >= 1 or anchor_visible):
                missing_col_edges.add((r_pos, c_boundary))

    if not missing_row_edges or not missing_col_edges:
        return {}

    adjacency = {}

    def add_edge(a, b):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    for b_pos, c_idx in missing_row_edges:
        add_edge((b_pos, c_idx), (b_pos + 1, c_idx))
    for r_pos, c_boundary in missing_col_edges:
        add_edge((r_pos, c_boundary), (r_pos, c_boundary + 1))

    overrides = {}
    visited = set()
    for start in list(adjacency):
        if start in visited:
            continue
        stack = [start]
        component = set()
        visited.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for nxt in adjacency.get(node, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

        row_positions = sorted({node[0] for node in component})
        col_indices = sorted({node[1] for node in component})
        if len(row_positions) < 2 or len(col_indices) < 2:
            continue

        min_r, max_r = row_positions[0], row_positions[-1]
        min_c, max_c = col_indices[0], col_indices[-1]
        rect_cells = {
            (r_pos, c_idx)
            for r_pos in range(min_r, max_r + 1)
            for c_idx in range(min_c, max_c + 1)
        }
        if rect_cells != component:
            # Mixed spans are rectangular table cells. L-shapes usually mean
            # independent row/column spans touching each other, not one cell.
            continue

        if any(
            (b_pos, c_idx) not in missing_row_edges
            for b_pos in range(min_r, max_r)
            for c_idx in range(min_c, max_c + 1)
        ):
            continue
        if any(
            (r_pos, c_boundary) not in missing_col_edges
            for r_pos in range(min_r, max_r + 1)
            for c_boundary in range(min_c, max_c)
        ):
            continue

        row_indices = [body_row_indices[r_pos] for r_pos in range(min_r, max_r + 1)]
        logical_col_keys = {
            _logical_col_key_for_span_recovery(final_cols[c_idx])
            for c_idx in range(min_c, max_c + 1)
        }
        if len(logical_col_keys) < 2:
            continue

        span_box = [
            min(final_cols[c_idx]["box"][0] for c_idx in range(min_c, max_c + 1)),
            min(final_rows[r_idx]["box"][1] for r_idx in row_indices),
            max(final_cols[c_idx]["box"][2] for c_idx in range(min_c, max_c + 1)),
            max(final_rows[r_idx]["box"][3] for r_idx in row_indices),
        ]
        text = _line_recovery_span_text(span_box, ocr_data)
        if not text:
            continue
        if _line_recovery_has_conflicting_cell_texts(
            row_indices,
            list(range(min_c, max_c + 1)),
            final_rows,
            final_cols,
            ocr_data,
        ):
            continue

        for r_idx in row_indices:
            for c_idx in range(min_c, max_c + 1):
                overrides[(r_idx, c_idx)] = text

    return overrides



def _build_header_cell_overrides(final_rows, final_cols, row_assignments, ocr_data):
    overrides = {}
    scores = {}
    if not final_rows or not final_cols:
        return overrides

    for r_idx, row in enumerate(final_rows):
        if not _is_header_row_type(row["type"]):
            continue

        for item in row.get("items", []):
            if _should_skip_header_item_for_propagation(final_rows, r_idx, row, item, final_cols):
                continue
            item_box = item["box"]
            item_text = get_ocr_text_in_box(item_box, row_assignments[r_idx], intersection_thresh=0.05)
            if not item_text:
                item_text = get_ocr_text_in_box(item_box, ocr_data, intersection_thresh=0.05)
            if not item_text:
                item_text = get_text_center_in_box(item_box, row_assignments[r_idx])
            if not item_text:
                item_text = get_text_center_in_box(item_box, ocr_data)
            item_text = item_text.strip()
            if not item_text:
                continue

            candidate_cols = [
                (entry["score"], entry["idx"])
                for entry in _column_support_candidates(item_box, final_cols, allow_soft=True)
            ]
            if not candidate_cols:
                continue

            for score, c_idx in candidate_cols:
                key = (r_idx, c_idx)
                if score > scores.get(key, float("-inf")):
                    overrides[key] = item_text
                    scores[key] = score

    return overrides


def _tall_header_item_end_row_index(header_item, final_rows, final_cols=None):
    if not header_item or not final_rows:
        return None
    if header_item.get("class") != CLASS_NAME_COLUMN_HEADER:
        return None

    header_rows = [
        (idx, row)
        for idx, row in enumerate(final_rows)
        if _is_header_row_type(row["type"])
    ]
    if len(header_rows) < 2:
        return None

    header_box = header_item.get("box")
    if not header_box:
        return None
    covered_header_rows = []
    for candidate_idx, candidate_row in header_rows:
        row_box = candidate_row["box"]
        inter_h = max(0.0, min(header_box[3], row_box[3]) - max(header_box[1], row_box[1]))
        row_h = max(1.0, float(row_box[3] - row_box[1]))
        if inter_h / row_h >= 0.25:
            covered_header_rows.append((candidate_idx, candidate_row))
    if len(covered_header_rows) < 2:
        return None

    bottom_y = float(header_box[3])
    end_row_idx, _end_row = min(
        covered_header_rows,
        key=lambda entry: (
            0 if entry[1]["box"][1] <= bottom_y <= entry[1]["box"][3] else 1,
            abs(float(entry[1]["box"][3]) - bottom_y),
            -float(entry[1]["box"][1]),
        ),
    )
    return end_row_idx


def _cell_has_non_end_tall_header_owner(row, r_idx, logic_cell_box, final_rows, final_cols):
    if not row or not _is_header_row_type(row.get("type")):
        return False
    for owner_idx, owner_row in enumerate(final_rows):
        if not _is_header_row_type(owner_row.get("type")):
            continue
        for item in owner_row.get("items", []):
            if item.get("class") != CLASS_NAME_COLUMN_HEADER:
                continue
            end_row_idx = _tall_header_item_end_row_index(item, final_rows, final_cols)
            if end_row_idx is None or r_idx >= end_row_idx:
                continue
            if _smaller_region_overlap_ratio(item.get("box"), logic_cell_box) >= 0.20:
                return True
    return False


def _ocr_item_blocked_by_non_end_tall_header(item_box, r_idx, final_rows, final_cols):
    for owner_row in final_rows:
        if not _is_header_row_type(owner_row.get("type")):
            continue
        for header_item in owner_row.get("items", []):
            if header_item.get("class") != CLASS_NAME_COLUMN_HEADER:
                continue
            end_row_idx = _tall_header_item_end_row_index(header_item, final_rows, final_cols)
            if end_row_idx is None or r_idx >= end_row_idx:
                continue
            if _smaller_region_overlap_ratio(header_item.get("box"), item_box) >= 0.20:
                return True
    return False


def _ocr_item_blocked_by_other_header_row(item_box, r_idx, final_rows):
    for owner_idx, owner_row in enumerate(final_rows):
        if owner_idx == r_idx or not _is_header_row_type(owner_row.get("type")):
            continue
        for header_item in owner_row.get("items", []):
            if header_item.get("class") not in {CLASS_NAME_COLUMN_HEADER, CLASS_NAME_SPANNING_COLUMN_HEADER}:
                continue
            if _smaller_region_overlap_ratio(header_item.get("box"), item_box) >= 0.20:
                return True
    return False


def _ocr_item_key(item):
    box = o_to_box(item)
    return (
        str(item.get("text", "")).strip(),
        tuple(round(float(value), 1) for value in box),
    )


def _strict_header_row_items(row_box, cell_box, final_cols, c_idx, source_items):
    if not row_box or not source_items:
        return []

    row_h = max(1.0, float(row_box[3] - row_box[1]))
    strict_row_box = [
        row_box[0],
        row_box[1] + min(2.0, 0.10 * row_h),
        row_box[2],
        row_box[3] - min(2.0, 0.10 * row_h),
    ]
    items = _row_cell_raw_items(strict_row_box, cell_box, final_cols, c_idx, source_items)
    kept = []
    for item in items:
        ibox = o_to_box(item)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        _, cy = get_box_center(ibox)
        if not (row_box[1] <= cy <= row_box[3]):
            continue
        item_h = max(1.0, float(ibox[3] - ibox[1]))
        inter_h = max(0.0, min(row_box[3], ibox[3]) - max(row_box[1], ibox[1]))
        if inter_h / item_h < 0.60:
            continue
        kept.append(item)
    if kept:
        return kept

    # Some valid spanning header OCR boxes cover two adjacent value columns.
    # Center-only assignment sees only one of those columns, so allow a header
    # item to serve an empty neighbor only when it materially covers that cell.
    for item in source_items:
        ibox = o_to_box(item)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        _, cy = get_box_center(ibox)
        if not (row_box[1] <= cy <= row_box[3]):
            continue
        item_h = max(1.0, float(ibox[3] - ibox[1]))
        inter_h = max(0.0, min(row_box[3], ibox[3]) - max(row_box[1], ibox[1]))
        if inter_h / item_h < 0.60:
            continue
        inter_w = max(0.0, min(cell_box[2], ibox[2]) - max(cell_box[0], ibox[0]))
        cell_w = max(1.0, float(cell_box[2] - cell_box[0]))
        item_w = max(1.0, float(ibox[2] - ibox[0]))
        if inter_w / cell_w >= 0.35 and item_w >= 0.80 * cell_w:
            kept.append(item)
    return kept


def _header_item_materially_covers_cell(item, cell_box):
    ibox = o_to_box(item)
    inter_w = max(0.0, min(cell_box[2], ibox[2]) - max(cell_box[0], ibox[0]))
    cell_w = max(1.0, float(cell_box[2] - cell_box[0]))
    return inter_w / cell_w >= 0.35


def _fill_orphan_header_ocr_cells(matrix, final_rows, final_cols, row_assignments, ocr_data):
    if not matrix or not final_rows or not final_cols or not row_assignments:
        return

    for r_idx, row in enumerate(final_rows[: len(matrix)]):
        if r_idx >= len(row_assignments) or not _is_header_row_type(row.get("type")):
            continue
        if not any(str(cell).strip() for cell in matrix[r_idx]):
            continue

        row_box = row["box"]
        used_item_keys = set()
        for c_idx, cell_text in enumerate(matrix[r_idx]):
            if not str(cell_text).strip():
                continue
            col = final_cols[c_idx]
            cell_box = [col["box"][0], row_box[1] - 1, col["box"][2], row_box[3] + 1]
            used_items = _strict_header_row_items(
                row_box,
                cell_box,
                final_cols,
                c_idx,
                row_assignments[r_idx],
            )
            for item in used_items:
                used_item_keys.add(_ocr_item_key(item))

        for c_idx, cell_text in enumerate(matrix[r_idx]):
            if str(cell_text).strip():
                continue
            col = final_cols[c_idx]
            logic_cell_box = [col["box"][0], row_box[1] - 1, col["box"][2], row_box[3] + 1]
            if _cell_has_non_end_tall_header_owner(row, r_idx, logic_cell_box, final_rows, final_cols):
                continue

            candidates = _strict_header_row_items(
                row_box,
                logic_cell_box,
                final_cols,
                c_idx,
                row_assignments[r_idx],
            )
            if not candidates:
                row_center_items = [
                    item for item in ocr_data or []
                    if row_box[1] <= get_box_center(o_to_box(item))[1] <= row_box[3]
                ]
                candidates = _strict_header_row_items(
                    row_box,
                    logic_cell_box,
                    final_cols,
                    c_idx,
                    row_center_items,
                )

            filtered = []
            for item in candidates:
                if (
                    _ocr_item_key(item) in used_item_keys
                    and not _header_item_materially_covers_cell(item, logic_cell_box)
                ):
                    continue
                if _ocr_item_blocked_by_non_end_tall_header(o_to_box(item), r_idx, final_rows, final_cols):
                    continue
                if _ocr_item_blocked_by_other_header_row(o_to_box(item), r_idx, final_rows):
                    continue
                filtered.append(item)

            if not filtered:
                continue
            filtered = _dedupe_near_duplicate_text_items(filtered)
            filtered = _sort_ocr_items_reading_order(filtered)
            text = " ".join(str(item.get("text", "")).strip() for item in filtered).strip()
            if not text:
                continue
            matrix[r_idx][c_idx] = text
            for item in filtered:
                used_item_keys.add(_ocr_item_key(item))


def _row_band_text_profile(row_box, final_cols, ocr_data):
    if not row_box or not final_cols or not ocr_data:
        return None

    row_items = get_ocr_items_in_box(row_box, ocr_data, intersection_thresh=0.05)
    if not row_items:
        return None

    populated = set()
    union_box = None
    text_parts = []

    for item in row_items:
        ibox = o_to_box(item)
        text_val = str(item.get("text", "")).strip()
        if text_val:
            text_parts.append(text_val)

        cx, _ = get_box_center(ibox)
        matched = None
        for c_idx, col in enumerate(final_cols):
            if col["box"][0] <= cx <= col["box"][2]:
                matched = c_idx
                break
        if matched is not None:
            populated.add(matched)

        if union_box is None:
            union_box = list(ibox)
        else:
            union_box = [
                min(union_box[0], ibox[0]),
                min(union_box[1], ibox[1]),
                max(union_box[2], ibox[2]),
                max(union_box[3], ibox[3]),
            ]

    if union_box is None:
        return None

    text_val = " ".join(text_parts).strip()

    return {
        "items": row_items,
        "text": text_val,
        "alpha": _alpha_frac(text_val),
        "digit": _digit_frac(text_val),
        "populated": populated,
        "max_col": max(populated) if populated else -1,
        "union_box": union_box,
    }


def _should_promote_row_to_row_header(row, final_cols, ocr_data):
    if not row or not final_cols or not ocr_data:
        return False

    row_box = row["box"]
    table_w = max(1.0, float(final_cols[-1]["box"][2] - final_cols[0]["box"][0]))
    row_w = max(1.0, float(row_box[2] - row_box[0]))
    if row_w < 0.55 * table_w:
        return False

    row_items = []
    for item in ocr_data:
        ibox = o_to_box(item)
        text_val = str(item.get("text", "")).strip()
        if not text_val:
            continue

        cx, cy = get_box_center(ibox)
        item_h = max(1.0, float(ibox[3] - ibox[1]))
        inter_h = _row_band_intersection_height(ibox, row_box)
        if (
            row_box[0] <= cx <= row_box[2]
            and row_box[1] <= cy <= row_box[3]
        ) or (inter_h / item_h) >= 0.45:
            row_items.append(item)

    if not row_items:
        return False

    profile = _row_band_text_profile(row_box, final_cols, row_items)
    if not profile or not profile.get("populated"):
        return False

    union_box = profile.get("union_box")
    if union_box is None:
        return False
    return True


def _should_drop_lone_title_row_header(row_header, body_rows, final_cols, ocr_data, header_bottom):
    if not row_header or not body_rows or not final_cols or len(final_cols) < 2 or not ocr_data:
        return False

    first_body_row = next((row for row in body_rows if row["type"] == CLASS_NAME_ROW), None)
    if first_body_row is None or row_header["y"] >= first_body_row["y"]:
        return False

    row_box = row_header["box"]
    table_w = max(1.0, float(final_cols[-1]["box"][2] - final_cols[0]["box"][0]))
    row_w = max(1.0, float(row_box[2] - row_box[0]))
    row_h = max(1.0, float(row_box[3] - row_box[1]))
    row_conf = float(row_header.get("confidence", 0.0))
    if row_w < 0.65 * table_w:
        return False

    if row_conf >= 0.75:
        return False

    if header_bottom is not None and row_box[1] > (header_bottom + max(12.0, 0.6 * row_h)):
        return False

    row_items = []
    for item in ocr_data:
        ibox = o_to_box(item)
        center = get_box_center(ibox)
        item_h = max(1.0, float(ibox[3] - ibox[1]))
        inter_h = _row_band_intersection_height(ibox, row_box)
        if (
            row_box[0] <= center[0] <= row_box[2]
            and row_box[1] <= center[1] <= row_box[3]
        ) or (inter_h / item_h) >= 0.45:
            row_items.append(item)

    if not row_items:
        return False

    populated_col_indices = set()
    for item in row_items:
        ibox = o_to_box(item)
        center = get_box_center(ibox)
        for c_idx, col in enumerate(final_cols[: min(3, len(final_cols))]):
            if col["box"][0] <= center[0] <= col["box"][2]:
                populated_col_indices.add(c_idx)
                break
    if len(populated_col_indices) > 1 and len(row_items) > 1:
        return False

    sample_rows = [row for row in body_rows if row["type"] == CLASS_NAME_ROW and row["y"] > row_header["y"]][:4]
    if len(sample_rows) < 3:
        return False

    dense_rows = 0
    for row in sample_rows:
        first_box = [final_cols[0]["box"][0], row["box"][1], final_cols[0]["box"][2], row["box"][3]]
        second_box = [final_cols[1]["box"][0], row["box"][1], final_cols[1]["box"][2], row["box"][3]]
        first_text = get_text_center_in_box(first_box, ocr_data)
        if not first_text:
            first_text = get_ocr_text_in_box(first_box, ocr_data, intersection_thresh=0.05)
        second_text = get_text_center_in_box(second_box, ocr_data)
        if not second_text:
            second_text = get_ocr_text_in_box(second_box, ocr_data, intersection_thresh=0.05)
        if first_text.strip() and second_text.strip():
            dense_rows += 1

    return dense_rows >= 3


def cluster_items_by_y(items, y_threshold):
    if not items:
        return []

    items = sorted(items, key=lambda item: item["y"])
    clusters = [[items[0]]]
    for item in items[1:]:
        if abs(item["y"] - clusters[-1][0]["y"]) < y_threshold:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def union_box(items):
    return [
        min(item["box"][0] for item in items),
        min(item["box"][1] for item in items),
        max(item["box"][2] for item in items),
        max(item["box"][3] for item in items),
    ]


def _transform_point(pt, img_w, img_h, mode):
    x, y = pt
    if mode == "identity":
        return [x, y]
    if mode == "rot180":
        return [img_w - x, img_h - y]
    if mode == "rot90cw":
        return [img_h - y, x]
    if mode == "rot90ccw":
        return [y, img_w - x]
    return [x, y]


def _transform_poly(poly, img_w, img_h, mode):
    return [_transform_point(pt, img_w, img_h, mode) for pt in poly]


def _transform_xyxy(box, img_w, img_h, mode):
    x1, y1, x2, y2 = box
    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    tpoly = _transform_poly(poly, img_w, img_h, mode)
    xs = [p[0] for p in tpoly]
    ys = [p[1] for p in tpoly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _ocr_center_inside_ratio(ocr_items, box):
    if not ocr_items:
        return 0.0
    inside = 0
    total = 0
    for item in ocr_items:
        txt = str(item.get("text", "")).strip()
        if not txt:
            continue
        total += 1
        ibox = o_to_box(item)
        cx, cy = get_box_center(ibox)
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            inside += 1
    if total == 0:
        return 0.0
    return inside / total


def _alpha_frac(s):
    if not s:
        return 0.0
    t = "".join(ch for ch in str(s) if not ch.isspace())
    if not t:
        return 0.0
    a = sum(ch.isalpha() for ch in t)
    return a / max(1, len(t))


def _digit_frac(s):
    if not s:
        return 0.0
    t = "".join(ch for ch in str(s) if not ch.isspace())
    if not t:
        return 0.0
    d = sum(ch.isdigit() for ch in t)
    return d / max(1, len(t))


def _shift_box(box, dx, dy):
    if isinstance(box[0], list):
        return [[pt[0] + dx, pt[1] + dy] for pt in box]
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def _shift_ocr_items(ocr_items, dx, dy):
    if dx == 0 and dy == 0:
        return ocr_items
    shifted = []
    for item in ocr_items:
        box = item.get("box")
        if not box:
            continue
        shifted.append({**item, "box": _shift_box(box, dx, dy)})
    return shifted


def _nearest_band_index(value, bands, axis):
    if not bands:
        return None

    for index, box in enumerate(bands):
        start = box[0] if axis == "x" else box[1]
        end = box[2] if axis == "x" else box[3]
        if start <= value <= end:
            return index

    if axis == "x":
        centers = [(box[0] + box[2]) / 2 for box in bands]
    else:
        centers = [(box[1] + box[3]) / 2 for box in bands]
    return min(range(len(centers)), key=lambda idx: abs(centers[idx] - value))


def _orientation_structure_score(ocr_items, table_box, row_boxes, col_boxes, header_box=None, row_types=None):
    if not ocr_items:
        return 0.0

    score = 0.0
    useful_items = 0
    last_row_idx = max(0, len(row_boxes) - 1)
    last_col_idx = max(0, len(col_boxes) - 1)
    top_row_count = min(2, len(row_boxes))
    left_col_count = min(2, len(col_boxes))

    for item in ocr_items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        box = o_to_box(item)
        cx, cy = get_box_center(box)
        if not (table_box[0] <= cx <= table_box[2] and table_box[1] <= cy <= table_box[3]):
            continue

        useful_items += 1
        alpha = _alpha_frac(text)
        digit = _digit_frac(text)
        bracket_like = "[" in text or "]" in text
        row_idx = _nearest_band_index(cy, row_boxes, "y")
        col_idx = _nearest_band_index(cx, col_boxes, "x")

        row_pos = (row_idx / last_row_idx) if last_row_idx else 0.0
        col_pos = (col_idx / last_col_idx) if last_col_idx else 0.0
        row_type = row_types[row_idx] if row_idx is not None and row_idx < len(row_types) else None

        score += alpha * (1.8 - row_pos)
        score += alpha * (1.4 - col_pos)
        score += digit * (0.5 + row_pos)
        score += digit * (0.7 + col_pos)

        if row_type in (CLASS_NAME_COLUMN_HEADER, CLASS_NAME_SPANNING_COLUMN_HEADER):
            if col_idx == 0:
                score += 1.8 * alpha
                score -= 0.8 * digit
            else:
                if bracket_like:
                    score += 2.4
                score += 0.9 * alpha
                score -= 1.4 * digit
        elif col_idx == 0:
            score += 1.2 * alpha
            score -= 0.4 * digit
        else:
            score += 1.0 * digit
            score -= 0.4 * alpha

        if row_idx == 0:
            if col_idx == 0:
                score += 1.0 * alpha
            elif bracket_like:
                score += 2.0
            else:
                score -= 1.1 * digit
        elif col_idx > 0 and bracket_like:
            score -= 0.8

        if top_row_count and row_idx < top_row_count:
            score += 0.9 * alpha + 0.2 * digit
        if left_col_count and col_idx < left_col_count:
            score += 0.8 * alpha

        if last_row_idx and row_idx == last_row_idx:
            score -= 0.6 * alpha
        if last_col_idx and col_idx == last_col_idx:
            score -= 0.6 * alpha

        if header_box and header_box[0] <= cx <= header_box[2] and header_box[1] <= cy <= header_box[3]:
            score += 1.0 * alpha + 0.15 * digit
            if bracket_like:
                score += 1.2

    if useful_items == 0:
        return 0.0

    score /= useful_items
    score += 0.5 * _ocr_center_inside_ratio(ocr_items, table_box)
    if header_box:
        score += 1.0 * _ocr_center_inside_ratio(ocr_items, header_box)
        score += _header_content_score(ocr_items, header_box)
    return score


def _ocr_bbox_xyxy(ocr_items):
    if not ocr_items:
        return None

    xs1, ys1, xs2, ys2 = [], [], [], []
    for item in ocr_items:
        box = o_to_box(item)
        xs1.append(box[0])
        ys1.append(box[1])
        xs2.append(box[2])
        ys2.append(box[3])

    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _horizontal_text_score(ocr_items):
    useful = 0
    score = 0.0
    for item in ocr_items:
        x1, y1, x2, y2 = o_to_box(item)
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        score += float(np.log(max(0.25, min(4.0, w / h))))
        useful += 1

    return score / useful if useful else 0.0


def _normalize_ocr_coordinates(ocr_data, table_box, img_w, img_h, header_box=None, row_boxes=None, col_boxes=None, row_types=None):
    if not ocr_data:
        return ocr_data, "identity"
    return ocr_data, "identity"

def _build_extraction_row_boxes(final_rows):
    if not final_rows:
        return []

    row_boxes = []
    for idx, row in enumerate(final_rows):
        x1, y1, x2, y2 = row["box"]

        if idx > 0:
            prev_bottom = final_rows[idx - 1]["box"][3]
            y1 = (y1 + prev_bottom) / 2

        if idx + 1 < len(final_rows):
            next_top = final_rows[idx + 1]["box"][1]
            y2 = (y2 + next_top) / 2

        if y2 <= y1:
            y1, y2 = row["box"][1], row["box"][3]

        row_boxes.append([
            x1,
            max(0, y1),
            x2,
            y2,
        ])

    return row_boxes


def _label_key(text):
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())
def _is_headerish_cell_text(text):
    text = str(text or "").strip()
    if not text:
        return False
    if len(text) > 32:
        return False
    alpha = _alpha_frac(text)
    digit = _digit_frac(text)
    if digit > 0.6 and alpha < 0.2 and "[" not in text:
        return False
    if text.isupper() and len(text) <= 14:
        return True
    return alpha >= 0.25


def _is_numericish_text(text):
    text = str(text or "").strip()
    if not text:
        return False
    alpha = _alpha_frac(text)
    digit = _digit_frac(text)
    if digit >= 0.6 and alpha < 0.2 and "[" not in text:
        return True
    return False





def _header_content_score(ocr_items, header_box):
    if not header_box or not ocr_items:
        return 0.0

    header_items = []
    for item in ocr_items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ibox = o_to_box(item)
        cx = (ibox[0] + ibox[2]) / 2
        cy = (ibox[1] + ibox[3]) / 2
        if header_box[0] <= cx <= header_box[2] and header_box[1] <= cy <= header_box[3]:
            header_items.append(text)

    if not header_items:
        return 0.0

    headerish = 0
    numericish = 0
    for text in header_items:
        if _is_headerish_cell_text(text):
            headerish += 1
        if _digit_frac(text) > 0.6 and _alpha_frac(text) < 0.2 and "[" not in text:
            numericish += 1

    total = max(1, len(header_items))
    return 1.6 * ((headerish - numericish) / total)



def _extract_unit_from_text(text):
    text = str(text or "").strip()
    if not text:
        return ""
    lower = text.lower().replace("??", "?")
    lower = lower.replace("?", "deg").replace("?", "ohm").replace("?", "ohm")
    if "temp" in lower:
        return "?C"
    if "deg" in lower and "c" in lower:
        return "?C"
    unit_pattern = r"(mv|uv|nv|v|ma|ua|na|a|mw|uw|w|khz|mhz|ghz|hz|ps|ns|us|ms|s|db|%|ohm|degc|k)"
    match = re.search(unit_pattern, lower)
    if not match:
        return ""
    token = match.group(0)
    token_map = {
        "mv": "mV",
        "uv": "uV",
        "nv": "nV",
        "v": "V",
        "ma": "mA",
        "ua": "uA",
        "na": "nA",
        "a": "A",
        "mw": "mW",
        "uw": "uW",
        "w": "W",
        "khz": "kHz",
        "mhz": "MHz",
        "ghz": "GHz",
        "hz": "Hz",
        "fs": "fs",
        "ps": "ps",
        "ns": "ns",
        "us": "us",
        "ms": "ms",
        "s": "s",
        "db": "dB",
        "%": "%",
        "ohm": "ohm",
        "degc": "?C",
        "k": "K",
    }
    return token_map.get(token, token)


def _unit_family_key(text):
    compact = "".join(ch for ch in str(text or "").strip() if not ch.isspace())
    if not compact:
        return ""

    family = compact.upper().replace("Â", "")
    family = family.replace("DEGC", "C")
    family = family.replace("DEG", "")
    family = family.replace("°", "")
    family = family.replace("º", "")
    family = family.replace("Ω", "Ω")
    family = family.replace("OHM", "Ω")
    return family


def _is_simple_unit_text(text):
    compact = "".join(ch for ch in str(text or "").strip() if not ch.isspace())
    if not compact:
        return False

    compact = compact.replace("Â", "")
    simple_pattern = (
        r"(?i)^(?:"
        r"[pnumkgt]?v|"
        r"[pnumkgt]?a|"
        r"[pnumkgt]?w|"
        r"[pnumkgt]?hz|"
        r"ohm|ω|Ω|"
        r"db|%|"
        r"fs|ps|ns|us|ms|s|"
        r"k|"
        r"(?:°|º)?c|f"
        r")$"
    )
    return bool(re.fullmatch(simple_pattern, compact))


def _unit_text_quality(text):
    compact = "".join(ch for ch in str(text or "").strip() if not ch.isspace())
    if not compact:
        return float("-inf")

    compact = compact.replace("Â", "")
    score = 0.0
    upper_compact = compact.upper()
    has_known_slash_unit = (
        "/" in compact
        and bool(re.search(r"(?i)(?:dbfs|hz|ohm|ω|Ω|(?:°|º)?c|degc|[vaw])", compact))
    )
    if has_known_slash_unit:
        score += 3.0
    elif "/" in compact:
        score += 0.3
    if any(marker in compact for marker in ("°", "º")) or "DEG" in compact.upper():
        score += 2.0
    if re.search(r"(?i)(?:ohm|ω|Ω|hz|db|%|[cfk]|[vaw])", compact):
        score += 1.0
    if "DBFS" in upper_compact:
        score += 1.5
    if _is_simple_unit_text(compact):
        score += 3.0
    if len(compact) <= 8:
        score += 0.3
    if re.fullmatch(r"[A-Za-z]{1,3}", compact) and "/" not in compact and not _is_simple_unit_text(compact):
        score -= 1.2
    return score


def _normalize_repeated_unit_column_noise(matrix, final_rows):
    if not matrix or not final_rows:
        return

    header_types = (CLASS_NAME_COLUMN_HEADER, CLASS_NAME_SPANNING_COLUMN_HEADER)
    body_row_indices = [
        idx for idx, row in enumerate(final_rows)
        if row["type"] == CLASS_NAME_ROW
    ]
    if not body_row_indices:
        return

    num_cols = max((len(row) for row in matrix), default=0)
    for c_idx in range(num_cols):
        header_text = " ".join(
            str(matrix[r_idx][c_idx]).strip()
            for r_idx, row in enumerate(final_rows)
            if row["type"] in header_types and c_idx < len(matrix[r_idx]) and str(matrix[r_idx][c_idx]).strip()
        ).lower()
        if "unit" not in header_text:
            inferred_header_bits = []
            for r_idx in range(min(len(matrix), 4)):
                if c_idx >= len(matrix[r_idx]):
                    continue
                cell_text = str(matrix[r_idx][c_idx]).strip()
                if not cell_text:
                    continue
                if final_rows[r_idx]["type"] in header_types or _headerish_row_score(matrix[r_idx]) >= 1.8:
                    inferred_header_bits.append(cell_text.lower())
            header_text = " ".join(inferred_header_bits)
        if "unit" not in header_text:
            continue

        observed = []
        for r_idx in body_row_indices:
            if c_idx >= len(matrix[r_idx]):
                continue
            text = str(matrix[r_idx][c_idx]).strip()
            if not text:
                continue
            observed.append(
                {
                    "row": r_idx,
                    "text": text,
                    "family": _unit_family_key(text),
                    "quality": _unit_text_quality(text),
                }
            )

        if len(observed) < 3:
            continue

        exemplar = max(observed, key=lambda entry: (entry["quality"], len(entry["text"])))
        exemplar_family = exemplar["family"]
        if not exemplar_family or exemplar["quality"] < 4.0 or "/" not in exemplar["text"]:
            continue
        exemplar_text = exemplar["text"]
        if exemplar_family == "C/W":
            exemplar_text = "°C/W"

        strong_families = {
            entry["family"]
            for entry in observed
            if entry["quality"] >= 2.0 or _is_simple_unit_text(entry["text"])
        }
        strong_families.discard("")
        if any(family != exemplar_family for family in strong_families):
            continue

        weak_outliers = [
            entry
            for entry in observed
            if entry["family"] != exemplar_family and entry["quality"] < 2.0
        ]
        if not weak_outliers:
            continue

        for entry in observed:
            if entry["family"] == exemplar_family or entry["quality"] < 2.0:
                matrix[entry["row"]][c_idx] = exemplar_text


def _normalize_temperature_range_cells(matrix):
    if not matrix:
        return

    temp_re = re.compile(
        r"(?P<a>[+\-−–—]?\d{1,3})\s*(?:[°º]\s*C?|C)?\s*to\s*(?P<b>[+\-−–—]?\d{1,3})\s*(?:[°º]\s*C?|C)?",
        re.IGNORECASE,
    )

    def _norm_num(raw):
        return str(raw).replace("−", "-").replace("–", "-").replace("—", "-").strip()

    for row in matrix:
        for c_idx, cell in enumerate(row):
            text = str(cell).strip()
            if not text or "to" not in text.lower():
                continue
            spaced_style = bool(re.search(r"\d\s+[°ºC]", text) or re.search(r"[°º]\s*C", text))

            def _replace(match):
                left = _norm_num(match.group("a"))
                right = _norm_num(match.group("b"))
                if spaced_style:
                    return f"{left} °C to {right} °C"
                return f"{left}°C to {right}°C"

            fixed = temp_re.sub(_replace, text)
            if fixed != text:
                row[c_idx] = fixed


def _normalize_dash_placeholders(matrix):
    if not matrix:
        return

    dash_chars = {"−", "–", "—", "‑"}
    for row in matrix:
        for c_idx, cell in enumerate(row):
            text = str(cell).strip()
            if text in dash_chars:
                row[c_idx] = "-"


def _normalize_unbalanced_parenthetical_cells(matrix):
    if not matrix:
        return

    for row in matrix:
        for c_idx, cell in enumerate(row):
            text = str(cell).strip()
            if not text or len(text) > 96:
                continue
            if text.count("(") != text.count(")") + 1:
                continue
            if text.endswith(")"):
                continue
            if _alpha_frac(text) < 0.15:
                continue
            row[c_idx] = text + ")"

def _column_stats(matrix, row_types, col_idx):
    body_types = (CLASS_NAME_ROW, CLASS_NAME_ROW_HEADER)
    values = [
        str(matrix[r][col_idx]).strip()
        for r, rtype in enumerate(row_types)
        if rtype in body_types and str(matrix[r][col_idx]).strip()
    ]
    if not values:
        return {"avg_len": 0.0, "alpha": 0.0, "digit": 0.0, "nonempty": 0}
    lengths = [len(v) for v in values]
    alpha = sum(_alpha_frac(v) for v in values) / len(values)
    digit = sum(_digit_frac(v) for v in values) / len(values)
    return {
        "avg_len": sum(lengths) / len(lengths),
        "alpha": alpha,
        "digit": digit,
        "nonempty": len(values),
    }


def _infer_header_and_units(matrix, header_idx, row_types):
    """Minimal post-processing: only preserve OCR text as-is.
    No longer overwrites headers with inferred labels."""
    pass

def _headerish_row_score(row):
    if not row:
        return float('-inf')

    non_empty = [str(cell).strip() for cell in row if str(cell).strip()]
    if len(non_empty) < max(2, len(row) // 3):
        return float('-inf')

    alpha = sum(_alpha_frac(cell) for cell in non_empty) / len(non_empty)
    digit = sum(_digit_frac(cell) for cell in non_empty) / len(non_empty)
    shortish = sum(1 for cell in non_empty if len(cell) <= 18) / len(non_empty)
    compact = sum(1 for cell in non_empty if len(cell) <= 24) / len(non_empty)
    first_text = str(row[0]).strip() if row else ''
    first_bonus = 0.8 if first_text and _alpha_frac(first_text) >= 0.18 else 0.0
    return 1.4 * alpha + 1.0 * shortish + 0.8 * compact + first_bonus - 1.0 * digit


def _labelish_column_score(matrix, col_idx):
    col_cells = []
    for row in matrix[:min(len(matrix), 8)]:
        if col_idx < len(row):
            cell = str(row[col_idx]).strip()
            if cell:
                col_cells.append(cell)
    if len(col_cells) < max(2, min(len(matrix), 8) // 3):
        return float('-inf')

    alpha = sum(_alpha_frac(cell) for cell in col_cells) / len(col_cells)
    digit = sum(_digit_frac(cell) for cell in col_cells) / len(col_cells)
    longish = sum(1 for cell in col_cells if len(cell) >= 10) / len(col_cells)
    compact = sum(1 for cell in col_cells if len(cell) <= 18) / len(col_cells)
    return 1.3 * alpha + 1.1 * longish + 0.4 * compact - 0.9 * digit
def _looks_like_wrapped_label_fragment(text):
    text = str(text or "").strip()
    if not text:
        return False
    compact = "".join(ch for ch in text if not ch.isspace())
    if 0 < len(compact) <= 2 and _alpha_frac(compact) >= 0.8:
        return True
    if len(text) > 56:
        return False
    if _alpha_frac(text) < 0.25 or _digit_frac(text) > 0.30:
        return False
    if _looks_like_symbol_code(text):
        return False
    if " " not in text and text.isalpha() and text.islower() and len(compact) <= 3:
        return False

    first_alpha = next((ch for ch in text if ch.isalpha()), "")
    if first_alpha and first_alpha.islower():
        if any(ch.isspace() for ch in text) or any(ch in text for ch in "([/-") or len(compact) >= 4:
            return True

    return text[0] in "([/-"


def _looks_like_symbol_code(text):
    text = str(text or "").strip()
    if not text:
        return False
    if len(text) > 28:
        return False
    if _is_numericish_text(text) or _is_simple_unit_text(text):
        return False

    compact = "".join(ch for ch in text if not ch.isspace())
    tokens = [token for token in re.split(r"\s+", text) if token]
    if not compact or not tokens or len(tokens) > 3:
        return False

    alpha = _alpha_frac(text)
    digit = _digit_frac(text)
    if alpha < 0.25 or digit > 0.45:
        return False

    alpha_chars = [ch for ch in compact if ch.isalpha()]
    if not alpha_chars:
        return False

    upper_like = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
    if any(ch in compact for ch in "_/()[]-"):
        return True
    return upper_like >= 0.45 and all(len(token) <= 16 for token in tokens)


def _append_label_fragment(base_text, extra_text):
    base = str(base_text or "").strip()
    extra = str(extra_text or "").strip()
    if not extra:
        return base
    if not base:
        return extra

    joiner = ""
    if not base.endswith("-"):
        joiner = "" if (len(extra) <= 2 and extra.isalpha() and base[-1:].isalpha()) else " "
    return f"{base}{joiner}{extra}".strip()


def _share_row_label_family(text_a, text_b):
    key_a = _label_key(text_a)
    key_b = _label_key(text_b)
    if not key_a or not key_b:
        return False

    shared_prefix = os.path.commonprefix([key_a, key_b])
    if len(shared_prefix) >= 6:
        return True

    tokens_a = {_label_key(token) for token in re.split(r"[^A-Za-z0-9]+", str(text_a or "")) if len(_label_key(token)) >= 3}
    tokens_b = {_label_key(token) for token in re.split(r"[^A-Za-z0-9]+", str(text_b or "")) if len(_label_key(token)) >= 3}
    return len(tokens_a & tokens_b) >= 2


def _is_standalone_mapping_row(row):
    if not row or len(row) < 2:
        return False

    first = str(row[0]).strip()
    second = str(row[1]).strip()
    if not first or not second:
        return False
    if any(str(cell).strip() for cell in row[2:]):
        return False

    compact = "".join(ch for ch in first if not ch.isspace())
    if not compact or len(compact) > 10:
        return False
    if len(_label_key(second)) < 4:
        return False
    if _digit_frac(second) > 0.30:
        return False

    # Short IDs, symbols, or pin numbers paired with a definition are already
    # complete logical rows and should not be collapsed as wrapped fragments.
    return (
        compact.isdigit()
        or _looks_like_symbol_code(first)
        or len(compact) <= 4
    )


def _looks_like_compact_grid_text(text):
    text = str(text or "").strip()
    if not text:
        return False
    if len(text) > 18:
        return False
    if _is_numericish_text(text) or _is_simple_unit_text(text) or _looks_like_symbol_code(text):
        return True
    if any(not ch.isalnum() and not ch.isspace() for ch in text):
        return True
    return (" " not in text) and _alpha_frac(text) >= 0.35 and _digit_frac(text) <= 0.45


def _should_keep_distinct_grid_rows(prev_row, cur_row):
    if not prev_row or not cur_row:
        return False
    if max(len(prev_row), len(cur_row)) < 4:
        return False

    prev_vals = [str(cell).strip() for cell in prev_row if str(cell).strip()]
    cur_vals = [str(cell).strip() for cell in cur_row if str(cell).strip()]
    if len(prev_vals) < 4 or len(cur_vals) < 4:
        return False

    prev_payload = [idx for idx in range(2, len(prev_row)) if str(prev_row[idx]).strip()]
    cur_payload = [idx for idx in range(2, len(cur_row)) if str(cur_row[idx]).strip()]
    if len(prev_payload) < 3 or len(cur_payload) < 3:
        return False

    shared = sorted(set(prev_payload) & set(cur_payload))
    if len(shared) < 3:
        return False

    diff_count = 0
    for idx in shared:
        if str(prev_row[idx]).strip() != str(cur_row[idx]).strip():
            diff_count += 1
    if diff_count < 2:
        return False

    prev_compact = sum(1 for cell in prev_vals if _looks_like_compact_grid_text(cell)) / len(prev_vals)
    cur_compact = sum(1 for cell in cur_vals if _looks_like_compact_grid_text(cell)) / len(cur_vals)
    return min(prev_compact, cur_compact) >= 0.7


def _row_looks_like_compact_grid(row):
    if not row:
        return False
    vals = [str(cell).strip() for cell in row if str(cell).strip()]
    if len(vals) < 4:
        return False
    compact = sum(1 for cell in vals if _looks_like_compact_grid_text(cell)) / len(vals)
    return compact >= 0.7


def _row_has_measurement_tail(row, start_idx=2):
    if not row or len(row) <= start_idx:
        return False

    tail_vals = [str(row[idx]).strip() for idx in range(start_idx, len(row)) if str(row[idx]).strip()]
    if len(tail_vals) < 2:
        return False

    numericish = sum(1 for text in tail_vals if _is_numericish_text(text))
    unitish = sum(1 for text in tail_vals if _is_simple_unit_text(text))
    return numericish >= 2 or (numericish >= 1 and (len(tail_vals) >= 3 or unitish >= 1))


def _merge_wrapped_first_column_rows(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return matrix, final_rows

    merged_matrix = []
    merged_rows = []

    row_count = len(matrix)
    for row_idx, (row, row_meta) in enumerate(zip(matrix, final_rows)):
        row_copy = list(row)
        row_meta_copy = dict(row_meta)

        if merged_matrix:
            prev_row = merged_matrix[-1]
            prev_meta = merged_rows[-1]
            non_empty_cells = [idx for idx, cell in enumerate(row_copy) if str(cell).strip()]
            prev_first = str(prev_row[0]).strip() if prev_row else ""
            cur_first = str(row_copy[0]).strip() if row_copy else ""
            prev_second = str(prev_row[1]).strip() if len(prev_row) > 1 else ""
            cur_second = str(row_copy[1]).strip() if len(row_copy) > 1 else ""
            next_row = matrix[row_idx + 1] if (row_idx + 1) < row_count and final_rows[row_idx + 1]["type"] == CLASS_NAME_ROW else None
            next_first = str(next_row[0]).strip() if next_row else ""
            prev_has_payload = any(str(cell).strip() for cell in prev_row[1:])
            cur_has_tail_payload = any(str(cell).strip() for cell in row_copy[2:])
            prev_measurement_tail = _row_has_measurement_tail(prev_row, 2)
            cur_measurement_tail = _row_has_measurement_tail(row_copy, 2)

            if (
                prev_meta["type"] == CLASS_NAME_ROW
                and row_meta_copy["type"] in (CLASS_NAME_ROW, CLASS_NAME_ROW_HEADER)
                and non_empty_cells == [0]
                and prev_first
                and prev_has_payload
                and _looks_like_wrapped_label_fragment(cur_first)
                and (
                    not prev_measurement_tail
                    or len(_label_key(cur_first)) <= 3
                    or cur_first[0] in "([/-"
                    or (
                        cur_first[:1].islower()
                        and len(_label_key(cur_first)) <= 18
                    )
                )
            ):
                if prev_measurement_tail and next_first and _share_row_label_family(prev_first, next_first):
                    merged_matrix.append(row_copy)
                    merged_rows.append(row_meta_copy)
                    continue

                joiner = ""
                if not prev_first.endswith("-"):
                    joiner = "" if (len(cur_first) <= 2 and cur_first.isalpha() and prev_first[-1:].isalpha()) else " "
                prev_row[0] = f"{prev_first}{joiner}{cur_first}".strip()

                if "box" in prev_meta and "box" in row_meta_copy:
                    prev_box = prev_meta["box"]
                    cur_box = row_meta_copy["box"]
                    merged_box = [
                        min(prev_box[0], cur_box[0]),
                        min(prev_box[1], cur_box[1]),
                        max(prev_box[2], cur_box[2]),
                        max(prev_box[3], cur_box[3]),
                    ]
                    prev_meta = dict(prev_meta)
                    prev_meta["box"] = merged_box
                    prev_meta["y"] = (merged_box[1] + merged_box[3]) / 2
                    merged_rows[-1] = prev_meta
                continue

            if (
                prev_meta["type"] == CLASS_NAME_ROW
                and row_meta_copy["type"] == CLASS_NAME_ROW
                and non_empty_cells
                and all(idx <= 1 for idx in non_empty_cells)
                and prev_has_payload
                and cur_second
                and _looks_like_continuation_text(cur_second)
                and (
                    not cur_first
                    or _looks_like_wrapped_label_fragment(cur_first)
                    or len("".join(ch for ch in cur_first if not ch.isspace())) <= 6
                )
            ):
                if _is_standalone_mapping_row(prev_row) and _is_standalone_mapping_row(row_copy):
                    prev_first_key = _mapping_row_lead_key(prev_first)
                    cur_first_key = _mapping_row_lead_key(cur_first)
                    if prev_first_key and cur_first_key and prev_first_key != cur_first_key:
                        merged_matrix.append(row_copy)
                        merged_rows.append(row_meta_copy)
                        continue

                    prev_second_key = _label_key(prev_second)
                    cur_second_key = _label_key(cur_second)
                    if (
                        prev_second_key
                        and cur_second_key
                        and cur_second_key != prev_second_key
                        and cur_second_key not in prev_second_key
                        and prev_second_key not in cur_second_key
                    ):
                        merged_matrix.append(row_copy)
                        merged_rows.append(row_meta_copy)
                        continue

                if (
                    len(prev_row) <= 2
                    and len(row_copy) <= 2
                    and prev_first
                    and cur_first
                    and prev_second
                    and cur_second
                ):
                    prev_first_key = _mapping_row_lead_key(prev_first)
                    cur_first_key = _mapping_row_lead_key(cur_first)
                    if (
                        prev_first_key
                        and cur_first_key
                        and prev_first_key != cur_first_key
                        and _looks_like_compact_grid_text(prev_second)
                        and _looks_like_compact_grid_text(cur_second)
                    ):
                        merged_matrix.append(row_copy)
                        merged_rows.append(row_meta_copy)
                        continue

                if cur_first:
                    joiner = ""
                    if not prev_first.endswith("-"):
                        joiner = "" if (len(cur_first) <= 2 and cur_first.isalpha() and prev_first[-1:].isalpha()) else " "
                    prev_row[0] = f"{prev_first}{joiner}{cur_first}".strip() if prev_first else cur_first
                if cur_second:
                    prev_row[1] = _append_continuation_text(prev_second, cur_second) if len(prev_row) > 1 else cur_second

                if "box" in prev_meta and "box" in row_meta_copy:
                    prev_box = prev_meta["box"]
                    cur_box = row_meta_copy["box"]
                    merged_box = [
                        min(prev_box[0], cur_box[0]),
                        min(prev_box[1], cur_box[1]),
                        max(prev_box[2], cur_box[2]),
                        max(prev_box[3], cur_box[3]),
                    ]
                    prev_meta = dict(prev_meta)
                    prev_meta["box"] = merged_box
                    prev_meta["y"] = (merged_box[1] + merged_box[3]) / 2
                    merged_rows[-1] = prev_meta
                continue

            if (
                prev_meta["type"] == CLASS_NAME_ROW
                and row_meta_copy["type"] == CLASS_NAME_ROW
                and prev_first
                and cur_first
                and prev_has_payload
                and cur_has_tail_payload
                and not (prev_measurement_tail and cur_measurement_tail)
                and _looks_like_wrapped_label_fragment(cur_first)
                and (
                    not cur_second
                    or _label_key(cur_second) == _label_key(prev_second)
                )
            ):
                if _should_keep_distinct_grid_rows(prev_row, row_copy):
                    merged_matrix.append(row_copy)
                    merged_rows.append(row_meta_copy)
                    continue

                same_first_key = _label_key(prev_first) and _label_key(prev_first) == _label_key(cur_first)
                prev_numeric_tail = any(
                    _is_numericish_text(str(prev_row[idx]).strip())
                    for idx in range(2, len(prev_row))
                )
                cur_numeric_tail = any(
                    _is_numericish_text(str(row_copy[idx]).strip())
                    for idx in range(2, len(row_copy))
                )
                if same_first_key and prev_numeric_tail and cur_numeric_tail:
                    merged_matrix.append(row_copy)
                    merged_rows.append(row_meta_copy)
                    continue

                joiner = ""
                if not prev_first.endswith("-"):
                    joiner = "" if (len(cur_first) <= 2 and cur_first.isalpha() and prev_first[-1:].isalpha()) else " "
                prev_row[0] = f"{prev_first}{joiner}{cur_first}".strip()

                can_merge = True
                for idx in range(2, len(row_copy)):
                    cur_text = str(row_copy[idx]).strip()
                    prev_text = str(prev_row[idx]).strip()
                    if not cur_text:
                        continue
                    if prev_text and _label_key(prev_text) == _label_key(cur_text):
                        continue
                    if prev_text:
                        if _is_numericish_text(cur_text):
                            can_merge = False
                            break
                        prev_row[idx] = _append_continuation_text(prev_text, cur_text)
                        continue
                    if _looks_like_continuation_text(cur_text) or _is_simple_unit_text(cur_text):
                        prev_row[idx] = cur_text
                        continue
                    can_merge = False
                    break

                if can_merge:
                    if "box" in prev_meta and "box" in row_meta_copy:
                        prev_box = prev_meta["box"]
                        cur_box = row_meta_copy["box"]
                        merged_box = [
                            min(prev_box[0], cur_box[0]),
                            min(prev_box[1], cur_box[1]),
                            max(prev_box[2], cur_box[2]),
                            max(prev_box[3], cur_box[3]),
                        ]
                        prev_meta = dict(prev_meta)
                        prev_meta["box"] = merged_box
                        prev_meta["y"] = (merged_box[1] + merged_box[3]) / 2
                        merged_rows[-1] = prev_meta
                    continue

            if (
                prev_meta["type"] == CLASS_NAME_ROW
                and row_meta_copy["type"] == CLASS_NAME_ROW
                and prev_first
                and cur_first
                and prev_has_payload
                and cur_has_tail_payload
                and len(row_copy) > 1
                and not str(row_copy[1]).strip()
                and _looks_like_wrapped_label_fragment(cur_first)
            ):
                joiner = ""
                if not prev_first.endswith("-"):
                    joiner = "" if (len(cur_first) <= 2 and cur_first.isalpha() and prev_first[-1:].isalpha()) else " "
                prev_row[0] = f"{prev_first}{joiner}{cur_first}".strip()
                row_copy[0] = ""

        merged_matrix.append(row_copy)
        merged_rows.append(row_meta_copy)

    return merged_matrix, merged_rows


def _looks_like_heading_style_row(row):
    if not row:
        return False

    non_empty = [idx for idx, cell in enumerate(row) if str(cell).strip()]
    if non_empty != [0]:
        return False

    text = str(row[0]).strip()
    if not text or len(text) > 42:
        return False
    if _digit_frac(text) > 0.08 or _alpha_frac(text) < 0.55:
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio >= 0.78


def _drop_leading_heading_style_rows(matrix, final_rows, final_cols=None, ocr_data=None):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return matrix, final_rows

    trimmed_matrix = list(matrix)
    trimmed_rows = list(final_rows)
    idx = 0
    while idx < len(trimmed_rows):
        if _is_header_row_type(trimmed_rows[idx]["type"]):
            idx += 1
            continue
        if _headerish_row_score(trimmed_matrix[idx]) >= 2.0:
            idx += 1
            continue
        break

    if idx >= len(trimmed_rows):
        return matrix, final_rows

    while idx < len(trimmed_rows):
        row_meta = trimmed_rows[idx]
        row = trimmed_matrix[idx]
        if row_meta["type"] not in (CLASS_NAME_ROW, CLASS_NAME_ROW_HEADER) or not _looks_like_heading_style_row(row):
            break
        if row_meta["type"] == CLASS_NAME_ROW_HEADER:
            break
        lookahead = []
        for next_idx in range(idx + 1, min(len(trimmed_rows), idx + 4)):
            if trimmed_rows[next_idx]["type"] != CLASS_NAME_ROW:
                continue
            lookahead.append(trimmed_matrix[next_idx])

        if len(lookahead) < 2:
            break
        if not all(_row_has_measurement_tail(next_row, 2) for next_row in lookahead[:2]):
            break

        trimmed_matrix.pop(idx)
        trimmed_rows.pop(idx)

    return trimmed_matrix, trimmed_rows


def _drop_adjacent_duplicate_row_header_rows(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return matrix, final_rows

    filtered_matrix = []
    filtered_rows = []
    for row, row_meta in zip(matrix, final_rows):
        if filtered_rows:
            prev_row = filtered_matrix[-1]
            prev_meta = filtered_rows[-1]
            prev_types = {prev_meta["type"], row_meta["type"]}
            prev_first = str(prev_row[0]).strip() if prev_row else ""
            cur_first = str(row[0]).strip() if row else ""
            prev_tail = any(str(cell).strip() for cell in prev_row[1:])
            cur_tail = any(str(cell).strip() for cell in row[1:])
            overlap = calculate_intersection_area(prev_meta["box"], row_meta["box"]) / (
                min(get_box_area(prev_meta["box"]), get_box_area(row_meta["box"])) + 1e-12
            )
            if (
                prev_types <= {CLASS_NAME_ROW, CLASS_NAME_ROW_HEADER}
                and CLASS_NAME_ROW_HEADER in prev_types
                and prev_first
                and prev_first == cur_first
                and not prev_tail
                and not cur_tail
                and overlap > 0.80
            ):
                if prev_meta["type"] == CLASS_NAME_ROW_HEADER and row_meta["type"] == CLASS_NAME_ROW:
                    filtered_rows[-1] = row_meta
                    filtered_matrix[-1] = list(row)
                continue

        filtered_matrix.append(list(row))
        filtered_rows.append(row_meta)

    return filtered_matrix, filtered_rows


def _propagate_repeated_first_column_labels(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return
    if not matrix[0] or len(matrix[0]) < 2:
        return

    r_idx = 0
    row_count = len(matrix)
    while r_idx < row_count:
        if final_rows[r_idx]["type"] != CLASS_NAME_ROW:
            r_idx += 1
            continue

        desc_key = _label_key(matrix[r_idx][1] if len(matrix[r_idx]) > 1 else "")
        if not desc_key:
            r_idx += 1
            continue

        run = [r_idx]
        next_idx = r_idx + 1
        while next_idx < row_count and final_rows[next_idx]["type"] == CLASS_NAME_ROW:
            next_desc_key = _label_key(matrix[next_idx][1] if len(matrix[next_idx]) > 1 else "")
            if next_desc_key != desc_key:
                break
            run.append(next_idx)
            next_idx += 1

        if len(run) >= 2:
            labels = []
            for idx in run:
                first = str(matrix[idx][0]).strip()
                if first:
                    labels.append(first)

            if labels and any(not str(matrix[idx][0]).strip() for idx in run):
                exemplar = max(labels, key=lambda text: (len(_label_key(text)), len(text)))
                exemplar_key = _label_key(exemplar)
                compatible = True
                for label in labels:
                    label_key = _label_key(label)
                    if not label_key:
                        compatible = False
                        break
                    if label_key == exemplar_key or label_key in exemplar_key or exemplar_key in label_key:
                        continue
                    compatible = False
                    break

                if compatible:
                    for idx in run:
                        if str(matrix[idx][0]).strip():
                            continue
                        if any(str(cell).strip() for cell in matrix[idx][2:]):
                            matrix[idx][0] = exemplar

        r_idx = next_idx


def _propagate_nearby_first_column_labels(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return
    if not matrix[0]:
        return

    row_count = len(matrix)

    def _nearby_labels(start_idx, step):
        found = []
        gap = 0
        idx = start_idx + step
        while 0 <= idx < row_count and gap < 4:
            if final_rows[idx]["type"] != CLASS_NAME_ROW:
                break
            first = str(matrix[idx][0]).strip()
            if first:
                found.append((idx, first))
                if len(found) >= 2:
                    break
            idx += step
            gap += 1
        return found

    def _compatible(a, b):
        ak = _label_key(a)
        bk = _label_key(b)
        if not ak or not bk:
            return False
        return ak == bk or ak in bk or bk in ak

    def _has_desc_diversity(entries):
        desc_keys = []
        for idx, _ in entries:
            desc = str(matrix[idx][1]).strip() if len(matrix[idx]) > 1 else ""
            desc_key = _label_key(desc)
            if desc_key:
                desc_keys.append(desc_key)
        return len(set(desc_keys)) >= 2

    for r_idx in range(row_count):
        if final_rows[r_idx]["type"] != CLASS_NAME_ROW:
            continue
        if str(matrix[r_idx][0]).strip():
            continue
        if not any(str(cell).strip() for cell in matrix[r_idx][1:]):
            continue

        prev_labels = _nearby_labels(r_idx, -1)
        next_labels = _nearby_labels(r_idx, 1)
        fill_label = ""

        if (
            prev_labels
            and next_labels
            and _compatible(prev_labels[0][1], next_labels[0][1])
            and _has_desc_diversity([prev_labels[0], next_labels[0]])
        ):
            fill_label = max(
                (prev_labels[0][1], next_labels[0][1]),
                key=lambda text: (len(_label_key(text)), len(text)),
            )
        elif (
            len(prev_labels) >= 2
            and _compatible(prev_labels[0][1], prev_labels[1][1])
            and _has_desc_diversity(prev_labels[:2])
        ):
            fill_label = max(
                (prev_labels[0][1], prev_labels[1][1]),
                key=lambda text: (len(_label_key(text)), len(text)),
            )
        elif (
            len(next_labels) >= 2
            and _compatible(next_labels[0][1], next_labels[1][1])
            and _has_desc_diversity(next_labels[:2])
        ):
            fill_label = max(
                (next_labels[0][1], next_labels[1][1]),
                key=lambda text: (len(_label_key(text)), len(text)),
            )

        if fill_label:
            matrix[r_idx][0] = fill_label


def _propagate_repeated_row_group_cells(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return
    if not matrix[0] or len(matrix[0]) < 3:
        return

    def _compatible_texts(values):
        keys = [_label_key(value) for value in values if _label_key(value)]
        if not keys:
            return False
        exemplar = max(values, key=lambda text: (len(_label_key(text)), len(text)))
        exemplar_key = _label_key(exemplar)
        for key in keys:
            if key == exemplar_key or key in exemplar_key or exemplar_key in key:
                continue
            return False
        return True

    row_count = len(matrix)
    r_idx = 0
    while r_idx < row_count:
        if final_rows[r_idx]["type"] != CLASS_NAME_ROW:
            r_idx += 1
            continue

        first = str(matrix[r_idx][0]).strip()
        first_key = _label_key(first)
        if not first_key:
            r_idx += 1
            continue

        run = [r_idx]
        next_idx = r_idx + 1
        while next_idx < row_count and final_rows[next_idx]["type"] == CLASS_NAME_ROW:
            next_first_key = _label_key(str(matrix[next_idx][0]).strip())
            if next_first_key != first_key:
                break
            run.append(next_idx)
            next_idx += 1

        if len(run) >= 2:
            for c_idx in (1, len(matrix[0]) - 1):
                if c_idx <= 0 or c_idx >= len(matrix[0]):
                    continue
                values = [str(matrix[idx][c_idx]).strip() for idx in run if str(matrix[idx][c_idx]).strip()]
                if not values or not any(not str(matrix[idx][c_idx]).strip() for idx in run):
                    continue
                if any(_is_numericish_text(value) for value in values):
                    continue
                if not _compatible_texts(values):
                    continue
                exemplar = max(values, key=lambda text: (len(_label_key(text)), len(text)))
                for idx in run:
                    if not str(matrix[idx][c_idx]).strip():
                        matrix[idx][c_idx] = exemplar

        r_idx = next_idx


def _header_family_keys_for_cols(matrix, final_rows):
    num_cols = max((len(row) for row in matrix), default=0)
    families = [""] * num_cols
    header_indices = [idx for idx, row in enumerate(final_rows) if _is_header_row_type(row["type"])]
    if not header_indices:
        return families

    for c_idx in range(num_cols):
        texts = []
        for r_idx in header_indices[-2:]:
            if c_idx >= len(matrix[r_idx]):
                continue
            text = str(matrix[r_idx][c_idx]).strip()
            if text:
                texts.append(text)
        if not texts:
            continue
        key = _label_key(texts[-1])
        if not key:
            continue
        families[c_idx] = re.sub(r"[^a-z]+", "", key)
    return families


def _looks_like_repeatable_grid_phrase(text):
    text = str(text or "").strip()
    if not text:
        return False
    if len(text) < 4 or len(text) > 32:
        return False
    if _is_numericish_text(text) or _is_simple_unit_text(text):
        return False
    if _looks_like_symbol_code(text):
        return False
    return _alpha_frac(text) >= 0.35


def _propagate_adjacent_sibling_grid_cells(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return
    if not matrix[0] or len(matrix[0]) < 3:
        return

    families = _header_family_keys_for_cols(matrix, final_rows)
    row_count = len(matrix)

    for r_idx in range(row_count):
        if final_rows[r_idx]["type"] != CLASS_NAME_ROW:
            continue

        row = matrix[r_idx]
        non_empty = [str(cell).strip() for cell in row if str(cell).strip()]
        if len(non_empty) < 5:
            continue

        repeated = {}
        for text in non_empty:
            key = _label_key(text)
            if key:
                repeated[key] = repeated.get(key, 0) + 1

        for c_idx in range(len(row) - 1):
            left = str(row[c_idx]).strip()
            right = str(row[c_idx + 1]).strip()
            if bool(left) == bool(right):
                continue

            family_a = families[c_idx] if c_idx < len(families) else ""
            family_b = families[c_idx + 1] if (c_idx + 1) < len(families) else ""
            shared_prefix = os.path.commonprefix([family_a, family_b]) if family_a and family_b else ""
            if not shared_prefix and family_a != family_b:
                continue
            if shared_prefix and len(shared_prefix) < 3:
                continue

            source = left or right
            if not _looks_like_repeatable_grid_phrase(source):
                continue
            if repeated.get(_label_key(source), 0) < 2:
                continue

            if not left:
                row[c_idx] = source
            else:
                row[c_idx + 1] = source


def _propagate_shared_blank_first_descriptions(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return
    if not matrix[0] or len(matrix[0]) < 2:
        return

    for r_idx in range(len(matrix) - 1):
        cur_row = final_rows[r_idx]
        next_row = final_rows[r_idx + 1]
        if cur_row["type"] != CLASS_NAME_ROW or next_row["type"] != CLASS_NAME_ROW:
            continue

        cur_first = str(matrix[r_idx][0]).strip()
        next_first = str(matrix[r_idx + 1][0]).strip()
        if cur_first or next_first:
            continue

        cur_desc = str(matrix[r_idx][1]).strip() if len(matrix[r_idx]) > 1 else ""
        next_desc = str(matrix[r_idx + 1][1]).strip() if len(matrix[r_idx + 1]) > 1 else ""
        if not cur_desc or not next_desc:
            continue

        cur_key = _label_key(cur_desc)
        next_key = _label_key(next_desc)
        if not cur_key or not next_key:
            continue
        if cur_key == next_key:
            continue

        if cur_key in next_key:
            longer = next_desc if len(next_key) > len(cur_key) else cur_desc
        elif next_key in cur_key:
            longer = cur_desc if len(cur_key) >= len(next_key) else next_desc
        else:
            continue

        cur_payload = any(str(cell).strip() for cell in matrix[r_idx][2:])
        next_payload = any(str(cell).strip() for cell in matrix[r_idx + 1][2:])
        if cur_payload and next_payload:
            matrix[r_idx][1] = longer
            matrix[r_idx + 1][1] = longer


def _clear_duplicate_adjacent_header_cells(matrix, final_rows, final_cols, row_assignments, ocr_data):
    if not matrix or not final_rows or len(matrix) != len(final_rows) or not final_cols:
        return

    for r_idx, row_meta in enumerate(final_rows):
        if not _is_header_row_type(row_meta["type"]):
            continue
        row = matrix[r_idx]
        row_items = row_assignments[r_idx] if r_idx < len(row_assignments) else []
        for c_idx in range(min(len(row) - 1, len(final_cols) - 1)):
            left = str(row[c_idx]).strip()
            right = str(row[c_idx + 1]).strip()
            if not left or not right:
                continue
            if _label_key(left) != _label_key(right):
                continue

            if _header_text_has_shared_span_support(
                left,
                row_meta,
                final_cols[c_idx]["box"],
                final_cols[c_idx + 1]["box"],
                row_items,
                ocr_data,
            ):
                continue

            left_support = _cell_text_has_direct_support(
                left,
                row_meta["box"],
                final_cols[c_idx]["box"],
                final_cols,
                c_idx,
                row_items,
                ocr_data,
            )
            right_support = _cell_text_has_direct_support(
                right,
                row_meta["box"],
                final_cols[c_idx + 1]["box"],
                final_cols,
                c_idx + 1,
                row_items,
                ocr_data,
            )
            if not left_support and right_support:
                row[c_idx] = ""


def _header_text_has_shared_span_support(text, row_meta, left_col_box, right_col_box, row_items, ocr_data):
    text_key = _label_key(text)
    if not text_key or not row_meta:
        return False

    for item in row_meta.get("items", []):
        item_box = item.get("box")
        if not item_box:
            continue

        item_w = max(1.0, float(item_box[2] - item_box[0]))
        left_w = max(1.0, float(left_col_box[2] - left_col_box[0]))
        right_w = max(1.0, float(right_col_box[2] - right_col_box[0]))
        inter_left = max(0.0, min(item_box[2], left_col_box[2]) - max(item_box[0], left_col_box[0]))
        inter_right = max(0.0, min(item_box[2], right_col_box[2]) - max(item_box[0], right_col_box[0]))
        if inter_left <= 0.0 or inter_right <= 0.0:
            continue
        if (
            (inter_left / item_w) < 0.12
            and (inter_left / left_w) < 0.20
        ):
            continue
        if (
            (inter_right / item_w) < 0.12
            and (inter_right / right_w) < 0.20
        ):
            continue

        item_text = get_ocr_text_in_box(item_box, row_items, intersection_thresh=0.05)
        if not item_text:
            item_text = get_ocr_text_in_box(item_box, ocr_data, intersection_thresh=0.05)
        if not item_text:
            item_text = get_text_center_in_box(item_box, row_items)
        if not item_text:
            item_text = get_text_center_in_box(item_box, ocr_data)

        item_key = _label_key(item_text)
        if not item_key:
            continue
        if item_key == text_key or item_key in text_key or text_key in item_key:
            return True

    return False


def _realign_singleton_lead_spill(matrix, final_rows, final_cols, row_assignments, ocr_data):
    if not matrix or not final_rows or len(matrix) != len(final_rows) or len(final_cols) < 2:
        return

    row_count = len(matrix)
    for r_idx in range(row_count):
        row_meta = final_rows[r_idx]
        if row_meta["type"] != CLASS_NAME_ROW:
            continue
        row = matrix[r_idx]
        if len(row) < 2:
            continue

        first = str(row[0]).strip()
        second = str(row[1]).strip()
        if first or not second:
            continue
        if not any(str(cell).strip() for cell in row[2:]):
            continue
        if _is_numericish_text(second) or len(_label_key(second)) < 8:
            continue

        prev_ok = (
            r_idx > 0
            and final_rows[r_idx - 1]["type"] == CLASS_NAME_ROW
            and str(matrix[r_idx - 1][0]).strip()
            and not str(matrix[r_idx - 1][1]).strip()
        )
        next_ok = (
            (r_idx + 1) < row_count
            and final_rows[r_idx + 1]["type"] == CLASS_NAME_ROW
            and str(matrix[r_idx + 1][0]).strip()
            and not str(matrix[r_idx + 1][1]).strip()
        )
        if not (prev_ok and next_ok):
            continue

        row_items = row_assignments[r_idx] if r_idx < len(row_assignments) else []
        first_support = _cell_text_has_direct_support(
            second,
            row_meta["box"],
            final_cols[0]["box"],
            final_cols,
            0,
            row_items,
            ocr_data,
        )
        second_support = _cell_text_has_direct_support(
            second,
            row_meta["box"],
            final_cols[1]["box"],
            final_cols,
            1,
            row_items,
            ocr_data,
        )
        if first_support and not second_support:
            row[0] = second
            row[1] = ""


def _cell_text_has_direct_support(text, row_box, col_box, final_cols, c_idx, row_items, ocr_data):
    text_key = _label_key(text)
    if not text_key:
        return False

    logic_cell_box = [col_box[0], row_box[1], col_box[2], row_box[3]]
    sources = []
    if row_items:
        sources.append(row_items)
    if ocr_data is not row_items and ocr_data:
        sources.append(ocr_data)

    for source in sources:
        items = _row_cell_raw_items(logic_cell_box, col_box, final_cols, c_idx, source)
        if not items:
            continue
        items = _sort_ocr_items_reading_order(items)
        joined = " ".join(
            str(item.get("text", "")).strip()
            for item in items
            if str(item.get("text", "")).strip()
        ).strip()
        joined_key = _label_key(joined)
        if not joined_key:
            continue
        if joined_key == text_key or joined_key in text_key or text_key in joined_key:
            return True

        if _looks_like_symbol_code(text):
            token_keys = [_label_key(token) for token in re.split(r"\s+", joined) if _label_key(token)]
            if any(token == text_key or token in text_key or text_key in token for token in token_keys):
                return True

    return False


def _clear_unsupported_repeated_lead_cells(matrix, final_rows, final_cols, row_assignments, ocr_data):
    if not matrix or not final_rows or len(matrix) != len(final_rows) or not final_cols:
        return

    lead_cols = min(2, max((len(row) for row in matrix), default=0))
    if lead_cols == 0:
        return

    row_count = min(len(matrix), len(final_rows))
    for r_idx in range(1, row_count):
        if final_rows[r_idx]["type"] != CLASS_NAME_ROW or final_rows[r_idx - 1]["type"] != CLASS_NAME_ROW:
            continue

        row = matrix[r_idx]
        prev_row = matrix[r_idx - 1]
        if not any(str(cell).strip() for cell in row[lead_cols:]):
            continue
        lead_label = str(row[0]).strip() if len(row) > 0 else ""
        lead_symbol = str(row[1]).strip() if len(row) > 1 else ""
        if not (
            lead_label
            and lead_symbol
            and not _looks_like_symbol_code(lead_label)
            and _looks_like_symbol_code(lead_symbol)
            and len(_label_key(lead_label)) >= 8
        ):
            continue

        row_box = final_rows[r_idx].get("box")
        if not row_box:
            continue
        row_items = row_assignments[r_idx] if r_idx < len(row_assignments) else []

        for c_idx in range(min(lead_cols, len(row), len(prev_row), len(final_cols))):
            cur_text = str(row[c_idx]).strip()
            prev_text = str(prev_row[c_idx]).strip()
            if not cur_text or not prev_text:
                continue

            cur_key = _label_key(cur_text)
            prev_key = _label_key(prev_text)
            if not cur_key or not prev_key:
                continue
            if cur_key != prev_key and cur_key not in prev_key and prev_key not in cur_key:
                continue

            if c_idx == 0 and _looks_like_wrapped_label_fragment(cur_text):
                continue
            if c_idx == 0 and _has_grouped_repeated_first_label_context(matrix, final_rows, r_idx):
                continue

            col_box = final_cols[c_idx]["box"]
            if _cell_text_has_direct_support(cur_text, row_box, col_box, final_cols, c_idx, row_items, ocr_data):
                continue

            row[c_idx] = ""


def _drop_empty_columns(matrix, final_rows, final_cols):
    if not matrix or not final_cols:
        return matrix, final_cols, []

    keep_indices = []
    dropped = []
    header_row_indices = [
        idx
        for idx, row_meta in enumerate(final_rows[: len(matrix)])
        if _is_header_row_type(row_meta.get("type"))
    ]

    for c_idx, col in enumerate(final_cols):
        col_cells = [
            str(row[c_idx]).strip()
            for row in matrix
            if c_idx < len(row)
        ]
        has_any_text = any(col_cells)
        has_header_text = any(
            c_idx < len(matrix[r_idx]) and str(matrix[r_idx][c_idx]).strip()
            for r_idx in header_row_indices
        )

        if has_any_text:
            keep_indices.append(c_idx)
            continue

        dropped.append(
            {
                "index": c_idx,
                "box": col.get("box"),
                "has_header_text": bool(has_header_text),
            }
        )

    if len(keep_indices) == len(final_cols):
        return matrix, final_cols, dropped

    pruned_matrix = [[row[c_idx] for c_idx in keep_indices] for row in matrix]
    pruned_cols = [final_cols[c_idx] for c_idx in keep_indices]
    return pruned_matrix, pruned_cols, dropped


def _has_grouped_repeated_first_label_context(matrix, final_rows, r_idx):
    if not matrix or not final_rows or r_idx >= len(matrix) or len(matrix[r_idx]) < 2:
        return False

    cur_row = matrix[r_idx]
    cur_label = str(cur_row[0]).strip()
    cur_key = _label_key(cur_label)
    if not cur_key:
        return False

    cur_desc_key = _label_key(str(cur_row[1]).strip())
    support = 0
    diverse_desc = set()
    start = max(0, r_idx - 2)
    end = min(len(matrix), r_idx + 3)

    for idx in range(start, end):
        if idx == r_idx or final_rows[idx]["type"] != CLASS_NAME_ROW:
            continue
        neighbor = matrix[idx]
        if not any(str(cell).strip() for cell in neighbor[2:]):
            continue

        neighbor_label = str(neighbor[0]).strip()
        neighbor_key = _label_key(neighbor_label)
        if not neighbor_key:
            continue
        if neighbor_key != cur_key and neighbor_key not in cur_key and cur_key not in neighbor_key:
            continue

        support += 1
        neighbor_desc_key = _label_key(str(neighbor[1]).strip()) if len(neighbor) > 1 else ""
        if neighbor_desc_key and neighbor_desc_key != cur_desc_key:
            diverse_desc.add(neighbor_desc_key)

    return support >= 1 and len(diverse_desc) >= 1


def _trim_body_raw_fallback_box(final_rows, r_idx, logic_cell_box):
    clipped = list(logic_cell_box)
    if r_idx > 0 and final_rows[r_idx - 1]["type"] == CLASS_NAME_ROW_HEADER:
        clipped[1] = max(clipped[1], final_rows[r_idx - 1]["box"][3] + 1)
    if (r_idx + 1) < len(final_rows) and final_rows[r_idx + 1]["type"] == CLASS_NAME_ROW_HEADER:
        clipped[3] = min(clipped[3], final_rows[r_idx + 1]["box"][1] - 2)
    if clipped[3] <= clipped[1]:
        return list(logic_cell_box)
    return clipped


def _normalize_wrapped_body_rows(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return

    row_count = len(matrix)
    for r_idx, row_meta in enumerate(final_rows):
        if row_meta["type"] != CLASS_NAME_ROW or not matrix[r_idx]:
            continue

        row = matrix[r_idx]
        first = str(row[0]).strip()
        if not first or not _looks_like_wrapped_label_fragment(first):
            continue

        second = str(row[1]).strip() if len(row) > 1 else ""
        tail_payload = any(str(cell).strip() for cell in row[2:])
        prev_idx = r_idx - 1
        next_idx = r_idx + 1
        prev_row = matrix[prev_idx] if prev_idx >= 0 and final_rows[prev_idx]["type"] == CLASS_NAME_ROW else None
        next_row = matrix[next_idx] if next_idx < row_count and final_rows[next_idx]["type"] == CLASS_NAME_ROW else None

        # Wrapped fragments like "threshold hysteresis" should keep their own row,
        # but leaked symbol codes must stay with the real symbol row instead.
        if second and not tail_payload and _looks_like_symbol_code(second):
            prev_second = str(prev_row[1]).strip() if prev_row is not None and len(prev_row) > 1 else ""
            next_second = str(next_row[1]).strip() if next_row is not None and len(next_row) > 1 else ""
            second_key = _label_key(second)
            prev_key = _label_key(prev_second)
            next_key = _label_key(next_second)
            if (
                (prev_key and (prev_key in second_key or second_key in prev_key))
                or (next_key and (next_key in second_key or second_key in next_key))
            ):
                row[1] = ""
                second = ""

        # If a wrapped label continues into the next logical row, copy the full
        # label to both rows instead of leaving one row with only a fragment.
        if (
            prev_row is not None
            and tail_payload
            and len(row) > 1
        ):
            prev_first = str(prev_row[0]).strip()
            prev_second = str(prev_row[1]).strip() if len(prev_row) > 1 else ""
            prev_tail = any(str(cell).strip() for cell in prev_row[1:])
            if prev_first and prev_tail:
                combined = _append_label_fragment(prev_first, first)
                if (
                    second
                    and _looks_like_symbol_code(second)
                    and not prev_second
                ):
                    prev_row[0] = combined
                    row[0] = combined
                    continue
                if not second and not prev_second:
                    prev_row[0] = combined
                    row[0] = combined


def _suppress_adjacent_body_text_spill(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return

    for r_idx, row_meta in enumerate(final_rows):
        if row_meta["type"] != CLASS_NAME_ROW:
            continue

        row = matrix[r_idx]
        if _row_looks_like_compact_grid(row):
            continue

        for c_idx in range(1, len(row)):
            prev_text = str(row[c_idx - 1]).strip()
            cur_text = str(row[c_idx]).strip()
            if not prev_text or not cur_text:
                continue
            if _is_numericish_text(prev_text) or _is_simple_unit_text(prev_text):
                continue
            if _is_numericish_text(cur_text) or _is_simple_unit_text(cur_text):
                continue

            prev_key = _label_key(prev_text)
            cur_key = _label_key(cur_text)
            if len(prev_key) < 8 or len(cur_key) < 8:
                continue
            if cur_key not in prev_key and prev_key not in cur_key:
                continue

            # Compact categorical truth-table style rows can legitimately repeat
            # the same long phrase across adjacent sibling cells. Do not blank
            # those repeated values just because later payload exists.
            row_non_empty = [str(cell).strip() for cell in row if str(cell).strip()]
            compact_like = sum(1 for text in row_non_empty if _looks_like_compact_grid_text(text))
            repeated_count = sum(1 for text in row_non_empty if _label_key(text) == cur_key)
            if (
                prev_key == cur_key
                and _looks_like_repeatable_grid_phrase(cur_text)
                and repeated_count >= 3
                and len(row_non_empty) >= 5
                and compact_like >= 3
            ):
                continue

            later_payload = any(str(cell).strip() for cell in row[c_idx + 1:])
            if later_payload or len(cur_key) <= len(prev_key):
                row[c_idx] = ""


def _looks_like_footnote_marker(text):
    text = str(text or "").strip()
    if not text:
        return False
    if re.fullmatch(r"[\(\[]?\d+[\)\]]?", text):
        return True
    return _label_key(text) in {"note", "notes", "footnote", "footnotes"}


def _looks_like_narrative_footnote(text):
    text = str(text or "").strip()
    if len(text) < 40:
        return False
    if _alpha_frac(text) < 0.45 or _digit_frac(text) > 0.12:
        return False
    if _is_numericish_text(text) or _is_simple_unit_text(text) or _looks_like_symbol_code(text):
        return False
    word_count = len(re.findall(r"[A-Za-z]{3,}", text))
    return word_count >= 6


def _drop_trailing_footnote_like_rows(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return matrix, final_rows

    trimmed_matrix = [list(row) for row in matrix]
    trimmed_rows = [dict(row) for row in final_rows]

    while trimmed_matrix:
        idx = len(trimmed_matrix) - 1
        row_meta = trimmed_rows[idx]
        row = trimmed_matrix[idx]
        if row_meta["type"] != CLASS_NAME_ROW:
            break

        first = str(row[0]).strip() if len(row) > 0 else ""
        second = str(row[1]).strip() if len(row) > 1 else ""
        if any(str(cell).strip() for cell in row[2:]):
            break
        if not second or not _looks_like_narrative_footnote(second):
            break
        if first and not _looks_like_footnote_marker(first):
            break

        prev_row = trimmed_matrix[idx - 1] if idx > 0 else None
        if prev_row is None or not _row_has_measurement_tail(prev_row, 2):
            break

        trimmed_matrix.pop()
        trimmed_rows.pop()

    return trimmed_matrix, trimmed_rows


def _looks_like_continuation_text(text):
    text = str(text or "").strip()
    if not text:
        return False
    if _looks_like_wrapped_label_fragment(text):
        return True
    first_char = text[0]
    if first_char.islower():
        return True
    return first_char in "([/-,:;"


def _append_continuation_text(base_text, extra_text):
    base = str(base_text or "").strip()
    extra = str(extra_text or "").strip()
    if not extra:
        return base
    if not base:
        return extra
    if _label_key(base) == _label_key(extra):
        return base

    joiner = "" if base.endswith("-") else " "
    if extra[0] in ",.;:)]%" and base[-1:] not in " ([/-":
        joiner = ""
    return f"{base}{joiner}{extra}".strip()


def _merge_body_continuation_rows(matrix, final_rows):
    if not matrix or not final_rows or len(matrix) != len(final_rows):
        return matrix, final_rows

    merged_matrix = []
    merged_rows = []

    for row, row_meta in zip(matrix, final_rows):
        row_copy = list(row)
        row_meta_copy = dict(row_meta)

        if merged_matrix:
            prev_row = merged_matrix[-1]
            prev_meta = merged_rows[-1]
            if prev_meta["type"] == CLASS_NAME_ROW and row_meta_copy["type"] == CLASS_NAME_ROW:
                lead_cols = min(2, len(row_copy))
                prev_lead = [str(prev_row[i]).strip() for i in range(lead_cols)]
                cur_lead = [str(row_copy[i]).strip() for i in range(lead_cols)]
                cur_payload = [idx for idx in range(lead_cols, len(row_copy)) if str(row_copy[idx]).strip()]
                prev_measurement_tail = _row_has_measurement_tail(prev_row, lead_cols)
                cur_measurement_tail = _row_has_measurement_tail(row_copy, lead_cols)

                key_mode = None
                if cur_payload:
                    if not any(cur_lead):
                        key_mode = "blank"
                    else:
                        key_ok = True
                        for idx in range(lead_cols):
                            cur_text = cur_lead[idx]
                            prev_text = prev_lead[idx]
                            if not cur_text:
                                continue
                            cur_key = _label_key(cur_text)
                            prev_key = _label_key(prev_text)
                            if not prev_key:
                                key_ok = False
                                break
                            if cur_key == prev_key or prev_key.endswith(cur_key) or cur_key.endswith(prev_key):
                                continue
                            key_ok = False
                            break
                        if key_ok:
                            key_mode = "dup"

                if prev_measurement_tail and cur_measurement_tail:
                    key_mode = None

                if key_mode == "dup":
                    # Repeated lead cells often indicate a valid multi-row parameter
                    # block. If the current row still carries numeric measurements,
                    # treat it as its own logical row instead of a wrap fragment.
                    has_numeric_payload = any(
                        _is_numericish_text(str(row_copy[idx]).strip())
                        for idx in cur_payload
                    )
                    if has_numeric_payload:
                        key_mode = None

                if key_mode is not None:
                    merged_any = False
                    can_merge = True
                    for idx in cur_payload:
                        cur_text = str(row_copy[idx]).strip()
                        prev_text = str(prev_row[idx]).strip()
                        if not cur_text:
                            continue
                        if prev_text and _label_key(prev_text) == _label_key(cur_text):
                            row_copy[idx] = ""
                            continue
                        if prev_text:
                            if _is_numericish_text(cur_text):
                                can_merge = False
                                break
                            if not _looks_like_continuation_text(cur_text):
                                can_merge = False
                                break
                            prev_row[idx] = _append_continuation_text(prev_text, cur_text)
                            row_copy[idx] = ""
                            merged_any = True
                            continue
                        if key_mode == "dup" and (_looks_like_continuation_text(cur_text) or _is_simple_unit_text(cur_text)):
                            prev_row[idx] = cur_text
                            row_copy[idx] = ""
                            merged_any = True
                            continue
                        if key_mode == "blank" and _is_simple_unit_text(cur_text):
                            prev_row[idx] = cur_text
                            row_copy[idx] = ""
                            merged_any = True
                            continue
                        can_merge = False
                        break

                    if can_merge and merged_any and not any(str(row_copy[idx]).strip() for idx in cur_payload):
                        if "box" in prev_meta and "box" in row_meta_copy:
                            prev_box = prev_meta["box"]
                            cur_box = row_meta_copy["box"]
                            merged_box = [
                                min(prev_box[0], cur_box[0]),
                                min(prev_box[1], cur_box[1]),
                                max(prev_box[2], cur_box[2]),
                                max(prev_box[3], cur_box[3]),
                            ]
                            prev_meta = dict(prev_meta)
                            prev_meta["box"] = merged_box
                            prev_meta["y"] = (merged_box[1] + merged_box[3]) / 2
                            merged_rows[-1] = prev_meta
                        continue

        merged_matrix.append(row_copy)
        merged_rows.append(row_meta_copy)

    return merged_matrix, merged_rows


def _get_span_text_for_cell(span_box, target_box, ocr_data):
    items = get_ocr_items_in_box(span_box, ocr_data, intersection_thresh=0.05)
    if not items:
        return ""

    def _belongs_to_target(item):
        item_box = o_to_box(item)
        cx, cy = get_box_center(item_box)
        if target_box[0] <= cx <= target_box[2] and target_box[1] <= cy <= target_box[3]:
            return True
        inter = calculate_intersection_area(target_box, item_box)
        return (inter / (get_box_area(item_box) + 1e-12)) > 0.35

    cell_items = [item for item in items if _belongs_to_target(item)]
    if cell_items:
        cell_items = _sort_ocr_items_reading_order(cell_items)
        return " ".join(item["text"] for item in cell_items).strip()

    centers = sorted(get_box_center(o_to_box(item))[0] for item in items)
    gaps = [b - a for a, b in zip(centers, centers[1:])]
    target_w = max(1.0, float(target_box[2] - target_box[0]))
    if len(items) <= 2 or (gaps and max(gaps) < max(24.0, 0.65 * target_w)):
        items = _sort_ocr_items_reading_order(items)
        return " ".join(item["text"] for item in items).strip()

    return ""


def get_filtered_structure_predictions(all_preds_obj, ocr_data):
    preds = all_preds_obj.get("predictions", [])
    all_preds = _collect_structure_predictions(preds)

    if not all_preds:
        return []

    table_box = [
        min(pred["box"][0] for pred in all_preds),
        min(pred["box"][1] for pred in all_preds),
        max(pred["box"][2] for pred in all_preds),
        max(pred["box"][3] for pred in all_preds),
    ]
    relaxed_col_header_preds = _collect_relaxed_detected_predictions(
        preds,
        CLASS_NAME_COLUMN_HEADER,
        table_box,
        min_conf=0.15,
    )
    strong_rows_seed = [
        pred for pred in all_preds
        if pred["class"] == CLASS_NAME_ROW and _prediction_bucket(pred) == "strong"
    ]
    cols_raw = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_COLUMN],
        CLASS_NAME_COLUMN,
        ocr_data,
        table_box,
        support_rows=strong_rows_seed,
        ios_thresh=IOS_THRESH_COLUMNS,
    )
    if cols_raw:
        cols_raw = merge_cols_by_x(cols_raw, x_threshold=10)
        cols_raw = group_nms(cols_raw, IOS_THRESH_COLUMNS)
    rows_raw = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_ROW],
        CLASS_NAME_ROW,
        ocr_data,
        table_box,
        support_cols=cols_raw,
        ios_thresh=IOS_THRESH_ROWS,
    )
    col_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_COLUMN_HEADER],
        CLASS_NAME_COLUMN_HEADER,
        ocr_data,
        table_box,
        support_rows=rows_raw,
        support_cols=cols_raw,
    )
    span_col_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPANNING_COLUMN_HEADER],
        CLASS_NAME_SPANNING_COLUMN_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    row_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_ROW_HEADER],
        CLASS_NAME_ROW_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    table_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_TABLE_HEADER],
        CLASS_NAME_TABLE_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    if not col_headers:
        col_headers = _recover_detected_column_headers(
            relaxed_col_header_preds,
            cols_raw,
            rows_raw,
        )
    col_headers = _split_header_predictions_by_recovery(
        col_headers,
        cols_raw,
        ocr_data,
        structure_gray=None,
        allow_ocr=True,
    )
    span_col_headers = _split_header_predictions_by_recovery(
        span_col_headers,
        cols_raw,
        ocr_data,
        structure_gray=None,
        allow_ocr=False,
    )
    span_col_headers = _filter_valid_spanning_column_headers(span_col_headers, col_headers)
    col_headers, span_col_headers = _resolve_same_band_header_class_conflicts(col_headers, span_col_headers)
    splits = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPLITTED_COLUMN],
        CLASS_NAME_SPLITTED_COLUMN,
        ocr_data,
        table_box,
        support_rows=rows_raw,
        support_cols=cols_raw,
    )
    cols_raw = _promote_orphan_split_columns(
        cols_raw,
        splits,
        rows_raw,
        col_headers + span_col_headers,
        table_box,
    )
    span_row_cells = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPANNING_ROW_CELL],
        CLASS_NAME_SPANNING_ROW_CELL,
        ocr_data,
        table_box,
        support_rows=rows_raw,
        support_cols=cols_raw,
    )
    span_col_cells = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPANNING_COLUMN_CELL],
        CLASS_NAME_SPANNING_COLUMN_CELL,
        ocr_data,
        table_box,
        support_rows=rows_raw,
        support_cols=cols_raw,
    )

    if cols_raw:
        cols_raw = merge_cols_by_x(cols_raw, x_threshold=10)
    if rows_raw:
        rows_raw = merge_rows_by_y(rows_raw, y_threshold=5)

    filtered = []
    for group in (
        cols_raw,
        rows_raw,
        col_headers,
        span_col_headers,
        row_headers,
        table_headers,
        splits,
        span_row_cells,
        span_col_cells,
    ):
        filtered.extend(group)

    return filtered


def get_strict_nms_visual_predictions(all_preds_obj, min_confidence=0.20):
    preds = all_preds_obj.get("predictions", [])
    all_preds = []
    for pred in _collect_structure_predictions(preds):
        if pred["confidence"] < min_confidence:
            continue
        if _prediction_bucket(pred) == "weak":
            continue
        all_preds.append(pred)

    if not all_preds:
        return []

    def _keep_class(cls, ios_thresh=IOS_THRESH_NMS_DEFAULT):
        kept = [pred for pred in all_preds if pred["class"] == cls]
        return group_nms(kept, ios_thresh)

    cols_raw = _keep_class(CLASS_NAME_COLUMN, IOS_THRESH_COLUMNS)
    if cols_raw:
        cols_raw = merge_cols_by_x(cols_raw, x_threshold=10)
        cols_raw = group_nms(cols_raw, IOS_THRESH_COLUMNS)

    rows_raw = _keep_class(CLASS_NAME_ROW, IOS_THRESH_ROWS)
    if rows_raw:
        rows_raw = merge_rows_by_y(rows_raw, y_threshold=5)
        rows_raw = group_nms(rows_raw, IOS_THRESH_ROWS)

    filtered = []
    for group in (
        cols_raw,
        rows_raw,
        _keep_class(CLASS_NAME_COLUMN_HEADER),
        _keep_class(CLASS_NAME_SPANNING_COLUMN_HEADER),
        _keep_class(CLASS_NAME_ROW_HEADER),
        _keep_class(CLASS_NAME_TABLE_HEADER),
        _keep_class(CLASS_NAME_SPLITTED_COLUMN),
        _keep_class(CLASS_NAME_SPANNING_ROW_CELL, IOS_THRESH_SPANS),
        _keep_class(CLASS_NAME_SPANNING_COLUMN_CELL, IOS_THRESH_SPANS),
    ):
        filtered.extend(group)

    return sorted(filtered, key=lambda item: item["confidence"], reverse=True)




def _get_unassigned_ocr_items(ocr_data, active_boxes, intersection_thresh=0.2):
    unassigned = []
    for item in ocr_data or []:
        item_box = o_to_box(item)
        item_area = get_box_area(item_box) + 1e-12
        assigned = False
        for box in active_boxes:
            inter = calculate_intersection_area(item_box, box)
            if (inter / item_area) >= intersection_thresh:
                assigned = True
                break
        if not assigned:
            unassigned.append(item)
    return unassigned


def _box_center_inside(box, outer_box, pad=0.0):
    if not box or not outer_box:
        return False
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return (
        float(outer_box[0]) - pad <= cx <= float(outer_box[2]) + pad
        and float(outer_box[1]) - pad <= cy <= float(outer_box[3]) + pad
    )


def _matrix_missing_ocr_items(ocr_data, matrix, final_rows, final_cols, table_box):
    if not ocr_data or not matrix or not final_rows or not final_cols:
        return []

    missing = []
    for item in ocr_data:
        text = str(item.get("text", "")).strip()
        if not _label_key(text):
            continue
        ibox = o_to_box(item)
        if table_box and not _box_center_inside(ibox, table_box, pad=2.0):
            continue

        item_cx, item_cy = get_box_center(ibox)
        assigned_here = False
        for r_idx, row in enumerate(final_rows[: len(matrix)]):
            if r_idx >= len(matrix):
                break
            row_box = row.get("box")
            if not row_box:
                continue
            inter_h = max(0.0, min(float(ibox[3]), float(row_box[3])) - max(float(ibox[1]), float(row_box[1])))
            item_h = max(1.0, float(ibox[3] - ibox[1]))
            if not (row_box[1] <= item_cy <= row_box[3] or inter_h / item_h >= 0.35):
                continue

            c_idx = _preferred_col_idx_for_item(ibox, final_cols)
            candidate_cols = [c_idx] if c_idx is not None else []
            if c_idx is None:
                candidate_cols = [
                    idx for idx, col in enumerate(final_cols)
                    if col["box"][0] <= item_cx <= col["box"][2]
                ]
            for col_idx in candidate_cols:
                if col_idx is None or col_idx < 0 or col_idx >= len(final_cols):
                    continue
                if col_idx >= len(matrix[r_idx]) or not str(matrix[r_idx][col_idx]).strip():
                    continue
                cell_box = [
                    final_cols[col_idx]["box"][0],
                    row_box[1],
                    final_cols[col_idx]["box"][2],
                    row_box[3],
                ]
                if _box_center_inside(ibox, cell_box, pad=1.0):
                    assigned_here = True
                    break
                inter_area = calculate_intersection_area(ibox, cell_box)
                if inter_area / (get_box_area(ibox) + 1e-12) >= 0.60:
                    assigned_here = True
                    break
            if assigned_here:
                break

        if not assigned_here:
            missing.append(item)

    return missing

def _gap_has_separator_line(structure_gray, gap_box, orientation="vertical"):
    lines = _gap_separator_lines(structure_gray, gap_box, orientation=orientation)
    if not lines:
        return None
    return max(lines, key=lambda item: item[1])


def _gap_separator_lines(structure_gray, gap_box, orientation="vertical"):
    if structure_gray is None or structure_gray.ndim != 2:
        return []
    img_h, img_w = structure_gray.shape[:2]
    
    x1 = max(0, int(round(gap_box[0])))
    y1 = max(0, int(round(gap_box[1])))
    x2 = min(img_w, int(round(gap_box[2])))
    y2 = min(img_h, int(round(gap_box[3])))
    
    if x2 <= x1 or y2 <= y1:
        return []
        
    strip = structure_gray[y1:y2, x1:x2]
    if strip.size == 0:
        return []

    if orientation == "vertical":
        dark_lines = (strip <= STRUCTURE_LINE_DARK_PIXEL_MAX).T
        min_ratio = 0.20
        offset = x1
    else:
        dark_lines = strip <= STRUCTURE_LINE_DARK_PIXEL_MAX
        min_ratio = 0.40
        offset = y1

    lines = []
    run_start = None
    run_ratios = []
    for idx, dark_line in enumerate(dark_lines):
        ratio = float(np.mean(dark_line))
        if ratio >= min_ratio:
            if run_start is None:
                run_start = idx
                run_ratios = []
            run_ratios.append(ratio)
        elif run_start is not None:
            run_end = idx - 1
            best_rel = int(np.argmax(run_ratios))
            best_idx = run_start + best_rel
            lines.append((offset + best_idx, run_ratios[best_rel]))
            run_start = None
            run_ratios = []

    if run_start is not None:
        best_rel = int(np.argmax(run_ratios))
        best_idx = run_start + best_rel
        lines.append((offset + best_idx, run_ratios[best_rel]))

    return lines


def _gap_horizontal_separator_lines_any_column(structure_gray, gap_box, active_cols):
    if not active_cols:
        return _gap_separator_lines(structure_gray, gap_box, "horizontal")

    candidates = []
    for col in active_cols:
        col_box = col.get("box") if isinstance(col, dict) else None
        if not col_box:
            continue
        col_x1 = max(float(gap_box[0]), float(col_box[0]))
        col_x2 = min(float(gap_box[2]), float(col_box[2]))
        if col_x2 - col_x1 < 3:
            continue
        col_gap_box = [col_x1, gap_box[1], col_x2, gap_box[3]]
        candidates.extend(_gap_separator_lines(structure_gray, col_gap_box, "horizontal"))

    if not candidates:
        return _gap_separator_lines(structure_gray, gap_box, "horizontal")

    merged = []
    for y_pos, ratio in sorted(candidates, key=lambda item: item[0]):
        if not merged or abs(y_pos - merged[-1]["best_y"]) > 2:
            merged.append({"best_y": y_pos, "best_ratio": ratio})
            continue
        if ratio > merged[-1]["best_ratio"]:
            merged[-1]["best_y"] = y_pos
            merged[-1]["best_ratio"] = ratio

    return [(item["best_y"], item["best_ratio"]) for item in merged]



def _gap_horizontal_separator_has_real_line_any_column(structure_gray, gap_box, active_cols, y):
    if structure_gray is None:
        return False
    if not active_cols:
        return _horizontal_separator_has_continuous_run(structure_gray, gap_box, y)

    for col in active_cols:
        col_box = col.get("box") if isinstance(col, dict) else None
        if not col_box:
            continue
        col_x1 = max(float(gap_box[0]), float(col_box[0]))
        col_x2 = min(float(gap_box[2]), float(col_box[2]))
        if col_x2 - col_x1 < 6.0:
            continue
        col_gap_box = [col_x1, gap_box[1], col_x2, gap_box[3]]
        if _horizontal_separator_has_continuous_run(structure_gray, col_gap_box, y):
            return True
    return False
def _make_synthetic_recovered_row(gap_box, row_y1, row_y2):
    return {
        "class": CLASS_NAME_ROW,
        "box": [gap_box[0], row_y1, gap_box[2], row_y2],
        "confidence": 0.95,
        "source_model": "synthetic_recovery",
        "x": (gap_box[0] + gap_box[2]) / 2,
        "y": (row_y1 + row_y2) / 2,
        "width": abs(gap_box[2] - gap_box[0]),
        "height": abs(row_y2 - row_y1),
    }


def _box_overlaps_row_header_band(box, row_header_boxes, min_overlap=0.72):
    if not box or not row_header_boxes:
        return False

    box_area = get_box_area(box) + 1e-12
    for rh_box in row_header_boxes:
        rh_area = get_box_area(rh_box) + 1e-12
        inter = calculate_intersection_area(box, rh_box)
        if inter / min(box_area, rh_area) >= min_overlap:
            return True
    return False


def _suppress_rows_overlapping_row_headers(body_rows, row_header_boxes):
    if not body_rows or not row_header_boxes:
        return body_rows

    filtered = []
    for row in body_rows:
        if (
            row.get("type") == CLASS_NAME_ROW
            and _box_overlaps_row_header_band(row.get("box"), row_header_boxes)
        ):
            continue
        filtered.append(row)
    return filtered


def _drop_span_column_overrides_on_row_headers(overrides, final_rows, row_header_boxes):
    if not overrides or not final_rows:
        return overrides

    filtered = {}
    for key, value in overrides.items():
        r_idx = key[0]
        if r_idx >= len(final_rows):
            continue
        row = final_rows[r_idx]
        if row.get("type") == CLASS_NAME_ROW_HEADER:
            continue
        if _box_overlaps_row_header_band(row.get("box"), row_header_boxes):
            continue
        filtered[key] = value
    return filtered

def _cluster_unassigned_tokens_vertically(unassigned_items):
    if not unassigned_items:
        return []
        
    items = sorted(unassigned_items, key=lambda i: o_to_box(i)[0])
    clusters = []
    
    for item in items:
        ibox = o_to_box(item)
        matched = False
        for cluster in clusters:
            cbox = cluster["box"]
            # Check X overlap
            if _x_overlap_ratio(ibox, cbox) > 0.1 or _x_overlap_ratio(cbox, ibox) > 0.1:
                # Check horizontal distance
                if min(abs(ibox[0] - cbox[2]), abs(cbox[0] - ibox[2])) < 50:
                    cluster["items"].append(item)
                    cluster["box"] = [
                        min(cbox[0], ibox[0]),
                        min(cbox[1], ibox[1]),
                        max(cbox[2], ibox[2]),
                        max(cbox[3], ibox[3]),
                    ]
                    matched = True
                    break
        if not matched:
            clusters.append({"box": ibox, "items": [item]})
            
    return clusters

def _cluster_unassigned_tokens_horizontally(unassigned_items):
    if not unassigned_items:
        return []
        
    items = sorted(unassigned_items, key=lambda i: o_to_box(i)[1])
    clusters = []
    
    for item in items:
        ibox = o_to_box(item)
        matched = False
        for cluster in clusters:
            cbox = cluster["box"]
            # Check Y overlap
            if _y_overlap_ratio(ibox, cbox) > 0.1 or _y_overlap_ratio(cbox, ibox) > 0.1:
                if min(abs(ibox[1] - cbox[3]), abs(cbox[1] - ibox[3])) < 50:
                    cluster["items"].append(item)
                    cluster["box"] = [
                        min(cbox[0], ibox[0]),
                        min(cbox[1], ibox[1]),
                        max(cbox[2], ibox[2]),
                        max(cbox[3], ibox[3]),
                    ]
                    matched = True
                    break
        if not matched:
            clusters.append({"box": ibox, "items": [item]})
            
    return clusters

def _recover_missing_columns(cols_raw, ocr_data, table_box, structure_gray, spanning_headers, protected_boxes=None):
    active_boxes = [c["box"] for c in cols_raw]
    if protected_boxes:
        active_boxes.extend(protected_boxes)
    unassigned = _get_unassigned_ocr_items(ocr_data, active_boxes)
    clusters = _cluster_unassigned_tokens_vertically(unassigned)
    
    new_cols = []
    new_splits = []
    
    if not cols_raw:
        return new_cols, new_splits
        
    min_col_x = min(c["box"][0] for c in cols_raw)
    max_col_x = max(c["box"][2] for c in cols_raw)
    
    table_h = table_box[3] - table_box[1] if table_box else 800
    
    for cluster in clusters:
        cbox = cluster["box"]
        
        # Only recover if the cluster is strictly between existing columns
        if cbox[2] <= min_col_x or cbox[0] >= max_col_x:
            continue
            
        # Find which gap this cluster falls in
        gap_x1, gap_x2 = table_box[0] if table_box else 0, table_box[2] if table_box else 1000
        for c in cols_raw:
            cb = c["box"]
            if cb[2] <= cbox[0]:
                gap_x1 = max(gap_x1, cb[2])
            if cb[0] >= cbox[2]:
                gap_x2 = min(gap_x2, cb[0])
                
        gap_box = [gap_x1, table_box[1] if table_box else 0, gap_x2, table_box[3] if table_box else 1000]
        
        # Check for separator line in gap
        sep_res = _gap_has_separator_line(structure_gray, gap_box, "vertical")
        sep_x = sep_res[0] if sep_res else None
        sep_ratio = sep_res[1] if sep_res else 0
        
        # Validation
        if len(cluster["items"]) > 1 or (len(cluster["items"]) == 1 and sep_x is not None):
            pred_class = CLASS_NAME_COLUMN
            
            # The 80% Rule & Hierarchy Check
            if sep_x is not None:
                if sep_ratio < 0.80:
                    pred_class = CLASS_NAME_SPLITTED_COLUMN
                else:
                    pred_class = CLASS_NAME_COLUMN
                    
                # Also keep spanning header check as a fallback for structural hierarchy
                if pred_class == CLASS_NAME_COLUMN:
                    for span in spanning_headers:
                        sbox = span["box"]
                        if sbox[0] <= cbox[0] and sbox[2] >= cbox[2]:
                            if cbox[1] >= sbox[3] - 10:
                                pred_class = CLASS_NAME_SPLITTED_COLUMN
                                break
            
            # If line exists and cluster crosses it, cut the cluster.
            # For simplicity, we just bound the cluster by the line if it's near.
            x1 = cbox[0]
            x2 = cbox[2]
            if sep_x is not None:
                if cbox[0] < sep_x < cbox[2]:
                    # Splitting OCR tokens based on sep_x
                    left_items = [i for i in cluster["items"] if o_to_box(i)[2] <= sep_x + 5]
                    right_items = [i for i in cluster["items"] if o_to_box(i)[0] >= sep_x - 5]
                    
                    if left_items:
                        lx1 = min(o_to_box(i)[0] for i in left_items)
                        lx2 = max(o_to_box(i)[2] for i in left_items)
                        new_pred = {
                            "class": pred_class,
                            "box": [lx1, gap_box[1], lx2, gap_box[3]],
                            "confidence": 0.95,
                            "source_model": "synthetic_recovery", "x": (gap_box[0]+gap_box[2])/2, "y": (gap_box[1]+gap_box[3])/2, "width": abs(gap_box[2]-gap_box[0]), "height": abs(gap_box[3]-gap_box[1])
                        }
                        if pred_class == CLASS_NAME_COLUMN: new_cols.append(new_pred)
                        else: new_splits.append(new_pred)
                        
                    if right_items:
                        rx1 = min(o_to_box(i)[0] for i in right_items)
                        rx2 = max(o_to_box(i)[2] for i in right_items)
                        new_pred = {
                            "class": pred_class,
                            "box": [rx1, gap_box[1], rx2, gap_box[3]],
                            "confidence": 0.95,
                            "source_model": "synthetic_recovery", "x": (gap_box[0]+gap_box[2])/2, "y": (gap_box[1]+gap_box[3])/2, "width": abs(gap_box[2]-gap_box[0]), "height": abs(gap_box[3]-gap_box[1])
                        }
                        if pred_class == CLASS_NAME_COLUMN: new_cols.append(new_pred)
                        else: new_splits.append(new_pred)
                    continue

            new_pred = {
                "class": pred_class,
                "box": [x1, gap_box[1], x2, gap_box[3]],
                "confidence": 0.95,
                "source_model": "synthetic_recovery", "x": (gap_box[0]+gap_box[2])/2, "y": (gap_box[1]+gap_box[3])/2, "width": abs(gap_box[2]-gap_box[0]), "height": abs(gap_box[3]-gap_box[1])
            }
            if pred_class == CLASS_NAME_COLUMN:
                new_cols.append(new_pred)
            else:
                new_splits.append(new_pred)
                
    return new_cols, new_splits

def _recover_missing_rows(
    rows_raw,
    ocr_data,
    table_box,
    structure_gray,
    active_cols,
    protected_boxes=None,
    gap_anchor_boxes=None,
    candidate_ocr_override=None,
):
    if not rows_raw:
        return []

    candidate_source = candidate_ocr_override if candidate_ocr_override is not None else ocr_data
    candidate_ocr = [
        item for item in candidate_source or []
        if str(item.get("text", "")).strip()
    ]
    if not candidate_ocr:
        return []
        
    recovery_anchors = list(rows_raw)
    for anchor_box in gap_anchor_boxes or []:
        if not anchor_box:
            continue
        recovery_anchors.append(
            {
                "box": list(anchor_box),
                "y": (float(anchor_box[1]) + float(anchor_box[3])) / 2.0,
            }
        )

    # The last body row can be missed even when the table columns continue
    # below the final detected row. Treat the lower end of the column grid as
    # a structural boundary so matrix-missing OCR in that bottom gap can still
    # create a recovered row.
    if active_cols:
        col_bottom = max(float(col["box"][3]) for col in active_cols if col.get("box"))
        existing_bottom = max(float(r["box"][3]) for r in recovery_anchors if r.get("box"))
        if col_bottom > existing_bottom + 2.0:
            x1 = table_box[0] if table_box else min(float(col["box"][0]) for col in active_cols if col.get("box"))
            x2 = table_box[2] if table_box else max(float(col["box"][2]) for col in active_cols if col.get("box"))
            recovery_anchors.append(
                {
                    "box": [x1, col_bottom, x2, col_bottom],
                    "y": col_bottom,
                    "source": "column_bottom_recovery_anchor",
                }
            )

    if len(recovery_anchors) < 2:
        return []

    min_row_y = min(r["box"][1] for r in recovery_anchors)
    max_row_y = max(r["box"][3] for r in recovery_anchors)

    # Recovery is gap-region based: do not reject OCR only because it slightly
    # overlaps a neighboring row. If its center falls between structural bands,
    # it can be a missing row candidate.
    internal_candidates = [
        item for item in candidate_ocr
        if min_row_y <= ((o_to_box(item)[1] + o_to_box(item)[3]) / 2.0) <= max_row_y
    ]
    if not internal_candidates:
        return []

    # Find gaps between all structural row bands, not only body rows. This
    # catches OCR bands between column headers, row headers, and body rows.
    ordered_rows = sorted(recovery_anchors, key=lambda r: r["y"])
    gaps = []
    for i in range(len(ordered_rows) - 1):
        gap_y1 = ordered_rows[i]["box"][3]
        gap_y2 = ordered_rows[i+1]["box"][1]
        if gap_y2 > gap_y1:
            gaps.append((gap_y1, gap_y2))

    new_rows = []
    for (gap_y1, gap_y2) in gaps:
        gap_items = []
        for item in internal_candidates:
            ibox = o_to_box(item)
            icy = (float(ibox[1]) + float(ibox[3])) / 2.0
            if (gap_y1 - 5.0) <= icy <= (gap_y2 + 5.0):
                gap_items.append(item)
        if not gap_items:
            continue
            
        gap_box = [table_box[0] if table_box else 0, gap_y1, table_box[2] if table_box else 1000, gap_y2]
        
        cbox = [
            min(o_to_box(i)[0] for i in gap_items),
            min(o_to_box(i)[1] for i in gap_items),
            max(o_to_box(i)[2] for i in gap_items),
            max(o_to_box(i)[3] for i in gap_items),
        ]
        
        # Check horizontal spread across columns
        col_hits = sum(1 for col in active_cols if _x_overlap_ratio(cbox, col["box"]) >= 0.05)
        sep_lines = _gap_horizontal_separator_lines_any_column(
            structure_gray,
            gap_box,
            active_cols,
        )
        sep_positions = sorted(
            sep_y
            for sep_y, _ in sep_lines
            if cbox[1] < sep_y < cbox[3]
            and not _line_y_crosses_ocr_text_in_band(
                sep_y,
                cbox[0],
                cbox[2],
                gap_items,
            )
            and _gap_horizontal_separator_has_real_line_any_column(
                structure_gray,
                gap_box,
                active_cols,
                sep_y,
            )
        )
        
        if col_hits >= 1 or sep_positions:
            if sep_positions:
                bounds = [float("-inf")] + sep_positions + [float("inf")]
                for lower, upper in zip(bounds, bounds[1:]):
                    seg_y1 = gap_y1 if lower == float("-inf") else max(gap_y1, lower)
                    seg_y2 = gap_y2 if upper == float("inf") else min(gap_y2, upper)
                    segment_box = [gap_box[0], seg_y1, gap_box[2], seg_y2]
                    segment_items = []
                    for item in gap_items:
                        ibox = o_to_box(item)
                        text_area = get_box_area(ibox) + 1e-12
                        inter_area = calculate_intersection_area(ibox, segment_box)
                        if inter_area / text_area >= 0.60:
                            segment_items.append(item)
                    if not segment_items:
                        continue
                    sy1 = min(o_to_box(i)[1] for i in segment_items)
                    sy2 = max(o_to_box(i)[3] for i in segment_items)
                    new_rows.append(_make_synthetic_recovered_row(gap_box, sy1, sy2))
            else:
                new_rows.append(_make_synthetic_recovered_row(gap_box, cbox[1], cbox[3]))

    return new_rows

def build_table_structure(
    all_preds_obj,
    ocr_data,
    structure_image_path=None,
    _row_recovery_items=None,
    _return_debug=False,
):
    preds = all_preds_obj.get("predictions", [])
    all_preds = _collect_structure_predictions(preds)

    if not all_preds:
        return []

    table_box = [
        min(pred["box"][0] for pred in all_preds),
        min(pred["box"][1] for pred in all_preds),
        max(pred["box"][2] for pred in all_preds),
        max(pred["box"][3] for pred in all_preds),
    ]
    if _row_recovery_items is None:
        probe_result = build_table_structure(
            all_preds_obj,
            ocr_data,
            structure_image_path=structure_image_path,
            _row_recovery_items=[],
            _return_debug=True,
        )
        if isinstance(probe_result, dict):
            _row_recovery_items = _matrix_missing_ocr_items(
                ocr_data,
                probe_result.get("matrix", []),
                probe_result.get("rows", []),
                probe_result.get("cols", []),
                table_box,
            )
        else:
            _row_recovery_items = []

    relaxed_col_header_preds = _collect_relaxed_detected_predictions(
        preds,
        CLASS_NAME_COLUMN_HEADER,
        table_box,
        min_conf=0.15,
    )
    strong_rows_seed = [
        pred for pred in all_preds
        if pred["class"] == CLASS_NAME_ROW and _prediction_bucket(pred) == "strong"
    ]
    cols_raw = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_COLUMN],
        CLASS_NAME_COLUMN,
        ocr_data,
        table_box,
        support_rows=strong_rows_seed,
        ios_thresh=IOS_THRESH_COLUMNS,
    )
    if cols_raw:
        cols_raw = merge_cols_by_x(cols_raw, x_threshold=10)
        cols_raw = group_nms(cols_raw, IOS_THRESH_COLUMNS)

    structure_gray = None
    if structure_image_path:
        structure_gray = cv2.imread(structure_image_path, cv2.IMREAD_GRAYSCALE)
        
    # MISSING COLUMN RECOVERY
    span_col_headers_temp = [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPANNING_COLUMN_HEADER]
    protected_col_boxes = [pred["box"] for pred in all_preds if pred["class"] in (
        CLASS_NAME_ROW_HEADER, CLASS_NAME_COLUMN_HEADER, CLASS_NAME_SPANNING_COLUMN_HEADER
    )]
    recovered_cols, recovered_splits = _recover_missing_columns(cols_raw, ocr_data, table_box, structure_gray, span_col_headers_temp, protected_col_boxes)
    cols_raw.extend(recovered_cols)
    cols_raw = group_nms(cols_raw, IOS_THRESH_COLUMNS) # Re-sort and NMS
    provisional_row_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_ROW_HEADER],
        CLASS_NAME_ROW_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    cols_raw = _split_columns_at_internal_separators(
        cols_raw,
        table_box,
        structure_gray,
        ignore_bands=[pred["box"] for pred in provisional_row_headers],
    )


    rows_raw = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_ROW],
        CLASS_NAME_ROW,
        ocr_data,
        table_box,
        support_cols=cols_raw,
        ios_thresh=IOS_THRESH_ROWS,
    )

    # MISSING ROW RECOVERY
    # Use selected structural header bands as anchors, not every raw/provisional
    # header prediction. Raw duplicates can overlap a real gap and incorrectly
    # block row recovery before they are later removed by NMS/support filtering.
    recovery_col_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_COLUMN_HEADER],
        CLASS_NAME_COLUMN_HEADER,
        ocr_data,
        table_box,
        support_rows=rows_raw,
        support_cols=cols_raw,
    )
    recovery_span_col_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPANNING_COLUMN_HEADER],
        CLASS_NAME_SPANNING_COLUMN_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    recovery_row_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_ROW_HEADER],
        CLASS_NAME_ROW_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    protected_row_boxes = [
        pred["box"]
        for pred in (recovery_col_headers + recovery_span_col_headers + recovery_row_headers)
    ]
    recovered_rows = []
    if _row_recovery_items:
        recovered_rows = _recover_missing_rows(
            rows_raw,
            ocr_data,
            table_box,
            structure_gray,
            cols_raw,
            protected_row_boxes,
            gap_anchor_boxes=protected_row_boxes,
            candidate_ocr_override=_row_recovery_items,
        )
    rows_raw.extend(recovered_rows)
    rows_raw = group_nms(rows_raw, IOS_THRESH_ROWS)

    col_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_COLUMN_HEADER],
        CLASS_NAME_COLUMN_HEADER,
        ocr_data,
        table_box,
        support_rows=rows_raw,
        support_cols=cols_raw,
    )
    span_col_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPANNING_COLUMN_HEADER],
        CLASS_NAME_SPANNING_COLUMN_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    row_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_ROW_HEADER],
        CLASS_NAME_ROW_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    table_headers = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_TABLE_HEADER],
        CLASS_NAME_TABLE_HEADER,
        ocr_data,
        table_box,
        support_cols=cols_raw,
    )
    if not col_headers:
        col_headers = _recover_detected_column_headers(
            relaxed_col_header_preds,
            cols_raw,
            rows_raw,
        )
    col_headers = _split_header_predictions_by_recovery(
        col_headers,
        cols_raw,
        ocr_data,
        structure_gray=structure_gray,
        allow_ocr=True,
    )
    span_col_headers = _split_header_predictions_by_recovery(
        span_col_headers,
        cols_raw,
        ocr_data,
        structure_gray=structure_gray,
        allow_ocr=False,
    )
    # Validate spanning column headers in the active reconstruction path too.
    # Without this, raw same-box column_header/spanning_column_header pairs can
    # materialize as duplicate stacked header rows.
    span_col_headers = _filter_valid_spanning_column_headers(span_col_headers, col_headers)
    col_headers, span_col_headers = _resolve_same_band_header_class_conflicts(col_headers, span_col_headers)

    raw_split_preds = [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPLITTED_COLUMN]
    recovered_splits.extend(
        _recover_split_columns_from_separator_lines(
            cols_raw,
            rows_raw,
            structure_gray,
            ocr_data,
            existing_splits=raw_split_preds + recovered_splits,
        )
    )
    all_preds.extend(recovered_splits) # so it gets picked up
    splits = _select_predictions_by_support(
        [pred for pred in all_preds if pred["class"] == CLASS_NAME_SPLITTED_COLUMN],
        CLASS_NAME_SPLITTED_COLUMN,
        ocr_data,
        table_box,
        support_rows=rows_raw,
        support_cols=cols_raw,
    )
    cols_raw = _promote_orphan_split_columns(
        cols_raw,
        splits,
        rows_raw,
        col_headers + span_col_headers,
        table_box,
    )
    raw_span_row_cell_preds = [
        pred for pred in all_preds if pred["class"] == CLASS_NAME_SPANNING_ROW_CELL
    ]
    raw_span_col_cell_preds = [
        pred for pred in all_preds if pred["class"] == CLASS_NAME_SPANNING_COLUMN_CELL
    ]

    if cols_raw:
        cols_raw = merge_cols_by_x(cols_raw, x_threshold=10)
    if rows_raw:
        rows_raw = merge_rows_by_y(rows_raw, y_threshold=5)

    parent_split_plans = _build_parent_split_plans(cols_raw, splits)

    cols_raw.sort(key=lambda col: col["x"])
    final_cols = []
    for col in cols_raw:
        parent_key = _box_anchor_key(col["box"])
        parent_plan = parent_split_plans.get(parent_key, {})
        if col.get("from_orphan_split"):
            final_cols.append({"box": col["box"], "is_split": False, "source": "detected"})
            continue

        splits_in_col = [
            split
            for split in splits
            if calculate_intersection_area(split["box"], col["box"])
            / (get_box_area(split["box"]) + 1e-12)
            > INTERSECTION_THRESH_SPLIT
        ]

        valid_splits = []
        for split in splits_in_col:
            if _is_valid_split_candidate(split, col):
                valid_splits.append(split)

        split_count = _infer_split_count(valid_splits, col["box"]) if valid_splits else 0
        if split_count < 2:
            final_cols.append({"box": col["box"], "is_split": False, "source": "detected"})
            continue

        c_x1, c_y1, c_x2, c_y2 = col["box"]
        sub_width = (c_x2 - c_x1) / split_count
        assign_boundaries = parent_plan.get("boundaries")
        for index in range(split_count):
            sx1 = c_x1 + index * sub_width
            sx2 = c_x1 + (index + 1) * sub_width
            visual_box = [sx1, c_y1, sx2, c_y2]
            assign_box = [sx1, c_y1, sx2, c_y2]
            if assign_boundaries and len(assign_boundaries) == split_count + 1:
                ax1 = max(c_x1, min(c_x2, float(assign_boundaries[index])))
                ax2 = max(c_x1, min(c_x2, float(assign_boundaries[index + 1])))
                if ax2 > ax1:
                    assign_box = [ax1, c_y1, ax2, c_y2]
                    visual_box = [ax1, c_y1, ax2, c_y2]
            final_cols.append(
                {
                    "box": visual_box,
                    "assign_box": assign_box,
                    "is_split": True,
                    "parent_box": col["box"],
                    "split_index": index,
                    "total_splits": split_count,
                    "source": "split_from_detection",
                }
            )

    if not final_cols:
        return []

    final_cols.sort(key=lambda col: col["box"][0])
    min_col_x = min(col["box"][0] for col in final_cols)
    max_col_x = max(col["box"][2] for col in final_cols)

    final_rows = []
    header_items_for_rows = list(col_headers or []) + list(span_col_headers or [])
    ocr_data = _augment_ocr_with_header_crop(
        ocr_data,
        structure_image_path,
        header_items_for_rows,
        min_col_x,
        max_col_x,
    )
    recovered_missing_headers = _recover_missing_header_slots_by_geometry(
        col_headers,
        span_col_headers,
        final_cols,
        ocr_data,
        structure_gray,
    )
    if recovered_missing_headers:
        recovered_col_headers = [
            item for item in recovered_missing_headers
            if item.get("class") == CLASS_NAME_COLUMN_HEADER
        ]
        recovered_span_headers = [
            item for item in recovered_missing_headers
            if item.get("class") == CLASS_NAME_SPANNING_COLUMN_HEADER
        ]
        if recovered_col_headers:
            col_headers = sorted(
                list(col_headers or []) + recovered_col_headers,
                key=lambda item: (item["box"][1], item["box"][0]),
            )
        if recovered_span_headers:
            span_col_headers = sorted(
                list(span_col_headers or []) + recovered_span_headers,
                key=lambda item: (item["box"][1], item["box"][0]),
            )
        header_items_for_rows = list(col_headers or []) + list(span_col_headers or [])
    recovered_missing_spans, span_col_headers = _recover_missing_spanning_header_slots_by_geometry(
        col_headers,
        span_col_headers,
        final_cols,
        ocr_data,
        structure_gray,
    )
    span_col_headers = _expand_spanning_headers_to_separator_boundaries(
        span_col_headers,
        col_headers,
        final_cols,
        ocr_data,
        structure_gray,
    )
    header_items_for_rows = list(col_headers or []) + list(span_col_headers or [])
    header_rows_definitions = _build_y_band_header_row_definitions(
        col_headers,
        span_col_headers,
        ocr_data,
        min_col_x,
        max_col_x,
        rows_raw=rows_raw,
    )
    header_rows_definitions = _filter_late_isolated_header_rows(
        header_rows_definitions,
        final_cols,
        rows_raw,
    )

    for hr in header_rows_definitions:
        final_rows.append(
            {
                "type": HEADER_ROW_TYPE,
                "header_class": hr["header_class"],
                "items": hr["items"],
                "y": (hr["box"][1] + hr["box"][3]) / 2,
                "box": [min_col_x, hr["box"][1], max_col_x, hr["box"][3]],
                "source": "detected",
            }
        )

    header_bottom = _top_header_stack_bottom(header_rows_definitions)
    body_rows = []
    for row in rows_raw:
        if header_bottom is not None and row["y"] <= (header_bottom - 1):
            continue
        body_rows.append(
            {
                "type": CLASS_NAME_ROW,
                "items": row.get("items", [row]),
                "y": row["y"],
                "box": row["box"],
                "source": "detected",
            }
        )

    first_col_box = final_cols[0]["box"] if final_cols else None
    table_header_boxes = [pred["box"] for pred in table_headers]
    valid_row_headers = []
    excluded_title_boxes = []
    for row_header in row_headers:
        rh_box = row_header["box"]
        if any(
            calculate_intersection_area(rh_box, th_box)
            / (min(get_box_area(rh_box), get_box_area(th_box)) + 1e-12)
            > 0.45
            for th_box in table_header_boxes
        ):
            continue
        if header_bottom is not None and row_header["y"] <= (header_bottom - 1):
            continue
        if first_col_box is not None:
            # Do not accept row_header just because its box touches/centers in
            # the first column; require OCR/profile support for the row band.
            if not _should_promote_row_to_row_header({"box": rh_box}, final_cols, ocr_data):
                continue
        valid_row_headers.append(row_header)

    valid_row_header_boxes = [row_header["box"] for row_header in valid_row_headers]
    body_rows = _suppress_rows_overlapping_row_headers(body_rows, valid_row_header_boxes)

    for row_header in valid_row_headers:
        body_rows.append(
            {
                "type": CLASS_NAME_ROW_HEADER,
                "items": [row_header],
                "y": row_header["y"],
                "box": row_header["box"],
                "source": "detected",
            }
        )
    body_rows.sort(key=lambda row: (row["y"], 0 if row["type"] == CLASS_NAME_ROW_HEADER else 1))

    structure_gray = None
    if structure_image_path:
        structure_gray = cv2.imread(structure_image_path, cv2.IMREAD_GRAYSCALE)
    body_rows = _split_body_rows_at_internal_separators(body_rows, final_cols, structure_gray, ocr_data)
    body_rows = _suppress_rows_overlapping_row_headers(body_rows, valid_row_header_boxes)

    final_rows.extend(body_rows)
    if not final_rows:
        return []

    # Spanning cells must be validated after the final body rows are built.
    # Separator-based row splitting can turn one detected row into multiple
    # logical rows; validating spans before that split drops real row-spans.
    final_body_rows_for_spans = [
        row for row in final_rows if row.get("type") == CLASS_NAME_ROW
    ]
    span_row_cells = _select_predictions_by_support(
        raw_span_row_cell_preds,
        CLASS_NAME_SPANNING_ROW_CELL,
        ocr_data,
        table_box,
        support_rows=final_body_rows_for_spans,
        support_cols=cols_raw,
    )
    span_col_cells = _select_predictions_by_support(
        raw_span_col_cell_preds,
        CLASS_NAME_SPANNING_COLUMN_CELL,
        ocr_data,
        table_box,
        support_rows=final_body_rows_for_spans,
        support_cols=cols_raw,
    )
    row_span_anchors = _anchor_row_spans_to_detected_structure(span_row_cells, cols_raw, splits)

    table_min_y = min(r["box"][1] for r in final_rows) if final_rows else 0
    table_max_y = max(r["box"][3] for r in final_rows) if final_rows else 0

    body_heights = [
        max(1.0, float(row["box"][3] - row["box"][1]))
        for row in final_rows
        if row["type"] in (CLASS_NAME_ROW, CLASS_NAME_ROW_HEADER)
    ]
    header_heights = [
        max(1.0, float(row["box"][3] - row["box"][1]))
        for row in final_rows
        if _is_header_row_type(row["type"])
    ]
    top_pad = 0.0
    if body_heights:
        top_pad = max(top_pad, 0.22 * float(np.median(body_heights)))
    if header_heights:
        top_pad = max(top_pad, 0.12 * float(np.median(header_heights)))
    table_min_y = max(0, table_min_y - min(18.0, top_pad))

    table_box = [min_col_x, table_min_y, max_col_x, table_max_y]

    img_meta = all_preds_obj.get("image", {}) if isinstance(all_preds_obj, dict) else {}
    img_w = int(img_meta.get("width") or 1280)
    img_h = int(img_meta.get("height") or 1280)

    header_box = None
    if header_rows_definitions:
        hb = union_box(header_rows_definitions)
        header_box = [min_col_x, hb[1], max_col_x, hb[3]]

    row_boxes = _build_extraction_row_boxes(final_rows)
    row_types = [row["type"] for row in final_rows]
    col_boxes = [col["box"] for col in final_cols]
    ocr_data, ocr_mode = _normalize_ocr_coordinates(
        ocr_data,
        table_box,
        img_w,
        img_h,
        header_box=header_box,
        row_boxes=row_boxes,
        col_boxes=col_boxes,
        row_types=row_types,
    )
    if structure_gray is None and structure_image_path:
        structure_gray = cv2.imread(structure_image_path, cv2.IMREAD_GRAYSCALE)

    ocr_created_columns = False
    body_rows_split = False
    header_row_count = sum(1 for row in final_rows if _is_header_row_type(row["type"]))
    matrix = [["" for _ in range(len(final_cols))] for _ in range(len(final_rows))]
    # 1. Identify "Anchors" (Column 0 text for each row) to guide assignment
    # Define row vertical ranges based strictly on Roboflow detections, abandoning gap-split midpoints
    # This precisely fulfills foundational rule #3
    row_y_ranges = []
    for r_idx, row in enumerate(final_rows):
        rbox = row["box"]
        row_y_ranges.append((rbox[1], rbox[3]))

    # 2. Assign items using Roboflow row geometry, preferring overlap and center-Y.
    row_assignments = [[] for _ in final_rows]
    row_centers = [((ry1 + ry2) / 2) for ry1, ry2 in row_y_ranges]
    row_min_y = min((ry1 for ry1, _ in row_y_ranges), default=0.0)
    row_max_y = max((ry2 for _, ry2 in row_y_ranges), default=0.0)
    avg_row_h = (sum(max(1.0, ry2 - ry1) for ry1, ry2 in row_y_ranges) / len(row_y_ranges)) if row_y_ranges else 1.0
    body_like_types = (CLASS_NAME_ROW, CLASS_NAME_ROW_HEADER)
    first_body_idx = next((i for i, row in enumerate(final_rows) if row["type"] in body_like_types), None)
    header_before_body = (
        first_body_idx is not None
        and any(row["type"] not in body_like_types for row in final_rows[:first_body_idx])
    )
    table_width = max(1.0, float(max_col_x - min_col_x))
    skipped_header_boxes = []
    wrapped_full_header_boxes = []
    for r_idx, row in enumerate(final_rows):
        if not _is_header_row_type(row["type"]):
            continue
        for item in row.get("items", []):
            if item.get("header_use_full_box_text"):
                wrapped_full_header_boxes.append(item.get("header_text_box", item["box"]))
            if _should_skip_header_item_for_propagation(final_rows, r_idx, row, item, final_cols):
                skipped_header_boxes.append(item["box"])
    excluded_header_boxes = (
        table_header_boxes
        + excluded_title_boxes
        + skipped_header_boxes
        + wrapped_full_header_boxes
    )
    first_assignable_row_top = min((row["box"][1] for row in final_rows), default=0.0)
    excluded_text_items_skipped = 0
    for item in ocr_data:
        ibox = o_to_box(item)
        iy1, iy2 = ibox[1], ibox[3]
        icy = (iy1 + iy2) / 2
        owner_col_idx = _preferred_col_idx_for_item(ibox, final_cols)
        if any(
            _ocr_item_hits_exclusion_box(ibox, th_box)
            or (
                ibox[1] < first_assignable_row_top
                and th_box[0] <= ((ibox[0] + ibox[2]) / 2) <= th_box[2]
                and calculate_intersection_area(ibox, th_box) > 0.0
            )
            for th_box in excluded_header_boxes
        ):
            excluded_text_items_skipped += 1
            continue
        if not _item_is_assignable_table_text(
            ibox,
            final_rows,
            row_y_ranges,
            row_centers,
            final_cols,
            first_assignable_row_top,
            row_max_y,
            avg_row_h,
        ):
            continue
        item_w = max(1.0, float(ibox[2] - ibox[0]))
        max_row_inter_h = max((_row_band_intersection_height(ibox, row["box"]) for row in final_rows), default=0.0)
        if (
            item_w >= (0.45 * table_width)
            and max_row_inter_h < max(4.0, 0.12 * max(1.0, float(iy2 - iy1)))
        ):
            continue

        preferred_header_row = None
        preferred_header_score = 0.0
        for candidate_idx, candidate_row in enumerate(final_rows):
            if not _is_header_row_type(candidate_row["type"]):
                continue
            if not (candidate_row["box"][1] <= icy <= candidate_row["box"][3]):
                continue
            for header_item in candidate_row.get("items", []):
                header_box = header_item["box"]
                inter = calculate_intersection_area(header_box, ibox)
                score = inter / (get_box_area(ibox) + 1e-12)
                if score > preferred_header_score:
                    preferred_header_score = score
                    preferred_header_row = candidate_idx

        if preferred_header_row is not None and preferred_header_score > 0.25:
            row_assignments[preferred_header_row].append(item)
            continue

        overlaps = []
        for r_idx, (ry1, ry2) in enumerate(row_y_ranges):
            inter_h = max(0.0, min(iy2, ry2) - max(iy1, ry1))
            if inter_h <= 0:
                continue
            item_h = max(1.0, iy2 - iy1)
            row_h = max(1.0, ry2 - ry1)
            overlap_score = min(inter_h / item_h, inter_h / row_h)
            overlaps.append((overlap_score, inter_h, r_idx))

        if overlaps:
            best_r = max(
                overlaps,
                key=lambda entry: (entry[0], entry[1], -abs(icy - row_centers[entry[2]])),
            )[2]
        else:
            contained_in = [r_idx for r_idx, (ry1, ry2) in enumerate(row_y_ranges) if ry1 <= icy <= ry2]
            if contained_in:
                best_r = min(contained_in, key=lambda i: abs(icy - row_centers[i]))
            else:
                nearest_idx = min(range(len(row_y_ranges)), key=lambda i: abs(icy - row_centers[i]))
                nearest_dist = abs(icy - row_centers[nearest_idx])
                nearest_row = final_rows[nearest_idx]
                nearest_ry1, _ = row_y_ranges[nearest_idx]
                if (
                    _is_header_row_type(nearest_row["type"])
                    and iy2 <= nearest_ry1
                ):
                    continue
                if (
                    (iy1 > row_max_y or iy2 < row_min_y)
                    and nearest_dist > max(32.0, 1.25 * avg_row_h)
                ):
                    continue
                best_r = nearest_idx

        best_row = final_rows[best_r]
        if (
            best_row.get("split_role") == "bottom"
            and owner_col_idx is not None
            and owner_col_idx >= int(best_row.get("split_leading_cols", 0))
            and icy < float(best_row.get("split_nonleading_min_y", float("inf")))
            and best_r > 0
        ):
            prev_row = final_rows[best_r - 1]
            if prev_row.get("split_parent_id") == best_row.get("split_parent_id"):
                best_r = best_r - 1
        row_assignments[best_r].append(item)

    body_row_maps = {}
    body_row_centers = {}
    row_header_maps = {}
    for r_idx, row in enumerate(final_rows):
        if row["type"] == CLASS_NAME_ROW:
            body_row_maps[r_idx] = _build_row_text_map(row["box"], final_cols, row_assignments[r_idx])
            body_row_centers[r_idx] = (row["box"][1] + row["box"][3]) / 2
        elif row["type"] == CLASS_NAME_ROW_HEADER:
            row_header_maps[r_idx] = _build_row_header_text_map(final_cols, row_assignments[r_idx])
    span_row_overrides = _build_span_cell_overrides(
        final_rows,
        final_cols,
        span_row_cells,
        ocr_data,
        span_axis="row",
        row_span_anchors=row_span_anchors,
        structure_gray=structure_gray,
    )
    span_col_overrides = _build_span_cell_overrides(
        final_rows,
        final_cols,
        span_col_cells,
        ocr_data,
        span_axis="col",
        structure_gray=structure_gray,
    )
    row_header_boxes_for_span = [
        row["box"] for row in final_rows if row.get("type") == CLASS_NAME_ROW_HEADER
    ]
    span_col_overrides = _drop_span_column_overrides_on_row_headers(
        span_col_overrides,
        final_rows,
        row_header_boxes_for_span,
    )
    span_both_overrides = {}
    if ENABLE_LINE_BASED_SPAN_CELL_RECOVERY and structure_gray is not None:
        real_span_cells = set(span_row_overrides.keys()) | set(span_col_overrides.keys())
        recovered_span_both_overrides = _recover_spanning_both_cells_from_grid_lines(
            final_rows,
            final_cols,
            ocr_data,
            structure_gray,
        )
        recovered_span_row_preds = _recover_spanning_row_cells_from_grid_lines(
            final_rows,
            final_cols,
            ocr_data,
            structure_gray,
        )
        recovered_span_col_preds = _recover_spanning_column_cells_from_grid_lines(
            final_rows,
            final_cols,
            ocr_data,
            structure_gray,
        )
        recovered_span_row_overrides = _build_span_cell_overrides(
            final_rows,
            final_cols,
            recovered_span_row_preds,
            ocr_data,
            span_axis="row",
            row_span_anchors=row_span_anchors,
            structure_gray=structure_gray,
        )
        recovered_span_col_overrides = _build_span_cell_overrides(
            final_rows,
            final_cols,
            recovered_span_col_preds,
            ocr_data,
            span_axis="col",
            structure_gray=structure_gray,
        )
        recovered_span_both_overrides = _drop_span_column_overrides_on_row_headers(
            recovered_span_both_overrides,
            final_rows,
            row_header_boxes_for_span,
        )
        recovered_span_col_overrides = _drop_span_column_overrides_on_row_headers(
            recovered_span_col_overrides,
            final_rows,
            row_header_boxes_for_span,
        )
        for key, value in recovered_span_both_overrides.items():
            if key not in real_span_cells:
                span_both_overrides.setdefault(key, value)
        for key, value in recovered_span_row_overrides.items():
            if key not in real_span_cells and key not in span_both_overrides:
                span_row_overrides.setdefault(key, value)
        for key, value in recovered_span_col_overrides.items():
            if key not in real_span_cells and key not in span_both_overrides and key not in span_row_overrides:
                span_col_overrides.setdefault(key, value)
    non_span_cross_column_used = False

    for r_idx, row in enumerate(final_rows):
        row_box = row["box"]
        row_text_map = None
        if row["type"] == CLASS_NAME_ROW:
            row_text_map = body_row_maps.get(r_idx)
        header_class = row.get("header_class")
        for c_idx, col in enumerate(final_cols):
            cell_box = [
                col["box"][0],
                max(0, row_box[1] - 1),
                col["box"][2],
                row_box[3] + 1,
            ]
            logic_cell_box = [
                col["box"][0],
                max(0, row["box"][1] - 1),
                col["box"][2],
                row["box"][3] + 1,
            ]
            text = ""
            if _is_header_row_type(row["type"]):
                header_items = [
                    item for item in row["items"]
                    if item["class"] == CLASS_NAME_COLUMN_HEADER
                    and not _should_skip_header_item_for_propagation(final_rows, r_idx, row, item, final_cols)
                    and (
                        _tall_header_item_end_row_index(item, final_rows, final_cols) is None
                        or _tall_header_item_end_row_index(item, final_rows, final_cols) == r_idx
                    )
                ]
                source_box = None
                source_item = None
                if col.get("is_split"):
                    matched_item = _find_split_parent_header_item(
                        header_items,
                        final_cols,
                        col,
                        logic_cell_box,
                    )
                    if matched_item is None:
                        parent_box = col.get("parent_box", col["box"])
                        parent_header_box = [
                            parent_box[0],
                            logic_cell_box[1],
                            parent_box[2],
                            logic_cell_box[3],
                        ]
                        matched_item = _find_best_row_item_for_col(header_items, parent_header_box)
                        if matched_item is None:
                                matched_item = _find_covering_header_item_for_col(header_items, parent_header_box)
                    if matched_item is not None:
                        source_item = matched_item
                        source_box = matched_item.get("header_text_box", matched_item["box"])
                else:
                    matched_item = _find_best_row_item_for_col(header_items, logic_cell_box)
                    if matched_item is None:
                        matched_item = _find_covering_header_item_for_col(header_items, logic_cell_box)
                    if matched_item is not None:
                        source_item = matched_item
                        source_box = matched_item.get("header_text_box", matched_item["box"])

                if source_box is not None:
                    tall_header_end_row = (
                        _tall_header_item_end_row_index(source_item, final_rows, final_cols)
                        if source_item is not None
                        else None
                    )
                    if source_item is not None and tall_header_end_row == r_idx:
                        text = _get_header_text_in_box_dedup(source_item["box"], ocr_data, intersection_thresh=0.20).strip()
                        if not text:
                            text = _get_header_text_in_box_dedup(source_item["box"], ocr_data, center_only=True).strip()
                    elif source_item is not None and source_item.get("header_use_full_box_text"):
                        text = _get_header_text_in_box_dedup(source_box, ocr_data, intersection_thresh=0.55).strip()
                        if not text:
                            text = _get_header_text_in_box_dedup(source_box, ocr_data, center_only=True).strip()
                    else:
                        text = _extract_box_text_prefer_row(
                            source_box,
                            row_assignments[r_idx],
                            ocr_data,
                            first_col=(c_idx == 0),
                        )

                if not text:
                    span_header_items = [
                        item for item in row["items"]
                        if item["class"] == CLASS_NAME_SPANNING_COLUMN_HEADER
                    ]
                    for item in span_header_items:
                        inter_cover = _smaller_region_overlap_ratio(item["box"], logic_cell_box)
                        if inter_cover >= INTERSECTION_THRESH_SPAN_CELL:
                            text = get_ocr_text_in_box(item["box"], row_assignments[r_idx]).strip()
                            if not text:
                                text = get_text_center_in_box(item["box"], row_assignments[r_idx]).strip()
                            if not text:
                                text = get_ocr_text_in_box(item["box"], ocr_data).strip()
                            if not text:
                                text = get_text_center_in_box(item["box"], ocr_data).strip()
                            break

                if not text and not _cell_has_non_end_tall_header_owner(row, r_idx, logic_cell_box, final_rows, final_cols):
                    cell_header_items = _row_cell_raw_items(logic_cell_box, cell_box, final_cols, c_idx, row_assignments[r_idx])
                    cell_header_items = [
                        item for item in cell_header_items
                        if not _ocr_item_blocked_by_other_header_row(o_to_box(item), r_idx, final_rows)
                    ]
                    if cell_header_items:
                        cell_header_items = _dedupe_near_duplicate_text_items(cell_header_items)
                        cell_header_items = _sort_ocr_items_reading_order(cell_header_items)
                        text = " ".join(item["text"] for item in cell_header_items).strip()

            # Row Header: OCR text may physically extend across adjacent columns,
            # but the final matrix normally stores it only in the first cell.
            # If the row-header band contains multiple separate OCR boxes that
            # clearly map to different columns, preserve those separate cells.
            elif row["type"] == CLASS_NAME_ROW_HEADER:
                row_header_map = row_header_maps.get(r_idx) or {}
                if row_header_map:
                    text = row_header_map.get(c_idx, "")
                elif c_idx == 0:
                    if row["items"]:
                        rh_box = row["items"][0]["box"]
                        if len(row["items"]) > 1:
                            xs = [item["box"][0] for item in row["items"]] + [item["box"][2] for item in row["items"]]
                            ys = [item["box"][1] for item in row["items"]] + [item["box"][3] for item in row["items"]]
                            rh_box = [min(xs), min(ys), max(xs), max(ys)]
                        text = get_ocr_text_in_box(rh_box, row_assignments[r_idx], intersection_thresh=0.05)
                        if not text:
                            text = get_text_center_in_box(rh_box, row_assignments[r_idx])
                        if not text:
                            text = get_ocr_text_in_box(rh_box, ocr_data, intersection_thresh=0.05)
                        if not text:
                            text = get_text_center_in_box(rh_box, ocr_data)
                    else:
                        text = get_ocr_text_in_box(cell_box, ocr_data)
                        if not text:
                            text = get_text_center_in_box(cell_box, ocr_data)
                        if not text:
                            text = get_ocr_text_in_box(cell_box, row_assignments[r_idx])
                        if not text:
                            text = get_text_center_in_box(cell_box, row_assignments[r_idx])
                else:
                    # Rule 4: All other cells in row header row must be empty
                    text = ""

            # Data Row
            elif row["type"] == CLASS_NAME_ROW:
                span_text = span_both_overrides.get((r_idx, c_idx))
                if not span_text:
                    span_text = span_row_overrides.get((r_idx, c_idx))
                if not span_text:
                    span_text = span_col_overrides.get((r_idx, c_idx))
                if span_text:
                    text = span_text
                else:
                    raw_items = _row_cell_raw_items(logic_cell_box, cell_box, final_cols, c_idx, row_assignments[r_idx])
                    if raw_items:
                        raw_items = _dedupe_near_duplicate_text_items(raw_items)
                        raw_items = _sort_ocr_items_reading_order(raw_items)
                        text = " ".join(item["text"] for item in raw_items).strip()
                    elif row_text_map is not None and c_idx in row_text_map and row_text_map[c_idx]:
                        text = row_text_map[c_idx]

            matrix[r_idx][c_idx] = text.strip()

    if matrix:
        blocked_cells = (
            set((span_both_overrides or {}).keys())
            | set((span_row_overrides or {}).keys())
            | set((span_col_overrides or {}).keys())
        )
        _reassign_split_group_text_by_center(
            matrix,
            final_rows,
            final_cols,
            row_assignments,
            blocked_cells=blocked_cells,
        )
        _fill_orphan_header_ocr_cells(
            matrix,
            final_rows,
            final_cols,
            row_assignments,
            ocr_data,
        )
        _propagate_split_parent_headers(matrix, final_rows, final_cols)
        _merge_header_fragments_across_missing_row_separator(
            matrix,
            final_rows,
            final_cols,
            structure_gray,
        )
        _repair_wide_body_item_spill(
            matrix,
            final_rows,
            final_cols,
            row_assignments,
            span_row_overrides={**span_row_overrides, **span_both_overrides},
            span_col_overrides={**span_col_overrides, **span_both_overrides},
        )
        _suppress_sparse_fragment_row_spill(
            matrix,
            final_rows,
            final_cols,
            row_assignments,
            blocked_cells=blocked_cells,
        )
        matrix, final_rows = _suppress_blank_leading_duplicate_bridge_rows(
            matrix,
            final_rows,
        )

        # Remove completely empty rows from the raw mapping to discard hallucinated boundary rows
        valid_matrix = []
        filtered_rows = []
        for r_idx, row in enumerate(matrix):
            if any(str(cell).strip() for cell in row):
                valid_matrix.append(row)
                filtered_rows.append(final_rows[r_idx])
        matrix = valid_matrix
        final_rows = filtered_rows

        matrix, final_cols, dropped_empty_columns = _drop_empty_columns(matrix, final_rows, final_cols)

    if _return_debug:
        return {
            "matrix": matrix,
            "rows": final_rows,
            "cols": final_cols,
        }
    return matrix

def save_csv(matrix, output_path):
    if not matrix:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        csv.writer(file_obj).writerows(matrix)

