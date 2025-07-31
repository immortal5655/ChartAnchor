import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import plotly.graph_objects as go
import pandas as pd

def clean_text(text: str) -> str:
    """
    Clean a given text string by:
    - Replacing non-breaking space characters with regular space
    - Removing line break tags
    - Removing all other HTML-like tags
    - Trimming and collapsing excessive spaces
    """
    if not isinstance(text, str):
        return text  # Return unchanged if not a string

    text = text.replace('\xa0', ' ')              # Replace non-breaking space
    text = text.replace('<br>', ' ')              # Replace line break with space
    text = re.sub(r'<[^>]+>', '', text)           # Remove HTML-style tags
    text = re.sub(r'\s+', ' ', text).strip()      # Collapse spaces and trim
    return text

def extract_axis_titles(fig: go.Figure) -> Dict[str, List[str]]:
    """
    Extract axis titles from a figure object.
    Handles Cartesian, polar, and 3D scene coordinate systems.
    Returns a dictionary where keys are axis types and values are lists of axis titles.
    """
    titles: Dict[str, List[str]] = defaultdict(list)

    cartesian_pat = re.compile(r"^[xy]axis(\d*)$")
    polar_pat     = re.compile(r"^polar(\d*)$")
    scene_pat     = re.compile(r"^scene(\d*)$")

    has_cartesian = False
    has_scene     = False

    # --- Extract titles from 2D Cartesian axes ---
    for attr in fig.layout:
        if cartesian_pat.match(attr):
            axis = getattr(fig.layout, attr, None)
            txt  = getattr(axis.title, "text", None) if axis and axis.title else None
            if txt:
                titles[attr[:5]].append(txt)
                has_cartesian = True

    # --- Extract titles from polar coordinate axes ---
    for attr in fig.layout:
        if not polar_pat.match(attr):
            continue
        polar = getattr(fig.layout, attr, None)
        if not polar:
            continue

        for key in ("angularaxis", "radialaxis"):
            ax = getattr(polar, key, None)
            txt = getattr(ax.title, "text", None) if ax and getattr(ax, "title", None) else None
            if txt:
                titles[key].append(txt)

    # --- Extract titles from 3D scene axes ---
    for attr in fig.layout:
        if not scene_pat.match(attr):
            continue
        scene = getattr(fig.layout, attr, None)
        if not scene:
            continue

        for key in ("xaxis", "yaxis", "zaxis"):
            ax = getattr(scene, key, None)
            txt = getattr(ax.title, "text", None) if ax and getattr(ax, "title", None) else None
            if txt:
                titles[key].append(txt)
                has_scene = True

    # --- Remove duplicates only if both 2D and 3D axes are present ---
    if has_cartesian and has_scene:
        for key in ("xaxis", "yaxis"):
            if key in titles:
                titles[key] = list(dict.fromkeys(titles[key]))  # Remove duplicates, preserve order

    return titles

def calculate_singal_axis_metrics(gold, gen):
    """
    Compute precision, recall, and F1 score for a single axis type.
    Matches labels based on exact equality.
    """
    n_correct = 0
    gen_copy = gen.copy()

    gold = list(set(gold))
    gen_copy = list(set(gen_copy))

    for label in gold:
        if label in gen_copy:
            n_correct += 1
            gen_copy.remove(label)

    if len(gold) == 0:
        if len(gen_copy) == 0:
            return {"precision": 1, "recall": 1, "f1": 1}
        else:
            return {"precision": 0, "recall": 0, "f1": 0}

    precision = n_correct / len(gen) if gen else 0
    recall = n_correct / len(gold)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_axis_metrics(fig_gold, fig_gen) -> Dict[str, float]:
    """
    Compare axis titles between two figures.
    Returns overall average precision, recall, and F1 score.
    """
    gold_names = extract_axis_titles(fig_gold)
    gen_names = extract_axis_titles(fig_gen)
    all_metrics = {}

    if len(gold_names) == 0:
        if len(gen_names) == 0:
            return {"precision": 1, "recall": 1, "f1": 1}
        else:
            return {"precision": 0, "recall": 0, "f1": 0}

    for key, gold_items in gold_names.items():
        if key not in gen_names:
            all_metrics[key] = {"precision": 0, "recall": 0, "f1": 0}
        else:
            gen_items = gen_names[key]
            all_metrics[key] = calculate_singal_axis_metrics(gold_items, gen_items)

    # Aggregate metrics over all axis types
    precision = sum(v["precision"] for v in all_metrics.values())
    recall = sum(v["recall"] for v in all_metrics.values())
    f1 = sum(v["f1"] for v in all_metrics.values())
    num_axis = len(all_metrics)

    precision = precision / num_axis if num_axis else 0
    recall = recall / num_axis if num_axis else 0
    f1 = f1 / num_axis if num_axis else 0

    return {"precision": precision, "recall": recall, "f1": f1}
