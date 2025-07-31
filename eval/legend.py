import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import plotly.graph_objects as go
from typing import Dict
import pandas as pd


def clean_text(text: str) -> str:
    """
    Batch clean a string by:
    - Removing non-breaking spaces (\xa0)
    - Removing <br> tags
    - Removing other HTML tags (e.g., <b>, <i>)
    - Collapsing multiple spaces into one and trimming leading/trailing whitespace
    """
    if not isinstance(text, str):
        return text  # Return as-is if not a string

    text = text.replace('\xa0', ' ')                  # Replace non-breaking space with normal space
    text = text.replace('<br>', ' ')                  # Replace <br> with space
    text = re.sub(r'<[^>]+>', '', text)               # Remove all HTML tags
    text = re.sub(r'\s+', ' ', text).strip()          # Collapse whitespace and strip ends
    return text

def extract_legend_labels(fig):
    """
    Extract a list of legend labels from a figure.
    Returns None if the legend is hidden.
    """
    try:
        if hasattr(fig.layout, 'showlegend') and fig.layout.showlegend is False:
            return None
    except:
        return None

    legend = fig.layout.legend
    if legend is not None and hasattr(legend, 'visible') and legend.visible is False:
        return None

    labels = []
    for trace in fig.data:
        if hasattr(trace, 'showlegend') and trace.showlegend is False:
            continue
        if hasattr(trace, 'name') and trace.name:
            labels.append(clean_text(trace.name))
    return labels

def calculate_legend_metrics(fig_gold, fig_gen):
    """
    Compare the legend labels between a gold figure and a generated figure.
    Returns a dictionary with precision, recall, and F1 score.
    """
    gold_legends = extract_legend_labels(fig_gold)
    gen_legends = extract_legend_labels(fig_gen)

    # ---- Matching logic ----
    if gold_legends is None and gen_legends is None:
        # Both legends are hidden → correct
        return {"precision": 1, "recall": 1, "f1": 1}

    if (gold_legends is None) != (gen_legends is None):
        # One is shown and one is hidden → incorrect
        return {"precision": 0, "recall": 0, "f1": 0}

    # If both are visible, proceed with label comparison
    gold_legends = gold_legends or []
    gen_legends = gen_legends or []

    gold_legends = list(set(gold_legends))
    gen_legends = list(set(gen_legends))

    n_correct = 0
    gen_copy = gen_legends.copy()

    for label in gold_legends:
        if label in gen_copy:
            n_correct += 1
            gen_copy.remove(label)

    if len(gold_legends) == 0:
        if len(gen_legends) == 0:
            return {"precision": 1, "recall": 1, "f1": 1}
        else:
            return {"precision": 0, "recall": 0, "f1": 0}

    precision = n_correct / len(gen_legends) if gen_legends else 0
    recall = n_correct / len(gold_legends)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}