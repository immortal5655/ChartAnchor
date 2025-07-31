import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import plotly.graph_objects as go
import pandas as pd

def clean_text(text: str) -> str:
    """
    Cleans a text string by:
    - Removing non-breaking space characters
    - Replacing line break tags with space
    - Removing any remaining HTML-like tags
    - Collapsing multiple spaces into one and trimming surrounding whitespace
    """
    if not isinstance(text, str):
        return text  # Return original if not a string

    text = text.replace('\xa0', ' ')             # Replace non-breaking space with regular space
    text = text.replace('<br>', ' ')             # Replace line break tags with space
    text = re.sub(r'<[^>]+>', '', text)          # Remove all HTML-style tags
    text = re.sub(r'\s+', ' ', text).strip()     # Collapse multiple spaces and trim
    return text

def clean_list(ll):
    """
    Processes a list by removing empty entries and converting numeric strings to floats.
    Keeps non-numeric values as-is.
    """
    cleaned = []
    for x in ll:
        if x in (None, ''):
            continue  # Skip empty values
        try:
            cleaned.append(float(x))  # Convert to float if possible
        except (ValueError, TypeError):
            cleaned.append(x)         # Leave unchanged if conversion fails
    return cleaned

def extract_annotations(fig):
    """
    Extracts text content from annotations in a visual figure.
    Cleans each annotation and returns a list of the processed results.
    """
    if not hasattr(fig.layout, "annotations") or fig.layout.annotations is None:
        return []

    return clean_list([
        clean_text(anno.text)
        for anno in fig.layout.annotations
        if hasattr(anno, "text") and anno.text
    ])

def calculate_annotations_metrics(fig_gold, fig_gen):
    """
    Compares annotation texts between a reference figure and a generated figure.
    Returns a dictionary with precision, recall, and F1 score metrics.
    """
    gold_annotations = extract_annotations(fig_gold)
    gen_annotations = extract_annotations(fig_gen)

    if len(gold_annotations) == 0:
        if len(gen_annotations) == 0:
            return {"precision": 1, "recall": 1, "f1": 1}
        else:
            return {"precision": 0, "recall": 0, "f1": 0}

    n_correct = 0
    gen_copy = gen_annotations.copy()

    for label in gold_annotations:
        if label in gen_copy:
            n_correct += 1
            gen_copy.remove(label)  # Avoid double-counting matches

    precision = n_correct / len(gen_annotations) if gen_annotations else 0
    recall = n_correct / len(gold_annotations)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}
