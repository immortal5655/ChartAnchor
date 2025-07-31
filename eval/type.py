import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import plotly.graph_objects as go
from typing import Dict
import pandas as pd

def clean_text(text: str) -> str:
    
    if not isinstance(text, str):
        return text  
    text = text.replace('\xa0', ' ')            
    text = text.replace('<br>', ' ')             
    text = re.sub(r'<[^>]+>', '', text)            
    text = re.sub(r'\s+', ' ', text).strip()       
    return text


def get_type(trace):
    mode = getattr(trace, 'mode', None)
    fill = getattr(trace, 'fill', None)
    ctype = getattr(trace, 'type', None)

    if ctype == 'scatter':
        if fill in  ['tozeroy', 'tozerox', 'tonexty', 'tonextx','toself', 'tonext']:
            return ('area')
        else:
            if mode == None:
                mode = 'lines+markers'
            if mode == 'lines':
                return('line')
            if mode in ['lines+markers','markers']:
                return('scatter')
    else:
        return ctype

def extract_chart_type_counts(fig):
    
    
    chart_type_counts = {}

    for trace in fig.data:
        chart_type = get_type(trace)
        if chart_type:
            chart_type_counts[chart_type] = chart_type_counts.get(chart_type, 0) + 1

    return chart_type_counts


def calculate_type_metrics(fig_gold, fig_gen) -> Dict[str, float]:
   
    gold_counts = extract_chart_type_counts(fig_gold)
    gen_counts = extract_chart_type_counts(fig_gen)
    
    return calculate_metrics(gold_counts,gen_counts)

def calculate_metrics(gold_counts,gen_counts):
   
    if not gen_counts or not gold_counts:
        return {"precision": 0, "recall": 0, "f1": 0}

    n_correct = 0
    total_gen = sum(gen_counts.values())
    total_gold = sum(gold_counts.values())

    if total_gen == 0 or total_gold == 0:
        return {"precision": 0, "recall": 0, "f1": 0}

    for chart_type, count in gen_counts.items():
        if chart_type in gold_counts:
            n_correct += min(count, gold_counts[chart_type])

    precision = n_correct / total_gen
    recall = n_correct / total_gold
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return {"precision": precision, "recall": recall, "f1": f1}
