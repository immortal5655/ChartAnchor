import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import plotly.graph_objects as go
from typing import Dict
import pandas as pd

def clean_text(text: str) -> str:
    """
    批量清洗字符串：
    - 去除\xa0 (non-breaking space)
    - 去除<br>标签
    - 去除其他HTML标签（比如 <b>、<i>）
    - 去除多余空格
    """
    if not isinstance(text, str):
        return text  # 不是字符串就原样返回

    text = text.replace('\xa0', ' ')             # 把\xa0变成普通空格
    text = text.replace('<br>', ' ')              # 把<br>变成空格
    text = re.sub(r'<[^>]+>', '', text)            # 用正则删除所有HTML标签
    text = re.sub(r'\s+', ' ', text).strip()       # 多空格合成一个空格，并去掉首尾空格
    return text

# ==== Type相关处理 ====
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
    """计算chart type的precision、recall和f1"""
    gold_counts = extract_chart_type_counts(fig_gold)
    gen_counts = extract_chart_type_counts(fig_gen)
    
    return calculate_metrics(gold_counts,gen_counts)

def calculate_metrics(gold_counts,gen_counts):
    """通用的精确率、召回率、F1计算"""
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
