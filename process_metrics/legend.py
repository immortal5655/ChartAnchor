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

def extract_legend_labels(fig):
    """从figure中提取legend的名字列表。如果legend隐藏，返回None。"""
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
    gold_legends = extract_legend_labels(fig_gold)
    gen_legends = extract_legend_labels(fig_gen)
   

    # ---- 判断逻辑 ----
    if gold_legends is None and gen_legends is None:
        # 都不显示，判定正确
        return {"precision": 1, "recall": 1, "f1": 1}
    
    if (gold_legends is None) != (gen_legends is None):
        # 一个显示，一个不显示，判定错误
        return {"precision": 0, "recall": 0, "f1": 0}
    
    # 如果两个都显示，继续正常比较
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

