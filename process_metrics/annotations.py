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
def clean_list(ll):
    cleaned = []
    for x in ll:
        if x in (None, ''):
            continue
        try:
            cleaned.append(float(x))   # 能转成 float 的，转
        except (ValueError, TypeError):
            cleaned.append(x)           # 转不了的，保留原样
    return cleaned



def extract_annotations(fig):
    """
    提取 Plotly 图中所有 annotations 的文本内容，返回一个列表
    """
    if not hasattr(fig.layout, "annotations") or fig.layout.annotations is None:
        return []

    return clean_list([clean_text(anno.text) for anno in fig.layout.annotations if hasattr(anno, "text") and anno.text])


def calculate_annotations_metrics(fig_gold, fig_gen):
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
            gen_copy.remove(label)

    precision = n_correct / len(gen_annotations) if gen_annotations else 0
    recall = n_correct / len(gold_annotations) 
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    return {"precision": precision, "recall": recall, "f1": f1}
    
