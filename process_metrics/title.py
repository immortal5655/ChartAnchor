import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import plotly.graph_objects as go
from typing import Dict
import pandas as pd
import Levenshtein

tol_word =3


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

def calculate_title_metrics(fig_gold, fig_gen) -> Dict[str, float]:
    """比较标题：都无→1，否则只有一边有或不相等→0"""
    try:
        gold_title = (
            fig_gold.layout.title.text
            if fig_gold.layout.title and fig_gold.layout.title.text
            else None
        )
        gen_title = (
            fig_gen.layout.title.text
            if fig_gen.layout.title and fig_gen.layout.title.text
            else None
        )
    except:
        return {"precision": 0, "recall": 0, "f1": 0}
    if gold_title is not None:
        gold_title = clean_text(gold_title)
    if gen_title is not None:
        gen_title = clean_text(gen_title)
   
    if gold_title is None and gen_title is None:
        correct = True
    elif (gold_title is None) != (gen_title is None):
        correct = False
    else:
        if Levenshtein.distance(gold_title, gen_title) <= tol_word:
            correct = True
        else:
            correct = False
  

    value = 1 if correct else 0
    return {"precision": value, "recall": value, "f1": value}
