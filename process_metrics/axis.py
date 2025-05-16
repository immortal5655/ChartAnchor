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





def extract_axis_titles(fig: go.Figure) -> Dict[str, List[str]]:
    """
    收集 Plotly Figure 的各类坐标轴标题。
    - 当同图里同时出现笛卡尔和 3-D scene 轴时，xaxis/yaxis 重复标题会被去重
    - 否则保持原样
    返回示例：
        {
            'xaxis'      : ['Species 1'],
            'yaxis'      : ['Species 2'],
            'zaxis'      : ['Week'],
            'angularaxis': ['Direction (°)']
        }
    """
    # 暂时用 list；是否去重留到最后判断
    titles: Dict[str, List[str]] = defaultdict(list)

    cartesian_pat = re.compile(r"^[xy]axis(\d*)$")
    polar_pat     = re.compile(r"^polar(\d*)$")
    scene_pat     = re.compile(r"^scene(\d*)$")

    has_cartesian = False
    has_scene     = False

    # ---------------- 1) 笛卡尔 -------------------------------------------------
    for attr in fig.layout:
        if cartesian_pat.match(attr):
            axis = getattr(fig.layout, attr, None)
            txt  = getattr(axis.title, "text", None) if axis and axis.title else None
            if txt:
                titles[attr[:5]].append(txt)
                has_cartesian = True

    # ---------------- 2) 极坐标 -------------------------------------------------
    for attr in fig.layout:
        if not polar_pat.match(attr):
            continue
        polar = getattr(fig.layout, attr, None)
        if not polar:
            continue

        for key, ax_name in (("angularaxis", "angularaxis"), ("radialaxis", "radialaxis")):
            ax  = getattr(polar, ax_name, None)
            txt = getattr(ax.title, "text", None) if ax and getattr(ax, "title", None) else None
            if txt:
                titles[key].append(txt)

    # ---------------- 3) 3-D scene ---------------------------------------------
    for attr in fig.layout:
        if not scene_pat.match(attr):
            continue
        scene = getattr(fig.layout, attr, None)
        if not scene:
            continue

        for key in ("xaxis", "yaxis", "zaxis"):
            ax  = getattr(scene, key, None)
            txt = getattr(ax.title, "text", None) if ax and getattr(ax, "title", None) else None
            if txt:
                titles[key].append(txt)
                has_scene = True

    # ---------------- 4) 只在 2-D & 3-D 混用时去重 ------------------------------
    if has_cartesian and has_scene:
        for key in ("xaxis", "yaxis"):
            if key in titles:
                titles[key] = list(dict.fromkeys(titles[key]))  # 保顺序去重

    return titles





def calculate_singal_axis_metrics(gold, gen):

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
    """比较标题：都无→1，否则只有一边有或不相等→0"""
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
    
    precision, recall, f1 = 0, 0, 0
    for k,v in all_metrics.items():
        precision += v["precision"]
        recall += v["recall"]
        f1 += v["f1"]
    num_axis = len(all_metrics)

    precision = precision/num_axis if num_axis else 0
    recall =  recall/num_axis if num_axis else 0
    f1 =  f1/num_axis if num_axis else 0
    return {"precision": precision, "recall": recall, "f1": f1}
       
        
        

