import plotly.graph_objects as go
import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
from typing import Dict
import pandas as pd


def detect_dual_axis_type(layout_dict) :
    """
    根据 layout_dict 判断是否存在笛卡尔双轴，以及属于哪种形式：
    - 'dual-x'   多 x 轴重叠（左右共用 y 轴）
    - 'dual-y'   多 y 轴重叠（上下共用 x 轴）
    - 'dual-xy'  同时 x、y 都有 overlay
    - None       没有双轴
    要求：layout_dict 必须是 fig.layout.to_plotly_json() 得到的纯 dict。
    """
    x_overlay = False
    y_overlay = False
    layout_dict =  layout_dict.to_plotly_json() 
    # 只遍历显式设置过的键，不会被默认值干扰
    for key, axis in layout_dict.items():
        if key.startswith(("xaxis", "yaxis")) and isinstance(axis, dict):
            overlaying = axis.get("overlaying")
            if key.startswith("xaxis") and overlaying == "x":
                x_overlay = True
            elif key.startswith("yaxis") and overlaying == "y":
                y_overlay = True

    if x_overlay and not y_overlay:
        return "dual-x"
    if y_overlay and not x_overlay:
        return "dual-y"
    if x_overlay and y_overlay:
        return "dual-xy"
    return None

import plotly.graph_objects as go

def detect_layout_type(fig: go.Figure) -> str:
    """
    判定 Plotly Figure 布局类型：
    - 'single'   单轴 / 单极坐标
    - 'dual'     笛卡尔双轴 (overlay)
    - 'subplots' 多极坐标、极+笛混排、grid 子图、或多组 x/y
    """
    layout = fig.layout.to_plotly_json()            # 只含显式键

    # ---------- 1) 统计“真正被使用”的 polar 坐标系 ----------
    used_polar = set()
    for tr in fig.data:
        if tr.type.endswith("polar"):
            # subplot 名缺省为 'polar'；否则 'polar2'、'polar3'...
            used_polar.add(getattr(tr, "subplot", "polar"))
    n_polar_used = len(used_polar)

    if n_polar_used > 1:                 # 多个被使用的 polar → 子图
        return "subplots"

    # ---------- 2) layout.grid(rows × cols) ----------
    grid = layout.get("grid")
    if grid and (grid.get("rows", 1) or 1) * (grid.get("columns", 1) or 1) > 1:
        return "subplots"

    # ---------- 3) 笛卡尔 overlaying → dual ----------
    for k, ax in layout.items():
        if k.startswith(("xaxis", "yaxis")) and isinstance(ax, dict):
            if ax.get("overlaying") in ("x", "y"):
                return "dual"

    # ---------- 4) 无极坐标：多 xaxis/yaxis 也是子图 ----------
    if n_polar_used == 0:
        x_cnt = sum(1 for k in layout if k.startswith("xaxis"))
        y_cnt = sum(1 for k in layout if k.startswith("yaxis"))
        if x_cnt > 1 or y_cnt > 1:
            return "subplots"
        return "single"

    # ---------- 5) 恰有 1 个 polar：若存在笛卡尔 trace ➔ 子图 ----------
    has_cartesian = any(not tr.type.endswith("polar") for tr in fig.data)
    return "subplots" if has_cartesian else "single"



def plotly_subplots_layout(fig: go.Figure) -> List[Dict[str, Any]]:
    """
    推断 Figure 的网格布局（cartesian + polar）。
    - 仅统计真正被 trace 使用的 xaxis/yaxis/polar
    - 若只有 polar 且它们都无 domain → 默认横向 1×N
    """
    out: List[Dict[str, Any]] = []

    layout: Dict[str, Any] = fig.layout.to_plotly_json()
    data   = fig.to_plotly_json()["data"]        # list[dict]

    # ---------- 1) 统计 trace 使用的坐标系 ----------
    used_x, used_y, used_polar = set(), set(), []
    for tr in data:
        if tr.get("type", "").endswith("polar"):
            subplot = tr.get("subplot", "polar")            # 'polar', 'polar2', ...
            used_polar.append(subplot)
        else:
            used_x.add(tr.get("xaxis", "x"))                # 'x', 'x2', ...
            used_y.add(tr.get("yaxis", "y"))

    used_polar = list(dict.fromkeys(used_polar))  # 去重且保顺序

    # ---------- 2) 收集 domain ----------
    x_dom, y_dom = {}, {}
    # (a) cartesian
    for name, ax in layout.items():
        if name.startswith("xaxis") and name.replace("axis", "") in used_x:
            dom = ax.get("domain")
            if dom: x_dom[name] = tuple(dom)
        if name.startswith("yaxis") and name.replace("axis", "") in used_y:
            dom = ax.get("domain")
            if dom: y_dom[name] = tuple(dom)

    # (b) polar
    polar_no_domain = []
    for p_name in used_polar:                           # 只看被用到的 polar
        p_dict = layout.get(p_name, {})
        dom = p_dict.get("domain")
        if dom and dom.get("x") and dom.get("y"):
            x_dom[p_name] = tuple(dom["x"])
            y_dom[p_name] = tuple(dom["y"])
        else:
            polar_no_domain.append(p_name)

    # ---------- 3) 特例：仅无-domain polar ----------
    if not x_dom and not y_dom and polar_no_domain:
        n = len(polar_no_domain)
        for idx, p_name in enumerate(polar_no_domain):
            out.append(dict(
                type="polar",
                name=p_name,
                nrows=1, ncols=n,
                row_start=0, row_end=0,
                col_start=idx, col_end=idx,
            ))
        return out

    # ---------- 4) 若无坐标系可推断 ----------
    if not x_dom and not y_dom:
        return out

    # ---------- 5) 正常行列映射 ----------
    all_x = sorted(set(x_dom.values()))
    all_y = sorted(set(y_dom.values()))
    if not all_x: all_x = [(0.0, 1.0)]
    if not all_y: all_y = [(0.0, 1.0)]
    ncols, nrows = len(all_x), len(all_y)
    col_map = {d: i for i, d in enumerate(all_x)}
    row_map = {d: i for i, d in enumerate(all_y)}

    # ---------- 6) 输出 cartesian 块 ----------
    added_cart = set()
    for tr in data:
        if tr.get("type", "").endswith("polar"): continue
        xname = f'xaxis{"" if tr.get("xaxis","x")=="x" else tr["xaxis"][1:]}'
        yname = f'yaxis{"" if tr.get("yaxis","y")=="y" else tr["yaxis"][1:]}'
        if (xname, yname) in added_cart:
            continue
       
        out.append((
            "cartesian",
            nrows, ncols,
            row_map.get(y_dom.get(yname, (0,1)), 0),
            row_map.get(y_dom.get(yname, (0,1)), 0),
            col_map.get(x_dom.get(xname, (0,1)), 0),
            col_map.get(x_dom.get(xname, (0,1)), 0),
        ))
        added_cart.add((xname, yname))

    # ---------- 7) 输出 polar 块 (有 domain 的) ----------
    for p_name in used_polar:
        if p_name in x_dom:         # 说明有 domain
            xinterval = x_dom[p_name]
            yinterval = y_dom[p_name]
            out.append((
            "polar",
            nrows, ncols,
            row_map.get(y_dom.get(p_name, (0,1)), 0),
            row_map.get(y_dom.get(p_name, (0,1)), 0),
            col_map.get(x_dom.get(p_name, (0,1)), 0),
            col_map.get(x_dom.get(p_name, (0,1)), 0),
        ))

    return out

def calculate_layout_metrics(fig_gold, fig_gen) -> Dict[str, float]:
    
    gold_type = detect_layout_type(fig_gold)
    gen_type = detect_layout_type(fig_gen)
   

    if gold_type != gen_type:
        return {"precision": 0, "recall": 0, "f1": 0}
    else:
        if gold_type == "single":
            return {"precision": 1, "recall": 1, "f1": 1}
        elif gold_type == "dual":
            gold_dual = detect_dual_axis_type(fig_gold.layout)
            gen_dual = detect_dual_axis_type(fig_gen.layout)
            if gold_dual != gen_dual:
                return {"precision": 0, "recall": 0, "f1": 0}
            return {"precision": 1, "recall": 1, "f1": 1}
        elif gold_type == "subplots":
            gold= plotly_subplots_layout(fig_gold)
            gen = plotly_subplots_layout(fig_gen)
        
            
            n_correct = 0
            gen_copy = gen.copy()

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

            
