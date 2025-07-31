import plotly.graph_objects as go
import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
from typing import Dict
import pandas as pd



def detect_dual_axis_type(layout_dict):
    """
    Detects whether the layout uses a dual Cartesian axis configuration and returns the type.

    Dual-axis types:
    - 'dual-x'   : Multiple x-axes overlaying the same y-axis (e.g., left/right)
    - 'dual-y'   : Multiple y-axes overlaying the same x-axis (e.g., top/bottom)
    - 'dual-xy'  : Both x and y axes are overlaid
    - None       : No dual-axis overlay detected

    Args:
        layout_dict (dict): The layout dictionary from `fig.layout.to_plotly_json()`.

    Returns:
        str or None: One of {'dual-x', 'dual-y', 'dual-xy'} or None if not dual-axis.
    """
    x_overlay = False
    y_overlay = False

    # Ensure we're working with the plain JSON dict representation
    layout_dict = layout_dict.to_plotly_json()

    # Check explicitly set axes only (ignores default-injected keys)
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
    Determine the layout type of a Plotly Figure.

    Returns one of:
    - 'single'   : Single axis or single polar plot
    - 'dual'     : Cartesian dual-axis layout (using overlaying)
    - 'subplots' : Multiple polars, mixed polar/cartesian, grid layout, or multiple x/y axes
    """
    layout = fig.layout.to_plotly_json()  # Only explicitly set keys are included

    # ---------- 1) Count polar coordinate systems actually used ----------
    used_polar = set()
    for tr in fig.data:
        if tr.type.endswith("polar"):
            # Default subplot name is 'polar'; others are 'polar2', 'polar3', etc.
            used_polar.add(getattr(tr, "subplot", "polar"))
    n_polar_used = len(used_polar)

    if n_polar_used > 1:
        return "subplots"  # Multiple polars in use → treated as subplots

    # ---------- 2) Check layout.grid(rows × columns) ----------
    grid = layout.get("grid")
    if grid and (grid.get("rows", 1) or 1) * (grid.get("columns", 1) or 1) > 1:
        return "subplots"

    # ---------- 3) Cartesian dual axis via overlaying ----------
    for k, ax in layout.items():
        if k.startswith(("xaxis", "yaxis")) and isinstance(ax, dict):
            if ax.get("overlaying") in ("x", "y"):
                return "dual"

    # ---------- 4) No polar traces: multiple x/y axes indicates subplots ----------
    if n_polar_used == 0:
        x_cnt = sum(1 for k in layout if k.startswith("xaxis"))
        y_cnt = sum(1 for k in layout if k.startswith("yaxis"))
        if x_cnt > 1 or y_cnt > 1:
            return "subplots"
        return "single"

    # ---------- 5) One polar used: if any cartesian trace exists, it's a subplot ----------
    has_cartesian = any(not tr.type.endswith("polar") for tr in fig.data)
    return "subplots" if has_cartesian else "single"




def plotly_subplots_layout(fig: go.Figure) -> List[Dict[str, Any]]:
    """
    Infers the subplot layout structure of a Plotly Figure (cartesian + polar).
    
    - Only includes coordinate systems actually used by the traces.
    - If only polar subplots are used and none of them define domain,
      assumes a horizontal layout (1 row × N columns).
    
    Returns:
        A list of layout blocks, each represented as a dictionary or tuple:
        - type: 'cartesian' or 'polar'
        - name: axis or subplot name
        - nrows, ncols: total grid dimensions
        - row_start, row_end, col_start, col_end: position in the grid
    """
    out: List[Dict[str, Any]] = []

    layout: Dict[str, Any] = fig.layout.to_plotly_json()
    data = fig.to_plotly_json()["data"]  # List of trace dictionaries

    # ---------- 1) Collect coordinate systems used by traces ----------
    used_x, used_y, used_polar = set(), set(), []
    for tr in data:
        if tr.get("type", "").endswith("polar"):
            subplot = tr.get("subplot", "polar")  # e.g. 'polar', 'polar2', ...
            used_polar.append(subplot)
        else:
            used_x.add(tr.get("xaxis", "x"))  # e.g. 'x', 'x2'
            used_y.add(tr.get("yaxis", "y"))

    used_polar = list(dict.fromkeys(used_polar))  # Remove duplicates, preserve order

    # ---------- 2) Collect domain info for used axes ----------
    x_dom, y_dom = {}, {}

    # (a) Cartesian axes
    for name, ax in layout.items():
        if name.startswith("xaxis") and name.replace("axis", "") in used_x:
            dom = ax.get("domain")
            if dom:
                x_dom[name] = tuple(dom)
        if name.startswith("yaxis") and name.replace("axis", "") in used_y:
            dom = ax.get("domain")
            if dom:
                y_dom[name] = tuple(dom)

    # (b) Polar axes
    polar_no_domain = []
    for p_name in used_polar:
        p_dict = layout.get(p_name, {})
        dom = p_dict.get("domain")
        if dom and dom.get("x") and dom.get("y"):
            x_dom[p_name] = tuple(dom["x"])
            y_dom[p_name] = tuple(dom["y"])
        else:
            polar_no_domain.append(p_name)

    # ---------- 3) Special case: only polar plots without domain ----------
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

    # ---------- 4) No usable axis domains ----------
    if not x_dom and not y_dom:
        return out

    # ---------- 5) Build grid mapping ----------
    all_x = sorted(set(x_dom.values()))
    all_y = sorted(set(y_dom.values()))
    if not all_x: all_x = [(0.0, 1.0)]
    if not all_y: all_y = [(0.0, 1.0)]

    ncols, nrows = len(all_x), len(all_y)
    col_map = {d: i for i, d in enumerate(all_x)}
    row_map = {d: i for i, d in enumerate(all_y)}

    # ---------- 6) Add cartesian layout blocks ----------
    added_cart = set()
    for tr in data:
        if tr.get("type", "").endswith("polar"):
            continue

        xname = f'xaxis{"" if tr.get("xaxis", "x") == "x" else tr["xaxis"][1:]}'
        yname = f'yaxis{"" if tr.get("yaxis", "y") == "y" else tr["yaxis"][1:]}'

        if (xname, yname) in added_cart:
            continue

        out.append((
            "cartesian",
            nrows, ncols,
            row_map.get(y_dom.get(yname, (0, 1)), 0),
            row_map.get(y_dom.get(yname, (0, 1)), 0),
            col_map.get(x_dom.get(xname, (0, 1)), 0),
            col_map.get(x_dom.get(xname, (0, 1)), 0),
        ))
        added_cart.add((xname, yname))

    # ---------- 7) Add polar layout blocks (with domain) ----------
    for p_name in used_polar:
        if p_name in x_dom:
            out.append((
                "polar",
                nrows, ncols,
                row_map.get(y_dom.get(p_name, (0, 1)), 0),
                row_map.get(y_dom.get(p_name, (0, 1)), 0),
                col_map.get(x_dom.get(p_name, (0, 1)), 0),
                col_map.get(x_dom.get(p_name, (0, 1)), 0),
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

            
