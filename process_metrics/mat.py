import os
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.patches import Rectangle
import numpy as np
import importlib.util
from importlib.util import spec_from_file_location, module_from_spec
import matplotlib.colors as mcolors
from matplotlib.text import Annotation
import ast, builtins, io, contextlib, types, os
from color import *
from data import *


EPS=1e-6
tol_str_num =3
tol_number = 0.1
tol_word = 3


def extract_from_ax(ax):
    result = []

    # --- Line ---
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        label = line.get_label()
        trace = {'type': 'line', 'x': x.tolist(), 'y': y.tolist()}
        if label and not label.startswith("_"):
            trace['label'] = label
        result.append(trace)

    # --- Area (fill_between) ---
    for coll in ax.collections:
        if isinstance(coll, PolyCollection) and not isinstance(coll, PathCollection):
            if coll.get_paths():
                verts = coll.get_paths()[0].vertices
                n = len(verts) // 2
                x, y = verts[:n, 0], verts[:n, 1]
                trace = {'type': 'area', 'x': x.tolist(), 'y': y.tolist()}
                label = coll.get_label()
                if label and not label.startswith("_"):
                    trace['label'] = label
                result.append(trace)

    # --- Scatter ---
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                x, y = zip(*offsets)
                trace = {'type': 'scatter', 'x': list(x), 'y': list(y)}
                label = coll.get_label()
                if label and not label.startswith("_"):
                    trace['label'] = label
                result.append(trace)

    # --- Bar / Barh ---
    EPS = 1e-9
    for patch in ax.patches:
        if not isinstance(patch, Rectangle): continue
        if not patch.get_transform().contains_branch(ax.transData): continue
        x, y = patch.get_x(), patch.get_y()
        w, h = patch.get_width(), patch.get_height()
        label = patch.get_label()
        trace = {}
        if abs(x) < EPS:  # barh
            trace = {'type': 'barh', 'x': [w], 'y': [y + h / 2]}
        elif abs(y) < EPS:  # bar
            trace = {'type': 'bar', 'x': [x + w / 2], 'y': [h]}
        if label and not label.startswith("_"):
            trace['label'] = label
        if trace:
            result.append(trace)
    return result

def run_and_extract(py_path):
    plt.close('all')
    spec = importlib.util.spec_from_file_location("mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[ERROR] {py_path}: {e}")
        return None

    fig = plt.gcf()
    if not fig.axes:
        return None
    ax = fig.gca()
    return extract_from_ax(ax)

def batch_extract_all(py_dir):
    all_data = {}
    for fname in os.listdir(py_dir):
        if fname.endswith(".py"):
            path = os.path.join(py_dir, fname)
            chart_data = run_and_extract(path)
            if chart_data:
                all_data[fname] = chart_data
    return all_data




def _as_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return None

def _hex(c):
    try:
        return mcolors.to_hex(mcolors.to_rgb(c))
    except Exception:
        return str(c)

def _exec_vars(code: str):
    safe_builtins = {k: getattr(builtins, k) for k in
                     ['abs','min','max','sum','range','len','list','float','int']}
    safe_builtins['__import__'] = __import__
    g = {'__builtins__': safe_builtins, 'np': __import__('numpy')}
    l = {}
    exec_lines = [ln for ln in code.splitlines()
                  if not ln.strip().startswith(('plt.savefig', 'plt.show', '#'))]
    try:
        exec("\n".join(exec_lines), g, l)
    except Exception as e:
        print(f"[exec_vars] Warning: {e}")
    return l

def _from_objects():
    fig = plt.gcf()
    if not fig.axes:
        return []
    ax = fig.gca()
    out = []

    for ln in ax.lines:
        label = ln.get_label()
        out.append({
            'type': 'line',
            'x': ln.get_xdata().tolist(),
            'y': ln.get_ydata().tolist(),
            'label': None if label.startswith('_') else label,
            'color': _hex(ln.get_color())
        })

    for coll in ax.collections:
        if isinstance(coll, PathCollection) and coll.get_offsets().shape[0] > 1:
            x, y = zip(*coll.get_offsets())
            out.append({
                'type': 'scatter',
                'x': list(x),
                'y': list(y),
                'label': None if coll.get_label().startswith('_') else coll.get_label(),
                'color': _hex(coll.get_facecolor()[0])
            })

    xtick = {round(t, 2): lab.get_text() for t, lab in zip(ax.get_xticks(), ax.get_xticklabels())}
    ytick = {round(t, 2): lab.get_text() for t, lab in zip(ax.get_yticks(), ax.get_yticklabels())}
    for p in ax.patches:
        if isinstance(p, Rectangle):
            x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            c = _hex(p.get_facecolor())
            if h > w:
                cat = xtick.get(round(x + w / 2, 2), x + w / 2)
                out.append({'type': 'bar', 'x': [cat], 'y': [h], 'label': None, 'color': c})
            elif w > h:
                cat = ytick.get(round(y + h / 2, 2), y + h / 2)
                out.append({'type': 'barh', 'x': [cat], 'y': [w], 'label': None, 'color': c})

    for pc in ax.collections:
        if isinstance(pc, PolyCollection):
            if pc.get_offsets().size:
                continue
            c_hex = _hex(pc.get_facecolor()[0])
            verts = pc.get_paths()[0].vertices
            n = len(verts) // 2
            x, y = zip(*verts[:n])
            out.append({'type': 'area', 'x': list(x), 'y': list(y), 'label': None, 'color': c_hex})

    return out

def smart_extract_plot_data(path: str, allow_exec=True):
    plt.close('all')
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    tree = ast.parse(code)
    res = []

    sym = {}

    def _resolve_static(node):
        if isinstance(node, ast.Name):
            return sym.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [_resolve_static(e) for e in node.elts]
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                # 支持 np.array([...])
                if func.attr == 'array' and len(node.args) == 1:
                    return _resolve_static(node.args[0])

                # ✅ 新增支持：np.arange(start, stop, step)
                if func.attr == 'arange':
                    try:
                        args = [eval(compile(ast.Expression(arg), filename="<ast>", mode="eval")) for arg in node.args]
                        return list(np.arange(*args))
                    except Exception as e:
                        print(f"[resolve_static] np.arange 解析失败: {e}")
        return None


    class Static(ast.NodeVisitor):
        def visit_Assign(self, n):
            target = n.targets[0]
            val = n.value

            # Handle list/tuple assignment
            if isinstance(target, ast.Name) and isinstance(val, (ast.List, ast.Tuple)):
                sym[target.id] = [getattr(e, 'value', getattr(e, 'n', getattr(e, 's', None))) for e in val.elts]

            # Handle np.array([...])
            if isinstance(target, ast.Name) and isinstance(val, ast.Call):
                if isinstance(val.func, ast.Attribute) and val.func.attr == 'array':
                    if len(val.args) == 1 and isinstance(val.args[0], (ast.List, ast.Tuple)):
                        sym[target.id] = [getattr(e, 'value', getattr(e, 'n', getattr(e, 's', None))) for e in val.args[0].elts]

            self.generic_visit(n)

        def visit_Call(self, n):
            if isinstance(n.func, ast.Attribute) and n.func.attr in {'plot', 'scatter', 'bar', 'barh', 'fill_between'}:
                args = n.args
                if len(args) >= 2:
                    x = _resolve_static(args[0])
                    y = _resolve_static(args[1])
                    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
                        chart_type = {'fill_between': 'area'}.get(n.func.attr, n.func.attr)
                        label = None
                        color = None
                        for kw in n.keywords:
                            if kw.arg == 'label' and isinstance(kw.value, (ast.Str, ast.Constant)):
                                label = getattr(kw.value, 's', getattr(kw.value, 'value', None))
                            if kw.arg == 'color' and isinstance(kw.value, (ast.Str, ast.Constant)):
                                color = getattr(kw.value, 's', getattr(kw.value, 'value', None))
                        res.append({'type': chart_type, 'x': x, 'y': y,
                                    'label': label, 'color': _hex(color) if color else None})
            self.generic_visit(n)

    Static().visit(tree)
    if res:
        return res

    if not allow_exec:
        return res

    # Runtime fallback
    mod = types.ModuleType("tmpmod")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(code, mod.__dict__)
        except Exception as e:
            print(f"[exec fallback] Error: {e}")
            return res

    res.extend(_from_objects())
    plt.close('all')
    return res


def resolve_color(color):
    try:
        rgb = mcolors.to_rgb(color)
        hex_val = mcolors.to_hex(rgb)
        name = next((n for n, h in mcolors.CSS4_COLORS.items() if mcolors.to_rgb(h) == rgb), None)
        return hex_val, name
    except Exception:
        return None, None

def extract_chart_properties_runtime(filepath):
    plt.close('all')  # 清除上一个图
    spec = spec_from_file_location("chart_module", filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    fig = plt.gcf()
    ax = fig.gca()

    properties = {
        "types": [],
        "colors": [],
        "legend": False,
        "legend_labels": [],
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "annotations": []
    }

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        legend = ax.legend()
    else:
        legend = None

    # Line
    for ln in ax.lines:
        ls = ln.get_linestyle() or ''
        chart_type = "scatter" if ls.strip().lower() in ('', 'none') else "line"
        hex_val, name = resolve_color(ln.get_color())
        lbl = ln.get_label()
        if lbl and not lbl.startswith("_"):
            properties["legend"] = True
            if lbl not in properties["legend_labels"]:
                properties["legend_labels"].append(lbl)
        properties["types"].append(chart_type)
        if hex_val:
            properties["colors"].append({
                "hex": hex_val, "name": name,
                "attribute": chart_type, "param": "color"
            })

    # Scatter (PathCollection)
    for coll in ax.collections:
        if isinstance(coll, PathCollection) and coll.get_offsets().shape[0] > 0:
            facecolors = coll.get_facecolor()
            if len(facecolors) == 0:
                continue
            hex_val, name = resolve_color(facecolors[0])
            lbl = coll.get_label()
            if lbl and not lbl.startswith("_"):
                properties["legend"] = True
                if lbl not in properties["legend_labels"]:
                    properties["legend_labels"].append(lbl)
            properties["types"].append("scatter")
            if hex_val:
                properties["colors"].append({
                    "hex": hex_val, "name": name,
                    "attribute": "scatter", "param": "facecolor"
                })

    # Bar / Barh
    EPS = 1e-9
    for patch in ax.patches:
        if not isinstance(patch, Rectangle):
            continue
        if not patch.get_transform().contains_branch(ax.transData):
            continue
        is_barh = abs(patch.get_x()) < EPS
        is_bar = abs(patch.get_y()) < EPS
        if not (is_bar or is_barh):
            continue
        chart_type = "barh" if is_barh else "bar"
        hex_val, name = resolve_color(patch.get_facecolor())
        lbl = patch.get_label() or ""
        if lbl and not lbl.startswith("_"):
            properties["legend"] = True
            if lbl not in properties["legend_labels"]:
                properties["legend_labels"].append(lbl)
        properties["types"].append(chart_type)
        if hex_val:
            properties["colors"].append({
                "hex": hex_val,
                "name": name,
                "attribute": chart_type,
                "param": "facecolor"
            })

    # Area (PolyCollection)
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            facecolors = coll.get_facecolor()
            if len(facecolors) == 0:
                continue
            hex_val, name = resolve_color(facecolors[0])
            properties["types"].append("area")
            if hex_val:
                properties["colors"].append({
                    "hex": hex_val,
                    "name": name,
                    "attribute": "area",
                    "param": "facecolor"
                })
            break  # 通常只有一个填充面

    # Annotations
    for child in ax.get_children():
        if isinstance(child, Annotation):
            properties["annotations"].append(child.get_text())

    plt.close(fig)
    return properties
import ast

import ast
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def to_hex_safe(color_str):
    """尝试将颜色字符串转换为 hex"""
    try:
        return mcolors.to_hex(mcolors.to_rgb(color_str))
    except Exception:
        return None

def extract_colors_from_ast(code: str):
    tree = ast.parse(code)
    color_dict = {"color": [], "facecolor": [], "edgecolor": []}

    class ColorExtractor(ast.NodeVisitor):
        def visit_Call(self, node):
            for kw in node.keywords:
                if kw.arg in color_dict:
                    val = None
                    if isinstance(kw.value, ast.Constant):
                        val = kw.value.value
                    elif isinstance(kw.value, ast.Str):  # for older Python versions
                        val = kw.value.s
                    hex_val = to_hex_safe(val)
                    if hex_val:
                        color_dict[kw.arg].append(hex_val)
            self.generic_visit(node)

    ColorExtractor().visit(tree)
    # 去重排序
    for k in color_dict:
        color_dict[k] = sorted(set(color_dict[k]))
    return {k: v for k, v in color_dict.items() if v}

def extract_colors_runtime():
    fig = plt.gcf()
    ax = fig.gca()
    color_dict = {"color": [], "facecolor": [], "edgecolor": []}

    def safe_to_hex(c):
        try:
            if isinstance(c, (list, tuple, np.ndarray)) and len(c) in [3, 4]:
                return mcolors.to_hex(c)
            elif isinstance(c, str):
                return mcolors.to_hex(mcolors.to_rgb(c))
        except Exception:
            return None

    # Lines
    for line in ax.lines:
        c = safe_to_hex(line.get_color())
        if c: color_dict["color"].append(c)

    # Patches (e.g., bars)
    for patch in ax.patches:
        for attr in ["facecolor", "edgecolor"]:
            val = getattr(patch, f"get_{attr}")()
            if isinstance(val, (list, tuple, np.ndarray)):
                val = val[0] if isinstance(val, (list, tuple)) and isinstance(val[0], (list, tuple, np.ndarray)) else val
            c = safe_to_hex(val)
            if c: color_dict[attr].append(c)

    # 去重 & 排序
    for k in color_dict:
        color_dict[k] = sorted(set(color_dict[k]))

    return {k: v for k, v in color_dict.items() if v}

def extract_all_colors(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # 静态
    static_colors = extract_colors_from_ast(code)

    # 运行代码
    plt.close('all')
    try:
        exec(code, {})  # 可加 sandbox 限制 builtins
    except Exception as e:
        print(f"Runtime error: {e}")

    runtime_colors = extract_colors_runtime()

    # 合并
    merged = {"color": [], "facecolor": [], "edgecolor": []}
    for key in merged:
        merged[key] = sorted(set(static_colors.get(key, []) + runtime_colors.get(key, [])))

    return {k: v for k, v in merged.items() if v}



def clean_text(text):
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
def calculate_title_metrics_mat(gold, gen):
    gold_title = gold['title']
    gen_title = gen['title']
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
def calculate_axis_metrics_mat(gold, gen):
    gold = [gold['xlabel'], gold['ylabel']]
    gen = [gen['xlabel'], gen['ylabel']]
    n_correct = 0
    gen_copy = gen.copy()
    
    gold_legends = list(set(gold))
    gen_legends = list(set(gen))


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
def calculate_legend_metrics_mat(gold,gen):

    gold_legend , gold_legend_labels = gold['legend'],gold['legend_labels']
    gen_legend , gen_legend_labels = gen['legend'],gen['legend_labels']
    if gold_legend!=gen_legend:
        return {"precision": 0, "recall": 0, "f1": 0}
    if gold_legend==False:
        return {"precision": 1, "recall": 1, "f1": 1}
    else:
        gold_legends = gold_legend_labels or []
        gen_legends = gen_legend_labels or []
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
def extract_chart_type_counts(types):
    
    
    chart_type_counts = {}

    for chart_type in types:
        if chart_type:
            chart_type_counts[chart_type] = chart_type_counts.get(chart_type, 0) + 1

    return chart_type_counts
def calculate_type_metrics_mat(gold, gen):

    gold_types = gold['types']
    gen_types = gen['types']
    gen_counts = extract_chart_type_counts(gen_types)
    gold_counts = extract_chart_type_counts(gold_types)
    
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


def calculate_color_metrics_mat(gold_path,gen_path):
    gen_group = extract_all_colors(gen_path)
    gold_group = extract_all_colors(gold_path)
    gold_keys = list(gold_group.keys())
    gen_keys = list(gen_group.keys())
    merged_color_group = list( set( gold_keys + gen_keys ) )
    for color in merged_color_group:
        if color not in gen_group:
            gen_group[color] = []
        if color not in gold_group:
            gold_group[color] = []
    sim = 0

    gen_length = sum(len(v) for v in gen_group.values())
    gold_length = sum(len(v) for v in gold_group.values())
    for color in merged_color_group:
        sim += compute_precision_hungarian(gold_group[color], gen_group[color])
        

    precision = sim / gen_length if gen_length != 0 else 0
    recall = sim / gold_length if gold_length != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

def get_tuples(data):
    tuples=[]
    for i in range(len(data)):
        item = data[i]
        x = item['x']
        y = item['y']
        if isinstance(x,list) and isinstance(y,list):
            min_len = min(len(x),len(y))
            for j in range(min_len):
                tuples.append((x[j],y[j]))
                if 'label' in item:
                    if item['label'] != None:
                        tuples.append((item['label'],x[j],y[j]))
                    else:
                        tuples.append((x[j],y[j]))
                else: 
                    tuples.append((x[j],y[j]))
        if (not isinstance(x,list)) and (not isinstance(y,list)):
            tuples.append((x,y))
            if 'label' in item:
                if item['label'] != None:
                    tuples.append((item['label'],x,y))
                else:
                    tuples.append((x,y))
            else:
                tuples.append((x,y))
    return tuples

def calculate_data_metrics_mat(gold_path,gen_path):
    gold_data = smart_extract_plot_data(gold_path)
    gen_data = smart_extract_plot_data(gen_path)
    gold = get_tuples(gold_data)
    gen = get_tuples(gen_data)
    intersection = intersection_with_tolerance(gold,gen,tol_str_num,tol_number)
    union = union_with_tolerance(gold,gen,tol_str_num,tol_number)
    sim = len(intersection)/(len(union)+1e-7)
    return {"f1":sim}

def calculate_mat_metrics(gold_path,gen_path):
    metrics = {
        "title": calculate_title_metrics_mat,
        "axis": calculate_axis_metrics_mat,
        "legend": calculate_legend_metrics_mat,
        "type": calculate_type_metrics_mat,
    }
    try:
        gold_chart_attr = extract_chart_properties_runtime(gold_path)
        gen_chart_attr = extract_chart_properties_runtime(gen_path)
    except:
        for key, func in metrics.items():
            result[key] = {"precision": 0, "recall": 0, "f1": 0}
        

    result = {}
    for key, func in metrics.items():
        try:
            result[key] = func(gold_chart_attr, gen_chart_attr)
        except Exception as e:
            result[key] = {"precision": 0, "recall": 0, "f1": 0}
            # print('error!')
            print(e)
            
    metrics = {
        "data": calculate_data_metrics_mat,
        'color':calculate_color_metrics_mat    
    }
    for key, func in metrics.items():
        try:
            result[key] = func(gold_path, gen_path)
        except Exception as e:
            result[key] = {"precision": 0, "recall": 0, "f1": 0}
            
            print(e)
            

    
    
    return result