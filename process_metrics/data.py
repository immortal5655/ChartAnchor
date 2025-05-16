import re
import string
from typing import Any, Callable, Optional, Sequence
from collections import Counter, defaultdict
import html
from dateutil.parser import parse
from datetime import datetime
import numpy as np
import Levenshtein
import editdistance
import pandas as pd
def is_empty(val):
    """判断是否为空值、None、NaN、空字符串或仅空格"""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == '':
        return True
    return False


data_attr_list = ['x','y','r','z','theta','labels','values','i','j','k','high','low','open','close','a','b','c','parents','dimensions','link','node']

import Levenshtein
from datetime import datetime
import math

EPS = 1e-6   # 防零除

# ---------- 1) 类型对齐 ----------
def try_float(x):
    """尝试把 x 转成 float，失败则抛 ValueError"""
    if isinstance(x, (int, float)):
        return float(x)
    # 支持带百分号/逗号/空格的字符串
    if isinstance(x, str):
        x_strip = x.replace('%', '').replace(',', '').replace('$', '').strip()
        return float(x_strip)
    raise ValueError

def ensure_same_type(e1, e2):
    """返回 (new_e1, new_e2) —— 同类型（float 或 str）"""
    try:
        f1, f2 = try_float(e1), try_float(e2)
        if not (math.isnan(f1) or math.isnan(f2)):
            return f1, f2         # 成功按 float
    except Exception:
        pass
    # 退化为 str
    return str(e1), str(e2)

def ensure_all_same_type(t1, t2):
    """对齐两个任意长度元组的每一维"""
    if len(t1) != len(t2):
        raise ValueError("Tuples must have same length.")
    out1, out2 = [], []
    for a, b in zip(t1, t2):
        na, nb = ensure_same_type(a, b)
        out1.append(na)
        out2.append(nb)
    return tuple(out1), tuple(out2)

# ---------- 2) 辅助比较 ----------
def same_numeric(a, b, tol_num):
    return abs(a - b) / (abs(b) + EPS) <= tol_num

def same_string(a, b, tol_word):
    return (
        Levenshtein.distance(a, b) <= tol_word
        or a in b
        or b in a
    )

# ---------- 3) 总入口 ----------
def compare_tuple(elem1, elem2, tol_word=3, tol_num=0.05):
    """
    - 对齐类型后比较。
    """
    def clean_string(text):
        if not isinstance(text, str):
            return text  
        text = re.sub(r'<[^>]*>', '', text)
        text = html.unescape(text)
        text = re.sub(r'[\s%$]', '', text)
        return text
        
    if len(elem1) != len(elem2):
        return False

    e1, e2 = ensure_all_same_type(elem1, elem2)
    # 逐键比较
    for a, b in zip(e1, e2):
        if isinstance(a, float) and isinstance(b, float):
            if not same_numeric(a, b, tol_num):
                return False
        else:   # str
            a = clean_string(a)
            b = clean_string(b)
            if not same_string(a, b, tol_word):
                return False
    return True

def min_nonzero_length(*lists):
    """
    返回多个有效列表中长度 > 0 的最小长度。
    忽略 None、非可迭代对象、空列表。如果都不满足条件，则返回 0。
    """
    non_zero_lengths = []
    for lst in lists:
        if lst is not None:
            try:
                l = len(lst)
                if l > 0:
                    non_zero_lengths.append(l)
            except TypeError:
                continue  # 忽略无法取 len 的对象

    return min(non_zero_lengths) if non_zero_lengths else 0

def safe_get(lst, i):
    return lst[i] if lst and i < len(lst) and lst[i] is not None else 0.0001


def parse_date_string(s):
    """
    判断一个字符串是否是日期。如果是则提取 '年-月-日' 格式；
    缺失的部分用空字符串代替；
    如果不是日期，返回原字符串。

    返回:
        'YYYY-MM-DD' 格式（带空字符串）或原始字符串
    """
    if not isinstance(s, str):
        return s  # 非字符串直接返回

    try:
        # 使用一个默认日期方便识别缺失项
        default_dt = datetime(1, 1, 1)
        dt = parse(s, default=default_dt, fuzzy=True)

        # 分别判断年/月/日是否真的出现在原字符串中（避免误识别默认值）
        s_lower = s.lower()

        # 年可能是 2023 或 23，处理两位或四位的模糊匹配
        year_part = str(dt.year) if str(dt.year) in s or str(dt.year)[-2:] in s else ''
        month_part = str(dt.month) if (dt.strftime('%B').lower() in s_lower or
                                       dt.strftime('%b').lower() in s_lower or
                                       f"{dt.month:02}" in s or
                                       str(dt.month) in s) else ''
        day_part = str(dt.day) if f"{dt.day:02}" in s or str(dt.day) in s else ''

        # 返回格式化结果
        return f"{year_part}-{month_part}-{day_part}"
    except Exception:
        return s  # 解析失败，返回原字符串
def should_skip_dict(d, key_list):
    """
    如果字典 d 的所有键都不在 key_list 中，返回 True（表示应跳过）。
    否则返回 False。
    
    参数:
        d (dict): 要判断的字典
        key_list (list or set): 合法的键名集合
        
    返回:
        bool: 是否跳过该字典
    """
    return all(k not in key_list for k in d.keys())

def clean_y_if_str_header(y):
    """
    如果列表 y 的第一个值是字符串，并且后续所有值都可以转换为 float，则删除第一个值

    参数:
        y (list): 输入列表

    返回:
        list: 如果满足条件，则返回删除第一个元素后的列表；否则返回原列表
    """
    if not y:
        return y

    if isinstance(y[0], str):
        try:
            # 尝试将 y[1:] 中所有值转换为 float
            [float(val) for val in y[1:]]
            return y[1:]
        except (ValueError, TypeError):
            pass
    return y

def clean_value(v):
    """单个值的清洗：去 %, $ 并保留原始类型"""
    if isinstance(v, str):
        v = v.strip()
        if v.endswith('%'):
            v = v[:-1]
        if v.startswith('$'):
            v = v[1:]
        try:
            v = v.replace(',', '')
        except:
            pass
    return v

def clean_and_convert_nested(values):
    """递归清洗并尝试转数字"""
    if isinstance(values, list):
        return [clean_and_convert_nested(v) for v in values]
    else:
        try:
            return float(clean_value(values).replace(' ', ''))
        except:
            return clean_value(values)

def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False

def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False
def smart_cast_list(data):
    
    if all(is_float(x) for x in data):
        return [float(x) for x in data]
    
    return data  # 原样返回
def remove_str_indices_by_y(y, *other_lists):
    """
    找出 y 中为字符串的索引，并从 y 及所有传入的其他列表中同步删除这些位置的数据。

    参数:
        y (list): 基准列表，找出其中为 str 的索引。
        *other_lists (list): 其余与 y 等长的列表，将删除相同索引位置的值。

    返回:
        - 如果只传入 y，返回处理后的 y。
        - 如果传入 y 和其他列表，返回处理后的 (y, *other_lists) 元组。
    """
    # 找出 y 中为 str 的索引
    str_indices = {i for i, val in enumerate(y) if isinstance(val, str)}

    # 定义过滤函数
    def filter_indices(lst):
        return [val for i, val in enumerate(lst) if i not in str_indices]

    # 处理 y 和其他列表
    cleaned_y = filter_indices(y)
    cleaned_others = [filter_indices(lst) for lst in other_lists]

    return cleaned_y if not other_lists else (cleaned_y, *cleaned_others)



def remove_none_indices(*lists):
    """
    删除所有列表中为 None 或空值（包括 '', 空格串, 空列表, NaN）的统一索引位置。
    如果传入的某个列表是 None，则保留为 None 不处理。

    返回单个列表或列表元组，对应原始结构。
    """
    if not lists:
        return []

    def is_empty(value):
        return (
            value is None or
            value == '' or
            (isinstance(value, str) and value.strip() == '') or
            (isinstance(value, list) and len(value) == 0) or
            (isinstance(value, float) and np.isnan(value))
        )

    # 筛选非 None 的有效列表
    valid_lists = [lst for lst in lists if lst is not None]

    if not valid_lists:
        return lists if len(lists) > 1 else []

    min_len = min(len(lst) for lst in valid_lists)

    indices_to_remove = set()
    for i in range(min_len):
        for lst in valid_lists:
            if is_empty(lst[i]):
                indices_to_remove.add(i)
                break

    # 构建结果：None 保留，其他按索引过滤
    result = []
    for lst in lists:
        if lst is None:
            result.append(None)
        else:
            new_lst = [item for idx, item in enumerate(lst) if idx not in indices_to_remove]
            result.append(new_lst)

    return result[0] if len(result) == 1 else tuple(result)

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

def process_pie(x_data, y_data):
    if x_data and y_data:
        min_len = min(len(x_data), len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        x_data, y_data = remove_none_indices(x_data, y_data)
        y_data = clean_and_convert_nested(y_data)
        y_data,x_data = remove_str_indices_by_y(y_data,x_data)
        
    elif y_data:
        y_data = remove_none_indices(y_data)
        y_data = clean_and_convert_nested(y_data)
        y_data = remove_str_indices_by_y(y_data)
    # y_data = smart_cast_list(y_data)
    # y_data = clean_and_convert_nested(y_data)
    sum_y = sum(y_data)
    y_data = [(item/sum_y)*100 for item in y_data]
    return x_data,y_data




def get_candle_ohlc_tuple(trace,add_name):
    tuple_list = []
    x_data = trace.x if hasattr(trace, 'x') else None
    if hasattr(trace, 'x'):
        try:
            x_data = [parse_date_string(str(x)) for x in x_data]
        except:
            pass
    x_data = ensure_list(x_data) if hasattr(trace, 'x') else None
    low_data = ensure_list(trace.low) if hasattr(trace, 'low') else None
    high_data = ensure_list(trace.high) if hasattr(trace, 'high') else None
    open_data = ensure_list(trace.open) if hasattr(trace, 'open') else None
    close_data = ensure_list(trace.close) if hasattr(trace, 'close') else None

    x_data, low_data, high_data, open_data, close_data = remove_none_indices(x_data, low_data, high_data, open_data, close_data)
    # low_data,x_data,high_data, open_data, close_data = remove_str_indices_by_y(low_data,x_data,high_data, open_data, close_data)
    x_data = clean_and_convert_nested(x_data)
    low_data = clean_and_convert_nested(low_data)
    high_data = clean_and_convert_nested(high_data)
    open_data = clean_and_convert_nested(open_data)
    close_data = clean_and_convert_nested(close_data)
    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    min_len = min_nonzero_length(x_data, low_data, high_data, open_data, close_data)
    for i in range(min_len):
        tuple_list.append((
            get_type(trace),
            method or 'name',
            str(parse_date_string(str(safe_get(x_data, i)))),
            safe_get(low_data, i),
            safe_get(high_data, i),
            safe_get(open_data, i),
            safe_get(close_data, i)
        ))
    return tuple_list

def get_scatter3d_tuple(trace, add_name):
    tuple_list = []
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    z_data = ensure_list(trace.z) if hasattr(trace, 'z') else None
    x_data, y_data, z_data = remove_none_indices(x_data, y_data, z_data)
    x_data = clean_and_convert_nested(x_data)
    y_data = clean_and_convert_nested(y_data)
    z_data = clean_and_convert_nested(z_data)
    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    min_len = min_nonzero_length(x_data, y_data, z_data)
    for i in range(min_len):
        tuple_list.append((
            'scatter3d',
            method or 'name',
            safe_get(x_data, i),
            safe_get(y_data, i),
            safe_get(z_data, i),
        ))
    return tuple_list
        
def get_mesh3d_tuple(trace,add_name):
    tuple_list = []
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    z_data = ensure_list(trace.z) if hasattr(trace, 'z') else None
    i_data = ensure_list(trace.i) if hasattr(trace, 'i') else None
    j_data = ensure_list(trace.j) if hasattr(trace, 'j') else None
    k_data = ensure_list(trace.k) if hasattr(trace, 'k') else None

    x_data, y_data, z_data, i_data, j_data, k_data = remove_none_indices(x_data, y_data, z_data, i_data, j_data, k_data)
    x_data = clean_and_convert_nested(x_data)
    y_data = clean_and_convert_nested(y_data)
    z_data = clean_and_convert_nested(z_data)
    i_data = clean_and_convert_nested(i_data)
    j_data = clean_and_convert_nested(j_data)
    k_data = clean_and_convert_nested(k_data)
    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    min_len = min_nonzero_length(x_data, y_data, z_data, i_data, j_data, k_data)
    for i in range(min_len):
        tuple_list.append((
            'mesh3d',
            method or 'name',
            safe_get(x_data, i),
            safe_get(y_data, i),
            safe_get(z_data, i),
            safe_get(i_data, i),
            safe_get(j_data, i),
            safe_get(k_data, i),
        ))
    return tuple_list
def get_cone_tuple(trace,add_name):
    tuple_list = []
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    z_data = ensure_list(trace.z) if hasattr(trace, 'z') else None
    u_data = ensure_list(trace.u) if hasattr(trace, 'u') else None
    v_data = ensure_list(trace.v) if hasattr(trace, 'v') else None
    w_data = ensure_list(trace.w) if hasattr(trace, 'w') else None
 
    x_data, y_data, z_data, u_data, v_data, w_data = remove_none_indices(x_data, y_data, z_data, u_data, v_data, w_data)
    x_data = clean_and_convert_nested(x_data)
    y_data = clean_and_convert_nested(y_data)
    z_data = clean_and_convert_nested(z_data)
    u_data = clean_and_convert_nested(u_data)
    v_data = clean_and_convert_nested(v_data)
    w_data = clean_and_convert_nested(w_data)
    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    min_len = min_nonzero_length(x_data, y_data, z_data, u_data, v_data, w_data)
    for i in range(min_len):
        tuple_list.append((
            'cone',
            method or 'name',
            safe_get(x_data, i),
            safe_get(y_data, i),
            safe_get(z_data, i),
            safe_get(u_data, i),
            safe_get(v_data, i),
            safe_get(w_data, i),
        ))
    return tuple_list

def is_empty(val):
    return val in [None, '', [], {}, np.nan] or (isinstance(val, float) and np.isnan(val))

def ensure_list(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    elif val is None:
        return []
    else:
        return [val]

def is_empty(val):
    return val is None or val == '' or (isinstance(val, float) and np.isnan(val))
    # 判断val是否为None或者空字符串
    return val is None or val == '' or (
        # 判断val是否为float类型并且是否为NaN
        isinstance(val, float) and np.isnan(val)
    )

def can_cast_to_float(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def get_surface_tuple(trace, add_name,drop_empty=True):
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    z_data = ensure_list(trace.z) if hasattr(trace, 'z') else None
    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
   
    if not z_data:
        return []
        

    z_arr = np.asarray(z_data, dtype=object)  # 不提前转 float，避免崩溃

    # -------------------- 1D 情况 --------------------
    if z_arr.ndim == 1:
        if not x_data or not y_data:
            raise ValueError("z 为 1D 时，必须提供 x 和 y")
        if len(x_data) != len(y_data) or len(x_data) != len(z_arr):
            x_data= np.arange(len(z_arr))
            y_data = np.arange(len(z_arr))
        return [
            ( 'surface', method or 'name',x_data[i], y_data[i], float(z_arr[i]))
            for i in range(len(z_arr))
            if not is_empty(x_data[i]) and not is_empty(y_data[i])
            and not is_empty(z_arr[i])
            and can_cast_to_float(z_arr[i])
        ]


    # -------------------- 2D 情况 --------------------
    elif z_arr.ndim == 2:
        m, n = z_arr.shape

        # 自动生成 x
        if not x_data:
            x_arr = np.broadcast_to(np.arange(n), (m, n))
        else:
            x_arr = np.asarray(x_data, dtype=object)
            if x_arr.ndim == 1:
                if len(x_arr) == n:
                    x_arr = np.broadcast_to(x_arr, (m, n))
                    orient = 'col'
                elif len(x_arr) == m:
                    x_arr = np.broadcast_to(x_arr[:, None], (m, n))
                    orient = 'row'
                else:
                    x_arr = np.broadcast_to(np.arange(n), (m, n))
            elif x_arr.shape != z_arr.shape:
                return []

        # 自动生成 y
        if not y_data:
            y_arr = np.broadcast_to(np.arange(m)[:, None], (m, n))
        else:
            y_arr = np.asarray(y_data, dtype=object)
            if y_arr.ndim == 1:
                if len(y_arr) == m:
                    y_arr = np.broadcast_to(y_arr[:, None], (m, n))
                elif len(y_arr) == n:
                    y_arr = np.broadcast_to(y_arr, (m, n))
                else:
                    y_arr = np.broadcast_to(np.arange(m)[:, None], (m, n))
            elif y_arr.shape != z_arr.shape:
                return []

        # ---------- 输出三元组 ----------
        triples = []
        for j in range(m):
            for i in range(n):
                x_val = x_arr[j, i]
                y_val = y_arr[j, i]
                z_val = z_arr[j, i]

                if is_empty(x_val) or is_empty(y_val) or is_empty(z_val):
                    continue
                try:
                    z_float = float(z_val)
                except (ValueError, TypeError):
                    continue

                triples.append(('surface',method or 'name',x_val, y_val, float(z_val)))

        return triples

    else:
        return []

def get_histogram2d_contou_tuple(trace, real_trace, add_name):
    tuple_list = []
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    z_data = ensure_list(trace.z) if hasattr(trace, 'z') else None
    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    x_data, y_data,z_data = remove_none_indices(x_data, y_data, z_data)
    x_data = clean_and_convert_nested(x_data)
    y_data = clean_and_convert_nested(y_data)
    x_data = clean_y_if_str_header(x_data)
    y_data = clean_y_if_str_header(y_data)
    if z_data:
        z_data = clean_and_convert_nested(z_data)
        z_data = clean_y_if_str_header(z_data)
        min_len = min_nonzero_length(x_data, y_data, z_data)
        for i in range(min_len):
            tuple_list.append((
                get_type(trace),
                method or 'name',
                safe_get(x_data, i),
                safe_get(y_data, i),
                safe_get(z_data, i),
            ))
        return tuple_list
    else:
        
        xbins = real_trace['xbins'].to_plotly_json()
        ybins = real_trace['ybins'].to_plotly_json()
        all_bins = (set(list(xbins.keys())) == set(['start','size','end'])) and (set(list(ybins.keys())) == set(['start','size','end']))
        if all_bins:
            try:
                x = pd.Series(x_data)
                y = pd.Series(y_data)
                x_numeric = pd.to_numeric(x)
                y_numeric = pd.to_numeric(y)
                x_start, x_end, x_size = xbins['start'], xbins['end'], xbins['size']
                y_start, y_end, y_size = ybins['start'], ybins['end'], ybins['size']

                x_edges = np.arange(x_start, x_end + x_size, x_size)
                y_edges = np.arange(y_start, y_end + y_size, y_size)
                hist, xedges, yedges = np.histogram2d(x_numeric, y_numeric, bins=[x_edges, y_edges])

                trace_type = get_type(trace)
                for i in range(hist.shape[0]):        # 遍历 x 方向的 bins
                    for j in range(hist.shape[1]):    # 遍历 y 方向的 bins
                        x,y = x_edges[i], y_edges[j]
                        count = hist[i, j]
                        triplets.append((trace_type,method or 'name', x,y,count))
                return triplets    
            except:
                pass
        min_len = min_nonzero_length(x_data, y_data)
        for i in range(min_len):
            tuple_list.append((
                get_type(trace),
                method or 'name',
                safe_get(x_data, i),
                safe_get(y_data, i),
            ))
        return tuple_list
        


    


def aggregate_histogram_data(x, bins=None, histnorm='count'):
    """
    对x进行直方图聚合，支持数值型/类别型数据。
    支持 histnorm='count', 'percent', 'probability', 'density', 'probability density'

    如果 bins 为 None，则即使是数值型也作为分类处理。

    返回：
        bin_names: list - 每个 bin 的中心值或分类标签
        bin_values: list - 每个 bin 的聚合值
        bin_edges: list or None
        bin_labels: list or None - 每个 bin 中的原始样本（分类时有）
        data_type: str - 'numeric' or 'category'
    """
    x_clean = pd.Series(x).dropna()

    # 1. 判断类型
    try:
        x_clean_numeric = pd.to_numeric(x_clean)
        inferred_type = 'numeric'
    except:
        inferred_type = 'category'

    # 2. 条件：没有bins就强制按分类处理
    if inferred_type == 'numeric' and bins !={}:
        x_clean = x_clean_numeric
        data_type = 'numeric'
    else:
        x_clean = x_clean.astype(str)  # 确保统一处理
        data_type = 'category'

    # 3. 数值型聚合（需要bins）
    if data_type == 'numeric':
        try:
            start = bins['start']
            end = bins['end']
            size = bins['size']
        except KeyError:
            raise ValueError("数值型 bins 必须包含 'start', 'end', 'size'")

        bin_edges = np.arange(start, end + size, size)
        counts, edges = np.histogram(x_clean, bins=bin_edges)
        total = counts.sum()
        widths = np.diff(edges)

        if histnorm == 'count':
            values = counts
        elif histnorm == 'percent':
            values = (counts / total) * 100
        elif histnorm == 'probability':
            values = counts / total
        elif histnorm == 'density':
            values = counts / widths
        elif histnorm == 'probability density':
            values = counts / (total * widths)
        else:
            histnorm == 'count'

        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        return centers, values.tolist(), edges.tolist(), None, data_type

    # 4. 分类聚合
    labels_dict = defaultdict(list)
    for val in x_clean:
        labels_dict[val].append(val)

    counts = {k: len(v) for k, v in labels_dict.items()}
    total = sum(counts.values())
    bin_count = len(counts)

    if histnorm == 'count':
        values = list(counts.values())
    elif histnorm == 'percent':
        values = [v / total * 100 for v in counts.values()]
    elif histnorm == 'probability':
        values = [v / total for v in counts.values()]
    elif histnorm == 'density':
        values = [v / total / bin_count for v in counts.values()]
    elif histnorm == 'probability density':
        values = [v / total for v in counts.values()]
    else:
        values = list(counts.values())
    return list(counts.keys()), values, None, list(labels_dict.values()), data_type

def get_histogram_tuple(trace, real_trace, mode,add_name):
    triples = []
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    orientation = trace.orientation if hasattr(trace, 'orientation') else 'v'
    if x_data:
        x = ensure_list(x_data)
        x = remove_none_indices(x)
        x = clean_and_convert_nested(x)
        x_data = clean_y_if_str_header(x)
    if y_data:
        x = ensure_list(y_data)
        x = remove_none_indices(x)
        x = clean_and_convert_nested(x)
        y_data = clean_y_if_str_header(x)
    
    histnorm = real_trace.histnorm
    
    
    if orientation == 'h':
        bins1 = real_trace['ybins'].to_plotly_json()
        bins2 = trace.ybins.to_plotly_json()
        if set(list(bins1.keys())) == set(['start','size','end']):
            bins=bins1
        elif set(list(bins2.keys())) == set(['start','size','end']):
            bins=bins2
        else:
            bins={}

        if histnorm != '' and histnorm!=None:
            x_data,y_data,_,_,htype = aggregate_histogram_data(y_data, bins, histnorm)
        else:
            x_data,y_data,_,_,htype = aggregate_histogram_data(y_data, bins)
    else:
        bins1 = real_trace['xbins'].to_plotly_json()
        bins2 = trace.xbins.to_plotly_json()
        if set(list(bins1.keys())) == set(['start','size','end']):
            bins=bins1
        elif set(list(bins2.keys())) == set(['start','size','end']):
            bins=bins2
        else:
            bins={}

        if histnorm != '' and histnorm!=None:
            x_data,y_data,_,_,htype = aggregate_histogram_data(x_data, bins, histnorm)
        else:
            x_data,y_data,_,_,htype = aggregate_histogram_data(x_data, bins)
    method = trace.name if hasattr(trace, 'name') else None
    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    for i in range(len(x_data)):
        triples.append(('histogram', htype, method or 'name',x_data[i], y_data[i]))
    return triples
    
def get_sankey_tuple(trace, add_name ):
    tuple_list = []
    nodes = ensure_list(trace.node.label) if hasattr(trace, 'node') and hasattr(trace.node, 'label') else None
    sources = ensure_list(trace.link.source) if hasattr(trace, 'link') and hasattr(trace.link, 'source') else None
    targets = ensure_list(trace.link.target) if hasattr(trace, 'link') and hasattr(trace.link, 'target') else None
    values = ensure_list(trace.link.value) if hasattr(trace, 'link') and hasattr(trace.link, 'value') else None

    # 清理和转换数据，移除 None 值
    sources, targets, values = remove_none_indices(sources, targets, values)

    # 对数据进行清洗和转换
    sources = clean_and_convert_nested(sources)
    targets = clean_and_convert_nested(targets)
    values = clean_and_convert_nested(values)

    # 获取 trace 的名字
    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    min_len = min_nonzero_length(sources, targets, values)
    sum_values = sum(values)
    values = [item/(sum_values+1e-8)*100 for item in values]
    # 生成元组列表
    for i in range(min_len):
        tuple_list.append((
            'sankey',                         # 图表类型
            method or 'name',                 # 图表方法（如果有名字，则使用名字）
            safe_get(sources, i),             # 源节点
            safe_get(targets, i),             # 目标节点
            safe_get(values, i),              # 连接数值
        ))

    return tuple_list
    
def get_box_tuple(trace):
    tuple_list = []

    # 获取每个箱形图的类别标签 (x) 和 y 数据
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None

    x_data, y_data = remove_none_indices(x_data, y_data)

    x_data = clean_and_convert_nested(x_data)
    y_data = clean_and_convert_nested(y_data)

    x_data = clean_y_if_str_header(x_data)
    y_data = clean_y_if_str_header(y_data)

    if x_data is None:
        sorted_data = sorted(y_data)
        Q1 = np.percentile(sorted_data, 25)  # 下四分位数
        Q3 = np.percentile(sorted_data, 75)  # 上四分位数
        median = np.median(sorted_data)     # 中位数
        IQR = Q3 - Q1                       # 四分位距
        min_value = min(sorted_data)        # 最小值
        max_value = max(sorted_data)        # 最大值
        
        # 计算离群值
        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR
        outliers = [x for x in sorted_data if x < lower_whisker or x > upper_whisker]

        # 获取 trace 的名字
        name = trace.name if hasattr(trace, 'name') else f'Trace'

        # 将总体统计数据整理成元组
        tuple_list.append((
            'boxplot',                  # 图表类型
            name,                       # 图表名称
            'All Data',                 # 统一的数据组
            Q1,                         # 下四分位数
            median,                     # 中位数
            Q3,                         # 上四分位数
            IQR,                        # 四分位距
            min_value,                  # 最小值
            max_value,                  # 最大值
            outliers                    # 离群值
        ))

    else:
        # 如果 x_data 存在，进行分组
        groups = {}
        for i in range(len(x_data)):
            group = x_data[i]
            value = y_data[i]
            if group not in groups:
                groups[group] = []
            groups[group].append(value)

        # 计算每个组的统计数据
        for group, values in groups.items():
            sorted_values = sorted(values)
            
            Q1 = np.percentile(sorted_values, 25)  # 下四分位数
            Q3 = np.percentile(sorted_values, 75)  # 上四分位数
            median = np.median(sorted_values)     # 中位数
            IQR = Q3 - Q1                         # 四分位距
            min_value = min(sorted_values)        # 最小值
            max_value = max(sorted_values)        # 最大值
            
            # 计算离群值
            lower_whisker = Q1 - 1.5 * IQR
            upper_whisker = Q3 + 1.5 * IQR
            outliers = [x for x in sorted_values if x < lower_whisker or x > upper_whisker]

            # 获取 trace 的名字
            name = trace.name if hasattr(trace, 'name') else f'Trace'

            # 将每个组的统计数据整理成元组
            tuple_list.append((
                'boxplot',                  # 图表类型
                name,                       # 图表名称
                group,                      # 当前组 (类别)
                Q1,                         # 下四分位数
                median,                     # 中位数
                Q3,                         # 上四分位数
                IQR,                        # 四分位距
                min_value,                  # 最小值
                max_value,                  # 最大值
                outliers                    # 离群值
            ))

    return tuple_list
def get_funnel_tuple(trace, add_name ):
    tuple_list = []

    # 获取 funnel 数据中的字符串属性（如步骤名称）和数值属性（如数量）
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    if y_data:
        if isinstance(y_data[0],list):
            y_data = y_data[0]
    else:
        y_data = [0] * len(x_data)

    x_data, y_data = remove_none_indices(x_data, y_data)
    x_data = clean_and_convert_nested(x_data)
    y_data = clean_and_convert_nested(y_data)
    min_len = min_nonzero_length(x_data, y_data)

    

    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    # 判断 x 和 y 的数据类型，决定元组的顺序
    for i in range(min_len):
        try:
            if isinstance(y_data[i], str) and isinstance(x_data[i], (int, float)):
                tuple_list.append((
                    'funnel',
                    method or 'name',                                # 图表类型
                    safe_get(y_data, i),                            # x 数据（字符串属性）
                    safe_get(x_data, i)                             # y 数据（数值属性）
                ))
            else:
                tuple_list.append((
                    'funnel',
                    method or 'name',   
                    safe_get(x_data, i),       # x 数据（数值属性）
                    safe_get(y_data, i)        # y 数据（数值属性）
                ))
        except:
            pass

     
    return tuple_list

from collections import defaultdict, deque

def compute_relative_tuples_auto(labels, parents, values=None, round_digits=4):
    tree = defaultdict(list)
    for label, parent in zip(labels, parents):
        tree[parent].append(label)

    all_nodes = set(labels)
    node_values = {}

    if values is not None:
        node_values = dict(zip(labels, values))
    else:
        # 非递归值推导：拓扑排序 + 自底向上求值
        child_count = defaultdict(int)
        for parent, child_list in tree.items():
            for child in child_list:
                child_count[parent] += 1

        # 初始化：叶子节点 = 1
        pending = set(labels)
        node_values = {}
        stack = deque()

        for label in labels:
            if label not in tree:  # 没有子节点
                node_values[label] = 1
                stack.append(label)

        # 迭代处理非叶节点
        while stack:
            node = stack.popleft()
            parent = parents[labels.index(node)]
            if parent in ["", None]:
                continue
            if parent not in node_values:
                node_values[parent] = 0
            node_values[parent] += node_values[node]
            child_count[parent] -= 1
            if child_count[parent] == 0:
                stack.append(parent)

        # 如果还有节点没计算到（循环或断层），强制设值
        for node in labels:
            if node not in node_values:
                node_values[node] = 1  # fallback 默认值

    # 计算比例
    relative_tuples = []
    for parent, children in tree.items():
        total = sum(node_values.get(child, 0) for child in children)
        for child in children:
            proportion = node_values[child] / total *100 if total > 0 else 0.0
            relative_tuples.append(("sunburst", parent, child, round(proportion, round_digits)))

    return relative_tuples
def fix_missing_parents(labels, parents):
    """
    修复 sunburst 中 labels 和 parents 长度不一致或缺失 parent 的问题。
    返回对齐后的 labels 和 parents 列表。
    """
    labels = list(labels)
    parents = list(parents)

    # 如果 parents 列缺失元素（比 labels 少），补上空字符串
    while len(parents) < len(labels):
        parents.insert(0, "")

    # 构建 label → parent 映射
    child_to_parent = dict(zip(labels, parents))

    # 检查未被列为子节点的根节点（没有作为任何人的 parent）
    all_labels = set(labels)
    referenced_parents = set(p for p in parents if p not in ["", None])

    unreferenced_roots = all_labels - referenced_parents
    for root in unreferenced_roots:
        if root not in child_to_parent:
            labels.insert(0, root)
            parents.insert(0, "")  # 加入根节点

    return labels, parents


def get_sunburst_tuple(trace):
    def normalize_sunburst_data(parents):
        new_parents = ["root" if p in ["", None] else p for p in parents]
        return new_parents

    def clean_string(text):
        if isinstance(text, list):
            return [clean_string(item) for item in text]
        if not isinstance(text, str):
            return text

        text = re.sub(r'<[^>]*>', '', text)
        text = html.unescape(text)
        text = re.sub(r'[\s%$\-]', '', text)
        return text.lower()

    def remove_headers_if_floats(*lists):
        def is_float_like(val):
            try:
                float(val)
                return True
            except:
                return False

        def is_header_float_case(lst):
            return (
                lst and isinstance(lst[0], str) and
                all(is_float_like(item) for item in lst[1:])
            )
        should_remove = any(is_header_float_case(lst) for lst in lists)
        return tuple(lst[1:] if should_remove and lst else lst for lst in lists)


   
    labels = ensure_list(trace.labels) if hasattr(trace, 'labels') else None
    parents = ensure_list(trace.parents) if hasattr(trace, 'parents') else None
    values = ensure_list(trace.values) if hasattr(trace, 'values') else None
  
    # 必须要有labels和parents属性
    if not labels or not parents :
        return []
    labels, parents = fix_missing_parents(labels, parents)

    labels = clean_string(labels)
    parents = clean_string(parents)

    # 根节点的parents属性为''，先将''改为root
    # parents = normalize_sunburst_data(parents)
    # labels, parents, values = remove_none_indices(labels, parents, values)

    labels = clean_and_convert_nested(labels)
    parents = clean_and_convert_nested(parents)
    cleaned_values = (
        clean_and_convert_nested(clean_string(values)) if values else None
    )
    labels, parents, cleaned_values = remove_headers_if_floats(labels, parents, cleaned_values)
    try:
        tuples = compute_relative_tuples_auto(labels, parents, cleaned_values, round_digits=4)
    except:
        return []


    return tuples

def get_scatterternary_tuple(trace,add_name):
    tuple_list = []

    a_data= ensure_list(trace.a) if hasattr(trace, 'a') else None
    b_data= ensure_list(trace.b) if hasattr(trace, 'b') else None
    c_data= ensure_list(trace.c) if hasattr(trace, 'c') else None
    a_data, b_data, c_data = remove_none_indices(a_data, b_data, c_data)
    a_data = clean_and_convert_nested(a_data)
    b_data = clean_and_convert_nested(b_data)
    c_data = clean_and_convert_nested(c_data)

    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    min_len = min_nonzero_length(a_data, b_data, c_data)
    for i in range(min_len):
        tuple_list.append((
            'scatterternary',
            method or 'name',
            safe_get(a_data, i),
            safe_get(b_data, i),
            safe_get(c_data, i),
        ))
    return tuple_list

def get_treemap_tuple(trace):
    def clean_string(text):
        if isinstance(text, list):
            return [clean_string(item) for item in text]
        if not isinstance(text, str):
            return text

        text = re.sub(r'<[^>]*>', '', text)
        text = html.unescape(text)
        text = re.sub(r'[\s%$\-]', '', text)
        return text.lower()

    def remove_headers_if_floats(*lists):
        def is_float_like(val):
            try:
                float(val)
                return True
            except:
                return False

        def is_header_float_case(lst):
            return (
                lst and isinstance(lst[0], str) and
                all(is_float_like(item) for item in lst[1:])
            )
        should_remove = any(is_header_float_case(lst) for lst in lists)
        return tuple(lst[1:] if should_remove and lst else lst for lst in lists)

    # 获取 labels, parents, values
    labels = ensure_list(trace.labels) if hasattr(trace, 'labels') else None
    parents = ensure_list(trace.parents) if hasattr(trace, 'parents') else None
    values = ensure_list(trace.values) if hasattr(trace, 'values') else None

    if not labels or not parents:
        return []

    labels, parents = fix_missing_parents(labels, parents)

    labels = clean_string(labels)
    parents = clean_string(parents)

    labels = clean_and_convert_nested(labels)
    parents = clean_and_convert_nested(parents)
    cleaned_values = clean_and_convert_nested(clean_string(values)) if values else None

    labels, parents, cleaned_values = remove_headers_if_floats(labels, parents, cleaned_values)

    try:
        tuples = compute_relative_tuples_auto(labels, parents, cleaned_values, round_digits=4)
    except Exception:
        return []

    # 将图类型从 sunburst 改为 treemap
    treemap_tuples = [('treemap', parent, child, value) for (_, parent, child, value) in tuples]
    return treemap_tuples


def get_carpet_tuple(trace, add_name):
    tuple_list = []

    a_data = ensure_list(trace.a) if hasattr(trace, 'a') else None
    b_data = ensure_list(trace.b) if hasattr(trace, 'b') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None  

    if not a_data or not b_data or not y_data:
        return []

    a_data = clean_and_convert_nested(a_data)
    b_data = clean_and_convert_nested(b_data)
    y_data = clean_and_convert_nested(y_data)
    x_data = clean_and_convert_nested(x_data) if x_data else None

    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None

    # 情况 1：y 是二维数组
    if isinstance(y_data, list) and all(isinstance(row, list) for row in y_data):
        for i, b_val in enumerate(b_data):
            if i >= len(y_data):
                continue
            y_row = y_data[i]
            x_row = x_data[i] if x_data and isinstance(x_data[0], list) else 0.0001
            for j, a_val in enumerate(a_data):
                y_val = safe_get(y_row, j)
                x_val = safe_get(x_row, j) if x_row else 0.0001
                tuple_list.append((
                    'carpet',
                    method or 'name',
                    a_val,
                    b_val,
                    x_val,
                    y_val
                ))
    else:
        # 情况 2：y 是一维数组，x 可以有也可以没有
        min_len = min_nonzero_length(a_data, b_data, y_data, x_data or [])
        for i in range(min_len):
            tuple_list.append((
                'carpet',
                method or 'name',
                safe_get(a_data, i),
                safe_get(b_data, i),
                safe_get(x_data, i) if x_data else 0.0001,
                safe_get(y_data, i)
            ))

    return tuple_list


def get_funnelarea_tuple(trace, add_name):
    tuple_list = []

    labels = ensure_list(trace.labels) if hasattr(trace, 'labels') else None
    values = ensure_list(trace.values) if hasattr(trace, 'values') else None
    text = ensure_list(trace.text) if hasattr(trace, 'text') else None
    labels, values = remove_none_indices(labels, values)

    labels = clean_and_convert_nested(labels)
    values = clean_and_convert_nested(values)

    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    if values:
        sum_y = sum(values)
        values = [(item/sum_y)*100 for item in values]
    if not labels and text:
        labels = text
    min_len = min_nonzero_length(labels, values)
    for i in range(min_len):
        tuple_list.append((
            'funnelarea',
            method or 'name',
            safe_get(labels, i),
            safe_get(values, i),
        ))
    return tuple_list

def get_parcoords_tuple(trace, add_name):
    tuple_list = []

    if add_name:
        method = getattr(trace, 'name', None)
    else:
        method = None

    dimensions = trace.dimensions
    dim_values = [clean_and_convert_nested(ensure_list(dim['values'])) for dim in dimensions]
    min_len = min(len(d) for d in dim_values)

    
    print(dim_values)
    for i in range(min_len):
        entry = ['parcoords', method or 'name']
        for dim_data in dim_values:
            entry.append(safe_get(dim_data, i))
        
        tuple_list.append(tuple(entry))

    return tuple_list

def convert_plotly_data_to_dict(fig, mode):
    def ensure_list(data):
        """确保输入数据转换为Python列表"""
        if data is None:
            return None
        if hasattr(data, 'tolist'):  # 处理NumPy数组
            return data.tolist()
        if isinstance(data, (list, tuple, np.ndarray)):  # 已经是列表或元组
            return list(data)
        return [data]  # 单个值转换为单元素列表
    
    def add_name(layout,n_trace):
        if n_trace ==1:
            if 'showlegend' in layout:
                if layout['showlegend'] == True:
                       return False
                else:
                    return False
            else:
                return False
        else:
            if 'showlegend' in layout:
                if layout['showlegend'] == False:
                    return False
                else:
                    return True
            else:
                return True
    
    result = []
    fig_data = fig.data
    layout = fig.layout.to_plotly_json()
    
    n_trace = len(fig_data)
    is_name = add_name(layout,n_trace)
    for ind in range(len(fig_data)):
        trace = fig_data[ind]
        trace_type = get_type(trace)
       
        trace_dict = trace.to_plotly_json()
        # 空trace
        if should_skip_dict(trace_dict, data_attr_list):
            continue
        if trace_type in ['barpolar','scatterpolar']:
            x_data = ensure_list(trace.theta) if hasattr(trace, 'theta') else None
            y_data = ensure_list(trace.r) if hasattr(trace, 'r') else None
        elif trace_type in ['pie']:
            x_data = ensure_list(trace.labels) if hasattr(trace, 'labels') else None
            y_data = ensure_list(trace.values) if hasattr(trace, 'values') else None
            x_data, y_data = process_pie(x_data, y_data)
        elif trace_type in ['candlestick','ohlc']:
            result.extend(get_candle_ohlc_tuple(trace,add_name=is_name))
            continue
        elif trace_type in ['scatter3d','line3d']:
            result.extend(get_scatter3d_tuple(trace,add_name = is_name))
            continue
        elif trace_type in ['mesh3d']:
            result.extend(get_mesh3d_tuple(trace,add_name=is_name))
            continue
        elif trace_type in ['surface','contour','heatmap']:
            result.extend(get_surface_tuple(trace, add_name = is_name))
            continue
        elif trace_type in ['cone']:
            result.extend(get_cone_tuple(trace,add_name=is_name))
            continue
        elif trace_type in ['histogram']:
            fig_filled = fig.full_figure_for_development()
            result.extend(get_histogram_tuple(trace, fig_filled.data[ind], mode, add_name = is_name))
            continue
        elif trace_type in ['sankey']:
            result.extend(get_sankey_tuple(trace, add_name = is_name))
            continue
        elif trace_type in ['funnel']:
            result.extend(get_funnel_tuple(trace,add_name = is_name))
            continue
        elif trace_type in ['sunburst']:
            result.extend(get_sunburst_tuple(trace))
            continue
        elif trace_type in ['histogram2d','histogram2dcontour']:
            fig_filled = fig.full_figure_for_development()
            result.extend(get_histogram2d_contou_tuple(trace, fig_filled.data[ind],add_name=is_name ))
            continue
        elif trace_type in ['scatterternary']:
            result.extend(get_scatterternary_tuple(trace,add_name = is_name))
            continue
        elif trace_type in ['funnelarea']:
            result.extend(get_funnelarea_tuple(trace,add_name = is_name))
            continue
        elif trace_type in ['parcoords']:
            result.extend(get_parcoords_tuple(trace,add_name = is_name))
            continue
        elif trace_type in ['treemap']:
            result.extend(get_treemap_tuple(trace))
            continue
        elif trace_type in ['carpet']:
            result.extend(get_carpet_tuple(trace,add_name = is_name))
            continue
        else:
            x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
            y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None

        # 转换x和y数据为列表
        if x_data:
            if isinstance(x_data[0], list):
                x_data = x_data[0]
        if y_data:
            if isinstance(y_data[0], list):
                y_data = y_data[0]
    
       
        if x_data and y_data:
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]
            x_data, y_data = remove_none_indices(x_data, y_data)
        # 处理gold模式下的类型转换
        
        if x_data:
            # x_data = remove_none_indices(x_data)
            x_data = clean_y_if_str_header(x_data)
            # x_data = smart_cast_list(x_data)
        if y_data:
            # y_data = remove_none_indices(y_data)
            y_data = clean_y_if_str_header(y_data)
            # y_data = smart_cast_list(y_data)
        
        
        # if mode == 'gold':
        #     if x_data:
        #         x_data = remove_none_indices(x_data)
        #         x_data = smart_cast_list(x_data)
        #     if y_data:
        #         y_data = remove_none_indices(y_data)
        #         y_data = smart_cast_list(y_data)
        
        # 获取方法名
        method = None
        if n_trace ==1:
            if 'showlegend' in layout:
                if layout['showlegend'] == True:
                       method = trace.name if hasattr(trace, 'name') else None
                else:
                    method = None
            else:
                method = None
        else:
            method = trace.name if hasattr(trace, 'name') else None
            if 'showlegend' in layout:
                if layout['showlegend'] == False:
                    method = None
        
        # 处理x和y数据
        if x_data and y_data:
            min_len = min(len(x_data), len(y_data))
            for i in range(min_len):
                x_val = x_data[i] if x_data[i]!=None else 0.0001
                y_val = y_data[i] if y_data[i]!=None else 0.0001
                result.append((trace_type, method or 'name', x_val, y_val))
        
        elif x_data and not y_data:
            for x_val in x_data:
                result.append((trace_type, method or 'name', x_val, 0.0001))
        
        elif y_data and not x_data:
            for y_val in y_data:
                result.append((trace_type, method or 'name', 0.0001, y_val))
       
    
    return result

def x_str_y_float(elem1,elem2, tol_word, tol_num):
    if ((Levenshtein.distance(''.join(elem1[:-1]),''.join(elem2[:-1])) <= tol_word) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num))or \
        ((''.join(elem1[:-1]) in ''.join(elem2[:-1])) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num)) or \
        ((''.join(elem2[:-1]) in ''.join(elem1[:-1])) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num)):
        return True
    else:
        return False
def x_str_y_str(elem1,elem2, tol_word, tol_num):
    if ((Levenshtein.distance(''.join(elem1[:-1]),''.join(elem2[:-1])) <= tol_word) and (Levenshtein.distance((elem1[-1]),(elem2[-1])) <= tol_word))or \
        ((''.join(elem1[:-1]) in ''.join(elem2[:-1])) and ((elem1[-1] in elem2[-1]) or (elem2[-1] in elem1[-1]))) or \
        ((''.join(elem2[:-1]) in ''.join(elem1[:-1])) and ((elem1[-1] in elem2[-1]) or (elem2[-1] in elem1[-1]))):
        return True
    else:
        return False

def x_float_y_float(elem1,elem2, tol_word, tol_num):
    if ((abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num) and (abs(elem1[1] - elem2[1]) / (abs(elem2[1])+0.000001) <= tol_num)):
        return True
    else:
        return False

def x_float_y_str(elem1,elem2, tol_word, tol_num):
    if ((Levenshtein.distance(''.join([elem1[0],elem1[-1]]),''.join([elem2[0],elem2[-1]])) <= tol_word) and (abs(elem1[1] - elem2[1]) / (abs(elem2[1])+0.000001) <= tol_num))or \
        ((''.join([elem1[0],elem1[-1]]) in ''.join([elem2[0],elem2[-1]])) and (abs(elem1[1] - elem2[1]) / (abs(elem2[1])+0.000001) <= tol_num)) or \
        ((''.join([elem2[0],elem2[-1]]) in ''.join([elem1[0],elem1[-1]])) and (abs(elem1[1] - elem2[1]) / (abs(elem2[1])+0.000001) <= tol_num)):
        return True
    else:
        return False

def x_float_candle(elem1,elem2, tol_word, tol_num):
    number = (abs(elem1[2] - elem2[2]) / (abs(elem2[2])+0.000001) <= tol_num) and (abs(elem1[3] - elem2[3]) / (abs(elem2[3])+0.000001) <= tol_num) and \
        (abs(elem1[4] - elem2[4]) / (abs(elem2[4])+0.000001) <= tol_num) and (abs(elem1[5] - elem2[5]) / (abs(elem2[5])+0.000001) <= tol_num)
    if ((Levenshtein.distance(''.join(elem1[:2]),''.join(elem2[:2])) <= tol_word) and number) or \
        ((''.join(elem1[:2]) in ''.join(elem2[:2])) and number) or \
        ((''.join(elem2[:2]) in ''.join(elem1[:2])) and number):
        return True
    else:
        return False

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def ensure_type(elem, idx, target_type):
    elem = list(elem)  # 元组转列表以便修改

    if target_type == "float":
        val = safe_float(elem[idx])
        if val is not None:
            elem[idx] = val
            return tuple(elem), True  # 修改后转回元组
        else:
            return tuple(elem), False

    elif target_type == "str":
        elem[idx] = str(elem[idx])
        return tuple(elem), True


def dispatch(elem1, elem2, tol_word, tol_num):
    # if len(elem1)!=len(elem2):
    #     return 0
    if elem1[0] == 'histogram':
        # (histogram,type,name,x,y) x为str说明是分类hist，需要计算label
        #(chhart_type,name,x,y)
        if elem1[1]!='category':
            elem1 = (elem1[2],elem1[4])
            elem2 = (elem2[1],elem2[3])
        else:
            elem1 = elem1[2:]
            elem2 = elem2[1:]
        if compare_tuple(elem1, elem2, tol_word, tol_num):
            return True
        else:
            return False
    # mesh3d
    # if len(elem1)==8:
    #     if compare_tuple(elem1, elem2, tol_word, tol_num):
    #         return True
    #     else:
    #         return False
    # # scatter3d，surface
    # if len(elem1) == 5:
    #     if compare_tuple(elem1, elem2, tol_word, tol_num):
    #         return True
    #     else:
    #         return False
    if len(elem1)==7:
        try:
            res = x_float_candle(elem1[1:], elem2[1:], tol_word, tol_num)
            return res
        except:
            return  0
    try:
        if compare_tuple(elem1[1:], elem2[1:], tol_word, tol_num):
            return True
        else:
            return False
    except:
        return False
   

def intersection_with_tolerance(gold, gen, tol_word, tol_num):
    sim_set = set()
    gold = list(set(gold))
    gen = list(set(gen))


    for elem1 in gold:
        for elem2 in gen:
            if dispatch(elem1, elem2, tol_word, tol_num):
                sim_set.add(elem1)
    return list(sim_set)

def union_with_tolerance(a, b, tol_word, tol_num):
    c = set(a) | set(b)
    d = set(a) & set(b)
    e = intersection_with_tolerance(a, b, tol_word, tol_num)
    f = set(e)
    g = c-(f-d)
    return list(g)


def calculate_data_metrics(fig_gold, fig_gen):
    tol_str_num =3
    tol_number = 0.05
    try:
        gold = list(set(convert_plotly_data_to_dict(fig_gold,'gold')))
    except:
        return None
    gen = list(set(convert_plotly_data_to_dict(fig_gen,'gen')))
    intersection = intersection_with_tolerance(gold,gen,tol_str_num,tol_number)
    union = union_with_tolerance(gold,gen,tol_str_num,tol_number)
    sim = len(intersection)/(len(union)+1e-7)
    return {"f1":sim}

