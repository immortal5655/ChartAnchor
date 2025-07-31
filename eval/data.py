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

EPS = 1e-6   

# ---------- 1) Type Alignment ----------
def try_float(x):
   
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        x_strip = x.replace('%', '').replace(',', '').replace('$', '').strip()
        return float(x_strip)
    raise ValueError

def ensure_same_type(e1, e2):
    
    try:
        f1, f2 = try_float(e1), try_float(e2)
        if not (math.isnan(f1) or math.isnan(f2)):
            return f1, f2         
    except Exception:
        pass
    return str(e1), str(e2)

def ensure_all_same_type(t1, t2):
    
    if len(t1) != len(t2):
        raise ValueError("Tuples must have same length.")
    out1, out2 = [], []
    for a, b in zip(t1, t2):
        na, nb = ensure_same_type(a, b)
        out1.append(na)
        out2.append(nb)
    return tuple(out1), tuple(out2)

# ---------- 2) Auxiliary Comparison ----------

def same_numeric(a, b, tol_num):
    return abs(a - b) / (abs(b) + EPS) <= tol_num

def same_string(a, b, tol_word):
    return (
        Levenshtein.distance(a, b) <= tol_word
        or a in b
        or b in a
    )

def compare_tuple(elem1, elem2, tol_word=3, tol_num=0.05):
    
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
    Return the smallest length among multiple valid non-empty lists.

    Ignores None, non-iterables, and empty lists.  
    Returns 0 if no valid list is found.
    """
    non_zero_lengths = []
    for lst in lists:
        if lst is not None:
            try:
                l = len(lst)
                if l > 0:
                    non_zero_lengths.append(l)
            except TypeError:
                continue  

    return min(non_zero_lengths) if non_zero_lengths else 0

def safe_get(lst, i):
    return lst[i] if lst and i < len(lst) and lst[i] is not None else 0.0001


from dateutil.parser import parse
from datetime import datetime

def parse_date_string(s):
    """
    Attempt to parse a fuzzy date string and return the components
    (year, month, day) only if they explicitly appear in the original string.

    Returns a formatted string like 'YYYY-MM-DD', with missing parts left empty.
    If parsing fails or input is not a string, returns the original input.
    """
    if not isinstance(s, str):
        return s  # Return as-is if not a string

    try:
        # Use a default date to help identify missing parts
        default_dt = datetime(1, 1, 1)
        dt = parse(s, default=default_dt, fuzzy=True)

        s_lower = s.lower()

        # Year: check if full year or last 2 digits exist in original string
        year_part = str(dt.year) if str(dt.year) in s or str(dt.year)[-2:] in s else ''

        # Month: match full month name, abbreviation, or numeric form
        month_part = str(dt.month) if (dt.strftime('%B').lower() in s_lower or
                                       dt.strftime('%b').lower() in s_lower or
                                       f"{dt.month:02}" in s or
                                       str(dt.month) in s) else ''

        # Day: match padded or plain numeric day
        day_part = str(dt.day) if f"{dt.day:02}" in s or str(dt.day) in s else ''

        return f"{year_part}-{month_part}-{day_part}"
    except Exception:
        return s  # On failure, return the original string



def should_skip_dict(d, key_list):

    return all(k not in key_list for k in d.keys())

def clean_y_if_str_header(y):
    if not y:
        return y

    if isinstance(y[0], str):
        try:
            [float(val) for val in y[1:]]
            return y[1:]
        except (ValueError, TypeError):
            pass
    return y

def clean_value(v):
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
    
    return data 
def remove_str_indices_by_y(y, *other_lists):
   
    str_indices = {i for i, val in enumerate(y) if isinstance(val, str)}

    def filter_indices(lst):
        return [val for i, val in enumerate(lst) if i not in str_indices]

    cleaned_y = filter_indices(y)
    cleaned_others = [filter_indices(lst) for lst in other_lists]

    return cleaned_y if not other_lists else (cleaned_y, *cleaned_others)



def remove_none_indices(*lists):
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
    

def can_cast_to_float(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def get_surface_tuple(trace, add_name, drop_empty=True):
    """
    Extract (x, y, z) data points from a 1D or 2D surface trace, returning them as a list of tuples.
    
    Each tuple is of the form: ('surface', method_name, x, y, z)

    Args:
        trace: The trace object containing x, y, z data (e.g., a surface plot).
        add_name (bool): If True, include trace.name as method identifier in the tuple.
        drop_empty (bool): Currently unused (reserved for future filtering logic).

    Returns:
        list of tuples: Each representing a data point on the surface.
                        Returns an empty list if no valid data is found.
    """
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    z_data = ensure_list(trace.z) if hasattr(trace, 'z') else None

    method = trace.name if add_name and hasattr(trace, 'name') else None

    if not z_data:
        return []

    z_arr = np.asarray(z_data, dtype=object)

    # -------------------- 1D case --------------------
    if z_arr.ndim == 1:
        if not x_data or not y_data:
            raise ValueError("When z is 1D, both x and y must be provided.")

        if len(x_data) != len(y_data) or len(x_data) != len(z_arr):
            # If dimensions don't match, fall back to automatic indexing
            x_data = np.arange(len(z_arr))
            y_data = np.arange(len(z_arr))

        return [
            ('surface', method or 'name', x_data[i], y_data[i], float(z_arr[i]))
            for i in range(len(z_arr))
            if not is_empty(x_data[i]) and not is_empty(y_data[i])
            and not is_empty(z_arr[i]) and can_cast_to_float(z_arr[i])
        ]

    # -------------------- 2D case --------------------
    elif z_arr.ndim == 2:
        m, n = z_arr.shape

        # Generate x grid
        if not x_data:
            x_arr = np.broadcast_to(np.arange(n), (m, n))
        else:
            x_arr = np.asarray(x_data, dtype=object)
            if x_arr.ndim == 1:
                if len(x_arr) == n:
                    x_arr = np.broadcast_to(x_arr, (m, n))
                elif len(x_arr) == m:
                    x_arr = np.broadcast_to(x_arr[:, None], (m, n))
                else:
                    x_arr = np.broadcast_to(np.arange(n), (m, n))
            elif x_arr.shape != z_arr.shape:
                return []

        # Generate y grid
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

        # ---------- Construct output tuples ----------
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

                triples.append(('surface', method or 'name', x_val, y_val, z_float))

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
        


    


from collections import defaultdict
import numpy as np
import pandas as pd

def aggregate_histogram_data(x, bins=None, histnorm='count'):
    """
    Aggregate data for histogram plotting, supporting both numeric and categorical input.
    Supports normalization modes: 'count', 'percent', 'probability', 'density', 'probability density'.

    If `bins` is None, even numeric data will be treated as categorical.

    Args:
        x (list-like): Input data to aggregate.
        bins (dict or None): For numeric data, a dictionary with keys 'start', 'end', and 'size'.
        histnorm (str): Type of normalization to apply.

    Returns:
        bin_names (list): Bin centers (for numeric) or category labels (for categorical).
        bin_values (list): Aggregated values per bin.
        bin_edges (list or None): Edges of numeric bins, or None for categorical.
        bin_labels (list or None): Original samples per bin (only for categorical).
        data_type (str): Either 'numeric' or 'category'.
    """
    x_clean = pd.Series(x).dropna()

    # Step 1: Infer type (numeric or category)
    try:
        x_clean_numeric = pd.to_numeric(x_clean)
        inferred_type = 'numeric'
    except:
        inferred_type = 'category'

    # Step 2: If bins is not provided, force categorical treatment
    if inferred_type == 'numeric' and bins != {}:
        x_clean = x_clean_numeric
        data_type = 'numeric'
    else:
        x_clean = x_clean.astype(str)  # Treat as category
        data_type = 'category'

    # Step 3: Numeric histogram aggregation
    if data_type == 'numeric':
        try:
            start = bins['start']
            end = bins['end']
            size = bins['size']
        except KeyError:
            raise ValueError("For numeric data, `bins` must contain 'start', 'end', and 'size'.")

        bin_edges = np.arange(start, end + size, size)
        counts, edges = np.histogram(x_clean, bins=bin_edges)
        total = counts.sum()
        widths = np.diff(edges)

        # Normalize based on histnorm
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
            values = counts  # default fallback

        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        return centers, values.tolist(), edges.tolist(), None, data_type

    # Step 4: Categorical histogram aggregation
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

   
    sources, targets, values = remove_none_indices(sources, targets, values)

  
    sources = clean_and_convert_nested(sources)
    targets = clean_and_convert_nested(targets)
    values = clean_and_convert_nested(values)


    if add_name:
        method = trace.name if hasattr(trace, 'name') else None
    else:
        method = None
    min_len = min_nonzero_length(sources, targets, values)
    sum_values = sum(values)
    values = [item/(sum_values+1e-8)*100 for item in values]
    
    for i in range(min_len):
        tuple_list.append((
            'sankey',                         
            method or 'name',                 
            safe_get(sources, i),             
            safe_get(targets, i),             
            safe_get(values, i),              
        ))

    return tuple_list
    

def get_funnel_tuple(trace, add_name ):
    tuple_list = []

   
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
   
    for i in range(min_len):
        try:
            if isinstance(y_data[i], str) and isinstance(x_data[i], (int, float)):
                tuple_list.append((
                    'funnel',
                    method or 'name',                               
                    safe_get(y_data, i),                           
                    safe_get(x_data, i)                             
                ))
            else:
                tuple_list.append((
                    'funnel',
                    method or 'name',   
                    safe_get(x_data, i),       
                    safe_get(y_data, i)        
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
       
        child_count = defaultdict(int)
        for parent, child_list in tree.items():
            for child in child_list:
                child_count[parent] += 1

       
        pending = set(labels)
        node_values = {}
        stack = deque()

        for label in labels:
            if label not in tree:  
                node_values[label] = 1
                stack.append(label)

       
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

       
        for node in labels:
            if node not in node_values:
                node_values[node] = 1  

   
    relative_tuples = []
    for parent, children in tree.items():
        total = sum(node_values.get(child, 0) for child in children)
        for child in children:
            proportion = node_values[child] / total *100 if total > 0 else 0.0
            relative_tuples.append(("sunburst", parent, child, round(proportion, round_digits)))

    return relative_tuples
def fix_missing_parents(labels, parents):
    """
    Fix mismatches or missing parent values in a sunburst chart's labels and parents.

    Ensures the `labels` and `parents` lists are aligned in length and all parent references exist.

    Args:
        labels (list): List of node labels.
        parents (list): Corresponding list of parent labels.

    Returns:
        tuple: A pair (labels, parents) with consistent and complete structure.
    """
    labels = list(labels)
    parents = list(parents)

    # If parents list is shorter, prepend empty strings to match the length
    while len(parents) < len(labels):
        parents.insert(0, "")

    # Create mapping: child label → parent label
    child_to_parent = dict(zip(labels, parents))

    # Identify root nodes not referenced as a parent by any other node
    all_labels = set(labels)
    referenced_parents = set(p for p in parents if p not in ["", None])
    unreferenced_roots = all_labels - referenced_parents

    # Insert missing root nodes (not present in mapping)
    for root in unreferenced_roots:
        if root not in child_to_parent:
            labels.insert(0, root)
            parents.insert(0, "")  # Add as top-level root

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
  
    
    if not labels or not parents :
        return []
    labels, parents = fix_missing_parents(labels, parents)

    labels = clean_string(labels)
    parents = clean_string(parents)


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

    treemap_tuples = [('treemap', parent, child, value) for (_, parent, child, value) in tuples]
    return treemap_tuples


def get_carpet_tuple(trace, add_name):
    """
    Extracts a list of tuples representing data points from a 'carpet' trace.

    Supports both 1D and 2D `y` values. Handles optional `x` values, and ensures missing
    data is padded with fallback values (e.g., 0.0001 for missing x).

    Args:
        trace: The Plotly carpet trace object containing attributes like a, b, x, y.
        add_name (bool): Whether to include the trace name in the output tuples.

    Returns:
        list of tuples: Each tuple has the structure:
            ('carpet', trace_name, a, b, x, y)
    """
    tuple_list = []

    a_data = ensure_list(trace.a) if hasattr(trace, 'a') else None
    b_data = ensure_list(trace.b) if hasattr(trace, 'b') else None
    y_data = ensure_list(trace.y) if hasattr(trace, 'y') else None
    x_data = ensure_list(trace.x) if hasattr(trace, 'x') else None  

    # If required data is missing, return empty
    if not a_data or not b_data or not y_data:
        return []

    # Clean and normalize input arrays
    a_data = clean_and_convert_nested(a_data)
    b_data = clean_and_convert_nested(b_data)
    y_data = clean_and_convert_nested(y_data)
    x_data = clean_and_convert_nested(x_data) if x_data else None

    method = trace.name if add_name and hasattr(trace, 'name') else None

    # Case 1: y is a 2D array (multiple rows)
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
        # Case 2: y is a 1D array; x may or may not be present
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
       
        if data is None:
            return None
        if hasattr(data, 'tolist'):  
            return data.tolist()
        if isinstance(data, (list, tuple, np.ndarray)):  
            return list(data)
        return [data]  
    
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
        
        
        if x_data:
            # x_data = remove_none_indices(x_data)
            x_data = clean_y_if_str_header(x_data)
            # x_data = smart_cast_list(x_data)
        if y_data:
            # y_data = remove_none_indices(y_data)
            y_data = clean_y_if_str_header(y_data)
            # y_data = smart_cast_list(y_data)
        
        
       
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
    elem = list(elem)  

    if target_type == "float":
        val = safe_float(elem[idx])
        if val is not None:
            elem[idx] = val
            return tuple(elem), True  
        else:
            return tuple(elem), False

    elif target_type == "str":
        elem[idx] = str(elem[idx])
        return tuple(elem), True


def dispatch(elem1, elem2, tol_word, tol_num):
  
    if elem1[0] == 'histogram':
        # (histogram, type, name, x, y) — if x is a string, it's a categorical histogram and labels should be computed
        # (chart_type, name, x, y)

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

