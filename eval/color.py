import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import plotly.graph_objects as go
from typing import Dict
import pandas as pd
from matplotlib import colors as mcolors
from typing import Any
from color_list import layout_color_dict, data_color_dict
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
import matplotlib.colors as mcolors
from plotly.colors import sample_colorscale
from typing import List, Union, Tuple, Any
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
from matplotlib.colors import to_rgb
from skimage.color import rgb2lab, deltaE_ciede2000


def parse_rgb_string(s: str) -> Tuple[float, float, float, float]:
    """Parse 'rgb(r, g, b)' or 'rgba(r, g, b, a)' into normalized RGBA tuple."""
    m = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d\.]+))?\s*\)', s)
    if not m:
        raise ValueError(f"Invalid RGB(A) string: {s}")
    r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
    a = float(m.group(4)) if m.group(4) is not None else 1.0
    return (r/255.0, g/255.0, b/255.0, a)

def to_lab(color_str: str) -> np.ndarray:
    """Convert a HEX or 'rgb(...)'/'rgba(...)' color string to CIE-Lab."""
    if color_str.strip().startswith("rgb"):
        rgba = parse_rgb_string(color_str)
    else:
        rgba = mcolors.to_rgba(color_str)
    lab = rgb2lab(np.array([[rgba[:3]]]))
    return lab[0, 0]

def normalize_colorscale(
    cs: Union[str, List[Union[str, List[Union[float, str]]]]]
) -> Union[str, List[List[Union[float, str]]]]:
    """
    Normalize various colorscale inputs into a format accepted by sample_colorscale:
    - String: return as-is (named colorscale)
    - List of color strings: convert to [[pos, color], ...]
    - List of [pos, color] pairs: ensure each pair is a list [pos, color]
    """
    if isinstance(cs, str):
        return cs
    if isinstance(cs, list) and all(isinstance(el, str) for el in cs):
        n = len(cs)
        if n < 2:
            raise ValueError("Colors list must contain at least two colors.")
        return [[i/(n-1), cs[i]] for i in range(n)]
    if isinstance(cs, list) and all(isinstance(el, (list, tuple)) and len(el)==2 for el in cs):
        return [[float(el[0]), el[1]] for el in cs]
    raise ValueError("Unsupported colorscale format.")

def compare_colorscales(
    cs1: Union[str, List[Any]],
    cs2: Union[str, List[Any]],
    n_samples: int = 100
) -> dict:
    """
    Compare two colorscales (named, explicit stops, or list of colors).
    Returns mean, max, and std of ΔE00 differences over samples.
    """
    norm1 = normalize_colorscale(cs1)
    norm2 = normalize_colorscale(cs2)

    t_vals = np.linspace(0, 1, n_samples).tolist()
    hex1 = sample_colorscale(norm1, t_vals)
    hex2 = sample_colorscale(norm2, t_vals)

    lab1 = np.array([to_lab(c) for c in hex1])
    lab2 = np.array([to_lab(c) for c in hex2])

    dEs = np.array([deltaE_ciede2000(lab1[i], lab2[i]) for i in range(n_samples)])
    return {
        "mean_dE00": float(dEs.mean()),
        "max_dE00": float(dEs.max()),
        "std_dE00": float(dEs.std()),
    }
import re
from matplotlib import colors as mcolors
from typing import Any

def to_hex(c: Any) -> str:
    
    if isinstance(c, str):
        if c.startswith('#'):
            return c.upper()
        
        rgb_match = re.fullmatch(r'rgba?\(([^)]+)\)', c.replace(' ', ''))
        if rgb_match:
            parts = list(map(float, rgb_match.group(1).split(',')))
            if len(parts) == 3:
                parts.append(1.0)  

            parts = [min(max(v, 0), 255)/255 for v in parts]
            return mcolors.to_hex(parts).upper()
        
        try:
            return mcolors.to_hex(mcolors.to_rgba(c)).upper()
        except ValueError:
            return c 
    
    if isinstance(c, (list, tuple)) and len(c) in (3, 4):
        try:
            return mcolors.to_hex(c).upper()
        except ValueError:
            return str(c)
    
    return str(c)


def get_nested_value(d, key_path):
    """
    Retrieve a value from a nested dictionary using a dot-separated key path, with null safety checks.

    Args:
        d: The input dictionary.
        key_path: Dot-separated string representing the path to the target key (e.g., "a.b.c").

    Returns:
        A tuple of (key_path, processed_value), or None if the path doesn't exist or encounters a None value.
    """
    if not d or not key_path:  # Check for empty dictionary or empty key path
        return None

    keys = key_path.split(".")
    current = d

    try:
        for key in keys:
            if current is None:  # Stop early if value becomes None
                return None

            if isinstance(current, dict):
                if key in current:
                    current = current[key]
                else:
                    return None
            else:
                return None  # Cannot proceed if current is not a dictionary

        # Process the final value
        if isinstance(current, list):
            if current:  # Check for non-empty list
                if isinstance(current[0], list):
                    return (key_path, [[x[0], to_hex(x[1])] for x in current if len(x) >= 2])
                else:
                    return (key_path, [to_hex(x) for x in current])
            else:
                return (key_path, [])  # Return empty list as-is

        return (key_path, to_hex(current)) if current is not None else None

    except (IndexError, AttributeError, TypeError):  # Handle unexpected structure or invalid access
        return None



def get_axis_nested_value(d, key_path):
   
    keys = key_path.split(".")[1:]
    
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return None
    
    return (key_path,to_hex(d))

    
   
def get_color_from_layout(layout, key, rename=None):
    """
    Try to get the color value from layout; fallback to template if not found.
    Return a tuple (renamed_key, hex_color).
    """
    
    value = getattr(layout, key, None)
    if value is None:
        template = getattr(layout, 'template',None)
        if template:
            value = getattr(template.layout, key)
    if value is not None:
        return (rename or key, to_hex(value))
    else:
        return None

def get_all_color(fig,  all_colors = [],fallback_keys: Optional[List[str]] = None) -> List[Tuple[str, Any]]:
    """
    Returns a list of (color_key, color_value) pairs.

    In 'gold' mode: no fallback is applied.  
    In 'gen' mode: fallback to the template is applied only for keys in fallback_keys.
    """

  
    if not fallback_keys:
        all_colors=[]
    layout = fig.layout.to_plotly_json()
  
    template_layout = fig.layout.template.layout.to_plotly_json()

    # layout_color_dict, data_color_dict assumed defined globally
    for k, v in layout_color_dict.items():
        if k == 'bgcolor':
            for color_name in v:
                color = None
                if fallback_keys:
                    if color_name not in fallback_keys:
                        continue
                    color = get_color_from_layout(fig.layout.template.layout, color_name) 
                else:
                    color = get_color_from_layout(fig.layout, color_name)
                if color:
                    all_colors.append(color)
        elif k == 'axiscolor':
            for color_name in v:
                for item_name, item_obj in layout.items():
                    if item_name.startswith(("xaxis", "yaxis", "zaxis")):
                        color = None
                        if fallback_keys:
                            if color_name not in fallback_keys:
                                continue
                            tmpl_item = template_layout.get(item_name, {})
                            color = get_axis_nested_value(tmpl_item, color_name)
                        else:
                            color = get_axis_nested_value(item_obj, color_name)
                        if color:
                            all_colors.append(color)


        else:
            for color_name in v:
                color = None
                if fallback_keys:
                    if color_name not in fallback_keys:
                        continue
                    color = get_nested_value(template_layout, color_name)
                else:
                    color = get_nested_value(layout, color_name)
                if color:
                    all_colors.append(color)
                


    for k, v in data_color_dict.items():
        for entry in fig.data:
            entry_dict = entry.to_plotly_json()
            for color_name in v:
                color = None
                if fallback_keys:
                    if color_name not in fallback_keys:
                        continue
                    trace_type = entry_dict.get('type') 
                   
                    template_traces = getattr(fig.layout.template.data,trace_type, [])
                    if template_traces:
                        entry_dict_template = template_traces[0].to_plotly_json()
                        color = get_nested_value(entry_dict_template, color_name)
                else:
                    color = get_nested_value(entry_dict, color_name)
                if color:
                    all_colors.append(color)
                
    return all_colors

def group_by_name(pairs):
   
    grouped = defaultdict(list)
    for name, color in pairs:
        grouped[name].append(color)
    return dict(grouped)



def is_single_color_valid(c: str) -> bool:
    if not isinstance(c, str):
        return False
    c = c.strip()
    if re.fullmatch(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})", c):
        return True
    if re.fullmatch(r"rgba?\((\d+,\s*){2}\d+(,\s*\d+(\.\d+)?)?\)", c):
        return True
    try:
        mcolors.to_rgba(c)
        return True
    except ValueError:
        return False

def is_color_or_colorscale_valid(value: Any) -> bool:
    """
    Check whether a value is a valid single color or a valid colorscale.
    """
    if isinstance(value, list):
       
        for item in value:
            if isinstance(item, list) and len(item) == 2:
                _, color = item
                if not is_single_color_valid(color):
                    return False
            else:
               
                return False
        return True
    else:
        return is_single_color_valid(value)

def extract_colorscale(group):
    colorscale = { 
        'coloraxis.colorscale':  [],
        'colorscale':  []
    }
    if 'coloraxis.colorscale' in group:
        colorscale['coloraxis.colorscale']=group['coloraxis.colorscale']
        group.pop('coloraxis.colorscale')
    if 'colorscale' in group:
        colorscale['colorscale']=group['colorscale']
        group.pop('colorscale')
    return colorscale,group


def compare_singal_colorscale(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    try:
        res = compare_colorscales(s1[0], s2[0], n_samples=200)
        similarity = max(0, 1 - res['mean_dE00'] / 100)
    except:
        return 0
    return similarity

    
    
def compare_all_colorscale(gold_colorscale, gen_colorscale):
    sim = 0
    sim+=compare_singal_colorscale(gold_colorscale['coloraxis.colorscale'], gen_colorscale['coloraxis.colorscale'])
    sim+=compare_singal_colorscale(gold_colorscale['colorscale'], gen_colorscale['colorscale'])
    return sim

def hex_color_similarity(c1: str, c2: str) -> float:
   
    rgb1 = to_rgb(c1)  # [0,1]
    rgb2 = to_rgb(c2)
    
    lab1 = rgb2lab([[rgb1]])[0][0]
    lab2 = rgb2lab([[rgb2]])[0][0]
    
    return deltaE_ciede2000(lab1, lab2) 

def hex_similarity_score(c1, c2):
    dE = hex_color_similarity(c1, c2)
    return max(0.0, 1.0 - dE / 100)  
    
def calculate_similarity_single(c1,c2):
   
    if c1.startswith("#") and c2.startswith("#"):
        return hex_similarity_score(c1, c2)
    elif not c1.startswith("#") and not c2.startswith("#"):

        return 1 if c1 == c2 else 0
    else:
        return 0

def compute_precision_hungarian(gold_colors: List[str], gen_colors: List[str], threshold: float = 0.0) -> float:
    """
    Compute precision for color matching using the Hungarian algorithm.
    - Automatically filters out invalid color values.
    - Uses `calculate_similarity_single` to compute similarity between colors.
    """

    def extract_and_validate(colors):
        return [c for c in colors if is_single_color_valid(c)]

    gen_colors = extract_and_validate(gen_colors)
    gold_colors = extract_and_validate(gold_colors)
    
    if len(gen_colors) == 0 or len(gold_colors) == 0:
        return 0.0


    sim_matrix = np.zeros((len(gen_colors), len(gold_colors)))
    for i, g1 in enumerate(gen_colors):
        for j, g2 in enumerate(gold_colors):
            sim_matrix[i, j] = calculate_similarity_single(g1, g2)  

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    matched_similarities = [sim_matrix[i, j] for i, j in zip(row_ind, col_ind)]

    matched_count = sum(sim for sim in matched_similarities if sim >= threshold)
    
    return matched_count

    
def calculate_color_metrics(fig_gold, fig_gen):
    gold_colors = get_all_color(fig_gold)
    gen_colors = get_all_color(fig_gen)
    gold_keys = [key for key, _ in gold_colors]
    gen_keys = [key for key, _ in gen_colors]
    gold_fallback_keys = list(set([key for key in gen_keys if key not in gold_keys]))
    gen_fallback_keys = list(set([key for key in gold_keys if key not in gen_keys]))
    if gen_fallback_keys:
        gen_colors = get_all_color(fig_gen,gen_colors,fallback_keys=gen_fallback_keys)
    if gold_fallback_keys:
        gold_colors = get_all_color(fig_gold,gold_colors,fallback_keys=gold_fallback_keys)
    
    gen_group = group_by_name(gen_colors)
    gold_group = group_by_name(gold_colors)
    merged_color_group = list( set( gold_keys + gen_keys ) )
    for color in merged_color_group:
        if color not in gen_group:
            gen_group[color] = []
        if color not in gold_group:
            gold_group[color] = []
    sim = 0

    gen_colorscale,gen_goup = extract_colorscale(gen_group)
    gold_colorscale,gold_goup = extract_colorscale(gold_group)

    sim+= compare_all_colorscale(gold_colorscale, gen_colorscale)
    
    for color in merged_color_group:
        if color == 'coloraxis.colorscale' or color =='colorscale' :
            continue
        if color == 'marker.colors':
            try:
                max_len = max(len(gold_group[color][0]),len(gen_group[color][0]))
                sim += (compute_precision_hungarian(gold_group[color][0], gen_group[color][0])/ max_len) if max_len else 0
            except:
                pass
        
        sim += compute_precision_hungarian(gold_group[color], gen_group[color])
    

    precision = sim / len(gen_colors) if len(gen_colors) != 0 else 0
    recall = sim / len(gold_colors) if len(gold_colors) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


   
    return {"precision": precision, "recall": recall, "f1": f1}
   