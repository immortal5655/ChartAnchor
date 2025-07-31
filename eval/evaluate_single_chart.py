import os
import runpy
import importlib.util
import warnings
import pandas as pd
from typing import List, Dict, Callable, Any
import traceback

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Clear warning registry from already-loaded modules
import sys
for mod in list(sys.modules.values()):
    if hasattr(mod, '__warningregistry__'):
        mod.__warningregistry__.clear()

# ========== Import metric functions ==========
from axis import calculate_axis_metrics
from type import calculate_type_metrics
from annotations import calculate_annotations_metrics
from legend import calculate_legend_metrics
from layout import calculate_layout_metrics
from color import calculate_color_metrics
from title import calculate_title_metrics
from data import calculate_data_metrics
from mat import calculate_mat_metrics

# ========== Load figure object ==========
def load_figure_from_py(file_path: str):
    """Load 'fig' object from a Python file"""
    spec = importlib.util.spec_from_file_location("loaded_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'fig'):
        return module.fig
    else:
        raise ValueError(f"No 'fig' object found in {file_path}")

# ========== Evaluate a single sample ==========
def evaluate_single_pair(
    gold_path: str,
    gen_path: str,
    metric_functions: Dict[str, Callable[[Any, Any], Dict[str, float]]],
    selected_tasks: List[str]
) -> Dict[str, float]:
    result = {k: 0 for k in selected_tasks}
    result['pass_rate'] = 0
    result['valid_num'] = 1

    if '_mat' in gold_path:
        try:
            runpy.run_path(gen_path, run_name="__main__")
            result['pass_rate'] = 1
        except Exception as e:
            print(f"⚠️ Error executing MAT chart: {e}")
            return result

        res = calculate_mat_metrics(gold_path, gen_path)
        for k, v in res.items():
            result[k] = v['f1']
        return result

    else:
        try:
            fig1 = load_figure_from_py(gold_path)
            fig2 = load_figure_from_py(gen_path)
            result['pass_rate'] = 1
        except Exception as e:
            print(f"❌ Failed to load 'fig' object: {e}")
            return result

        for k in selected_tasks:
            try:
                metrics = metric_functions[k](fig1, fig2)
            except Exception as e:
                print(f"❌ Error computing metric '{k}': {e}")
                result[k] = 0
                continue
            if metrics is not None:
                result[k] = metrics.get("f1", 0)
            if k == 'data' and metrics is None:
                result['valid_num'] = 0

        return result

# ========== Main entry ==========
if __name__ == "__main__":
    
    gold_path =    'your gold python file path'
    gen_path =    'your generate python file path'

    selected_tasks = ["legend", "title", "axis", "annotations", "layout", "type", "color", "data"]

    if '_mat' in gold_path:
        selected_tasks = [task for task in selected_tasks if task not in ['annotations', 'layout']]

    metric_functions = {
        "legend": calculate_legend_metrics,
        "title": calculate_title_metrics,
        "axis": calculate_axis_metrics,
        "annotations": calculate_annotations_metrics,
        "layout": calculate_layout_metrics,
        "type": calculate_type_metrics,
        "color": calculate_color_metrics,
        "data": calculate_data_metrics,
    }

    result = evaluate_single_pair(
        gold_path,
        gen_path,
        metric_functions=metric_functions,
        selected_tasks=selected_tasks
    )

    print("\nEvaluation result for a single sample:")
    for k, v in result.items():
        print(f"{k:12s}: {v:.4f}")
