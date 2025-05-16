
import warnings
import sys
import os
import runpy

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# 清除 warning 缓存（防止已经加载的模块重复发出）
for mod in list(sys.modules.values()):
    if hasattr(mod, '__warningregistry__'):
        mod.__warningregistry__.clear()
import warnings
warnings.filterwarnings(
    "ignore",
    message="plotly.graph_objs.Data is deprecated.*",
    category=DeprecationWarning,
    module=r"plotly\.graph_objs\._deprecations"
)

import os
import importlib.util
import os
import importlib.util
import pandas as pd
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from typing import List, Dict, Optional, Callable, Any
import re
from tqdm import tqdm
from axis import calculate_axis_metrics
from type import calculate_type_metrics
from annotations import calculate_annotations_metrics
from legend import calculate_legend_metrics
from layout import calculate_layout_metrics

from color import calculate_color_metrics
from title import calculate_title_metrics
from data import calculate_data_metrics
from mat import calculate_mat_metrics
# ==== 低层工具函数 ====

def load_figure_from_py(file_path: str):
    """从Python文件中加载fig对象"""
    spec = importlib.util.spec_from_file_location("loaded_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'fig'):
        return module.fig
    else:
        raise ValueError(f"No 'fig' object found in {file_path}")

def evaluate_single_file(gold_path: str, gen_path: str, metric_functions: Dict[str, Callable[[Any, Any], Dict[str, float]]], selected_tasks: List[str]) -> Dict[str, float]:
    result = {k: 0 for k in selected_tasks}
    result['pass_rate'] = 0
    result['valid_num'] = 1
    
    if 'other' in gold_path:
        try:
            runpy.run_path(gen_path, run_name="__main__")
            result['pass_rate'] = 1
        except:
            return result
        res = calculate_mat_metrics(gold_path, gen_path)
        for k,v in res.items():
            result[k] = v['f1']
        return result
    else:
        try:
            
            fig1 = load_figure_from_py(gold_path)
            fig2 = load_figure_from_py(gen_path)
            result['pass_rate'] = 1
        except:
            return result
        for k in selected_tasks:
            try:
                metrics = metric_functions[k](fig1, fig2)
            except:
                result[k] = 0
                continue
            if metrics is not None:
                result[k] = metrics["f1"]
            if k=='data':
                if metrics is None:
                    result['valid_num'] = 0
    
        return result
def evaluate_chart_type(
    chart: str,
    base_gold_dir: str,
    base_gen_dir: str,
    output_csv: str,
    metric_functions: Dict[str, Callable[[Any, Any], Dict[str, float]]],
    selected_tasks: List[str],
    max_workers: int = 4,
    model_name: str = ""
):
    gold_dir = os.path.join(base_gold_dir, chart)
    gen_dir = os.path.join(base_gen_dir, chart)
    files = [f for f in os.listdir(gold_dir) if f.endswith('.py')]

    tasks = [(os.path.join(gold_dir, f), os.path.join(gen_dir, f)) for f in files]

    metrics_accum = {k: [] for k in selected_tasks}
    metrics_accum['valid_num'] = []
    metrics_accum['pass_rate'] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_single_file, gold, gen, metric_functions, selected_tasks)
            for gold, gen in tasks
        ]
        with tqdm(total=len(futures), desc=f"[{model_name}] {chart}", ncols=80) as pbar:
            for future in as_completed(futures):
                result = future.result()
                for k in selected_tasks:
                    metrics_accum[k].append(result[k])
                metrics_accum['valid_num'].append(result.get('valid_num', 0))
                metrics_accum['pass_rate'].append(result.get('pass_rate', 0))
                pbar.update(1)

    # 聚合
    averaged = {
        'chart_type': chart,
        'chart_num': len(files),
        'valid_num': sum(metrics_accum['valid_num']),
        'pass_rate': sum(metrics_accum['pass_rate']),
    }
    for k in selected_tasks:
        averaged[k] = sum(metrics_accum[k]) if metrics_accum[k] else 0

    pd.DataFrame([averaged]).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
    print(f"✅ 完成 chart_type: {chart}，结果已写入 {output_csv}")

    
if __name__ == "__main__":
    models = [
        'InternVL3-9B-Results-merged'
    ]
   
    # models = os.listdir('/home/disk1/lixinhang/code/benchmark/gen/res-code-merge-py')
    for model in models:
        gold_base = 
        gen_base = 
        output_path = 

        chart_types = sorted(os.listdir(gold_base))
        
      
        for chart in chart_types:
            evaluate_chart_type(
                chart,
                gold_base,
                gen_base,
                output_path,
                metric_functions={
                    "legend": calculate_legend_metrics,
                    "title": calculate_title_metrics,
                    "axis": calculate_axis_metrics,
                    "annotations": calculate_annotations_metrics,
                    "layout": calculate_layout_metrics,
                    "type": calculate_type_metrics,
                    "color": calculate_color_metrics,
                    "data": calculate_data_metrics,
                },
                selected_tasks=["legend", "title", "axis", "annotations", "layout", "type", "color", "data"],
                model_name=model,
                max_workers=4
            )
