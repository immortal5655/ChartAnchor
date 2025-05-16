import json
import os
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
import importlib
import math
from data import *

EPS = 1e-6  
import json
from collections import defaultdict

# ------- 方法一：结构化表格 ----------
def extract_and_flatten_final(sample: dict):
    def is_2d(val):
        return isinstance(val, list) and val and all(isinstance(i, list) for i in val)
    
    tables, shared = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for k, v in sample.items():
        try:
            parts = k.split(';')
            for p in parts:
                tbl, col = p.strip().rsplit('-', 1)
                if ';' in k:
                    shared[tbl][col] = v
                else:
                    tables[tbl][col] = v
        except ValueError:
            continue

    for t, cols in shared.items():
        tables[t].update(cols)

    dict_out, tuple_out = {}, {}
    for tbl, cols in tables.items():
        has_mat = any(is_2d(v) for v in cols.values())
        recs, tpls = [], []

        if has_mat:
            mat_key = next(k for k, v in cols.items() if is_2d(v))
            n_rows = len(cols[mat_key])
            for i in range(n_rows):
                for j in range(len(cols[mat_key][i])):
                    row = {}
                    for col, val in cols.items():
                        if is_2d(val):
                            row[col] = val[i][j] if j < len(val[i]) else None
                        elif isinstance(val, list):
                            row[col] = val[i] if i < len(val) else None
                        else:
                            row[col] = val
                    row = {k: row[k] for k in sorted(row)}
                    recs.append(row)
                    tpls.append(tuple(row.values()))
        else:
            lists = {k: v for k, v in cols.items() if isinstance(v, list)}
            min_len = min(map(len, lists.values())) if lists else 1
            for i in range(min_len):
                row = {col: (val[i] if isinstance(val, list) else val) for col, val in cols.items()}
                row = {k: row[k] for k in sorted(row)}
                recs.append(row)
                tpls.append(tuple(row.values()))

        dict_out[tbl], tuple_out[tbl] = recs, tpls

    return dict_out, tuple_out

def parse_rows_to_tables(data_rows):
    from collections import defaultdict

    tables = defaultdict(list)

    for row in data_rows:
        per_table = defaultdict(dict)

        for key, value in row.items():
            parts = key.split(';')
            for part in parts:
                part = html.unescape(part.strip())  # 解码 HTML，例如 &nbsp;
                if '-' not in part:
                    continue  # 跳过格式错误的列名

                try:
                    table, col = part.rsplit('-', 1)
                except ValueError:
                    continue  # 安全处理异常

                table = table.strip()
                col = col.strip()
                per_table[table][col] = value

        for table, rec in per_table.items():
            sorted_rec = {k: rec[k] for k in sorted(rec)}
            tables[table].append(sorted_rec)

    # 构建 tuple_out
    tuple_out = {}
    for table, recs in tables.items():
        tuple_out[table] = [tuple(r[k] for k in sorted(r)) for r in recs]

    return tables, tuple_out


# ------- 方法二：普通结构 ----------
def sort_and_zip_dict(data: dict):
    # 按 key 排序后取出值列表
    sorted_keys = sorted(data.keys())
    sorted_values = [data[k] for k in sorted_keys]
    
    # 将值按位置配对成元组
    tuples = list(zip(*sorted_values))
    return tuples
def sort_and_extract(data):
    """
    对列表中的每个字典的key排序后提取其值，形成元组，最后返回元组列表。
    """
    result = []
    for item in data:
        # 按 key 排序后提取对应的 value
        sorted_values = tuple(item[key] for key in sorted(item.keys()))
        result.append(sorted_values)
    return result

# ------- 判断函数 ----------
def all_keys_have_dash(d: dict):
    return all('-' in k for k in d.keys())

def intersection_with_tolerance(gold, gen, tol_word, tol_num):
    sim_set = set()
    gold = list(set(gold))
    gen = list(set(gen))


    for elem1 in gold:
        for elem2 in gen:
            if compare_tuple(elem1, elem2, tol_word, tol_num):
                sim_set.add(elem1)
    return list(sim_set)

def union_with_tolerance(a, b, tol_word, tol_num):
    c = set(a) | set(b)
    d = set(a) & set(b)
    e = intersection_with_tolerance(a, b, tol_word, tol_num)
    f = set(e)
    g = c-(f-d)
    return list(g)

def iou(gold, gen, tol_word, tol_num):
    intersection = intersection_with_tolerance(gold, gen, tol_word, tol_num)
    union = union_with_tolerance(gold, gen, tol_word, tol_num)
    sim = len(intersection)/(len(union)+1e-7)
    return sim

tolerance_levels = {
    "strict": {"tol_word": 0, "tol_num": 0.0},
    "medium": {"tol_word": 3, "tol_num": 0.05},
    "high": {"tol_word": 5, "tol_num": 0.1}
}

models = os.listdir(path)
models.sort()

for model in models:
    print(f"Model: {model}")
    chart_types = os.listdir('path')
    chart_types.sort()

    merged_results = []

    for chart in chart_types:
        print(f"  Chart Type: {chart}")
        info_path = f'path'
        if not os.path.exists(info_path):
            continue

        gold_tables = json.load(open(info_path, 'r'))
        total_count = len(gold_tables)

        row_result = {
            "chart_type": chart,
            "chart_count": total_count
        }

        for tol_name, tol_vals in tolerance_levels.items():
            pass_count = 0
            sum_iou = 0
            sum_p = 0
            sum_r = 0
            sum_f1 = 0

            for ind in range(total_count):
                gold_table = gold_tables[ind]['simple_table']
                fid = gold_tables[ind]['fid']
                gen_path = f'path.json'

                try:
                    gen_table = json.load(open(gen_path, 'r'))
                    pass_count += 1
                except:
                    continue

                try:
                    if all_keys_have_dash(gold_table):
                        _, gold_tuple = extract_and_flatten_final(gold_table)
                        _, gen_tuple = parse_rows_to_tables(gen_table)
                        correct_s = set()
                        gold_num = sum(len(v) for v in gold_tuple.values())
                        gen_num = sum(len(v) for v in gen_tuple.values())


                        for key in gold_tuple:
                            if key not in gen_tuple:
                                continue
                            for elem1 in gold_tuple[key]:
                                for elem2 in gen_tuple[key]:
                                    if compare_tuple(elem1, elem2,
                                                     tol_word=tol_vals["tol_word"],
                                                     tol_num=tol_vals["tol_num"]):
                                        correct_s.add(elem1)

                        correct = len(correct_s)
                        gold_all = [e for v in gold_tuple.values() for e in v]
                        gen_all = [e for v in gen_tuple.values() for e in v]
                    else:
                        gold_list = sort_and_zip_dict(gold_table)
                        gen_list = sort_and_extract(gen_table)
                        correct_s = set()
                        for elem1 in gold_list:
                            for elem2 in gen_list:
                                if compare_tuple(elem1, elem2,
                                                 tol_word=tol_vals["tol_word"],
                                                 tol_num=tol_vals["tol_num"]):
                                    correct_s.add(elem1)
                        gold_num = len(gold_list)
                        gen_num = len(gen_list)
                        correct = len(correct_s)
                        gold_all = gold_list
                        gen_all = gen_list

                    precision = correct / gen_num if gen_num else 0
                    recall = correct / gold_num if gold_num else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
                    sim = iou(gold_all, gen_all,
                              tol_word=tol_vals["tol_word"],
                              tol_num=tol_vals["tol_num"])
                    

                except:
                    continue

              
                sum_iou += sim
                sum_p += precision
                sum_r += recall
                sum_f1 += f1

            row_result[f"sum_{tol_name}_p"] = round(sum_p, 4)
            row_result[f"sum_{tol_name}_r"] = round(sum_r, 4)
            row_result[f"sum_{tol_name}_f1"] = round(sum_f1, 4)
            row_result[f"sum_{tol_name}_iou"] = round(sum_iou, 4)

            # 只保留 medium 的 pass_count，用作统一 pass_rate 展示
            if tol_name == "medium":
                row_result["pass_count"] = pass_count

        merged_results.append(row_result)

    # 指定列顺序
    col_order = ["chart_type", "chart_count", "pass_count",
                 "sum_strict_p", "sum_strict_r", "sum_strict_f1", "sum_strict_iou",
                 "sum_medium_p", "sum_medium_r", "sum_medium_f1", "sum_medium_iou",
                 "sum_high_p", "sum_high_r", "sum_high_f1", "sum_high_iou"]
    df = pd.DataFrame(merged_results)
    df = df[col_order]
    df.to_csv(f"path/{model}_table_metrics_sum.csv", index=False)
    print(f"✅ Saved to {model}_table_metrics_sum.csv")
