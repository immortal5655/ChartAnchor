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
    tables = defaultdict(list)
    for row in data_rows:
        per_table = defaultdict(dict)
        for key, value in row.items():
            parts = key.split(';')
            for part in parts:
                part = html.unescape(part.strip())
                if '-' not in part:
                    continue
                try:
                    table, col = part.rsplit('-', 1)
                except ValueError:
                    continue
                table = table.strip()
                col = col.strip()
                per_table[table][col] = value
        for table, rec in per_table.items():
            sorted_rec = {k: rec[k] for k in sorted(rec)}
            tables[table].append(sorted_rec)

    tuple_out = {}
    for table, recs in tables.items():
        tuple_out[table] = [tuple(r[k] for k in sorted(r)) for r in recs]

    return tables, tuple_out

def sort_and_zip_dict(data: dict):
    sorted_keys = sorted(data.keys())
    sorted_values = [data[k] for k in sorted_keys]
    tuples = list(zip(*sorted_values))
    return tuples

def sort_and_extract(data):
    return [tuple(item[k] for k in sorted(item.keys())) for item in data]

def all_keys_have_dash(d: dict):
    return all('-' in k for k in d.keys())

def iou(gold, gen, tol_word, tol_num):
    intersection = set()
    for e1 in gold:
        for e2 in gen:
            if compare_tuple(e1, e2, tol_word, tol_num):
                intersection.add(e1)
    union = set(gold) | set(gen)
    return len(intersection) / (len(union) + 1e-7)

#------- Main evaluation logic -------
def evaluate_single_sample(gold_table: dict, gen_table: dict) -> dict:
    tolerance_levels = {
        "strict": {"tol_word": 0, "tol_num": 0.0},
        "medium": {"tol_word": 3, "tol_num": 0.05},
        "high": {"tol_word": 5, "tol_num": 0.1}
    }

    results = {}

    for tol_name, tol_vals in tolerance_levels.items():
        try:
            if all_keys_have_dash(gold_table):
                _, gold_tuple = extract_and_flatten_final(gold_table)
                _, gen_tuple = parse_rows_to_tables(gen_table)
                gold_all = [e for v in gold_tuple.values() for e in v]
                gen_all = [e for v in gen_tuple.values() for e in v]
            else:
                gold_all = sort_and_zip_dict(gold_table)
                gen_all = sort_and_extract(gen_table)

            correct = 0
            for elem1 in gold_all:
                for elem2 in gen_all:
                    if compare_tuple(elem1, elem2,
                                     tol_word=tol_vals["tol_word"],
                                     tol_num=tol_vals["tol_num"]):
                        correct += 1
                        break  # Match only once

            precision = correct / len(gen_all) if gen_all else 0
            recall = correct / len(gold_all) if gold_all else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            iou_score = iou(gold_all, gen_all,
                            tol_word=tol_vals["tol_word"],
                            tol_num=tol_vals["tol_num"])

            results[tol_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "iou": round(iou_score, 4)
            }
        except Exception as e:
            results[tol_name] = {"precision": 0, "recall": 0, "f1": 0, "iou": 0}
            print(f"❌ Evaluation failed at {tol_name} level: {e}")

    return results



# ============ Example ============
if __name__ == "__main__":
    # Replace with actual paths
    gold_path = 'gold path'
    gen_path =  'gen path'

    if not os.path.exists(gold_path) or not os.path.exists(gen_path):
        print("Invalid path. Please check the input paths.")
        exit(1)

    gold_table = json.load(open(gold_path))
    gen_table = json.load(open(gen_path))

    result = evaluate_single_sample(gold_table, gen_table)

    print("\nEvaluation Results (Single Sample):")
    for level, scores in result.items():
        print(f"  [{level.upper()}] P: {scores['precision']:.4f} | R: {scores['recall']:.4f} | "
              f"F1: {scores['f1']:.4f} | IoU: {scores['iou']:.4f}")
