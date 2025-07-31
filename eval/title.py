import re
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import plotly.graph_objects as go
from typing import Dict
import pandas as pd
import Levenshtein

tol_word =3


def clean_text(text: str) -> str:
   
    if not isinstance(text, str):
        return text  

    text = text.replace('\xa0', ' ')            
    text = text.replace('<br>', ' ')              
    text = re.sub(r'<[^>]+>', '', text)            
    text = re.sub(r'\s+', ' ', text).strip()       
    return text

def calculate_title_metrics(fig_gold, fig_gen) -> Dict[str, float]:
    
    try:
        gold_title = (
            fig_gold.layout.title.text
            if fig_gold.layout.title and fig_gold.layout.title.text
            else None
        )
        gen_title = (
            fig_gen.layout.title.text
            if fig_gen.layout.title and fig_gen.layout.title.text
            else None
        )
    except:
        return {"precision": 0, "recall": 0, "f1": 0}
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
