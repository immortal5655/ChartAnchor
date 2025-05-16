
# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
多进程 × 多卡推理（4 卡 → 2 进程，每进程 2 卡）
------------------------------------------------
* 每个子进程用 2 张卡独占加载模型
* 主进程仅负责切任务、启动子进程，不占显存
"""

import os, argparse, multiprocessing as mp
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import csv




# ---------- inference ---------------------------------------------------------
def inference(mode: str,
              chart_type: str,
              img_path: str,
              model,
              tokenizer,
              gen_kwargs,
              device,
              torch_type) -> str:
    """单张图推理"""
    if mode == "code":
        if 'other' in chart_type:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Matplotlib library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."
        else:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Plotly library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."

    else:  # table
        base = os.path.splitext(os.path.basename(img_path))[0]
        prompt_path = os.path.join(
            "/root/paddlejob/workspace/env_run/benchmark/prompts",
            chart_type, base + ".txt"
        )
        with open(prompt_path) as f:
            question = f.read().strip()

    img = Image.open(img_path).convert("RGB")
    query = f"USER: {question} ASSISTANT:"
    feat = model.build_conversation_input_ids(
        tokenizer, query=query, images=[img], template_version="chat"
    )

    inputs = {
        "input_ids":      feat["input_ids"].unsqueeze(0).to(device),
        "token_type_ids": feat["token_type_ids"].unsqueeze(0).to(device),
        "attention_mask": feat["attention_mask"].unsqueeze(0).to(device),
        "images":        [[feat["images"][0].to(device).to(torch_type)]],
    }

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
        out = out[:, inputs["input_ids"].shape[1]:]          # 去掉 prompt
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ---------- 子进程 -------------------------------------------------------------
def worker(gpu_ids: List[int],
           task_chunk: List[Tuple[str, str]],
           opt: argparse.Namespace,
           rank: int):
    # 1) GPU 隔离
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # 2) 进程本地 device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_type = (torch.bfloat16 if torch.cuda.is_available() and
                  torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

    # 3) 模型 & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        opt.model_path, trust_remote_code=True, padding_side="left"
    )
    with init_empty_weights():
        model_tmp = AutoModelForCausalLM.from_pretrained(
            opt.model_path, torch_dtype=torch_type, trust_remote_code=True
        )

    max_mem = {i: "24GiB" for i in range(len(gpu_ids))}
    dev_map = infer_auto_device_map(
        model_tmp, max_memory=max_mem,
        no_split_module_classes=["CogVLMDecoderLayer", "TransformerLayer", "Block"],
    )
    model = load_checkpoint_and_dispatch(
        model_tmp, opt.model_path, device_map=dev_map, dtype=torch_type
    ).eval()

    gen_kwargs = dict(
        max_new_tokens=opt.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or 128002,
        top_p=0.95, temperature=0.1, do_sample=True,
    )

    # 4) 推理循环
    for ct, img_path in tqdm(task_chunk, position=rank, desc=f"worker-{rank}"):
        try:
            text = inference(opt.mode, ct, img_path,
                             model, tokenizer, gen_kwargs,
                             device, torch_type)
            out_dir = os.path.join(opt.output_dir, ct)
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[GPU{gpu_ids}] 处理 {img_path} 出错: {e}")


# ---------- 主进程 -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/cogvlm2-llama3-chat-19B")
    parser.add_argument("--base_image_dir",
                        default="/root/paddlejob/workspace/env_run/benchmark/input_image_table/human_image")
    parser.add_argument("--output_dir",
                        default="/root/paddlejob/workspace/env_run/benchmark/res-table/CogVLM2_results")
    parser.add_argument("--chart_types",
                        default="scatterternary,sunburst,surface,treemap,violin,waterfall")
    parser.add_argument("--mode", choices=["table", "code"], default="table")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    opt = parser.parse_args()

    # 收集任务


    tasks: List[Tuple[str, str]] = []
    for ct in [x.strip() for x in opt.chart_types.split(",") if x.strip()]:
        sub = os.path.join(opt.base_image_dir, ct)
        for f in os.listdir(sub):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
               
                tasks.append((ct, os.path.join(sub, f)))
    print("Total images:", len(tasks))

    # 两进程 × 两卡
    chunks = [tasks[i::2] for i in range(2)]
    gpu_groups = [[0, 1], [2, 3]]

    procs = []
    for rk, (gids, chunk) in enumerate(zip(gpu_groups, chunks)):
        p = mp.Process(target=worker, args=(gids, chunk, opt, rk))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    print("All done.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()



