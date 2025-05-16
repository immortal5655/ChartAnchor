import os
import time
import argparse
from tqdm import tqdm
import torch
from pathlib import Path
from PIL import Image
import torch.distributed as dist
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def qwen_image2plotly_inference(mode, chart_type,image_path, model, processor, gen_kwargs):
    """
    使用 Qwen2.5-VL 对给定图表图像进行推理，输出可执行的 Plotly Python 代码。
    """
    if mode == 'code':
        if 'other' in chart_type:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Matplotlib library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."
        else:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Plotly library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."
    if mode == 'table':
        base_name = Path(image_path).stem
        prompt_path = Path('/root/paddlejob/workspace/env_run/benchmark/prompts') / chart_type / f"{base_name}.txt"
        # 明确使用 UTF‑8
        with prompt_path.open('r', encoding='utf-8') as f:
            question = f.read().strip()  
        
    
    messages = [{'role': 'user', 'content': [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=4096, do_sample=True,top_p=0.95,temperature=0.1)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-VL-7B-Instruct",
                        help="Path to the Qwen2.5-VL model.")
    parser.add_argument("--base_image_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/input_image/human_image",
                        help="Directory containing chart images in subfolders.")
    parser.add_argument("--output_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/res-code/Qwen2.5-VL-7B_Results",
                        help="Directory to save the generation outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--mode", type=str,help="Maximum number of new tokens to generate.")
    parser.add_argument("--chart_types", type=str,  # 默认支持多个类型，逗号分隔
    help="Comma-separated list of chart types to process. E.g. 'bar,box,violin'"
)
    args = parser.parse_args()

    # ------------------------- 初始化分布式 ------------------------- #
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        print("World size:", world_size)
        print("Loading Qwen2.5-VL model and processor on rank=0 ...")

    # ------------------------- 加载模型与处理器 ------------------------- #
    # 显存充足，可直接把整份模型都搬到 rank 对应的 GPU

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval().cuda(local_rank)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # ------------------------- 收集所有要处理的图像 ------------------------- #
    chart_types = [x.strip() for x in args.chart_types.split(',') if x.strip()]
    all_image_paths = []
    for chart_type in chart_types:
        chart_dir = os.path.join(args.base_image_dir, chart_type)
        if not os.path.exists(chart_dir):
            continue
        for fname in os.listdir(chart_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                all_image_paths.append((chart_type, os.path.join(chart_dir, fname)))

    if local_rank == 0:
        print("Found total {} images.".format(len(all_image_paths)))

    if not all_image_paths:
        dist.barrier()
        dist.destroy_process_group()
        return

    # 每个进程单独建输出子目录，避免文件冲突
    local_output_dir = os.path.join(args.output_dir, f"rank_{local_rank}")
    os.makedirs(local_output_dir, exist_ok=True)

    # ------------------------- 把数据切给各进程 ------------------------- #
    local_data = all_image_paths[local_rank::world_size]
    total_time = 0.0
    num_samples = 0

    start_time_global = time.time()
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.1,top_p=0.95)
    for chart_type, image_path in tqdm(local_data, desc=f"[Rank {local_rank}] Processing"):
        start_time = time.time()
        try:
            result_text = qwen_image2plotly_inference(
                args.mode, chart_type,
                image_path, 
                model, 
                processor,
                gen_kwargs
            )

            # 写文件
            out_subdir = os.path.join(local_output_dir, chart_type)
            os.makedirs(out_subdir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(out_subdir, base_name + ".txt")

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result_text)

            end_time = time.time()
            total_time += (end_time - start_time)
            num_samples += 1
        except Exception as e:
            print(f"[Rank {local_rank}] Error processing {image_path}: {e}")
    
       
    end_time_global = time.time()
    if num_samples > 0:
        avg_time = total_time / num_samples
        print(f"[Rank {local_rank}] Processed {num_samples} images, avg time {avg_time:.4f}s, total {end_time_global - start_time_global:.4f}s")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
