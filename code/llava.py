
from PIL import Image
from tqdm import tqdm  # 用于显示进度条
import os
import argparse
import time
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import torch
import torch.distributed as dist
import datetime
import csv

def extract_png_from_csv(file_path):
    """
    从 CSV 文件中提取所有非表头、以 .png 结尾的字符串项

    参数:
        file_path: str - CSV 文件路径

    返回:
        List[str] - 所有以 .png 结尾的字符串
    """
    png_entries = []

    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头
        for row in reader:
            for cell in row:
                if isinstance(cell, str) and cell.strip().lower().endswith('.png'):
                    png_entries.append(cell.strip())

    return png_entries

def inference(mode, chart_type,image_path,model,processor,generation_config):
    image = Image.open(image_path)
    if mode == 'code':
        if 'other' in chart_type:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Matplotlib library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."
        else:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Plotly library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."

    if mode == 'table':
        base_name = os.path.splitext(os.path.basename(image_path))[0]
       
        prompt_path = os.path.join('/root/paddlejob/workspace/env_run/benchmark/prompts',chart_type, base_name + ".txt")
        f = open(prompt_path,'r')
        question = f.read().strip()
    conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]
    
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch.bfloat16).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=4096,do_sample=True, temperature=0.1,top_p=1)
    res = processor.decode(output[0], skip_special_tokens=True)
    return res
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/llava-v1.6-mistral-7b-hf",
                        help="Path to the Qwen2.5-VL model.")
    parser.add_argument("--base_image_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/input_image/human_image",
                        help="Directory containing chart images in subfolders.")
    parser.add_argument("--output_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/llava-v1.6-mistral-7b-hf_Results",
                        help="Directory to save the generation outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--mode", type=str, default='code',
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--chart_types", type=str,  # 默认支持多个类型，逗号分隔
    help="Comma-separated list of chart types to process. E.g. 'bar,box,violin'"
)
    args = parser.parse_args()

        # 设置采样参数

    dist.init_process_group(
    backend="nccl",
    init_method="env://",
    timeout=datetime.timedelta(seconds=1800)
    )
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        print("World size:", world_size)
        print("Loading DeepSeek-VL2 model on rank=0 ...")
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda(local_rank).eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_path)

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.1,top_p=1)

    chart_types = [
        'barpolar','histogram','histogram2d','histogram2dcontour','ohlc','scatter','scatter3d', 'scatterpolar','streamtube','surface','waterfall', 'areachart','bar','box','candlestick','carpet','cone','contour','funnel','funnelarea','heatmap','line','line3d','mesh3d','pie','treemap','violin','sankey','scatterternary','parcoords'
    ]
    chart_types = ['parcoords']
  
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
        print(f"Found total {len(all_image_paths)} images.")

    # 如果没有图像，直接结束
    if not all_image_paths:
        dist.barrier()
        dist.destroy_process_group()
        return

    # 每个进程单独建输出子目录，避免文件冲突
    local_output_dir = os.path.join(args.output_dir, f"rank_{local_rank}")
    os.makedirs(local_output_dir, exist_ok=True)

    # ------------------------- 切分数据给各进程 ------------------------- #
    local_data = all_image_paths[local_rank::world_size]
    total_time = 0.0
    num_samples = 0

    start_time_global = time.time()

    # 遍历并推理
    for chart_type, image_path in tqdm(local_data, desc=f"[Rank {local_rank}] Processing"):
        start_time = time.time()
        try:
            
            result_text = inference(args.mode, chart_type,image_path,model,processor,generation_config)
            
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

