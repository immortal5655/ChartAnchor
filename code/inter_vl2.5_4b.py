import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer,AutoConfig
import os
import json
import time
from tqdm import tqdm 
import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def intern_vl3_chart2code_inference(mode, chart_type,image_path,model,tokenizer,generation_config):
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(model.device)
    if mode == 'code':
        if 'other' in chart_type:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Matplotlib library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."
        else:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Plotly library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."

      
    if mode == 'table':
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        prompt_path = os.path.join('/root/paddlejob/workspace/env_run/benchmark/prompts',chart_type, base_name + ".txt")
        f = open(prompt_path,'r')
        question = '<image>\n'+f.read().strip()
        
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/InternVL2.5-4B",
                        help="HuggingFace Hub or local path of the DeepSeek-VL2 model.")
    parser.add_argument("--base_image_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/input_image/human_image",
                        help="Directory containing chart images in subfolders.")
    parser.add_argument("--output_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/InternVL3-14B-Results",
                        help="Directory to save the generation outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--mode", type=str,help="Maximum number of new tokens to generate.")
    parser.add_argument("--chart_types", type=str,  # 默认支持多个类型，逗号分隔
    help="Comma-separated list of chart types to process. E.g. 'bar,box,violin'"
)

    args = parser.parse_args()

    # ------------------------- 初始化分布式 ------------------------- #
    import datetime

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
        print("Loading intern2.5-4b model on rank=0 ...")

    # ------------------------- 加载 DeepSeek-VL2 模型与处理器 ------------------------- #
    # 如果显存充足，可以直接用 .cuda(local_rank)，将整份模型都放到该卡
    # 如果显存可能不够，可以使用 device_map="auto" 的方式
    model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    ).cuda(local_rank).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.1,top_p=0.95)
    chart_types = [x.strip() for x in args.chart_types.split(',') if x.strip()]
    ll = extract_png_from_csv('/root/paddlejob/workspace/env_run/benchmark/redo.csv')


    all_image_paths = []
    for chart_type in chart_types:
        chart_dir = os.path.join(args.base_image_dir, chart_type)
        if not os.path.exists(chart_dir):
            continue
        for fname in os.listdir(chart_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                if (fname in ll) or chart_type in ['barpolar','histogram','histogram2d','histogram2dcontour','ohlc','parcoords','sankey','scatter','scatter3d','scatterpolar','scatterternary','sunburst','surface','waterfall','line_other','areachart_other','scatter_other','bar_other']:
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
            # 调用我们封装的 deepseek_chart2code_inference
            result_text = intern_vl3_chart2code_inference(args.mode, chart_type,image_path,model,tokenizer,generation_config)
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




