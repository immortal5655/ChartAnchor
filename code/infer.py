import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import json
import time
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
from tqdm import tqdm  # 用于显示进度条
import os



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

model_name = 'InternVL2_5-4B'
path = '/root/paddlejob/workspace/env_run/output/models/'+model_name
res_dir = '/root/paddlejob/workspace/env_run/benchmark/'+model_name+'_results'
os.makedirs(res_dir,exist_ok=True)
# device_map = split_model('InternVL2-8B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    ).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
generation_config = dict(max_new_tokens=4096, do_sample=True)

base_image_dir = '/root/paddlejob/workspace/env_run/benchmark/input_image/human_image'
chart_types = ['areachart','bar','box','candlestick','carpet','cone','contour','funnel','funnelarea','heatmap', 'line', 'line3d','mesh3d','pie', 'treemap', 'violin']


# 收集所有图像路径
all_image_paths = []
for chart in chart_types:
    chart_dir = os.path.join(base_image_dir, chart)
    for image in os.listdir(chart_dir):
        all_image_paths.append((chart, os.path.join(chart_dir, image)))

# 总体进度条
total_time = 0
num = 0
with torch.inference_mode():
    for chart, image_path in tqdm(all_image_paths, desc="Processing all images"):
        try:
            start_time = time.time()
            image_name = os.path.basename(image_path)
            output_dir = os.path.join(res_dir, chart)
            os.makedirs(output_dir, exist_ok=True)
            print(output_dir)
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            question = '<image>\nYou are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Plotly library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process.'
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            with open(f'{output_dir}/{image_name.split(".png")[0]}.txt', 'w', encoding='utf-8') as f:
                f.write(response)
            end_time = time.time()
            total_time += (end_time - start_time)
            num += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    if num > 0:
        avg_time = total_time / num
        with open(res_dir+'/overall_average_time.txt', 'w') as f:
            f.write(f"Average time per sample: {avg_time:.4f} seconds\nTotal samples: {num}")