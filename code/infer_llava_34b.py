

from PIL import Image
import requests
from vllm import LLM, SamplingParams
import math
from tqdm import tqdm  # 用于显示进度条
import os
import argparse
import time
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


# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def inference(mode, chart_type,image_path,llm,sampling_params):
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
    #------------------------------------------
    prompt = '[User] <image>\n '+question + ' Here is an image of a chart, Please output table:[Assistant:]'
    image = Image.open(image_path)
    outputs = llm.generate([{
    "prompt": prompt,
    "multi_modal_data": {"image": image}}],sampling_params=sampling_params
    )
    return outputs[0].outputs[0].text
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/llava-v1.6-34b-hf",
                        help="Path to the Qwen2.5-VL model.")
    parser.add_argument("--base_image_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/input_image_table/human_image",
                        help="Directory containing chart images in subfolders.")
    parser.add_argument("--output_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/res-table/llava-v1.6-34b-hf_Results",
                        help="Directory to save the generation outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--mode", type=str, default='table',
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--chart_types", type=str,  default="areachart,bar,barpolar,box,candlestick,carpet,cone,contour,funnel,funnelarea,heatmap,histogram,histogram2d,histogram2dcontour,line,line3d,mesh3d,ohlc,parcoords,pie,sankey,scatter,scatter3d,scatterpolar,scatterternary,sunburst,surface,treemap,violin,waterfall,areachart_other,scatter_other,bar_other,line_other",  help="Comma-separated list of chart types to process. E.g. 'bar,box,violin'"
)
# "areachart,bar,barpolar,box,candlestick,carpet,cone,contour,funnel,funnelarea,heatmap,histogram,histogram2d,histogram2dcontour,line,line3d,mesh3d,ohlc,parcoords,pie,sankey,scatter,scatter3d,scatterpolar,scatterternary,sunburst,surface,treemap,violin,waterfall"
    args = parser.parse_args()
        # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
       
        max_tokens=args.max_new_tokens,
        stop_token_ids=[],
    )


    llm = LLM(model=args.model_path,tensor_parallel_size=4,  dtype="bfloat16",trust_remote_code=True,gpu_memory_utilization=0.8, max_model_len=4096,limit_mm_per_prompt={"image": 1, "video": 0})
    chart_types = [x.strip() for x in args.chart_types.split(',') if x.strip()]
    all_image_paths = []
    num = 0
    total_time = 0
    ll = extract_png_from_csv('/root/paddlejob/workspace/env_run/benchmark/redo.csv')
    for chart_type in chart_types:
        chart_dir = os.path.join(args.base_image_dir, chart_type)
        if not os.path.exists(chart_dir):
            continue
        for fname in os.listdir(chart_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_image_paths.append((chart_type, os.path.join(chart_dir, fname)))
    print(len(all_image_paths))
    start_time = time.time()
    for chart, image_path in tqdm(all_image_paths, desc="Processing all images"):
        try:
            image_name = os.path.basename(image_path)
            output_dir = os.path.join(args.output_dir, chart)
            os.makedirs(output_dir, exist_ok=True)
            generated_text = inference(args.mode, chart,image_path,llm,sampling_params)
            
            with open(f'{output_dir}/{image_name.split(".png")[0]}.txt', 'w', encoding='utf-8') as f:
                f.write(generated_text)
            num += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    end_time = time.time()
    total_time = (end_time - start_time)
    if num > 0:
        avg_time = total_time / num
        with open(args.output_dir+'/overall_average_time.txt', 'w') as f:
            f.write(f"Average time per sample: {avg_time:.4f} seconds\nTotal samples: {num}")


if __name__ == "__main__":
    main()


