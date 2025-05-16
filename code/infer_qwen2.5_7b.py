import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
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
def qwen_image2plotly_inference(mode, chart_type,image_path, model, processor, max_new_tokens=4096):
    """
    使用 Qwen2.5-VL 对给定图表图像进行推理，输出可执行的 Plotly Python 代码。
    """
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
    user_message = [
        {"type": "text",  "text": question+". Here is an image of a chart: "},
        {"type": "image", "image": f"file://{image_path}"}
    ]
    messages = [
        {"role": "user",   "content": user_message}
    ]
    # # system 指令：只输出完整可执行的代码，不要解释
    # system_message = (
    #     "You are an AI assistant capable of understanding charts and quickly transforming "
    #     "chart information into Python plotting code. Please deduce or extract the data "
    #     "shown in the chart, accurately restore the chart type, axis labels, ranges, legend, "
    #     "title, color scheme, and other details. Only provide the complete, directly executable "
    #     "Python code for plotting (using Plotly), without any explanation or reasoning process."
    # )

    # # user 内容中插入这张图
    # user_message = [
    #     {"type": "text",  "text": "Here is an image of a chart: "},
    #     {"type": "image", "image": f"file://{image_path}"}
    # ]

    # 拼成多轮对话
    # messages = [
    #     {"role": "system", "content": system_message},
    #     {"role": "user",   "content": user_message}
    # ]

    # 将对话结构转换成 Qwen2.5-VL 可消费的序列，并附加生成标记
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 分离出 image/video 输入
    image_inputs, video_inputs = process_vision_info(messages)

    # 打包成可送入模型的输入
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 推理生成
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,temperature=0.1,top_p=0.95)

    # 截取仅包含新生成的 tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]

    # 解码成文本
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text

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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,  # 不使用自动切分
    ).eval().cuda(local_rank)

    processor = AutoProcessor.from_pretrained(args.model_path)

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
    
    for chart_type, image_path in tqdm(local_data, desc=f"[Rank {local_rank}] Processing"):
        start_time = time.time()
        try:
            result_text = qwen_image2plotly_inference(
                args.mode, chart_type,
                image_path, 
                model, 
                processor,
                max_new_tokens=args.max_new_tokens
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
