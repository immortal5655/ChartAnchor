import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist

# DeepSeek-VL2 相关
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import csv
import traceback                          # ←①
from datetime import timedelta 

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

def deepseek_chart2code_inference(mode,chart_type, image_path, vl_gpt, vl_chat_processor, tokenizer, max_new_tokens=4096):
    if mode == 'code':
        if 'other' in chart_type:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Matplotlib library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."
        else:
            question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Plotly library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."

    if mode == 'table':
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        prompt_path = os.path.join('/root/paddlejob/workspace/env_run/benchmark/prompts',chart_type, base_name + ".txt")
        f = open(prompt_path,'r')
        question = f.read()
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>"+question,
            "images": [image_path],
        },
        {
            "role": "Assistant",
            "content": ""
        },
    ]
  

    # 1) 加载图像得到 PIL 对象
    pil_images = load_pil_images(conversation)

    # 2) 调用 Processor 得到可投喂模型的输入
    #    system_prompt="" 表示不再额外添加系统提示，你也可以按需自定义
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)
    
    # 3) 先跑 image encoder，得到图像embedding
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    # 4) 生成输出
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        use_cache=True,temperature=0.1,top_p=1
    )

    # 5) 解码文本
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../models/deepseek-vl2",
                        help="HuggingFace Hub or local path of the DeepSeek-VL2 model.")
    parser.add_argument("--base_image_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/input_image/human_image",
                        help="Directory containing chart images in subfolders.")
    parser.add_argument("--output_dir", type=str,
                        default="/root/paddlejob/workspace/env_run/benchmark/DeepSeek-VL2-Results",
                        help="Directory to save the generation outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--mode", type=str,help="Maximum number of new tokens to generate.")
    parser.add_argument("--chart_types", type=str,  # 默认支持多个类型，逗号分隔
    help="Comma-separated list of chart types to process. E.g. 'bar,box,violin'"
)
    args = parser.parse_args()

    # ------------------------- 初始化分布式 ------------------------- #
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=2)           # ★ enlarged timeout
    )
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    group_gloo = dist.new_group(backend="gloo")  # ★ gloo group


    if local_rank == 0:
        print("World size:", world_size)
        print("Loading DeepSeek-VL2 model on rank=0 ...")

    # ------------------------- 加载 DeepSeek-VL2 模型与处理器 ------------------------- #
    # 如果显存充足，可以直接用 .cuda(local_rank)，将整份模型都放到该卡
    # 如果显存可能不够，可以使用 device_map="auto" 的方式
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda(local_rank).eval()

    # ------------------------- 收集所有要处理的图像 ------------------------- #
    # chart_types = [
    #     'areachart','bar','box','candlestick','carpet','cone','contour','funnel',
    #     'funnelarea','heatmap','line','line3d','mesh3d','pie','treemap','violin'
    # ]
    chart_types = [
        'barpolar','histogram','histogram2d','histogram2dcontour','ohlc','scatter','scatter3d',
        'scatterpolar','streamtube','surface','waterfall'
    ]
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
        print(f"Found total {len(all_image_paths)} images.", flush=True)

    if not all_image_paths:
        dist.destroy_process_group()
        return
    local_output_dir = os.path.join(args.output_dir, f"rank_{local_rank}")
    os.makedirs(local_output_dir, exist_ok=True)

    # ------------------------- 主循环 ------------------------- #
    local_data = all_image_paths[local_rank::world_size]
    total_time = 0.0
    num_samples = 0
    start_time_global = time.time()
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      do_sample=True,
                      temperature=0.1,
                      top_p=0.95)

    try:
        for idx, (chart_type, image_path) in enumerate(
                tqdm(local_data, desc=f"[Rank {local_rank}] Processing")):
            t0 = time.time()
            try:
                result_text = deepseek_chart2code_inference(
                args.mode,
                chart_type,
                image_path,
                vl_gpt,
                vl_chat_processor,
                tokenizer,
                max_new_tokens=args.max_new_tokens
            )

                # -- 写文件 --
                out_subdir = os.path.join(local_output_dir, chart_type)
                os.makedirs(out_subdir, exist_ok=True)
                base = os.path.splitext(os.path.basename(image_path))[0]
                with open(os.path.join(out_subdir, base + ".txt"),
                          "w", encoding="utf-8") as f:
                    f.write(result_text)

                total_time += time.time() - t0     # ←②
                num_samples += 1                   # ←②
            except Exception as e:
                traceback.print_exc()
                print(f"[Rank {local_rank}] Error {image_path}: {e}", flush=True)
                continue

            # 每 50 张同步一次
            if (idx + 1) % 50 == 0:
                dist.barrier()

    finally:
        # ------------------------- 收尾 ------------------------- #
        if num_samples:
            avg = total_time / num_samples
            print(f"[Rank {local_rank}] Done {num_samples} imgs | "
                  f"avg {avg:.3f}s | total {time.time()-start_time_global:.1f}s",
                  flush=True)

        try:
            dist.barrier(group=group_gloo, timeout=timedelta(minutes=10))
        except Exception:
            pass
        dist.destroy_process_group()               # ←③

if __name__ == "__main__":
    main()