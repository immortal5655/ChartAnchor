# """
# This is a demo for using CogVLM2 in CLI using multi-GPU with lower memory.
# If your single GPU is not enough to drive this model, you can use this demo to run this model on multiple graphics cards with limited video memory.
# Here, we default that your graphics card has 24GB of video memory, which is not enough to load the FP16 / BF16 model.
# so , need to use two graphics cards to load. We set '23GiB' for each GPU to avoid out of memory.
# GPUs less than 2 is recommended and need more than 16GB of video memory.

# test success in 3 GPUs with 16GB video memory.
# +---------------------------------------------------------------------------------------+
# | Processes:                                                                            |
# |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
# |        ID   ID                                                             Usage      |
# |=======================================================================================|
# |    1   N/A  N/A   1890574      C   python                                    13066MiB |
# |    2   N/A  N/A   1890574      C   python                                    14560MiB |
# |    3   N/A  N/A   1890574      C   python                                    11164MiB |
# +---------------------------------------------------------------------------------------+
# """
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# import torch
# from PIL import Image
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

# MODEL_PATH = "models/cogvlm2-llama3-chat-19B"
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
#     0] >= 8 else torch.float16

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_PATH,
#     trust_remote_code=True
# )
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=TORCH_TYPE,
#         trust_remote_code=True,
#     )

# num_gpus = torch.cuda.device_count()
# print(num_gpus)
# max_memory_per_gpu = "28GiB"
# if num_gpus > 2:
#     max_memory_per_gpu = f"{round(42 / num_gpus)}GiB"

# device_map = infer_auto_device_map(
#     model=model,
#     max_memory={i: max_memory_per_gpu for i in range(num_gpus)},
#     no_split_module_classes=["CogVLMDecoderLayer","TransformerLayer"]
# )
# model = load_checkpoint_and_dispatch(model, MODEL_PATH, device_map=device_map, dtype=TORCH_TYPE)
# model = model.eval()

# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"


# image_path = "/root/paddlejob/workspace/env_run/benchmark/input_image_table/human_image/bar/bugwheels94:7.png"
# if image_path == '':
#     print('You did not enter image path, the following will be a plain text conversation.')
#     image = None
#     text_only_first_query = True
# else:
#     image = Image.open(image_path).convert('RGB')

# history = []


# query = "You are a vision-capable data analyst AI with expert-level ability to interpret and estimate numerical data from visual charts. Based on the provided chart image, extract the original data as accurately and completely as possible by visually estimating values using axis scales, grid lines, and chart elements. Focus specifically on extracting data for the following column names: ['MongoDB-x; Mongoose-x', 'Mongoose-y', 'MongoDB-y']. Present the extracted data in a structured, column-wise format enclosed within <Table Begin> and <Table End>. For each column, use the following structure: <Column Begin> {Column Name} <Column Value Begin> value1 | value2 | value3 | ... <Column Value End> <Column End>. Do not skip or refuse to estimate values — always output a complete table, even if some values are approximate. For example, the output should look like: <Table Begin> <Column Begin> Year <Column Value Begin> 2020 | 2021 | 2022 <Column Value End> <Column End> <Column Begin> Sales <Column Value Begin> 1000 | 1500 | 1800 <Column Value End> <Column End> <Table End>. Now extract and output the full table based on the chart using the specified column names only."


# if image is None:
#     if text_only_first_query:
#         query = text_only_template.format(query)
#         text_only_first_query = False
#     else:
#         old_prompt = ''
#         for _, (old_query, response) in enumerate(history):
#             old_prompt += old_query + " " + response + "\n"
#         query = old_prompt + "USER: {} ASSISTANT:".format(query)
# if image is None:
#     input_by_model = model.build_conversation_input_ids(
#         tokenizer,
#         query=query,
#         history=history,
#         template_version='chat'
#     )
# else:
#     input_by_model = model.build_conversation_input_ids(
#         tokenizer,
#         query=query,
#         history=history,
#         images=[image],
#         template_version='chat'
#     )
# inputs = {
#     'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
#     'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
#     'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
#     'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
# }

# gen_kwargs = {
#     "max_new_tokens": 2048,
#     "pad_token_id": 128002,
#     "top_p": 0.95,
#     "temperature":0.1,
#     "do_sample":True
# }
# with torch.no_grad():
#     outputs = model.generate(**inputs, **gen_kwargs)
#     print(outputs.cpu().tolist())     
#     outputs = outputs[:, inputs['input_ids'].shape[1]:]
#     response = tokenizer.decode(outputs[0])
    
#     print("\nCogVLM2:", response)

import os
from lmdeploy.vl import load_image
# os.environ["LMDEPLOY_USE_PAGED_ATTENTION"] = "0"   # 强制走普通 attention
# os.environ["LMDEPLOY_CUDA_GRAPH"] = "0" 

from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig,ChatTemplateConfig
# cfg = PytorchEngineConfig(tp=4, eager_mode=True)

# pipe = pipeline('models/cogvlm2-llama3-chat-19B', backend_config=cfg)
backend_config = PytorchEngineConfig(session_len=4096,dtype='bfloat16',tp=4, eager_mode=True)
gen_config = GenerationConfig(top_p=0.95,
                              temperature=0.1,
                              max_new_tokens=2048,
                              do_sample=True,
                              )
pipe = pipeline('models/cogvlm2-llama3-chat-19B',
                backend_config=backend_config, log_level='INFO',chat_template_config=ChatTemplateConfig(model_name='cogvlm'))
prompts = "You are a vision-capable data analyst AI with expert-level ability to interpret and estimate numerical data from visual charts. Based on the provided chart image, extract the original data as accurately and completely as possible by visually estimating values using axis scales, grid lines, and chart elements. Focus specifically on extracting data for the following column names: ['MongoDB-x; Mongoose-x', 'Mongoose-y', 'MongoDB-y']. Present the extracted data in a structured, column-wise format enclosed within <Table Begin> and <Table End>. For each column, use the following structure: <Column Begin> Column Name <Column Value Begin> value1 | value2 | value3 | ... <Column Value End> <Column End>. Do not skip or refuse to estimate values — always output a complete table, even if some values are approximate.Now extract and output the full table based on the chart using the specified column names only."
image = load_image("/root/paddlejob/workspace/env_run/benchmark/input_image_table/human_image/bar/bugwheels94:7.png")

response = pipe((prompts,[image]), gen_config=gen_config)
print(response)

# from lmdeploy import pipeline, PytorchEngineConfig   # 再导入
# from lmdeploy.vl import load_image
# import math
# import numpy as np
# from PIL import Image
# import os
# import json
# import time
# from tqdm import tqdm 
# import argparse

# from lmdeploy import pipeline
# from lmdeploy.vl import load_image


# if __name__ == "__main__":
#     pipe = pipeline('models/cogvlm2-llama3-chat-19B')

#     image = load_image("/root/paddlejob/workspace/env_run/benchmark/input_image_table/human_image/areachart/AhmedEle:47.png")
#     q = "You are a vision-capable data analyst AI with expert-level ability to interpret and estimate numerical data from visual charts. Based on the provided chart image, extract the original data as accurately and completely as possible by visually estimating values using axis scales, grid lines, and chart elements. Focus specifically on extracting data for the following column names: ['Time(Seconds)-x', 'Time(Seconds)-y']. Present the extracted data in a structured, column-wise format enclosed within <Table Begin> and <Table End>. For each column, use the following structure: <Column Begin> {Column Name} <Column Value Begin> value1 | value2 | value3 | ... <Column Value End> <Column End>. Do not skip or refuse to estimate values — always output a complete table, even if some values are approximate. For example, the output should look like: <Table Begin> <Column Begin> Year <Column Value Begin> 2020 | 2021 | 2022 <Column Value End> <Column End> <Column Begin> Sales <Column Value Begin> 1000 | 1500 | 1800 <Column Value End> <Column End> <Table End>. Now extract and output the full table based on the chart using the specified column names only"
#     response = pipe((q, image) )
#     # response = pipe(("describe this image", image) )
#     print(response)


# def inference(mode, chart_type,image_path,pipe):
#     if mode == 'code':
#         question = "You are an AI assistant capable of understanding charts and quickly transforming chart information into Python plotting code. I have an image of a chart, and I would like you to use the Plotly library to recreate it. Please deduce or extract the data shown in the chart, accurately restore the chart type, axis labels, ranges, legend, title, color scheme, and other details. Only provide the complete, directly executable Python code for plotting, without any explanation or reasoning process."
#     if mode == 'table':
#         base_name = os.path.splitext(os.path.basename(image_path))[0]
#         print(base_name)
#         prompt_path = os.path.join('/root/paddlejob/workspace/env_run/benchmark/prompts',chart_type, base_name + ".txt")
#         f = open(prompt_path,'r')
#         question = f.readline()
#     image = load_image(image_path)
#     response = pipe((question, image))
#     print(question)
#     print(response)
           
#     return response

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, default="models/cogvlm2-llama3-chat-19B",
#                         help="Path to the Qwen2.5-VL model.")
#     parser.add_argument("--base_image_dir", type=str,
#                         default="/root/paddlejob/workspace/env_run/benchmark/input_image_table/human_image",
#                         help="Directory containing chart images in subfolders.")
#     parser.add_argument("--output_dir", type=str,
#                         default="/root/paddlejob/workspace/env_run/benchmark/res-table/cogvlm2-llama3-chat-19B_Results",
#                         help="Directory to save the generation outputs.")
#     parser.add_argument("--max_new_tokens", type=int, default=4096,
#                         help="Maximum number of new tokens to generate.")
#     parser.add_argument("--mode", type=str, default='table',
#                         help="Maximum number of new tokens to generate.")
#     parser.add_argument("--chart_types", type=str,  default="areachart",  help="Comma-separated list of chart types to process. E.g. 'bar,box,violin'"
# )
#     args = parser.parse_args()
   
#     cfg = PytorchEngineConfig(tp=4, eager_mode=True)

#     pipe = pipeline('models/cogvlm2-llama3-chat-19B', backend_config=cfg)

#     chart_types = [x.strip() for x in args.chart_types.split(',') if x.strip()]
#     all_image_paths = []
#     num = 0
#     total_time = 0
    
#     for chart_type in chart_types:
#         chart_dir = os.path.join(args.base_image_dir, chart_type)
#         if not os.path.exists(chart_dir):
#             continue
#         for fname in os.listdir(chart_dir):
#             if fname.lower().endswith((".png", ".jpg", ".jpeg")):
#                 all_image_paths.append((chart_type, os.path.join(chart_dir, fname)))
#     print(len(all_image_paths))
#     all_image_paths = all_image_paths[:10]
#     start_time = time.time()
    
#     for chart, image_path in tqdm(all_image_paths, desc="Processing all images"):
#         try:
#             image_name = os.path.basename(image_path)
#             output_dir = os.path.join(args.output_dir, chart)
#             os.makedirs(output_dir, exist_ok=True)
#             generated_text = inference(args.mode, chart,image_path,pipe)
#             with open(f'{output_dir}/{image_name.split(".png")[0]}.txt', 'w', encoding='utf-8') as f:
#                 f.write(generated_text.text)
#             num += 1
#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")
#             continue
#     end_time = time.time()
#     pipe.close()
#     total_time = (end_time - start_time)
#     if num > 0:
#         avg_time = total_time / num
#         with open(args.output_dir+'/overall_average_time.txt', 'w') as f:
#             f.write(f"Average time per sample: {avg_time:.4f} seconds\nTotal samples: {num}")
# if __name__ == "__main__":
    
#     main()
       # 同步关闭线程池 / event-loop
