import os
import torch
from PIL import Image
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel
from tqdm import tqdm

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(20)
print(f"✅ Using device: {device}, threads: {torch.get_num_threads()}")

# 初始化模型和处理器
model = CLIPModel.from_pretrained("models/clip-vit-base-patch16").to(device)
processor = AutoProcessor.from_pretrained("models/clip-vit-base-patch16")

# 路径设置
gt_root = Path("/home/disk1/lixinhang/code/benchmark/human_image_final")
model_root = Path("/home/disk1/lixinhang/code/benchmark/gen/res-code-merge-image")
model_names = ["claude_code"]  # 可替换为多个模型名

# 批量计算相似度，支持 tqdm
def compute_clip_score_batch(image_pairs, batch_size=8, desc=""):
    scores = []
    loop = tqdm(
        range(0, len(image_pairs), batch_size),
        desc=f"[{desc}] Batches",
        leave=False,
        ncols=80
    )

    for i in loop:
        batch = image_pairs[i:i + batch_size]
        imgs1, imgs2 = [], []

        for img1_path, img2_path in batch:
            try:
                imgs1.append(Image.open(img1_path).convert("RGB"))
                imgs2.append(Image.open(img2_path).convert("RGB"))
            except Exception as e:
                print(f"❌ Error loading: {img1_path}, {img2_path}, {e}")
                scores.append(0.0)
                continue

        try:
            with torch.no_grad():
                inputs1 = processor(images=imgs1, return_tensors="pt", padding=True).to(device)
                inputs2 = processor(images=imgs2, return_tensors="pt", padding=True).to(device)
                feat1 = F.normalize(model.get_image_features(**inputs1), dim=-1)
                feat2 = F.normalize(model.get_image_features(**inputs2), dim=-1)
                sim = F.cosine_similarity(feat1, feat2)
                scores.extend(((sim + 1) / 2).tolist())
        except Exception as e:
            print(f"❌ Batch error: {e}")
            scores.extend([0.0] * len(batch))

    return scores

# 主评估流程
for model_name in tqdm(model_names, desc="Processing Models"):
    model_dir = model_root / model_name
    rows = []

    chart_types = sorted([p for p in gt_root.iterdir() if p.is_dir()])
    for chart_type_dir in tqdm(chart_types, desc=f"[{model_name}] Chart Types", ncols=80):
        chart_type = chart_type_dir.name
        gt_dir = chart_type_dir
        pred_dir = model_dir / chart_type

        # 收集图像对（两者都存在才计入）
        image_pairs = [
            (gt_img, pred_dir / gt_img.name)
            for gt_img in gt_dir.glob("*.png")
            if (pred_dir / gt_img.name).exists()
        ]

        if not image_pairs:
            print(f"⚠️ No matched images for chart type: {chart_type}")
            rows.append([chart_type, 0, 0.0])
            continue

        # 计算 CLIP 分数
        scores = compute_clip_score_batch(image_pairs, batch_size=8, desc=chart_type)
        rows.append([chart_type, len(scores), sum(scores)])

    # 保存为 CSV
    df = pd.DataFrame(rows, columns=["chart_type", "gt_image_count", "total_clip_score"])
    output_path = f"/home/disk1/lixinhang/code/benchmark/res_metrics/res_image_metrics/clip_score_{model_name}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved results to: {output_path}")
