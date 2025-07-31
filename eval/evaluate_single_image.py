import os
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)
print(f"Using device: {device}, threads: {torch.get_num_threads()}")

# Initialize model and processor
model_path = "models/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_path).to(device)
processor = AutoProcessor.from_pretrained(model_path)

# ========== Set image paths ==========
gt_image_path = "/path/to/gt.png"         # Ground Truth image
pred_image_path = "/path/to/pred.png"     # Predicted image

# ========== Load images and compute similarity ==========
def compute_clip_score(gt_path: str, pred_path: str) -> float:
    try:
        img1 = Image.open(gt_path).convert("RGB")
        img2 = Image.open(pred_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return 0.0

    try:
        with torch.no_grad():
            inputs = processor(images=[img1, img2], return_tensors="pt", padding=True).to(device)
            features = model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1)
            score = F.cosine_similarity(features[0], features[1], dim=0)
            score = ((score + 1) / 2).item()  # Map to [0, 1]
            return score
    except Exception as e:
        print(f"Failed to compute similarity: {e}")
        return 0.0

# ========== Run evaluation ==========
if __name__ == "__main__":
    score = compute_clip_score(gt_image_path, pred_image_path)
    print(f"\nCLIP image similarity score: {score:.4f}")
