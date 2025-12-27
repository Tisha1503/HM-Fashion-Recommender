import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import polars as pl
import os
import numpy as np

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))

preprocess = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMAGE_ROOT = "images" 

train = pl.read_parquet("train_split.parquet")
top_items = (
    train["article_id"]
    .value_counts()
    .sort("count", descending=True)
    .head(500)["article_id"]
    .to_list()
)

visual_data = []

print(f"Starting extraction for 500 items...")

for aid in top_items:
    clean_id = str(aid).zfill(10) 
    
    subfolder = clean_id[:3]
    path = os.path.join(IMAGE_ROOT, subfolder, f"{clean_id}.jpg")
    
    if os.path.exists(path):
        try:
            img = Image.open(path).convert('RGB')
            t_img = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = feature_extractor(t_img)
            vector = output.flatten().numpy()
            visual_data.append({"article_id": clean_id, "feature_vector": vector})
        except:
            continue
    
    if len(visual_data) % 50 == 0 and len(visual_data) > 0:
        print(f"Processed {len(visual_data)} images...")

if len(visual_data) > 0:
    np.save("visual_features.npy", visual_data)
    print(f"\nSUCCESS: Extracted {len(visual_data)} visual fingerprints!")
else:
    print("\nStill not finding images. Check if your IDs in the folder start with 0.")