from transformers import CLIPProcessor, CLIPModel
import torch

# automatically downloads model data
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# do single pass on dummy data
inputs = processor(text=["dummy"], images=torch.zeros((3, 128, 128), dtype=torch.uint8), return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
