import torch
import gradio as gr
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def similarity(image, text):
    lines = text.splitlines()
    inputs = processor(text=lines, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    similarities = outputs.logits_per_image.view(-1)
    return "\n".join(f"{line}: {similarity}" for line, similarity in zip(lines, similarities))

demo = gr.Interface(fn=similarity, inputs=["image", "text"], outputs="text")
demo.launch()
