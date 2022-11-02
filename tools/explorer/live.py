import torch
import gradio as gr
from transformers import CLIPProcessor, CLIPModel
import shlex

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def similarity(image, text, order):
    lines = text.splitlines()
    if len(lines) == 0:
        return "", ""
    inputs = processor(text=lines, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    similarities = outputs.logits_per_image.view(-1)

    if order:
        tfm = lambda xs: sorted(xs, reverse=True)
    else:
        tfm = lambda xs: xs

    # TODO add better indication of detection

    clicmd = shlex.join([
        "pluginctl",
        "run",
        "--name", "clip-app",
        "waggle/clip-app:0.9.1",
        "--",
        "--input=bottom",
        *lines])

    return "\n".join(f"{line}: {similarity}" for similarity, line in tfm(zip(similarities, lines))), clicmd


demo = gr.Interface(
    title="CLIP Explorer",
    description="Input an image and lines of text then press submit to output the image-text similarity scores.",
    fn=similarity,
    inputs=[gr.Image(label="Webcam", source="webcam", streaming=True), gr.TextArea(label="Text descriptions"), gr.Checkbox(value=True, label="Order by similarity score?")],
    outputs=[gr.TextArea(label="Image-text similarity scores"), gr.TextArea(label="Pluginctl command")],
    live=True,
)

if __name__ == "__main__":
    demo.launch()
