import torch
import gradio as gr
from transformers import CLIPProcessor, CLIPModel
import shlex

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def similarity(image, text, threshold, order):
    lines = text.splitlines()
    if len(lines) == 0:
        return "", ""
    inputs = processor(text=lines, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    similarities = outputs.logits_per_image.view(-1)

    # convert to plain list of floats for display
    similarities = [s.item() for s in similarities]

    if order:
        tfm = lambda xs: sorted(xs, reverse=True)
    else:
        tfm = lambda xs: xs

    detections = [(f"{line}: {similarity:0.2f}", "yes" if similarity > threshold else "no") for similarity, line in tfm(zip(similarities, lines))]

    # TODO add better indication of detection
    clicmd = shlex.join([
        "pluginctl",
        "run",
        "--name", "clip-app",
        "waggle/clip-app:0.9.1",
        "--",
        "--input=bottom",
        f"--threshold={threshold}",
        *lines,
    ])

    return detections, clicmd


demo = gr.Interface(
    title="CLIP Explorer",
    description="Input an image and lines of text then press submit to output the image-text similarity scores.",
    fn=similarity,
    inputs=[
        gr.Image(label="Webcam", source="webcam", streaming=True),
        gr.TextArea(label="Text descriptions"),
        gr.Slider(0, 40, 26, label="Similarity threshold"),
        gr.Checkbox(value=True, label="Order by similarity score?"),
    ],
    outputs=[
        gr.HighlightedText(label="Image-text similarity scores", color_map={
            "yes": "green",
            "no": "red",
        }),
        gr.TextArea(label="Pluginctl command"),
    ],
    # outputs=[gr.TextArea(label="Image-text similarity scores"), gr.TextArea(label="Pluginctl command")],
    live=True,
)

if __name__ == "__main__":
    demo.launch()
