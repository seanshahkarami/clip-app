from transformers import CLIPProcessor, CLIPModel
from shutil import make_archive

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

processor.save_pretrained("./openai-clip-vit-base-patch32")
model.save_pretrained("./openai-clip-vit-base-patch32")

make_archive(
    base_name="openai-clip-vit-base-patch32",
    format="tar",
    root_dir=".",
    base_dir="openai-clip-vit-base-patch32",
)
