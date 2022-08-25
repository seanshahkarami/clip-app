import argparse

from transformers import CLIPProcessor, CLIPModel
from waggle.data.vision import Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=0, help="input source")
    parser.add_argument("--threshold", default=0.90, type=float, help="threshold for publishing a detection")
    parser.add_argument("text", nargs="+", help="list of text descriptions to match")
    args = parser.parse_args()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    with Camera(args.input) as camera:
        for sample in camera.stream():
            inputs = processor(text=args.text, images=sample.data, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

            for prob, logits, description in sorted(zip(probs.view(-1), logits_per_image.view(-1), args.text)):
                # TODO use similarity score instead of softmax for thresholding
                marker = "*" if prob > args.threshold else " "
                print(f"{prob:0.3f} {logits:0.3f} {marker} {description} ")
            print(flush=True)


if __name__ == "__main__":
    main()
