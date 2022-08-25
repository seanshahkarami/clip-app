import argparse
import torch

from transformers import CLIPProcessor, CLIPModel
from waggle.plugin import Plugin
from waggle.data.vision import Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=0, help="input source")
    parser.add_argument("--threshold-type", default="similarity", choices=["similarity", "softmax"], help="which type of value to check threshold on")
    parser.add_argument("--threshold", type=float, help="threshold for publishing a detection")
    parser.add_argument("text", nargs="+", help="list of text descriptions to match")
    args = parser.parse_args()

    if args.threshold_type == "similarity" and args.threshold is None:
        args.threshold = 28.0
    elif args.threshold_type == "softmax" and args.threshold is None:
        args.threshold = 0.90

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    with Plugin() as plugin, Camera(args.input) as camera:
        for snapshot in camera.stream():
            inputs = processor(text=args.text, images=snapshot.data, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

            for prob, logits, description in sorted(zip(probs.view(-1), logits_per_image.view(-1), args.text)):
                # TODO prefer similarity score to softmax prob for thresholding. software prob can give unituitive
                # results - for example, when a single text is provided, that will always be published.
                matched = (
                    (args.threshold_type == "similarity" and logits > args.threshold) or
                    (args.threshold_type == "softmax" and prob > args.threshold)
                )

                if matched:
                    plugin.publish("image.clip.prediction", f"{description},{logits:0.3f},{prob:0.3f}", timestamp=snapshot.timestamp)
                
                marker = "*" if matched else " "
                print(f"{logits:0.3f} {prob:0.3f} {marker} {description} ")
            print(flush=True)


if __name__ == "__main__":
    main()
