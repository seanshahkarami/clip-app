import argparse

from transformers import CLIPProcessor, CLIPModel
from waggle.plugin import Plugin
from waggle.data.vision import Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=0, help="input source")
    parser.add_argument("--threshold", default=0.90, type=float, help="threshold for publishing a detection")
    parser.add_argument("text", nargs="+", help="list of text descriptions to match")
    args = parser.parse_args()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    with Plugin() as plugin, Camera(args.input) as camera:
        for snapshot in camera.stream():
            inputs = processor(text=args.text, images=snapshot.data, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

            for prob, logits, description in sorted(zip(probs.view(-1), logits_per_image.view(-1), args.text)):
                # TODO use similarity score instead of text softmax for thresholding. currently will give unituitive
                # results - for example, when a single text is provided, that will always be published.
                if prob > args.threshold:
                    plugin.publish("image.clip.prediction", f"{description}:{prob:0.3f}:{logits:0.3f}", timestamp=snapshot.timestamp)
                    marker = "*"
                else:
                    marker = " "
                print(f"{prob:0.3f} {logits:0.3f} {marker} {description} ")
            print(flush=True)


if __name__ == "__main__":
    main()
