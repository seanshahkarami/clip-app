import argparse
import logging
import torch
from transformers import CLIPProcessor, CLIPModel
from waggle.plugin import Plugin
from waggle.data.vision import Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    parser.add_argument("--input", default=0, help="input source")
    parser.add_argument("--threshold-type", default="similarity", choices=["similarity", "softmax"], help="which type of value to check threshold on")
    parser.add_argument("--threshold", type=float, help="threshold for publishing a detection")
    parser.add_argument("text", nargs="+", help="list of text descriptions to match")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S")

    if args.threshold_type == "similarity" and args.threshold is None:
        args.threshold = 28.0
    elif args.threshold_type == "softmax" and args.threshold is None:
        args.threshold = 0.90

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("using device: %s", device)

    logging.info("loading models...")
    processor = CLIPProcessor.from_pretrained("./openai-clip-vit-base-patch32/")
    model = CLIPModel.from_pretrained("./openai-clip-vit-base-patch32/").to(device)
    logging.info("done loading models!")

    with Plugin() as plugin, Camera(args.input) as camera:
        logging.info("processing stream...")
        for snapshot in camera.stream():
            inputs = processor(text=args.text, images=snapshot.data, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

            results = []

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
                results.append(f"{logits:0.3f} {prob:0.3f} {marker} {description}")

            logging.info("inference results:\n%s\n", "\n".join(results))


if __name__ == "__main__":
    main()
