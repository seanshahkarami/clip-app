import argparse
import logging
import torch
from threading import Thread, Lock
from urllib.request import urlopen
import time
from transformers import CLIPProcessor, CLIPModel
from waggle.plugin import Plugin
from waggle.data.vision import Camera
from collections import defaultdict


class TextPromptWatcher:
    """
    TextPromptWatcher manages a list of text prompts and can watch a remote URL to update the list.
    """

    def __init__(self, text_prompts, poll_url_interval):
        self.text_prompts = text_prompts
        self.lock = Lock()
        self.poll_url_interval = poll_url_interval

    def get_text_prompts(self):
        with self.lock:
            return self.text_prompts

    def watch_url(self, url):
        while True:
            try:
                with urlopen(url) as f:
                    content = f.read()
                text_prompts = content.decode().splitlines()
                with self.lock:
                    self.text_prompts = text_prompts
            except Exception:
                logging.exception("failed to update text prompts")
            time.sleep(self.poll_url_interval)


class RateLimiter:

    def __init__(self, interval):
        self.time = time.monotonic()

    def aquire(self):
        now = time.monotonic()
        if now - self.time < 1.0:
            return False
        self.time = now
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    parser.add_argument("--cpu", action="store_true", help="use cpu instead of accelerator")
    parser.add_argument("--retry-stream", action="store_true", help="always retry the video stream")
    parser.add_argument("--input", default=0, help="input source")
    parser.add_argument("--threshold-type", default="similarity", choices=["similarity", "softmax"], help="which type of value to check threshold on")
    parser.add_argument("--threshold", type=float, help="threshold for publishing a detection")
    parser.add_argument("--watch-text-url", help="url of text file to watch for new prompts")
    parser.add_argument("--watch-text-interval", type=int, default=10, help="interval to poll url of text file")
    parser.add_argument("text_prompts", nargs="+", help="list of text descriptions to match")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S")

    if args.threshold_type == "similarity" and args.threshold is None:
        args.threshold = 28.0
    elif args.threshold_type == "softmax" and args.threshold is None:
        args.threshold = 0.90

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logging.info("using device %s", device)

    logging.info("loading models...")
    processor = CLIPProcessor.from_pretrained("./openai-clip-vit-base-patch32/")
    model = CLIPModel.from_pretrained("./openai-clip-vit-base-patch32/").to(device)
    logging.info("done loading models")

    text_prompt_watcher = TextPromptWatcher(args.text_prompts, args.watch_text_interval)

    if args.watch_text_url is not None:
        Thread(target=text_prompt_watcher.watch_url, args=(args.watch_text_url,), daemon=True).start()

    # NOTE makes no attempt to garbage collect these. assuming number of prompts won't get crazy high.
    rate_limiters = defaultdict(RateLimiter)

    with Plugin() as plugin:
        while True:
            with Camera(args.input) as camera:
                logging.info("processing stream...")
                for snapshot in camera.stream():
                    # get latest cached text prompts
                    text_prompts = text_prompt_watcher.get_text_prompts()

                    logging.info("running inference...")
                    inputs = processor(text=text_prompts, images=snapshot.data, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

                    results = []

                    for prob, logits, description in sorted(zip(probs.view(-1), logits_per_image.view(-1), text_prompts)):
                        # TODO prefer similarity score to softmax prob for thresholding. software prob can give unituitive
                        # results - for example, when a single text is provided, that will always be published.
                        matched = (
                            (args.threshold_type == "similarity" and logits > args.threshold) or
                            (args.threshold_type == "softmax" and prob > args.threshold)
                        )

                        if matched and rate_limiters[description].aquire():
                            plugin.publish("image.clip.prediction", f"{description},{logits:0.3f},{prob:0.3f}", timestamp=snapshot.timestamp)

                        marker = "*" if matched else " "
                        results.append(f"{logits:0.3f} {prob:0.3f} {marker} {description}")

                    logging.info("inference results are\n\n%s\n", "\n".join(results))

            if not args.retry_stream:
                break

            time.sleep(3)


if __name__ == "__main__":
    main()
