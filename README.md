# CLIP App

This app perform zero-shot image classification using [OpenAI's CLIP](https://huggingface.co/docs/transformers/model_doc/clip) hosted by [Hugging Face](https://huggingface.co).

## Usage

```sh
# install dependencies
pip3 install --upgrade -r requirements.txt

# run app with some sample text
python3 main.py \
    "a person drinking coffee" \
    "a person making a call" \
    "a person jogging" \
    "a construction crew fixing the road" \
    "a red sports car" \
    "a busy intersection"
```

This will open a camera stream and print the similarity and softmax scores of each text desciption for each frame.

```txt
12.499 0.000   a red sports car 
16.413 0.001   a busy intersection 
17.943 0.006   a construction crew fixing the road 
20.251 0.065   a person jogging 
21.546 0.237   a person making a call 
22.612 0.690   a person drinking coffee 

12.850 0.000   a red sports car 
16.526 0.002   a busy intersection 
17.970 0.007   a construction crew fixing the road 
20.424 0.076   a person jogging 
21.518 0.226   a person making a call 
22.633 0.690   a person drinking coffee 
...
```

_Note: I'm still working out the reasonable range for similarity scores. They do not seem to be [-1, 1] or [-100, 100] as I expected._
