# CLIP App

This app perform zero-shot image classification using [OpenAI's CLIP](https://huggingface.co/docs/transformers/model_doc/clip) hosted by [Hugging Face](https://huggingface.co).

## Usage

```sh
# install dependencies
pip3 install --upgrade -r requirements.txt

# run app with some sample text
python3 main.py "a washing machine" "a person drinking water" "a red car" "a busy intersection"
```

This will open a camera stream and print the softmax and similarity scores of each text desciption for each frame.

```txt
0.022 18.801   a busy intersection 
0.036 19.287   a red car 
0.119 20.481   a washing machine 
0.823 22.417   a person drinking water 

0.023 18.912   a busy intersection 
0.034 19.280   a red car 
0.129 20.616   a washing machine 
0.814 22.457   a person drinking water 

0.028 19.043   a red car 
0.030 19.107   a busy intersection 
0.099 20.299   a washing machine 
0.842 22.434   a person drinking water 
...
```
