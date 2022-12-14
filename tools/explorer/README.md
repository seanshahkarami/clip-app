# CLIP Explorer

This is a [Gradio](https://gradio.app) app for quickly trying our similarity scores for image-text pairs. Simply run the app and open your browser:

```sh
# install dependencies
pip3 install --upgrade -r requirements.txt

# run app
gradio main.py

# you'll be prompted to open your browser in just a moment
```

Now, you can provide an image and multiple lines of text to compute a similarity score for!

![Screenshot from app](./app.png)

If you'd like to run this live using your webcam, simply run the `live.py` app instead:

```sh
gradio live.py
```
