FROM ecpe4s/waggle-ml

WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir --force-reinstall opencv-python-headless
#   ^ this is a hack to replace the "gui" version of opencv pulled in as a dep with headless again

# download and unpack model from storage
RUN wget https://web.lcrc.anl.gov/public/waggle/models/seanshahkarami/openai-clip-vit-base-patch32.tar && \
    tar xf openai-clip-vit-base-patch32.tar && \
    rm openai-clip-vit-base-patch32.tar
#   ^ this is a hack to get around missing libcurand at build time on some platforms

COPY main.py .
ENTRYPOINT ["python3", "main.py"]
