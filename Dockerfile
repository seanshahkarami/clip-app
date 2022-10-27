# FROM python:3.10 AS builder
# RUN pip3 install --no-cache-dir transformers==4.18.0 pillow torch
# COPY download-models.py .
# RUN python3 download-models.py
# # ^ this is a hack to download model cache while avoiding missing libcurand exception on arm64

FROM ecpe4s/waggle-ml
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir --force-reinstall opencv-python-headless
#   ^ this is a hack to replace the "gui" version of opencv pulled in as a dep with headless again

COPY download-models.py .
RUN python3 download-models.py

# COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY main.py .
ENTRYPOINT ["python3", "main.py"]
