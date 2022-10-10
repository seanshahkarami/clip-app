FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt && \
    pip3 install --force-reinstall opencv-python-headless
#   ^ this is a hack to replace the "gui" version of opencv pulled in as a dep with headless again

COPY download-models.py .
RUN python3 download-models.py

COPY . .
ENTRYPOINT ["python3", "main.py"]
