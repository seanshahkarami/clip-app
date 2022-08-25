FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt

COPY download-models.py .
RUN python3 download-models.py

COPY . .
ENTRYPOINT ["python3", "main.py"]
