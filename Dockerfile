FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py haarcascade_frontalface_default.xml ./
COPY templates ./templates
RUN mkdir -p /data && chmod 777 /data
EXPOSE 7860
# single worker: model + files are shared in-process; threads handle concurrent frames
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120
