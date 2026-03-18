FROM python:3.11-slim

WORKDIR /app

# Install system deps for rembg/onnxruntime
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the rembg AI model so first request isn't slow
RUN python -c "from rembg import remove; from PIL import Image; import io; remove(Image.new('RGB',(10,10)))" || true

COPY . .

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300
