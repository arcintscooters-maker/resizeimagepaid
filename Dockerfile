FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set ONNX threads for optimal CPU inference
ENV ONNXRUNTIME_NUM_THREADS=4
ENV OMP_NUM_THREADS=4

# Pre-download models at build time so first request is fast
RUN python -c "from rembg import remove, new_session; from PIL import Image; s=new_session('birefnet-general-lite'); remove(Image.new('RGB',(10,10)), session=s); print('BiRefNet-lite ready')" || true
RUN python -c "from rembg import remove, new_session; from PIL import Image; s=new_session('u2net'); remove(Image.new('RGB',(10,10)), session=s); print('u2net ready')" || true

COPY . .

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 300 --preload
