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
ENV PYTHONUNBUFFERED=1
# Limit glibc malloc arenas so freed memory is actually returned to the OS
# (fewer per-thread arenas holding onto freed inference buffers)
ENV MALLOC_ARENA_MAX=2

# Pre-download models at build time so first request is fast
RUN python -c "from rembg import remove, new_session; from PIL import Image; s=new_session('birefnet-general-lite'); remove(Image.new('RGB',(10,10)), session=s); print('BiRefNet-lite ready')" || true

COPY . .

EXPOSE 8080

# Single worker: heavy ONNX inference is serialized in-process by a semaphore,
# so 1 worker guarantees at most one multi-GB inference at a time (2 workers
# could double it). 4 threads keep light requests (page loads, plain resizes)
# concurrent while an inference runs. --max-requests recycles the worker
# periodically as a final backstop that returns all native memory to the OS.
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300 --max-requests 50 --max-requests-jitter 15 --preload
