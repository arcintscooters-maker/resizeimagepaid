FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BiRefNet model
RUN python -c "from rembg import remove, new_session; from PIL import Image; s=new_session('birefnet-general'); remove(Image.new('RGB',(10,10)), session=s)" || true

COPY . .

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 8 --timeout 300 --preload
