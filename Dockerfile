# 1. Base Image: Hafif resmi Python 3.11 Linux (~150 MB)
FROM python:3.11-slim

WORKDIR /app

# 2. Blackwell mimarisi (sm_120 / RTX 50 serisi) destekli resmi PyTorch CUDA 12.8 kurulumu
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# 3. Web sunucusu kütüphaneleri
RUN pip install --no-cache-dir fastapi uvicorn pydantic numpy

# 4. Sunucu dosyalarını kopyala
COPY server/ /app/

# 5. Portu aç
EXPOSE 8000

# 6. Sunucuyu başlat
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
