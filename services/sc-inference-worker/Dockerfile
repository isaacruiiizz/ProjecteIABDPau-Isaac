FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# PyTorch amb suport CUDA 12.4 (compatible amb runtime CUDA 12.6)
# S'instal·la abans de rfdetr per garantir la versió GPU en lloc de la CPU per defecte
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && python3.11 -m pip install -r requirements.txt

COPY app/ ./app/

CMD ["python3", "-m", "app.main"]
