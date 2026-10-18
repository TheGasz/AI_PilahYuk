FROM python:3.11-slim

WORKDIR /app

# Dependency sistem minimal untuk Pillow/torchvision
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (Azure Container Apps tanpa GPU). Index CPU -> image jauh lebih kecil
# & startup lebih cepat daripada build CUDA default. app.py otomatis fallback ke CPU.
# Versi di-pin agar build reprodusibel & menghindari perubahan API transformers
# (mis. get_text_features/output model) yang pernah memecah kode.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchvision==0.19.1

RUN pip install --no-cache-dir \
    "transformers==4.44.2" \
    "pillow>=10,<12" \
    "fastapi==0.115.*" "uvicorn[standard]==0.30.*" \
    python-multipart "numpy<2"

# Pre-download checkpoint SigLIP ke cache image -> startup instan, tak perlu unduh saat runtime
ENV HF_HOME=/app/hf_cache
RUN python -c "from transformers import SiglipVisionModel, SiglipImageProcessor; \
    SiglipImageProcessor.from_pretrained('google/siglip-so400m-patch14-384'); \
    SiglipVisionModel.from_pretrained('google/siglip-so400m-patch14-384')"

# Salin artefak head (.pt) dan kode aplikasi
COPY ./artifacts /app/artifacts
COPY ./app.py /app/app.py

EXPOSE 8000

# 1 worker per replika; scaling horizontal oleh Azure Container Apps.
# Backbone SigLIP besar -> hindari banyak worker dalam satu container (boros RAM).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
