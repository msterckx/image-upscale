FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    curl \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA 12.4 / Python 3.10 stack, matching the compatibility approach used
# for the Voice Studio worker.
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install BasicSR from source after patching its setup.py get_version().
RUN set -eux; \
    TMPDIR="$(mktemp -d)"; \
    curl -fsSL \
      "https://files.pythonhosted.org/packages/source/b/basicsr/basicsr-1.4.2.tar.gz" \
      -o "${TMPDIR}/basicsr.tar.gz"; \
    mkdir -p "${TMPDIR}/basicsr"; \
    tar xzf "${TMPDIR}/basicsr.tar.gz" -C "${TMPDIR}/basicsr" --strip-components=1; \
    python -c "from pathlib import Path; p=Path('${TMPDIR}/basicsr/setup.py'); s=p.read_text(); old=\"def get_version():\\n    with open(version_file, 'r') as f:\\n        exec(compile(f.read(), version_file, 'exec'))\\n    return locals()['__version__']\"; new=\"def get_version():\\n    ns = {}\\n    with open(version_file, 'r') as f:\\n        exec(compile(f.read(), version_file, 'exec'), ns)\\n    return ns['__version__']\"; assert old in s or 'ns = {}' in s; p.write_text(s.replace(old,new))"; \
    pip install --no-cache-dir --no-deps --no-build-isolation "${TMPDIR}/basicsr"; \
    rm -rf "${TMPDIR}"

RUN pip install --no-cache-dir \
    numpy \
    "opencv-python-headless>=4.8.0" \
    facexlib==0.3.0 \
    realesrgan==0.3.0 \
    "Pillow>=10.0.0" \
    runpod

WORKDIR /app

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
