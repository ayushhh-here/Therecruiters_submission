# SageMaker invokes this image with the `train` argument; train.py accepts it.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/program

WORKDIR /opt/program

# LightGBM wheels need the OpenMP runtime in the Debian slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY train.py ./train.py

RUN mkdir -p /opt/ml/model /opt/ml/output \
    && chown -R appuser:appuser /opt/program /opt/ml

USER appuser
ENTRYPOINT ["python", "train.py"]
CMD ["train"]
