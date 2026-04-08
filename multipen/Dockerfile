# Wildfire Detection OpenEnv Environment

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY . /app/env

WORKDIR /app/env

# Install dependencies directly
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]