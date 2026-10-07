FROM python:3.12-slim

# Keep runtime image deterministic and avoid bytecode/cache writes.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/agent
COPY pyproject.toml README.md ./
COPY app ./app
COPY evaluation ./evaluation
COPY fixtures ./fixtures
COPY web ./web
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /data /run/secrets \
    && chown -R agent:agent /srv/agent /data

# Run the API without root privileges; the worker uses the same image.
USER agent
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
