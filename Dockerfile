FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

# Loopback by default: under host networking there is no port mapping to hide
# behind, so binding 0.0.0.0 would publish the dashboard on every interface.
ENV BIND_HOST=127.0.0.1 \
    BIND_PORT=8501

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('BIND_PORT','8501')+'/healthz',timeout=4)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host \"$BIND_HOST\" --port \"$BIND_PORT\" --log-level warning"]
