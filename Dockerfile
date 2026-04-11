FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH="/app"

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY . /app

RUN pip install -e .

EXPOSE 7860

CMD ["python", "-m", "server.app"]