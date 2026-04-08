FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# This line fixes the import error!
ENV PYTHONPATH="/app"

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install openenv-core pydantic openai fastapi uvicorn

EXPOSE 7860

CMD ["python", "server/app.py"]