FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

# Added fastapi and uvicorn to the install list
RUN pip install --upgrade pip && \
    pip install openenv-core pydantic openai fastapi uvicorn

EXPOSE 7860

# Run the actual OpenEnv server app, NOT the static file server
CMD ["python", "server/app.py"]