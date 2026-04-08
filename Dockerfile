FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install openenv-core pydantic openai

EXPOSE 7860

# Keep the container alive on the expected Hugging Face Spaces port.
CMD ["python", "-m", "http.server", "7860"]
