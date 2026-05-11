FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache .

COPY app/ app/
COPY .env* ./
COPY main.py main.py

EXPOSE 8000

CMD ["python", "main.py"]
