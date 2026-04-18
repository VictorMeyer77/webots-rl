FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache .

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
