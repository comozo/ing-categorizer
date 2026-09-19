FROM python:3.13-slim

WORKDIR /app

RUN groupadd --gid 1001 app && useradd --uid 1001 --gid 1001 -m app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
