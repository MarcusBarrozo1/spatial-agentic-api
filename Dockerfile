FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]

