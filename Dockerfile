FROM python:3.11-slim
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev
WORKDIR /app
COPY requirements.txt .
COPY /pol_raster_geral .
COPY data_loader.py .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "data_loader.py"]

