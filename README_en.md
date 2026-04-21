# Spatial Agentic RAG: Computer Vision Applied to Credit Risk

An enterprise-grade orchestration pipeline designed to automate agricultural credit risk analysis by bridging Large Language Models (LLMs) with Geospatial Computer Vision.

## 🎯 The Vision
Traditional AgTech workflows require manual intervention (QGIS, satellite imagery downloading, and visual inspections). This project implements an **Agentic Workflow**, where an LLM acts as the orchestrator, autonomously triggering Dockerized Computer Vision pipelines (Rasterio/NumPy) to segment farm parcels, calculate vegetative health (NDVI), and cross-reference spatial knowledge bases (Enterprise RAG).

## 🏗️ Architecture Layers
This system is designed in 4 distinct layers:
- **Layer 1: FastAPI Orchestration Gateway (Implemented)** 
Fully decoupled microservice architecture handling HTTP requests.
Pydantic validation for incoming Agentic payloads (farm IDs, raster paths).
Dynamic directory routing for localized prediction outputs, ensuring full traceability for downstream LLM agents.
- **Layer 2: Vision Engine (U-Net MLOps) (Implemented)** 
Custom-trained U-Net for semantic segmentation of active agricultural pivots.
Addressed severe Class Imbalance using Smart Sampling and Custom Loss Functions (Weighted Sparse Categorical Crossentropy).
Implemented real-time Synchronized Data Augmentation (NumPy) to break spatial memory and prevent overfitting.
Full-scene inference capabilities using Tiling, Padding, and Stitching matrix operations.
- **Layer 3: The Brain (Roadmap)** - An LLM orchestrator (LangChain/LlamaIndex) utilizing Tool Calling to fetch data autonomously.
- **Layer 4: Persistence (Roadmap)** - Hybrid memory using PostGIS (vector geometries) and ChromaDB (credit policy embeddings).

## 🚀 Current MVP Features
- [x] **Geospatial Data Ingestion:** Reads vector (`.shp`, `.geojson`) and raster (`.tif`) data seamlessly.
- [x] **On-the-fly CRS Alignment:** Mathematically resolves coordinate system conflicts in-memory.
- [x] **Multi-Band Spatial ETL:** Stacking and ingestion of multiple spectral bands of Sentinel-2 for deep spectral analysis.
- [x] **Ground Truth Generation:** Polygon conversion automation (Shapefiles) in georeferenced binary masks (`rasterize`).
- [x] **Tensor Extraction:** Crops, masks, and slices satellite imagery in synchronized tensors `(256, 256, Channels)` and `(256, 256, 1)`, ready for ingestion.
- [x] **Deep Learning Modelling(U-Net):** Personalized semantic architecture with *Bottleneck*, *Upsampling*, (`Conv2DTranspose`) and *Skip Connections* for precision segmentation.
 - [x] **MLOps Orchestration:** Automatized trining cicle with *EarlyStopping* and saveguard of *Model Checkpoints* (`.keras`).
- [x] **Agent-Ready API:** FastAPI endpoints with Pydantic schemas, isolated in Docker environment (`python:3.11-slim`).

## 🛠️ Tech Stack
**Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic  
**Geospatial & ML:** Rasterio, GeoPandas, NumPy
**Machine Learning:** TensorFlow, Keras, NumPy    
**DevOps:** Docker

## ⚙️ How to Run Locally

### 1. Build the MLOps Container
```bash
docker build -t spatial-agent-api .
```

### 2. Run the Server
```bash
docker run -p 8000:8000 spatial-agent-api
```
#### Access the interactive API documentation at http://localhost:8000/docs.


