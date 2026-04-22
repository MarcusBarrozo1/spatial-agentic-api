*Read this in [Portuguese](README.md).*

# 🛰️ Spatial Agentic RAG: Geospatial Automation and Risk Analysis
A multi-agent microservice orchestration pipeline (Agentic AI) designed to automate agricultural credit risk analysis, connecting ultra-fast inference Large Language Models (LLMs) with Geospatial Computer Vision.

## 🎯 The Vision
Traditional AgTech workflows require heavy manual intervention (QGIS, satellite image downloads, and visual inspections). This project implements an Agentic Workflow, where an LLM acts as an orchestrator, reasoning about user demands and autonomously triggering Dockerized Computer Vision pipelines. The system segments agricultural plots, identifies active infrastructure, calculates vegetative health (NDVI), and issues technical reports, eliminating operational bottlenecks.

## 🏗️ Architecture Layers (Microservices)
This system was designed with a focus on scalability, responsibility isolation, and StateGraphs, divided into 4 structural layers:

🟩 **Layer 1: Orchestration Gateway (FastAPI)** - [In Production]
Acts as the RESTFUL server that encapsulates the inference engine.

Decoupling: Isolates heavy tensor processing from the Agent's communication interface.

Strict Validation: Uses Pydantic to validate geographic payloads sent by the LLM (farm_id, raster_bands).

Dynamic Traceability: Autonomous directory routing, ensuring that resulting GeoTIFFs are saved and dynamically mapped for downstream consumption.

Virtualization: Fully packaged in Docker (python:3.11-slim), ensuring hardware agnosticism and easy deployment.

🟦 **Layer 2: Computer Vision Engine (U-Net MLOps)** - [In Production]
The mathematical core of the project. A U-Net convolutional neural network trained from scratch for Semantic Segmentation of irrigated crops using Sentinel-2 bands (B2, B3, B4, B8).

Smart Sampling & Class Imbalance: Implementation of custom extractors that dynamically balance positive patches (pivots) and negative (background), preventing statistical bias from background.

Custom Loss Function: Use of Weighted Sparse Categorical Crossentropy computed pixel by pixel, severely penalizing errors in minority classes.

Synchronized Data Augmentation: Real-time matrix augmentation (X and Y) via NumPy, breaking the model's "spatial memory" to prevent Overfitting.

Full-Scene Inference Engine: Native pipeline (inference.py) of giant image slicing (Padding -> Tiling) and matrix reconstruction (Stitching -> Cropping) for complete scene prediction in seconds.

🟪 **Layer 3: Agentic Brain (LangGraph & Groq)** - [In Production]
The intelligent orchestrator that makes decisions based on the user's prompt.

StateGraph (State Machine): Non-linear routing flow using langgraph, allowing the model to autonomously decide when to invoke external tools (Tool Calling).

Tool Node (agent_tools.py): Autonomous tools that trigger POST requests to Layer 1 (Docker), interpret results, and generate raw data extraction.

LLM Core (Groq/Llama 3): Use of the llama-3.3-70b-versatile model via Groq API, delivering exceptional logical reasoning capacity (Function Calling) with near-zero response latency.

🟨 **Layer 4: Persistence (Enterprise RAG)** - [Roadmap]
Hybrid Memory: Planning for integration with PostGIS (historical vector geometries) and ChromaDB (embeddings of agricultural credit policies) for real-time context.

## 🛠️ Engineering Highlights & MLOps
Label Noise Correction: During audits, the model detected the pattern of pivots without active crop (NDVI ~0.08). The Ground Truth was surgically recalibrated, proving that the architecture learns active chlorophyll signatures (NDVI > 0.5) and not just geometric shapes, ensuring real risk analysis.

Multi-Band Spatial ETL: Ingestion and stacking of multiple perfectly spatially aligned spectral bands in memory (Rasterio/NumPy).

Feature Suppression vs. Efficiency: Documented analysis on neural weight allocation in high-distinction geometries (circles) at the expense of irregular polygons in reduced datasets.

## ⚙️ Initialization and Execution
### 1. Environment Requirements
- Make sure local ports (e.g., 8000) are not in use.

- GIS software (like QGIS) should not be locking files in the `/predictions` folder to avoid `Permission Denied` errors in the container.

### 2. Start the Vision Service (Docker Compose)
Initialize Layers 1 and 2 (API + U-Net Model):

```bash
# Builds and starts the Vision API in isolation
docker-compose up --build
```
The FastAPI will be listening for requests on the local port.

### 3. Agent Orchestration
In a separate terminal (local virtual environment), configure keys and start the Brain:

```bash
# Install agent dependencies
pip install -r requirements.txt

# Configure Groq access key
export GROQ_API_KEY="your_groq_key_here"

# Start the Geospatial Assistant
python agent.py
```

## 📊 Benchmark Results (V1)
- Inference Time (Full-Scene): ~2.4s per complete scene of 1334x1746 px in the slicing pipeline.

- Prediction Target: Crops in vegetative stage.

- Validation Metric: Average NDVI detected of ~0.58 in predicted areas (success in filtering exposed soil).

- Stability: 100% success in asynchronous dispatch via API and serialization/deserialization of `.keras` inside the Linux container.

## 🗺️ Next Steps (Roadmap)
[ ] Sentinel Hub API Integration: Allow the Agent to automatically download images based on coordinate requests (Lat/Lon) sent by the user, eliminating dependence on local sample files.

[ ] Time Series Analysis: Orchestration of multiple temporal frames for detection of phenological anomalies in the crop cycle.