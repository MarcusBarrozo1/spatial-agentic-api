*Read this in [English](README_en.md).*

# Spatial Agentic RAG: Visão Computacional Aplicada à Análise de Risco

Um pipeline de orquestração de nível empresarial projetado para automatizar a análise de risco de crédito agrícola conectando Large Language Models (LLMs) com Visão Computacional Geoespacial.

## 🎯 A Visão
Os fluxos de trabalho tradicionais de AgTech exigem intervenção manual (QGIS, download de imagens de satélite e inspeções visuais). Este projeto implementa um **Agentic Workflow**, onde um LLM atua como orquestrador, acionando autonomamente pipelines de Visão Computacional Dockerizados (Rasterio/NumPy) para segmentar parcelas agrícolas, calcular a saúde vegetativa (NDVI) e fazer referência cruzada a bases de conhecimento espaciais (RAG Empresarial).

## 🏗️ Camadas de Arquitetura
Este sistema é projetado em 4 camadas distintas:
- **Camada 1: Interface e API (Implementada)** - Um gateway FastAPI que recebe intenção em linguagem natural ou payloads estruturados.
- **Camada 2: Ferramentas de Visão (Implementadas)** - Um motor geoespacial Dockerizado que lida com operações GIS pesadas (reprojeção CRS, mascaramento raster, geração de tensores) sem conflitos de dependências C++ (GDAL/PROJ).
- **Camada 3: O Cérebro (Plano)** - Um orquestrador LLM (LangChain/LlamaIndex) utilizando Chamada de Ferramenta para buscar dados autonomamente.
- **Camada 4: Persistência (Plano)** - Memória híbrida usando PostGIS (geometrias vetoriais) e ChromaDB (embeddings de política de crédito).

## 🚀 Recursos Atuais do MVP
- [x] **Ingestão de Dados Geoespaciais:** Lê dados vetoriais (`.shp`, `.geojson`) e raster (`.tif`) de forma integrada.
- [x] **Alinhamento CRS em Tempo Real:** Resolve matematicamente conflitos de sistema de coordenadas na memória.
- [x] **Extração de Tensores:** Corta, mascara e fatia imagens de satélite em tensores `(256, 256, 1)` `float32`, prontos para ingestão de Aprendizado Profundo (U-Net/Mask R-CNN).
- [x] **Agent-Ready API:** Endpoints FastAPI com esquemas Pydantic, prontos para serem consumidos por chamada de função LLM.
- [x] **Containerização:** Totalmente isolado em um ambiente Docker `python:3.11-slim` para garantir reprodutibilidade de MLOps.

## 🛠️ Pilha Tecnológica
**Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic  
**Geoespacial e ML:** Rasterio, GeoPandas, NumPy  
**DevOps:** Docker

## ⚙️ Como Executar Localmente

### 1. Construir o Container MLOps
```bash
docker build -t spatial-agent-api .
```

### 2. Executar o Servidor
```bash
docker run -p 8000:8000 spatial-agent-api
```
#### Acesse a documentação interativa da API em http://localhost:8000/docs.