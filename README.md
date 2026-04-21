*Read this in [English](README_en.md).*

# Spatial Agentic RAG: Visão Computacional Aplicada à Análise de Risco

Um pipeline de orquestração de nível empresarial projetado para automatizar a análise de risco de crédito agrícola conectando Large Language Models (LLMs) com Visão Computacional Geoespacial.

## 🎯 A Visão
Os fluxos de trabalho tradicionais de AgTech exigem intervenção manual (QGIS, download de imagens de satélite e inspeções visuais). Este projeto implementa um **Agentic Workflow**, onde um LLM atua como orquestrador, acionando autonomamente pipelines de Visão Computacional Dockerizados (Rasterio/NumPy) para segmentar parcelas agrícolas, calcular a saúde vegetativa (NDVI) e fazer referência cruzada a bases de conhecimento espaciais (RAG Empresarial).

## 🏗️ Camadas de Arquitetura
Este sistema é projetado em 4 camadas distintas:
- **Camada 1: Gateway de Orquestração (FastAPI) (Implementada)** - 
Arquitetura de microsserviço desacoplada para gerenciamento de requisições HTTP.
Validação via Pydantic para os payloads enviados pelo Agente LLM.
Roteamento dinâmico de diretórios, garantindo rastreabilidade exata dos arquivos de saída para o Cérebro LangChain.
- **Camada 2: Motor de Visão Computacional (Implementadas)** - 
U-Net customizada para segmentação semântica de pivôs agrícolas com cultura ativa (filtro de clorofila).
Resolução de Desbalanceamento de Classes via Smart Sampling e Função de Perda Customizada (Pesos por pixel).
Data Augmentation sincronizado e dinâmico para evitar overfitting espacial.
Inferência em cenas completas (Full-Scene) aplicando rotinas matriciais de Fatiamento (Tiling) e Costura (Stitching).
- **Camada 3: O Cérebro (Plano)** - Um orquestrador LLM (LangChain/LlamaIndex) utilizando Chamada de Ferramenta para buscar dados autonomamente.
- **Camada 4: Persistência (Plano)** - Memória híbrida usando PostGIS (geometrias vetoriais) e ChromaDB (embeddings de política de crédito).

## 🚀 Recursos Atuais do MVP
- [x] **Ingestão de Dados Geoespaciais:** Lê dados vetoriais (`.shp`, `.geojson`) e raster (`.tif`) de forma integrada.
- [x] **Alinhamento CRS em Tempo Real:** Resolve matematicamente conflitos de sistema de coordenadas na memória.
- [x] **Multi-Band Spatial ETL:** Ingestão e empilhamento (*stacking*) de múltiplas bandas espectrais do Sentinel-2 (ex: B2, B3, B4, B8) para análise espectral profunda.
- [x] **Geração de Ground Truth:** Automação da conversão de polígonos (Shapefiles) em máscaras binárias georreferenciadas (`rasterize`).
- [x] **Extração de Tensores:** Corta, mascara e fatia imagens de satélite em tensores sincronizados `(256, 256, Canais)` e `(256, 256, 1)`, prontos para ingestão.
- [x] **Modelagem Deep Learning (U-Net):** Arquitetura semântica personalizada com *Bottleneck*, *Upsampling* (`Conv2DTranspose`) e *Skip Connections* para segmentação de precisão.
- [x] **Orquestração MLOps:** Ciclo de treino automatizado com *EarlyStopping* e salvaguarda de *Model Checkpoints* (`.keras`).
- [x] **Agent-Ready API:** Endpoints FastAPI com esquemas Pydantic, isolados em ambiente Docker (`python:3.11-slim`).

## 🛠️ Pilha Tecnológica
**Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic  
**Geoespacial e ML:** Rasterio, GeoPandas, NumPy
**Machine Learning:** TensorFlow, Keras, NumPy  
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