*Read this in [English](README_en.md).*

# 🛰️ Spatial Agentic RAG: Automação Geoespacial e Análise de Risco
Um pipeline de orquestração de microsserviço multiagente (Agentic AI) projetado para automatizar a análise de risco de crédito agrícola, conectando Large Language Models (LLMs) de inferência ultrarrápida com Visão Computacional Geoespacial.

## 🎯 A Visão
Os fluxos de trabalho tradicionais em AgTech exigem intervenção manual pesada (QGIS, download de imagens de satélite e inspeções visuais). Este projeto implementa um Agentic Workflow, onde um LLM atua como orquestrador, raciocinando sobre as demandas do usuário e acionando autonomamente pipelines de Visão Computacional Dockerizados. O sistema segmenta parcelas agrícolas, identifica infraestruturas ativas, calcula a saúde vegetativa (NDVI) e emite laudos técnicos, eliminando o gargalo operacional.

## 🏗️ Camadas de Arquitetura (Microservices)
Este sistema foi desenhado com foco em escalabilidade, isolamento de responsabilidades e gerenciamento de Estado (StateGraph), dividido em 4 camadas estruturais:

🟩 **Camada 1: Gateway de Orquestração (FastAPI)** - [Em Produção]
Atua como o servidor RESTFUL que encapsula o motor de inferência.

Desacoplamento: Isola o processamento pesado de tensores da interface de comunicação do Agente.

Validação Estrita: Utiliza Pydantic para validar os payloads geográficos enviados pelo LLM (farm_id, raster_bands).

Rastreabilidade Dinâmica: Roteamento autônomo de diretórios, garantindo que os GeoTIFFs resultantes sejam salvos e mapeados dinamicamente para consumo downstream.

Virtualização: Totalmente empacotado em Docker (python:3.11-slim), garantindo agnosticidade de hardware e facilidade de deploy.

🟦 **Camada 2: Motor de Visão Computacional (U-Net MLOps)** - [Em Produção]
O núcleo matemático do projeto. Uma rede neural convolucional U-Net treinada do zero para Segmentação Semântica de culturas irrigadas usando bandas do Sentinel-2 (B2, B3, B4, B8).

Smart Sampling & Class Imbalance: Implementação de extratores personalizados que equilibram dinamicamente patches positivos (pivôs) e negativos (background), prevenindo viés estatístico de fundo.

Custom Loss Function: Uso de Weighted Sparse Categorical Crossentropy computada pixel a pixel, penalizando severamente erros nas classes minoritárias.

Synchronized Data Augmentation: Aumento de dados matriciais (X e Y) em tempo real via NumPy, quebrando a "memória espacial" do modelo para prevenir Overfitting.

Full-Scene Inference Engine: Pipeline nativo (inference.py) de fatiamento de imagens gigantes (Padding -> Tiling) e reconstrução matricial (Stitching -> Cropping) para predição de cenas completas em segundos.

🟪 **Camada 3: Cérebro Agentic (LangGraph & Groq)** - [Em Produção]
O orquestrador inteligente que toma decisões com base no prompt do usuário.

StateGraph (Máquina de Estado): Fluxo de roteamento não-linear usando langgraph, permitindo que o modelo decida de forma autônoma quando invocar ferramentas externas (Tool Calling).

Tool Node (agent_tools.py): Ferramentas autônomas que disparam requisições POST para a Camada 1 (Docker), interpretam os resultados e geram a extração de dados brutos.

LLM Core (Groq/Llama 3): Utilização do modelo llama-3.3-70b-versatile via Groq API, entregando excepcional capacidade de raciocínio lógico (Function Calling) com latência de resposta próxima a zero.

🟨 **Camada 4: Persistência (RAG Empresarial)** - [Roadmap]
Memória Híbrida: Planejamento para integração com PostGIS (geometrias vetoriais históricas) e ChromaDB (embeddings de políticas de crédito agrícola) para contexto em tempo real.

## 🛠️ Destaques de Engenharia & MLOps
Correção de Ruído de Rótulo (Label Noise): Durante as auditorias, o modelo detectou o padrão de pivôs sem cultura ativa (NDVI ~0.08). O Ground Truth foi recalibrado cirurgicamente, comprovando que a arquitetura aprende assinaturas de clorofila ativa (NDVI > 0.5) e não apenas formas geométricas, garantindo análise de risco real.

Multi-Band Spatial ETL: Ingestão e empilhamento (stacking) de múltiplas bandas espectrais perfeitamente alinhadas espacialmente na memória (Rasterio/NumPy).

Supressão de Features vs. Eficiência: Análise documentada sobre a alocação de pesos neurais em geometrias de alta distinção (círculos) em detrimento de polígonos irregulares em datasets reduzidos.

## ⚙️ Inicialização e Execução
### 1. Requisitos do ambiente
- Certifique-se de que as portas locais (ex: 8000) não estejam em uso.

- Softwares GIS (como o QGIS) não devem estar travando os arquivos na pasta `/predictions` para evitar erros de `Permission Denied` no container.

### 2. Subir o Serviço de Visão (Docker Compose)
Inicialize as Camadas 1 e 2 (API + Modelo U-Net):

```bash
# Constrói e inicia a API de Visão de forma isolada
docker-compose up --build
```
A FastAPI estará ouvindo requisições na porta local.

### 3. Orquestração do Agente
Em um terminal separado (ambiente virtual local), configure as chaves e inicie o Cérebro:

```bash
# Instale as dependências do agente
pip install -r requirements.txt

# Configure a chave de acesso da Groq
export GROQ_API_KEY="sua_chave_groq_aqui"

# Inicie o Assistente Geoespacial
python agent.py
```

## 📊 Resultados de Referência (V1)
- Tempo de Inferência (Full-Scene): ~2.4s por cena completa de 1334x1746 px no pipeline de fatiamento.

- Alvo de Predição: Culturas em estágio vegetativo.

- Métrica de Validação: NDVI Médio detectado de ~0.58 nas áreas previstas (sucesso na filtragem de solo exposto).

- Estabilidade: 100% de sucesso no despache assíncrono via API e serialização/desserialização do `.keras` dentro do container Linux.

## 🗺️ Próximos Passos (Roadmap)
[ ] Integração Sentinel Hub API: Permitir que o Agente faça o download automático de imagens baseando-se em requisições de coordenadas (Lat/Lon) enviadas pelo usuário, eliminando a dependência de arquivos de amostra locais.

[ ] Análise de Séries Temporais: Orquestração de múltiplos frames temporais para detecção de anomalias fenológicas no ciclo da safra.