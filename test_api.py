import requests
import json

# URL do endpoint que criamos no main.py
API_URL = "http://localhost:8000/api/v1/segment"

# O "Payload" (JSON) que o Agente LLM enviará após ler o prompt do usuário
payload = {
    "farm_id": "fazenda_teste_001",
    "raster_bands": [
        "sample_data/B2_S2.tif",
        "sample_data/B3_S2.tif",
        "sample_data/B4_S2.tif",
        "sample_data/B8_S2.tif"
    ],
    "output_filename": "predicao_teste.tif"
}

print("🤖 [Agente Simulado] Iniciando requisição para a API de Visão...")
print(f"📦 Payload: {json.dumps(payload, indent=2)}")

try:
    # Disparando a requisição POST
    response = requests.post(API_URL, json=payload)
    
    # Exibindo o resultado retornado pela API
    if response.status_code == 200:
        print("\n✅ [API Resposta] Sucesso!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n❌ [API Resposta] Erro {response.status_code}:")
        print(response.text)

except Exception as e:
    print(f"\n⚠️ Falha ao conectar com a API. O servidor Uvicorn está rodando? Erro: {e}")