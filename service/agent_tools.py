import requests
from langchain_core.tools import tool

# URL da nossa FastAPI rodando localmente
API_URL = "http://localhost:8000/api/v1/segment"

@tool
def run_vision_segmentation(farm_id: str, target_crop: str = "unknown") -> str:
    """
    Use this tool to run the Deep Learning Vision Engine on a specific farm.
    Always use this when the user asks to analyze, segment, or get NDVI/area metrics for a farm.
    
    Args:
        farm_id: The unique identifier of the farm (e.g., 'fazenda_teste_001').
        target_crop: The crop type, if mentioned.
    """
    print(f"\n⚙️ [Tool Execution] Acionando motor de visão para {farm_id}...")
    
    # O mesmo payload que validamos no test_api.py
    payload = {
        "farm_id": farm_id,
        "raster_bands": [
            "sample_data/B2_S2.tif",
            "sample_data/B3_S2.tif",
            "sample_data/B4_S2.tif",
            "sample_data/B8_S2.tif"
        ],
        "output_filename": f"{farm_id}_prediction.tif"
    }

    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return (f"SUCESSO. A predição foi concluída.\n"
                f"Arquivo TIF gerado: {result.get('output_path')}\n"
                f"Tempo de Execução: {result.get('execution_time_seconds')}s\n"
                f"Pixels de Pivô Agrícola Detectados: {result.get('detected_pixels')}\n"
                f"NDVI Médio da Área de Cultura: {result.get('mean_ndvi')}"
            )
        else:
            return f"API Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Failed to connect to the Vision API: {str(e)}"