from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
from core.inference import run_inference
import os
import time

# 1. API Initialization
app = FastAPI(
    title="Spatial Agentic RAG API", 
    description="Core API for Vision Pipeline and Credit Risk Automation",
    version="1.0.0"
)

# 2. Input Schema (What the client/LLM will send us)
class InferenceRequest(BaseModel):
    farm_id: str
    raster_bands: List[str]
    output_filename: str = "prediction.tif"

# 3. Output Schema (What we return to the client/LLM)
class InferenceResponse(BaseModel): 
    farm_id: str
    status: str
    execution_time_seconds: float
    output_path: str
    mean_ndvi: float
    detected_pixels: int

# 4. Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "API is online. Awaiting spatial payloads."}

# 5. The Core Endpoint
@app.post("/api/v1/segment", response_model=InferenceResponse)
def process_farm(payload: InferenceRequest):
    """
    Receives farm data, triggers the raster/numpy pipeline, 
    and returns analytical metrics (Area, NDVI).
    """
    try:
        start_time = time.time()
        
        for band_path in payload.raster_bands:
            if not os.path.exists(band_path):
                raise HTTPException(status_code=400, detail=f"Raster file not found: {band_path}")
        
        output_dir = f"predictions/{payload.farm_id}"
        os.makedirs(output_dir, exist_ok=True)
        final_output_path = os.path.join(output_dir, payload.output_filename)
        
        print(f"[API] Dispatching vision engine for farm: {payload.farm_id}")
        
        result_dict = run_inference(
            model_path='saved_models/spatial_unet_v1.keras', 
            raster_bands=payload.raster_bands,
            output_path=final_output_path
        )
        
        exec_time = round(time.time() - start_time, 2)
        print(f"[API] Inference completed in {exec_time}s. Saved at {result_dict}")
        
        return InferenceResponse(
            status = "success",
            farm_id = payload.farm_id,
            execution_time_seconds = exec_time,
            output_path = result_dict["output_path"],
            mean_ndvi = result_dict["mean_ndvi"],
            detected_pixels = result_dict["detected_pixels"]
        )
        
    except Exception as e:
        print(f"[API ERROR]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)