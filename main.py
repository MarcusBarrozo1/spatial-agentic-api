from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 1. API Initialization
app = FastAPI(
    title="Spatial Agentic RAG API", 
    description="Core API for Vision Pipeline and Credit Risk Automation",
    version="1.0.0"
)

# 2. Input Schema (What the client/LLM will send us)
class FarmPayload(BaseModel):
    farm_id: str
    target_crop: str

# 3. Output Schema (What we return to the client/LLM)
class VisionResponse(BaseModel):
    farm_id: str
    status: str
    total_area_hectares: float
    mean_ndvi: float

# 4. Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "API is online. Awaiting spatial payloads."}

# 5. The Core Endpoint
@app.post("/api/v1/process-farm", response_model=VisionResponse)
def process_farm(payload: FarmPayload):
    """
    Receives farm data, triggers the raster/numpy pipeline, 
    and returns analytical metrics (Area, NDVI).
    """
    try:
        # TODO: Integrate the logic from data_loader.py here in the next step
        print(f"Processing farm {payload.farm_id} for {payload.target_crop}...")
        
        # Mocking the response for initial architecture test
        return VisionResponse(
            farm_id=payload.farm_id,
            status="success",
            total_area_hectares=52.5,
            mean_ndvi=0.82
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)