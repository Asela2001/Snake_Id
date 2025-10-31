from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
import logging
import requests
from pydantic import BaseModel
from typing import Optional

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS for dev (restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for POST body of /snake-details
class SnakeDetailsRequest(BaseModel):
    snake_name: str
    detail_type: str
    country: Optional[str] = "general"

# Load model at startup

# Load model at startup
try:
    model = YOLO('models/snake_id.onnx', task='detect')
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    raise  # Re-raise to prevent partial startup

# Gemini API configuration (keep key secure; use env vars in prod)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = "AIzaSyCzTnS3amkfj0K6QpAst7hTmFTZF7KhYG0"  # Replace with env var in production

@app.get("/")
async def root():
    """Health check."""
    return {"message": "Snake Detection API is running!", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if len(image_data) == 0:
            return JSONResponse(status_code=400, content={"error": "Empty file."})
        
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"Processing image: {file.filename}")
        
        results = model(image)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return {"message": "No snake detected.", "predictions": []}
        
        predictions = []
        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            bbox = [round(x, 4) for x in box.xywh.tolist()[0]]  # Normalized xywh, rounded
            predictions.append({
                "label": model.names[cls],
                "confidence": round(conf, 4),
                "bbox": bbox
            })
        
        logger.info(f"Found {len(predictions)} predictions.")
        return {"predictions": predictions}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

class SnakeDetailsRequest(BaseModel):
    snake_name: str
    detail_type: str
    country: str = "general"

    
@app.get("/snake-details")
async def get_snake_details(
    snake_name: str = Query(..., description="Name of the snake species"),
    detail_type: str = Query(..., description="Type of details (e.g., 'facts', 'info')"),
    country: str = Query("general", description="Country or region where found")
):
    """
    Fetch snake details using Gemini API.
    Returns processed bullet-point facts.
    """
    try:
        prompt = f"""
        Provide {detail_type} about {snake_name} found in {country} with these rules:
        1. Maximum 10 short sentences
        2. Use relevant emojis
        3. Format as bullet points
        4. Include: identification, habitat, behavior, and danger level
        5. Keep scientific but friendly tone
        
        Example format:
        • 🐍 Species: Eastern Brown Snake (Pseudonaja textilis)
        • 🌍 Habitat: Woodlands and grasslands of Australia
        • ⚡ Danger: Highly venomous (2nd most venomous land snake)
        • 📏 Size: Typically 1.5-2m long
        """

        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            },
            headers={"Content-Type": "application/json"}
        )
        
        full_response = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        
        # Post-process to ensure concise output
        lines = [line.strip() for line in full_response.splitlines() if line.strip()]
        processed_response = "\n\n".join(lines[:10])  # Take first 10 non-empty lines and join with spacing
        
        logger.info(f"Generated details for {snake_name} in {country}")
        return {"details": processed_response}
        logger.info(f"Generated details for {snake_name} in {country}")
        return {"details": processed_response}
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to fetch details from AI service."})
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected Gemini response format: {e}")
        return JSONResponse(status_code=500, content={"error": "Invalid response from AI service."})
    except Exception as e:
        logger.error(f"Snake details error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    # Use import string for reload support (no warning)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")