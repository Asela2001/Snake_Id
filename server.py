from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Configure CORS properly for production
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "https://your-expo-app.exp.host",
#         "http://localhost:19006"
#     ],
#     allow_methods=["POST"],
#     allow_headers=["*"],
# )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, restrict in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load model once at startup
model = YOLO('models/snake_id.onnx', task='detect')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        results = model(image)
        
        predictions = [
            {
                "label": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xywh.tolist()[0]
            }
            for box in results[0].boxes
        ]
        
        return {"predictions": predictions}
    
    except Exception as e:
        return {"error": str(e)}