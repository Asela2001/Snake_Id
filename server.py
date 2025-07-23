from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Allow all origins for testing (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load the YOLO model once at startup (adjust path as needed)
model = YOLO('models/snake_id.onnx', task='detect')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        results = model(image)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return {"message": "No snake detected with high confidence."}

        predictions = []
        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            bbox = box.xywh.tolist()[0]
            predictions.append({
                "label": model.names[cls],
                "confidence": conf,
                "bbox": bbox
            })

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}
