from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load ONNX model
session = ort.InferenceSession("models/snake_id.onnx")

# Get model input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Labels (manually write your 6 classes)
labels = ["cobra", "viper", "python", "russell_viper", "green_pit_viper", "rat_snake"]

def preprocess(image):
    image = image.resize((640, 640))  # Resize to model input size
    img_array = np.array(image).astype(np.float32)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[..., :3]
    img_array = img_array.transpose(2, 0, 1) / 255.0  # CHW, normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = preprocess(image)
        outputs = session.run([output_name], {input_name: input_tensor})[0]

        predictions = []
        for pred in outputs[0]:  # adjust if shape is [N, num_boxes, 6] or [N, 6]
            conf = float(pred[4])
            if conf > 0.5:  # confidence threshold
                cls = int(pred[5])
                predictions.append({
                    "label": labels[cls],
                    "confidence": conf,
                    "bbox": [float(x) for x in pred[0:4]]
                })

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}
