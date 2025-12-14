from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import shutil
import uuid
import os
import mimetypes

from app.backend.integrations.llm import generate_text, build_date_description_prompt
from app.backend.integrations.tts import generate_tts


app = FastAPI()

# Allow communication with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load pre-trained model
model = YOLO("date_fruit_model.pt")

# Static image directory for YOLO outputs and audio files
STATIC_IMAGE_DIR = "static/images"
os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Saudi Date Classifier API MADE BY: Abdulrahman Almejna\nLinkedin: https://www.linkedin.com/in/abdulrahman-almejna-1786b21b4/"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save temp file
    mime_type = file.content_type or "image/jpeg"
    extension = mimetypes.guess_extension(mime_type) or ".jpg"
    temp_filename = f"temp_{uuid.uuid4()}{extension}"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("📥 Received image:", temp_filename)

    # YOLO prediction
    results = model(temp_filename)
    names = results[0].names
    boxes = results[0].boxes

    # Classify the class
    if boxes is not None and boxes.cls.numel() > 0:
        top_class_index = int(boxes.cls[0].item())
        predicted_class = names[top_class_index]
    else:
        predicted_class = "Unknown"

    print("🧠 Predicted class:", predicted_class)

    # Save plotted YOLO image
    output_filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(STATIC_IMAGE_DIR, output_filename)

    plotted = results[0].plot()
    Image.fromarray(plotted).save(output_path)
    print("✅ Saved processed image at:", output_path)

    # Remove temp file
    os.remove(temp_filename)

    image_url = f"/static/images/{output_filename}"

    # Only YOLO result here (fast)
    return {
        "class": predicted_class,
        "image_url": image_url,
    }


@app.post("/describe")
async def describe(date_type: str):
    # Basic fallback if class is Unknown
    if date_type == "Unknown":
        return {
            "description": "ما قدرت أوصف التمرة لأن النوع ما هو واضح.",
            "audio_url": None,
        }

    # Build LLM prompt based on date type + knowledge
    prompt = build_date_description_prompt(date_type)

    # Generate text with LLM
    description = generate_text(prompt)

    # Generate TTS audio from description
    audio_file_path = generate_tts(description)
    print("[DEBUG] Generated TTS audio at:", audio_file_path)

    return {
        "description": description,
        "audio_url": f"/{audio_file_path}" if audio_file_path else None,
    }
