# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# --- 1. Load the YOLOv8 model once on startup ---
# This is efficient as the model is loaded into memory only one time.
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")  # "n" is for the nano version - small and fast
print("Model loaded successfully.")


# --- 2. Initialize the FastAPI app ---
app = FastAPI(title="YOLOv8 Object Detection API")


# --- 3. Define API Endpoints ---
@app.get("/", include_in_schema=False)
@app.get("/")
def read_root():
    """ A simple endpoint to confirm the server is running. """
    return {"status": "ok", "message": "YOLOv8 API is running successfully."}


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Receives an image file, performs object detection, and returns the
    image with bounding boxes and labels drawn on it.
    """
    # Validate that the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Read the image content from the uploaded file
    contents = await file.read()
    
    # Open the image using Pillow
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # --- Perform detection ---
    # The model() call runs the detection process.
    results = model(image)

    # --- Draw bounding boxes on the image ---
    draw = ImageDraw.Draw(image)
    
    # The results object contains the detected boxes
    # results[0].boxes gives access to the bounding box data
    for box in results[0].boxes:
        # Get coordinates, confidence, and class ID
        xyxy = box.xyxy[0].tolist()  # Bounding box coordinates (x1, y1, x2, y2)
        conf = box.conf[0].item()    # Detection confidence
        cls = int(box.cls[0].item()) # Class ID
        
        # Get the class name from the model's names list
        class_name = model.names[cls]
        
        # Define color for the bounding box (you can customize this)
        color = "red"
        
        # Draw the rectangle
        draw.rectangle(xyxy, outline=color, width=2)
        
        # Draw the label with confidence
        label = f"{class_name}: {conf:.2f}"
        
        # Use a default font or specify a path to a .ttf file
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((xyxy[0], xyxy[1] - 15), label, fill=color, font=font)

    # --- Prepare the image for response ---
    # Save the modified image to an in-memory buffer
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Move the cursor to the beginning of the buffer

    # Return the image as a streaming response
    return StreamingResponse(img_byte_arr, media_type="image/png")