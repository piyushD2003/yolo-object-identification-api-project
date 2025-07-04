# --- STAGE 1: The Builder Stage ---
    FROM python:3.10-slim AS builder

    # Install all system dependencies needed for the build
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    
    COPY requirements.txt constraints.txt ./
    
    # Install all python packages
    RUN pip install --no-cache-dir -r requirements.txt \
        -c constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu
    
    
    # --- STAGE 2: The Final, Lean Runtime Stage ---
    FROM python:3.10-slim
    
    # --- THIS IS THE CRITICAL FIX ---
    # Install ONLY the RUNTIME system dependencies in the final image.
    # These are the .so files that OpenCV needs to actually run.
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    # Copy the installed python packages from the builder
    COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
    
    # Copy our application code
    COPY main.py .
    
    # Pre-download the model
    RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    
    EXPOSE 8000
    CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]