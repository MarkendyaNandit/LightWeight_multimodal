FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Run the Uvicorn server on the port Hugging Face expects (7860)
CMD ["uvicorn", "web_app.main:app", "--host", "0.0.0.0", "--port", "7860"]
