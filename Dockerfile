FROM python:3.9-slim-bullseye

# Set working directory
WORKDIR /app

# Install dependenciess
COPY requirements.txt /app/requirements.txt

#TODO  Change this in future for faster build time
RUN apt-get update && \
    apt-get install -y --no-install-recommends debian-archive-keyring && \
    apt-get update

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

