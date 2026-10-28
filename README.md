# Realtime Streaming Speech-to-Text (STT) Server

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Sherpa-ONNX](https://img.shields.io/badge/Sherpa--ONNX-Zipformer-orange.svg)](https://github.com/k2-fsa/sherpa-onnx)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

A high-performance, low-latency, real-time streaming speech-to-text server powered by **FastAPI**, **Sherpa-ONNX** (Streaming Zipformer), and **WebSockets**. The project includes a modern browser client supporting AudioWorklet 16 kHz PCM streaming, real-time endpoint detection, conversation management with SQLite persistence, audio playback, and transcript export.

---

## 🌟 Key Features

- **True Streaming ASR**: Real-time acoustic processing with Sherpa-ONNX Zipformer model; no large chunk delays.
- **Low Latency & CPU Efficient**: Utilizes ONNX Runtime CPU inference with configurable thread allocation.
- **AudioWorklet Client**: Captures microphone input natively, resamples to 16 kHz mono PCM in the browser, and streams over WebSockets.
- **Conversation & Audio Persistence**:
  - Automatically records conversations and segment timestamps into SQLite with Write-Ahead Logging (WAL) for high concurrency.
  - Option to archive full audio as 16-bit WAV files.
- **Interactive Web UI**:
  - **Live Mode**: Real-time partial and final transcription.
  - **History Mode**: Search, filter by date, playback recorded audio, rename titles, and export transcripts (.txt).
- **Production-Ready**:
  - Containerized with optimized multi-stage Dockerfile and non-root security.
  - Docker Compose setup with persistent volumes for models and database.
  - FastAPI modern `lifespan` architecture, CORS middleware, and `/healthz` monitoring.
  - Nginx reverse proxy configuration template with WebSocket upgrade support.
  - Automated GitHub Actions CI workflow.

---

## 🏗️ Architecture

```
+-------------------------------------------------------------+
| Browser Client (HTML5 / AudioWorklet)                       |
| 1. Capture Mic -> Downsample to 16 kHz 16-bit PCM Mono      |
| 2. Send Binary PCM frames via WebSocket                     |
+------------------------------+------------------------------+
                               |
                        WebSocket /ws/asr
                               |
+------------------------------v------------------------------+
| FastAPI Application Server                                  |
| - Async WebSocket Connection Handler                        |
| - Audio WAV Stream Archiver (Optional)                      |
| - Sherpa-ONNX Online Recognizer (Zipformer Transducer)      |
| - SQLite Database (WAL Mode for high concurrency)           |
+-------------------------------------------------------------+
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/SpiritPhu/streaming-stt-eng.git
cd streaming-stt-eng

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

### 3. Run the Server

```bash
python app.py
# Or using uvicorn directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`**, allow microphone access, and click **Start** to begin speaking.

*(Note: On the first launch, the Zipformer ONNX model files (~70MB) will be automatically downloaded from Hugging Face into `./models/zipformer/`)*.

---

## 🐳 Production Deployment

### Option 1: Docker Compose (Recommended)

1. **Start the service in detached mode:**
   ```bash
   docker compose up -d --build
   ```

2. **Inspect logs:**
   ```bash
   docker compose logs -f streaming-stt
   ```

3. **Verify health check:**
   ```bash
   curl http://localhost:8000/healthz
   ```

4. **Volumes:**
   - `./data`: Stores `conversations.db` and recorded `.wav` files.
   - `./models`: Stores downloaded ONNX model weights so restarts do not redownload them.

### Option 2: Docker CLI

```bash
# Build the Docker image
docker build -t streaming-stt:latest .

# Run container with persistent volumes
docker run -d \
  --name streaming-stt-app \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e LOG_LEVEL=INFO \
  -e NUM_THREADS=2 \
  streaming-stt:latest
```

### Option 3: Systemd Service (Linux Ubuntu/Debian)

Create a systemd unit file `/etc/systemd/system/streaming-stt.service`:

```ini
[Unit]
Description=Realtime Streaming Speech-to-Text Server
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/streaming-stt-eng
Environment="PATH=/opt/streaming-stt-eng/.venv/bin"
EnvironmentFile=/opt/streaming-stt-eng/.env
ExecStart=/opt/streaming-stt-eng/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now streaming-stt
```

---

## 🔒 Nginx Reverse Proxy & SSL Setup

For production use (especially HTTPS, required for browser microphone permissions on non-localhost domains), configure Nginx with WebSocket upgrade headers. A template is provided in `nginx.conf.example`:

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

upstream streaming_stt_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name stt.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/stt.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/stt.yourdomain.com/privkey.pem;

    # Real-time WebSocket streaming
    location /ws/asr {
        proxy_pass http://streaming_stt_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # Static assets and REST API
    location / {
        proxy_pass http://streaming_stt_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## ⚙️ Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Server bind IP address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_REPO_ID` | `csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | Hugging Face repo ID for Sherpa-ONNX model |
| `SAMPLE_RATE` | `16000` | Target audio sample rate in Hz |
| `NUM_THREADS` | `2` | CPU thread count allocated for ONNX Runtime decoding |
| `SAVE_AUDIO` | `true` | Save conversation sessions as `.wav` files |
| `DATA_DIR` | `./data` | Directory for SQLite DB and recordings |
| `MODEL_DIR` | `./models/zipformer` | Directory storing ONNX model files |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated or `*`) |

---

## 📡 API & WebSocket Reference

### WebSocket Endpoint

- **`ws://localhost:8000/ws/asr`**
  - **Input**: Raw 16-bit signed PCM mono audio bytes at 16,000 Hz.
  - **Output (JSON)**:
    ```json
    {
      "text": "hello world",
      "is_final": true,
      "conversation_id": 12
    }
    ```

### REST Endpoints

- **`GET /healthz`**: Liveness and readiness health check.
- **`GET /api/conversations?date=YYYY-MM-DD&search=query`**: List saved conversations.
- **`GET /api/conversations/{id}`**: Retrieve full details and segmented transcript of a conversation.
- **`GET /api/conversations/{id}/audio`**: Stream audio WAV file for a conversation.
- **`GET /api/conversations/{id}/export`**: Download transcript as `.txt` file.
- **`PATCH /api/conversations/{id}`**: Update conversation title.
- **`DELETE /api/conversations/{id}`**: Delete conversation and associated audio file.

---

## 📄 License

This project is licensed under the MIT License.
