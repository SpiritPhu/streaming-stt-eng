"""Realtime streaming speech-to-text server using sherpa-onnx + FastAPI WebSocket.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in the browser, grant microphone access,
and click Start to see live transcription.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sherpa_onnx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from db import ConversationStore

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("streaming_stt_app")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
RECORDINGS_DIR = DATA_DIR / "recordings"

MODEL_REPO_ID = os.environ.get(
    "MODEL_REPO_ID", "csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models" / "zipformer"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
NUM_THREADS = int(os.environ.get("NUM_THREADS", "2"))
SAVE_AUDIO = os.environ.get("SAVE_AUDIO", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Streaming Speech-to-Text")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

store = ConversationStore(DATA_DIR / "conversations.db")


class RenameRequest(BaseModel):
    title: str

# Shared recognizer instance; each connection gets its own stream.
# A lock guards decode calls because the underlying onnxruntime sessions
# are not guaranteed safe for unsynchronized concurrent decoding.
_recognizer: sherpa_onnx.OnlineRecognizer | None = None
_decode_lock = asyncio.Lock()


def _download_model() -> dict[str, str]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        files[name] = hf_hub_download(
            repo_id=MODEL_REPO_ID, filename=name, local_dir=str(MODEL_DIR)
        )
    return files


def get_recognizer() -> sherpa_onnx.OnlineRecognizer:
    global _recognizer
    if _recognizer is None:
        logger.info("Loading model %s into %s", MODEL_REPO_ID, MODEL_DIR)
        files = _download_model()
        _recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=files["tokens.txt"],
            encoder=files["encoder.onnx"],
            decoder=files["decoder.onnx"],
            joiner=files["joiner.onnx"],
            num_threads=NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            provider="cpu",
        )
        logger.info("Recognizer ready")
    return _recognizer


@app.on_event("startup")
def _warm_up() -> None:
    get_recognizer()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_loaded": _recognizer is not None}


@app.get("/api/conversations")
def api_list_conversations(date: Optional[str] = None, search: Optional[str] = None) -> list:
    return store.list_conversations(date=date, search=search)


@app.get("/api/conversations/{conversation_id}")
def api_get_conversation(conversation_id: int) -> dict:
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.get("/api/conversations/{conversation_id}/export")
def api_export_conversation(conversation_id: int) -> PlainTextResponse:
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    text = "\n".join(s["text"] for s in conv["segments"])
    filename = f"conversation_{conversation_id}.txt"
    return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/conversations/{conversation_id}/audio")
def api_get_audio(conversation_id: int) -> FileResponse:
    conv = store.get_conversation(conversation_id)
    if conv is None or not conv.get("audio_path"):
        raise HTTPException(status_code=404, detail="Audio not available")
    path = Path(conv["audio_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing")
    return FileResponse(str(path), media_type="audio/wav")


@app.patch("/api/conversations/{conversation_id}")
def api_rename_conversation(conversation_id: int, payload: RenameRequest) -> dict:
    if store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    store.rename_conversation(conversation_id, payload.title)
    return {"status": "ok"}


@app.delete("/api/conversations/{conversation_id}")
def api_delete_conversation(conversation_id: int) -> dict:
    audio_path = store.delete_conversation(conversation_id)
    if audio_path:
        Path(audio_path).unlink(missing_ok=True)
    return {"status": "ok"}


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket) -> None:
    await websocket.accept()
    recognizer = get_recognizer()
    stream = recognizer.create_stream()
    session_id = str(uuid.uuid4())
    conversation_id = await asyncio.to_thread(store.create_conversation, session_id)
    logger.info("Client connected: %s (conversation #%s)", websocket.client, conversation_id)

    seq = 0
    wav_writer: wave.Wave_write | None = None
    audio_path: Path | None = None
    if SAVE_AUDIO:
        date_dir = RECORDINGS_DIR / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        audio_path = date_dir / f"{conversation_id}.wav"
        wav_writer = wave.open(str(audio_path), "wb")
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(SAMPLE_RATE)

    try:
        while True:
            message = await websocket.receive_bytes()
            if wav_writer is not None:
                wav_writer.writeframes(message)

            # Frontend sends little-endian 16-bit PCM mono at SAMPLE_RATE.
            samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0

            async with _decode_lock:
                stream.accept_waveform(SAMPLE_RATE, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                text = recognizer.get_result(stream).strip()
                is_endpoint = recognizer.is_endpoint(stream)
                if is_endpoint:
                    recognizer.reset(stream)

            if is_endpoint and text:
                seq += 1
                await asyncio.to_thread(store.add_segment, conversation_id, seq, text)

            await websocket.send_json(
                {"text": text, "is_final": is_endpoint, "conversation_id": conversation_id}
            )
    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", websocket.client)
    except Exception:
        logger.exception("Error while processing audio stream")
        await websocket.close(code=1011)
    finally:
        if wav_writer is not None:
            wav_writer.close()
        await asyncio.to_thread(
            store.finalize_conversation, conversation_id, str(audio_path) if audio_path else None
        )
