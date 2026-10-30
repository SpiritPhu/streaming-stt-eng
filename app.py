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
# Default threads to the available CPU cores (capped) so a single stream
# decodes well below real time on a CPU-only host.
NUM_THREADS = int(os.environ.get("NUM_THREADS", str(min(4, os.cpu_count() or 2))))
PROVIDER = os.environ.get("PROVIDER", "cpu")
ENCODER_FILE = os.environ.get("ENCODER_FILE", "encoder.onnx")
DECODER_FILE = os.environ.get("DECODER_FILE", "decoder.onnx")
JOINER_FILE = os.environ.get("JOINER_FILE", "joiner.onnx")
TOKENS_FILE = os.environ.get("TOKENS_FILE", "tokens.txt")
SAVE_AUDIO = os.environ.get("SAVE_AUDIO", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Streaming Speech-to-Text")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

store = ConversationStore(DATA_DIR / "conversations.db")


class RenameRequest(BaseModel):
    title: str


# Shared recognizer instance; each connection gets its own stream.
# ONNX Runtime sessions are safe for concurrent Run() calls and each stream
# holds its own decoding state, so decoding can run from worker threads
# (one decoder task per connection) without a global lock.
_recognizer: sherpa_onnx.OnlineRecognizer | None = None


def _download_model() -> dict[str, str]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, name in (
        ("encoder", ENCODER_FILE),
        ("decoder", DECODER_FILE),
        ("joiner", JOINER_FILE),
        ("tokens", TOKENS_FILE),
    ):
        files[key] = hf_hub_download(
            repo_id=MODEL_REPO_ID, filename=name, local_dir=str(MODEL_DIR)
        )
    return files


def get_recognizer() -> sherpa_onnx.OnlineRecognizer:
    global _recognizer
    if _recognizer is None:
        logger.info("Loading model %s into %s", MODEL_REPO_ID, MODEL_DIR)
        files = _download_model()
        _recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=files["tokens"],
            encoder=files["encoder"],
            decoder=files["decoder"],
            joiner=files["joiner"],
            num_threads=NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            provider=PROVIDER,
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

    # Bounded queue decouples socket reads from CPU decoding and applies
    # backpressure instead of growing memory without limit.
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=128)

    def process_chunk(pcm: bytes) -> tuple[str, bool]:
        """Ingest + decode one chunk in a worker thread (keeps the loop free)."""
        if wav_writer is not None:
            wav_writer.writeframes(pcm)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        stream.accept_waveform(SAMPLE_RATE, samples)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        text = recognizer.get_result(stream).strip()
        is_endpoint = recognizer.is_endpoint(stream)
        if is_endpoint:
            recognizer.reset(stream)
        return text, is_endpoint

    async def receiver() -> None:
        try:
            while True:
                message = await websocket.receive_bytes()
                await audio_queue.put(message)
        except WebSocketDisconnect:
            logger.info("Client disconnected: %s", websocket.client)
        finally:
            await audio_queue.put(None)

    async def decoder() -> None:
        seq = 0
        while True:
            message = await audio_queue.get()
            if message is None:
                break
            text, is_endpoint = await asyncio.to_thread(process_chunk, message)
            if is_endpoint and text:
                seq += 1
                await asyncio.to_thread(store.add_segment, conversation_id, seq, text)
            await websocket.send_json(
                {"text": text, "is_final": is_endpoint, "conversation_id": conversation_id}
            )

    try:
        await asyncio.gather(receiver(), decoder())
    except Exception:
        logger.exception("Error while processing audio stream")
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass
    finally:
        if wav_writer is not None:
            wav_writer.close()
        await asyncio.to_thread(
            store.finalize_conversation, conversation_id, str(audio_path) if audio_path else None
        )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
