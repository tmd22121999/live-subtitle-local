import asyncio
import json
import time
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from config import CFG
from pipeline import Pipeline, validate_source_language

app = FastAPI(title='Local Live Subtitle Translator')
# Local-only server. WebSocket additionally checks origin; do not expose to LAN.
app.add_middleware(CORSMiddleware, allow_origins=[], allow_methods=[], allow_headers=[])

@app.get('/health')
def health():
    return {'ok':True, 'sample_rate':CFG.sample_rate}

class Segmenter:
    """Simple RMS speech gate. Replace with Silero VAD for production noise robustness."""
    def __init__(self):
        self.blocks = []
        self.speech = 0
        self.silence = 0
        self.active = False

    def feed(self, pcm):
        duration = len(pcm) / CFG.sample_rate
        speaking = float(np.sqrt(np.mean(pcm * pcm))) > CFG.energy_threshold
        if speaking:
            self.active = True
            self.speech += duration
            self.silence = 0
        elif self.active:
            self.silence += duration
        if self.active:
            self.blocks.append(pcm)
        length = sum(len(b) for b in self.blocks) / CFG.sample_rate
        if self.active and (length >= CFG.max_segment_seconds or self.silence >= CFG.silence_seconds):
            output = np.concatenate(self.blocks)
            valid = self.speech >= CFG.min_speech_seconds
            self.blocks = []; self.speech = 0; self.silence = 0; self.active = False
            return output if valid else None
        return None

class TranslationSession:
    """Manages active translation session state, segmenting, and dynamic language configuration."""
    def __init__(self, pipeline=None, segmenter=None):
        self.pipeline = pipeline or Pipeline()
        self.segmenter = segmenter or Segmenter()
        self.source_language = 'auto'

    def handle_text_message(self, text: str) -> dict | None:
        """Handle incoming JSON text message. Returns response dict if an error occurred."""
        try:
            msg = json.loads(text)
        except Exception:
            return None
        msg_type = msg.get('type')
        if msg_type == 'reset':
            self.segmenter = Segmenter()
            self.pipeline.context.clear()
            return None
        elif msg_type == 'config':
            lang = msg.get('source_language')
            if lang is not None:
                if validate_source_language(lang):
                    self.source_language = lang.lower()
                    return None
                else:
                    return {'type': 'error', 'message': f'Invalid source_language: {lang}'}
        return None

    def handle_audio_chunk(self, samples: np.ndarray) -> tuple[np.ndarray, str] | None:
        """Feed audio chunk into segmenter. If segment is complete, returns (segment, language)."""
        segment = self.segmenter.feed(samples)
        if segment is not None:
            # Tag segment with current source_language atomically
            return (segment, self.source_language)
        return None

@app.websocket('/ws')
async def ws_audio(ws: WebSocket):
    origin = ws.headers.get('origin', '')
    if origin and not origin.startswith('chrome-extension://'):
        await ws.close(code=1008)
        return
    await ws.accept()
    session = TranslationSession()
    work = asyncio.Queue(maxsize=2)

    async def worker():
        while True:
            try:
                segment, seg_lang = await work.get()
            except asyncio.CancelledError:
                break
            try:
                result = await session.pipeline.process(segment, source_language=seg_lang)
                if result:
                    await ws.send_json({'type':'subtitle', **result, 'timestamp':time.time()})
            except Exception as exc:
                await ws.send_json({'type':'error','message':str(exc)[:300]})
            finally:
                work.task_done()

    task = asyncio.create_task(worker())
    try:
        while True:
            packet = await ws.receive()
            if packet['type'] == 'websocket.disconnect':
                break
            if packet.get('text'):
                err = session.handle_text_message(packet['text'])
                if err:
                    await ws.send_json(err)
                continue
            binary = packet.get('bytes')
            if not binary or len(binary) % 4:
                continue
            samples = np.frombuffer(binary, dtype='<f4').copy()
            if len(samples) > 16000 * 2:
                continue
            item = session.handle_audio_chunk(samples)
            if item is not None:
                if work.full():
                    try:
                        work.get_nowait()
                        work.task_done()
                    except asyncio.QueueEmpty:
                        pass
                work.put_nowait(item)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
