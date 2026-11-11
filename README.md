# Local Live Subtitle Translator — Windows / AMD RX 7800 XT

**Status:** end-to-end MVP source code (Chrome tab audio → local WebSocket → segmenter → whisper.cpp → NLLB → adaptive llama.cpp refinement → overlay). **Not yet verified on RX 7800 XT or against real YouTube/anime playback.**

## Requirements

- Windows 10/11, Chrome or Chromium with MV3 `tabCapture` + offscreen API
- Python 3.11–3.12 recommended (PyTorch compatibility varies by Python version)
- Git, CMake, Visual Studio C++ build tools, Vulkan SDK, recent AMD driver
- Free disk space for Whisper small, NLLB-200-distilled-600M and optional Qwen3 1.7B Q4
- Local backend binds `127.0.0.1` only

## Architecture & Pipeline

```
Chrome Tab Audio Capture (AudioWorklet 16kHz mono)
   │
   ▼
Local WebSocket (ws://127.0.0.1:8766/ws)
   │
   ▼
Audio Segmenter (RMS energy VAD gate)
   │
   ▼
whisper.cpp Vulkan (STT with configurable -l <lang> or -l auto)
   │
   ▼
NLLB-200 Translator (Multilingual -> vie_Latn, bypassed for Vietnamese)
   │
   ▼
Adaptive Refiner (Optional local Qwen3 via llama.cpp Vulkan)
   │
   ▼
Vietnamese Subtitle Overlay (DOM Shadow Root on active tab)
```

## Source Language Selection

The Chrome Extension popup provides a **Source Language** dropdown:

| Display | Language Code | NLLB Target Code |
|---|---|---|
| Auto Detect | `auto` | Dynamic based on Whisper detection |
| Japanese | `ja` | `jpn_Jpan` |
| English | `en` | `eng_Latn` |
| Korean | `ko` | `kor_Hang` |
| Chinese | `zh` | `zho_Hans` |
| French | `fr` | `fra_Latn` |
| Spanish | `es` | `spa_Latn` |
| Vietnamese | `vi` | `vie_Latn` (Translation bypassed) |

### Features:
- **Persistence:** Selected language is saved in `chrome.storage.local` and restored upon reopening.
- **Dynamic Session Switching:** Changing the dropdown during an active stream sends a session configuration update (`{"type": "config", "source_language": "..."}`) over the WebSocket. The backend atomically applies the new language starting with the subsequent audio segment without restarting or interrupting playback.
- **Detected Language Indicator:** When in **Auto Detect** mode, the popup UI displays the detected language (e.g. `Detected: Japanese (ja)`) in real time as subtitle events arrive.
- **Vietnamese Bypass:** When the source language is Vietnamese (`vi`), NLLB translation and refinement are automatically bypassed, preserving the original transcription verbatim.

## Step 1: whisper.cpp Vulkan

Build whisper.cpp (separate repository):

```powershell
git clone https://github.com/ggml-org/whisper.cpp C:\whisper.cpp
cd C:\whisper.cpp
cmake -S . -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j 6
.\models\download-ggml-model.cmd small
```

Find the binary with `Get-ChildItem .\build -Recurse -Filter whisper-cli.exe`. Test it with a 16kHz mono WAV and confirm Vulkan is actually detected. Set `WHISPER_CLI` and `WHISPER_MODEL` to absolute paths. **Do not assume GPU acceleration is active merely because compilation succeeded.** whisper.cpp command line flags/JSON shape can change across releases; adjust the adapter in `backend/pipeline.py` if necessary.

### Model Loading Overhead & Persistent Inference Evaluation

- **Subprocess Overhead:** Currently, `WhisperCLI` executes `whisper-cli.exe` via `subprocess.run` for each speech segment. Each invocation incurs C++ runtime startup, Vulkan device initialization, and model weight reload (~460MB for small), creating a baseline latency of ~200ms–800ms per segment.
- **Persistent Architecture Evaluation:** For production lower latency, consider:
  1. `whisper-server`: A built-in HTTP server shipped with whisper.cpp that maintains the model resident in VRAM.
  2. In-process bindings: Python bindings such as `pywhispercpp` or `whisper-cpp-python` that load `whisper_context` once.
- **Dynamic Configuration:** Explicit language selection (`-l ja`, `-l en`, etc.) eliminates Whisper's internal language identification pass, reducing inference latency compared to `-l auto`.

## Step 2: Python backend

```powershell
cd live-subtitle-local\backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:WHISPER_CLI='C:\whisper.cpp\build\bin\Release\whisper-cli.exe'
$env:WHISPER_MODEL='C:\whisper.cpp\models\ggml-small.bin'
$env:REFINE_ENABLED='false' # first verify STT + translation before enabling refinement
python -m uvicorn main:app --host 127.0.0.1 --port 8766
```

First translation will download NLLB-200-distilled-600M from Hugging Face and load it into CPU RAM; this can take time. Once cached, it runs offline. **NLLB is not optimized for short conversational subtitles; compare quality with a multilingual LLM before relying on it.** For faster CPU inference consider ONNX/CTranslate2 conversion as a later optimization.

## Step 3: Chrome extension

Open `chrome://extensions` → Developer mode → Load unpacked → select `live-subtitle-local/extension`. Open an ordinary http(s) YouTube or video tab, choose the desired Source Language, and click **Start translation**. The selected tab's audio is captured; subtitles are displayed on the same tab. Stop before starting another tab. DRM-protected players, Chrome internal pages and some embedded sites may restrict capture or injection.

## Step 4: Optional refinement via llama.cpp Vulkan

Download a compatible Qwen3-1.7B-Instruct GGUF Q4 model (verify its actual repository/model identity and licensing before use). Build/download **llama.cpp Vulkan** and run e.g.:

```powershell
llama-server.exe -m C:\models\qwen3-1.7b-q4.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 2048
```

Then restart backend with `$env:REFINE_ENABLED='true'`. The refiner sends source, raw translation, language, previous dialogue context, and optional glossary to the local OpenAI-compatible API. It only runs when a simple heuristic finds a reason. On timeout, bad response, or validation failure it falls back to raw translation. The `REFINE_TIMEOUT` default is 1.5 seconds, **not** a measured GPU latency.

## Configuration & Environment Variables

- `WHISPER_CLI`: Path to `whisper-cli.exe`.
- `WHISPER_MODEL`: Path to `ggml-*.bin` model file.
- `TRANSLATION_MODEL`: NLLB model identifier (default: `facebook/nllb-200-distilled-600M`).
- `REFINE_URL`: Endpoint for refinement LLM (default: `http://127.0.0.1:8080/v1/chat/completions`).
- `REFINE_MODEL`: Model name for refiner (default: `local`).
- `REFINE_ENABLED`: Enable/disable refinement stage (`true`/`false`).
- `REFINE_TIMEOUT`: Timeout in seconds for refinement requests (default: `1.5`).
- `ENERGY_THRESHOLD`: RMS speech threshold for segmenter (default: `0.012`).
- `SILENCE_SECONDS`: Silence duration to trigger segment end (default: `0.65`).
- `MIN_SPEECH_SECONDS`: Minimum speech duration required for valid segment (default: `0.5`).
- `MAX_SEGMENT_SECONDS`: Hard cutoff length for a segment (default: `6.0`).
- `VIETNAMESE_CLEANUP_ENABLED`: Enable optional text cleanup for Vietnamese source speech (`true`/`false`, default: `false`).

## Known Limitations & Technical Details

1. **Chinese Script Handling:** Whisper transcribes Chinese (`zh`) speech into either Simplified Chinese (`zh-CN`) or Traditional Chinese (`zh-TW`, `zh-HK`) characters. NLLB-200 maps `zh` to `zho_Hans` (Simplified Chinese script). If Traditional characters or Cantonese dialect forms are recognized, NLLB translation accuracy may degrade unless converted to Simplified script prior to translation.
2. **Voice Activity Detection:** RMS energy-based thresholding is sensitive to background music and sound effects. For noisy anime or movies, a neural VAD (e.g., Silero VAD) is recommended.
3. **Resampling:** The AudioWorklet uses basic decimation from the AudioContext rate down to 16kHz mono.
4. **Timing Metrics:** Each subtitle event emitted by the backend includes detailed latency metrics:
   ```json
   "metrics": {
     "stt_ms": 320.5,
     "translation_ms": 110.2,
     "refinement_ms": 0.0,
     "total_ms": 435.8
   }
   ```

## Test

```powershell
.\.venv\Scripts\pytest tests -v
```
