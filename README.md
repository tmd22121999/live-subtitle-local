# Local Live Subtitle Translator — Windows / AMD RX 7800 XT

**Status:** end-to-end MVP source code (Chrome tab audio → local WebSocket → segmenter → whisper.cpp → NLLB → adaptive llama.cpp refinement → overlay). **Not yet verified on RX 7800 XT or against real YouTube/anime playback.**

## Requirements

- Windows 10/11, Chrome or Chromium with MV3 `tabCapture` + offscreen API
- Python 3.11–3.12 recommended (PyTorch compatibility varies by Python version)
- Git, CMake, Visual Studio C++ build tools, Vulkan SDK, recent AMD driver
- Free disk space for Whisper small, NLLB-200-distilled-600M and optional Qwen3 1.7B Q4
- Local backend binds `127.0.0.1` only

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

Open `chrome://extensions` → Developer mode → Load unpacked → select `live-subtitle-local/extension`. Open an ordinary http(s) YouTube or video tab, click extension icon → Start translation. The selected tab's audio is captured; subtitles are displayed on the same tab. Stop before starting another tab. DRM-protected players, Chrome internal pages and some embedded sites may restrict capture or injection.

## Step 4: Optional refinement via llama.cpp Vulkan

Download a compatible Qwen3-1.7B-Instruct GGUF Q4 model (verify its actual repository/model identity and licensing before use). Build/download **llama.cpp Vulkan** and run e.g.:

```powershell
llama-server.exe -m C:\models\qwen3-1.7b-q4.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 2048
```

Then restart backend with `$env:REFINE_ENABLED='true'`. The refiner sends source, raw translation and up to three previous subtitle pairs to the local OpenAI-compatible API. It only runs when a simple heuristic finds a reason. On timeout, bad response, or validation failure it falls back to raw translation. The `REFINE_TIMEOUT` default is 1.5 seconds, **not** a measured GPU latency. Tune it after benchmarking. Models with thinking modes may require disabling thinking via their chat template for lower latency and cleaner output.

## Configuration

Environment variables: `WHISPER_CLI`, `WHISPER_MODEL`, `TRANSLATION_MODEL`, `REFINE_URL`, `REFINE_MODEL`, `REFINE_ENABLED`, `REFINE_TIMEOUT`, `ENERGY_THRESHOLD`, `SILENCE_SECONDS`, `MIN_SPEECH_SECONDS`, `MAX_SEGMENT_SECONDS`.

## Known limitations / important engineering work

1. **Voice activity detection uses RMS energy, not Silero VAD.** Music and effects can trigger STT. Replace it with a real VAD for anime; this is an MVP trade-off.
2. **AudioWorklet uses simple decimation rather than anti-aliased resampling.** Replace with polyphase or AudioContext resampling for high-quality capture.
3. Segmenter does not provide overlapping windows or sophisticated timestamps; subtitle display is delayed by end-of-segment detection and inference time. No source video clock synchronization.
4. NLLB uses CPU by default; its first model load can block a subtitle. No GPU path implemented for NLLB.
5. The heuristic and validation do **not** establish semantic equivalence. Wrong STT/translation can remain wrong or be made worse by refinement. For safety-critical content, do not use this pipeline.
6. The UI shows only the final subtitle, not a progressively revised partial; queue length 2 drops oldest segments when overloaded.
7. Chrome MV3 service-worker lifecycle and browser version behavior need real-world testing. Restart capture after extension reloads. No packaged installer or automatic model download script.
8. The local WebSocket checks the extension origin prefix but does not authenticate an extension ID. Keep the backend bound to loopback; add an installation-scoped token for stronger local access control.
9. This is not a production-ready extension. Add tests on real anime/YouTube and measure P50/P95 latency, word error rate and translation fidelity.

## Test

```powershell
cd live-subtitle-local
python -m pytest tests -q
```
