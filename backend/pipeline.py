"""Offline STT -> multilingual translation -> optional bounded refinement.

Architectural Notes & Documented Limitations:
---------------------------------------------
1. Whisper Model Loading Overhead vs Persistent Inference:
   Currently, WhisperCLI invokes `whisper-cli.exe` per audio segment via `subprocess.run`.
   Each execution must initialize the C++ runtime, Vulkan compute device, and reload the
   GGML model weights (~460MB for small) from disk/page cache. This introduces a baseline
   overhead of approximately 200ms - 800ms per audio segment.
   Persistent inference alternatives:
   - Run whisper.cpp's `whisper-server` (built-in HTTP/streaming endpoint) to keep model
     weights resident in GPU VRAM and stream segments or post WAV buffers.
   - Use in-process C-bindings (e.g. pywhispercpp).
   In this release, subprocess invocation is preserved for compatibility without major
   architecture replacement, but explicit source language specification (`-l <lang>`)
   eliminates Whisper's costly internal language identification pass when fixed mode is chosen.

2. Chinese Script Handling Limitations:
   Whisper speech recognition for Chinese (`zh`) may produce either Simplified Chinese
   (zh-CN) or Traditional Chinese (zh-TW, zh-HK) depending on the audio characteristics
   and vocabulary. NLLB-200 maps `zh` to `zho_Hans` (Simplified Chinese script).
   If Whisper transcribes Traditional Chinese characters, NLLB's tokenizer may encounter
   unfamiliar tokens or produce degraded translations unless converted to Simplified
   Chinese (e.g. via OpenCC) beforehand.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from collections import deque
from pathlib import Path

import httpx
import numpy as np
from config import CFG

# Centralized Whisper language code -> NLLB language code mapping.
LANGUAGE_MAP = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "vi": "vie_Latn",
}

# Alias for backwards compatibility with existing modules/tests
LANGS = LANGUAGE_MAP

# Valid source languages selectable by the client
SUPPORTED_SOURCE_LANGUAGES = {"auto", "ja", "en", "ko", "zh", "fr", "es", "vi"}


def validate_source_language(lang: str) -> bool:
    """Validate whether a language code is an accepted source language."""
    return isinstance(lang, str) and lang.lower() in SUPPORTED_SOURCE_LANGUAGES


class WhisperCLI:
    def __init__(self, cli=None, model=None):
        self.cli = cli or CFG.whisper_cli
        self.model = model or CFG.whisper_model

    def build_cmd(self, wav_path: Path, output_path: Path, language: str = 'auto') -> list[str]:
        """Generate safe command array for whisper-cli invocation."""
        lang = (language or 'auto').lower()
        if lang != 'auto' and lang not in LANGUAGE_MAP:
            raise ValueError(f'Unsupported language for transcription: {lang}')
        return [
            self.cli,
            '-m', self.model,
            '-f', str(wav_path),
            '-oj',
            '-of', str(output_path),
            '-l', lang,
            '-nt',
        ]

    def _transcribe(self, samples, language='auto'):
        lang = (language or 'auto').lower()
        if lang != 'auto' and lang not in LANGUAGE_MAP:
            raise ValueError(f'Unsupported language for transcription: {lang}')
        if not shutil.which(self.cli):
            raise RuntimeError('whisper-cli not found. Set WHISPER_CLI to the full path of whisper-cli.exe.')
        if not Path(self.model).is_file():
            raise RuntimeError('Whisper model not found. Set WHISPER_MODEL to the full path of the model .bin file.')
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / 'segment.wav'
            with wave.open(str(wav), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(CFG.sample_rate)
                f.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
            output = Path(directory) / 'result'
            cmd = self.build_cmd(wav, output, language=lang)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
            if proc.returncode:
                raise RuntimeError(f'whisper.cpp failed: {proc.stderr[-500:]}')
            data = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
            text = ' '.join(s.get('text', '').strip() for s in data.get('transcription', []))
            # whisper.cpp JSON uses result.language in current releases.
            detected = data.get('result', {}).get('language', 'auto')
            actual_lang = lang if lang != 'auto' else (detected or 'auto')
            return text.strip(), actual_lang

    async def transcribe(self, samples, language='auto'):
        return await asyncio.to_thread(self._transcribe, samples, language)


class NLLBTranslator:
    def __init__(self, model_name=None):
        self.model_name = model_name or CFG.translation_model
        self.tokenizer = None
        self.model = None
        self.lock = asyncio.Lock()

    def _load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()

    def _translate(self, text, language):
        # Vietnamese translation bypass: preserve original transcribed text
        if language == 'vi':
            return text
        src = LANGUAGE_MAP.get(language)
        if not src:
            raise ValueError(f'Unsupported language: {language}')
        self._load()
        import torch
        self.tokenizer.src_lang = src
        batch = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        target_id = self.tokenizer.convert_tokens_to_ids('vie_Latn')
        with torch.inference_mode():
            tokens = self.model.generate(**batch, forced_bos_token_id=target_id, max_new_tokens=180)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    async def translate(self, text, language):
        async with self.lock:
            return await asyncio.to_thread(self._translate, text, language)


class OllamaTranslator:
    def __init__(self, model_name):
        self.model_name = model_name

    async def translate(self, text, language):
        if language == 'vi':
            return text

        system = (f'Translate the following subtitle from language code {language} into natural Vietnamese. '
                  'Preserve meaning, names, and numbers. Return only the Vietnamese subtitle.')
        try:
            async with httpx.AsyncClient(timeout=CFG.ollama_translation_timeout) as client:
                response = await client.post(CFG.ollama_translation_url, json={
                    'model': self.model_name,
                    'messages': [
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': text},
                    ],
                    'stream': False,
                })
                response.raise_for_status()
                translation = response.json()['choices'][0]['message']['content']
        except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError) as exc:
            raise RuntimeError(f'Ollama translation failed for {self.model_name}: {exc}') from exc

        if not isinstance(translation, str) or not translation.strip():
            raise RuntimeError(f'Ollama returned an empty subtitle for {self.model_name}')
        return translation.strip()


class AdaptiveRefiner:
    def __init__(self, url=None, timeout=None):
        self.url = url or CFG.refine_url
        self.timeout = timeout if timeout is not None else CFG.refine_timeout

    @staticmethod
    def should_refine(source, translated, context):
        if not source or not translated or len(translated) < 12:
            return False
        suspicious = ('một người mà', 'không thể giúp được', 'đã được làm bởi', 'cái điều mà')
        return (len(translated) > 65 or any(s in translated.lower() for s in suspicious)
                or (len(source) > 40 and bool(context)))

    @staticmethod
    def safe(candidate, raw):
        if not candidate or len(candidate) > max(140, len(raw) * 1.65):
            return False
        if len(candidate) < max(3, len(raw) * 0.35):
            return False
        if any(mark in candidate for mark in ('```', '<|', '\n\n')):
            return False
        # Preserve digit sequences; these checks are defensive, not semantic equivalence.
        if re.findall(r'\d+', raw) != re.findall(r'\d+', candidate):
            return False
        return True

    async def refine(self, source, raw, language, context, glossary=None):
        if not CFG.refine_enabled or not self.should_refine(source, raw, context):
            return raw, False
        # Vietnamese bypass: do not hallucinate or alter original Vietnamese transcription
        # unless an explicit cleanup feature is enabled
        if language == 'vi' and not getattr(CFG, 'vietnamese_cleanup_enabled', False):
            return raw, False
        system = ('Bạn biên tập phụ đề tiếng Việt. Giữ nguyên nghĩa câu gốc, tên riêng, '
                  'số và sắc thái; viết tự nhiên, ngắn, không tự thêm nội dung. '
                  'Nếu bản dịch đã ổn, giữ nguyên. Chỉ xuất câu phụ đề tiếng Việt.')
        context_str = " | ".join(context[-3:]) if context else ""
        user = f'Ngôn ngữ: {language}\nCâu gốc: {source}\nDịch thô: {raw}\nNgữ cảnh trước: {context_str}'
        if glossary:
            glossary_str = ", ".join(f"{k} -> {v}" for k, v in glossary.items())
            user += f'\nThuật ngữ: {glossary_str}'
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.url, json={'model': CFG.refine_model,
                    'messages': [{'role':'system','content':system},{'role':'user','content':user}],
                    'temperature':0.1,'max_tokens':256,'stream':False})
                r.raise_for_status()
                candidate = r.json()['choices'][0]['message']['content'].strip().strip('"')
            if self.safe(candidate, raw):
                return candidate, candidate != raw
        except (httpx.HTTPError, KeyError, ValueError, IndexError) as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
            pass
        return raw, False


class Pipeline:
    def __init__(self, stt=None, translator=None, refiner=None, glossary=None):
        self.stt = stt or WhisperCLI()
        self.translator = translator or (
            OllamaTranslator(CFG.ollama_translation_model)
            if CFG.ollama_translation_model else NLLBTranslator()
        )
        self.refiner = refiner or AdaptiveRefiner()
        self.glossary = glossary or {}
        self.context = deque(maxlen=5)

    async def process(self, samples, source_language='auto'):
        t_start = time.perf_counter()

        # 1. Speech-to-text
        t_stt = time.perf_counter()
        source, language = await self.stt.transcribe(samples, language=source_language)
        stt_ms = round((time.perf_counter() - t_stt) * 1000, 1)

        if not source:
            return None

        # Check if detected or configured language is supported by translation mapping
        if language not in LANGUAGE_MAP:
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)
            return {
                'error': f'Language {language} is not supported',
                'source': source,
                'language': language,
                'metrics': {
                    'stt_ms': stt_ms,
                    'translation_ms': 0.0,
                    'refinement_ms': 0.0,
                    'total_ms': total_ms,
                }
            }

        # 2. Translation
        t_trans = time.perf_counter()
        raw = await self.translator.translate(source, language)
        trans_ms = round((time.perf_counter() - t_trans) * 1000, 1)

        # 3. Adaptive Refinement
        t_refine = time.perf_counter()
        result, refined = await self.refiner.refine(
            source, raw, language, list(self.context), glossary=self.glossary
        )
        refine_ms = round((time.perf_counter() - t_refine) * 1000, 1)

        self.context.append(f'{source} => {result}')
        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return {
            'source': source,
            'language': language,
            'raw_vi': raw,
            'text_vi': result,
            'refined': refined,
            'metrics': {
                'stt_ms': stt_ms,
                'translation_ms': trans_ms,
                'refinement_ms': refine_ms,
                'total_ms': total_ms,
            }
        }
