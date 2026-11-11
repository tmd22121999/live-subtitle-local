"""Offline STT -> multilingual translation -> optional bounded refinement."""
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections import deque
from pathlib import Path

import httpx
import numpy as np
from config import CFG

# Whisper language -> NLLB language. Unmapped languages are rejected explicitly.
LANGS = {'en':'eng_Latn','ja':'jpn_Jpan','ko':'kor_Hang','zh':'zho_Hans',
         'fr':'fra_Latn','de':'deu_Latn','es':'spa_Latn','ru':'rus_Cyrl',
         'th':'tha_Thai','id':'ind_Latn','vi':'vie_Latn','pt':'por_Latn',
         'it':'ita_Latn','ar':'arb_Arab','hi':'hin_Deva','tr':'tur_Latn',
         'pl':'pol_Latn','nl':'nld_Latn','uk':'ukr_Cyrl'}

class WhisperCLI:
    def __init__(self, cli=None, model=None):
        self.cli = cli or CFG.whisper_cli
        self.model = model or CFG.whisper_model

    def _transcribe(self, samples):
        if not shutil.which(self.cli):
            raise RuntimeError('whisper-cli not found. Set WHISPER_CLI to the full path of whisper-cli.exe.')
        if not Path(self.model).is_file():
            raise RuntimeError('Whisper model not found. Set WHISPER_MODEL to the full path of the model .bin file.')
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / 'segment.wav'
            with wave.open(str(wav), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(CFG.sample_rate)
                f.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
            output = Path(directory) / 'result'
            proc = subprocess.run([self.cli, '-m', self.model, '-f', str(wav),
                '-oj', '-of', str(output), '-l', 'auto', '-nt'],
                capture_output=True, text=True, timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if proc.returncode:
                raise RuntimeError(f'whisper.cpp failed: {proc.stderr[-500:]}')
            data = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
            text = ' '.join(s.get('text', '').strip() for s in data.get('transcription', []))
            # whisper.cpp JSON uses result.language in current releases.
            lang = data.get('result', {}).get('language', 'auto')
            return text.strip(), lang

    async def transcribe(self, samples):
        return await asyncio.to_thread(self._transcribe, samples)

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
        self._load()
        src = LANGS.get(language)
        if not src:
            raise ValueError(f'Unsupported language: {language}')
        if language == 'vi':
            return text
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

    async def refine(self, source, raw, language, context):
        if not CFG.refine_enabled or not self.should_refine(source, raw, context):
            return raw, False
        system = ('Bạn biên tập phụ đề tiếng Việt. Giữ nguyên nghĩa câu gốc, tên riêng, '
                  'số và sắc thái; viết tự nhiên, ngắn, không tự thêm nội dung. '
                  'Nếu bản dịch đã ổn, giữ nguyên. Chỉ xuất câu phụ đề tiếng Việt.')
        user = f'Ngôn ngữ: {language}\nCâu gốc: {source}\nDịch thô: {raw}\nNgữ cảnh trước: {" | ".join(context[-3:])}'
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.url, json={'model': CFG.refine_model,
                    'messages': [{'role':'system','content':system},{'role':'user','content':user}],
                    'temperature':0.1,'max_tokens':100,'stream':False})
                r.raise_for_status()
                candidate = r.json()['choices'][0]['message']['content'].strip().strip('"')
            if self.safe(candidate, raw):
                return candidate, candidate != raw
        except (httpx.HTTPError, KeyError, ValueError, IndexError):
            pass
        return raw, False

class Pipeline:
    def __init__(self, stt=None, translator=None, refiner=None):
        self.stt = stt or WhisperCLI()
        self.translator = translator or NLLBTranslator()
        self.refiner = refiner or AdaptiveRefiner()
        self.context = deque(maxlen=5)

    async def process(self, samples):
        source, language = await self.stt.transcribe(samples)
        if not source:
            return None
        if language not in LANGS:
            return {'error':f'Language {language} is not configured', 'source':source}
        raw = await self.translator.translate(source, language)
        result, refined = await self.refiner.refine(source, raw, language, list(self.context))
        self.context.append(f'{source} => {result}')
        return {'source':source, 'language':language, 'raw_vi':raw,
                'text_vi':result, 'refined':refined}
