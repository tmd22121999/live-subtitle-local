import os
from dataclasses import dataclass

@dataclass
class Config:
    whisper_cli: str = os.getenv('WHISPER_CLI', r'\whisper.cpp\build\bin\Release\whisper-cli.exe')
    whisper_model: str = os.getenv('WHISPER_MODEL', r'\whisper.cpp\models\ggml-small.bin')
    translation_model: str = os.getenv('TRANSLATION_MODEL', 'facebook/nllb-200-distilled-600M')
    refine_url: str = os.getenv('REFINE_URL', 'http://127.0.0.1:8080/v1/chat/completions')
    refine_model: str = os.getenv('REFINE_MODEL', 'local')
    refine_enabled: bool = os.getenv('REFINE_ENABLED', 'true').lower() == 'true'
    refine_timeout: float = float(os.getenv('REFINE_TIMEOUT', '1.5'))
    min_speech_seconds: float = float(os.getenv('MIN_SPEECH_SECONDS', '0.5'))
    max_segment_seconds: float = float(os.getenv('MAX_SEGMENT_SECONDS', '6'))
    silence_seconds: float = float(os.getenv('SILENCE_SECONDS', '0.65'))
    energy_threshold: float = float(os.getenv('ENERGY_THRESHOLD', '0.012'))
    sample_rate: int = 16000

CFG = Config()
