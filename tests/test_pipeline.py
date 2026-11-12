import json
import sys
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from config import CFG
from main import Segmenter, TranslationSession
from pipeline import (
    AdaptiveRefiner,
    LANGUAGE_MAP,
    LANGS,
    NLLBTranslator,
    Pipeline,
    SUPPORTED_SOURCE_LANGUAGES,
    WhisperCLI,
    validate_source_language,
)


@pytest.mark.parametrize('language, source, expected', [
    ('en', 'See you tomorrow.', 'Hẹn gặp lại ngày mai.'),
    ('vi', 'Chào bạn.', 'Chào bạn.'),
])
def test_pipeline_uses_configured_ollama_translation_model(monkeypatch, language, source, expected):
    monkeypatch.setattr(CFG, 'ollama_translation_model', 'custom/subtitles:local', raising=False)
    monkeypatch.setattr(CFG, 'refine_enabled', False)
    stt = MagicMock()
    stt.transcribe = AsyncMock(return_value=(source, language))
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            'choices': [{'message': {'content': ' Hẹn gặp lại ngày mai. '}}],
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(NLLBTranslator, 'translate', AsyncMock(side_effect=lambda text, lang: text))
    with patch('pipeline.httpx.AsyncClient', return_value=client):
        result = run_coro(Pipeline(stt=stt).process(np.zeros(1600)))

    assert result['text_vi'] == expected
    if language == 'vi':
        assert requests == []
    else:
        assert len(requests) == 1
        assert requests[0]['model'] == 'custom/subtitles:local'
        assert requests[0]['messages'][-1]['content'] == source


def run_coro(coro):
    """Executes a coroutine to completion without requiring a network event loop."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


# 1. Existing tests preserved
def test_languages():
    assert LANGS['ja'] == 'jpn_Jpan'
    assert LANGS['en'] == 'eng_Latn'


def test_refiner_rules():
    assert not AdaptiveRefiner.should_refine('Hi', 'Chào', [])
    assert AdaptiveRefiner.should_refine('Long sentence', 'Đây là một người mà tôi không thể giúp được', [])
    assert not AdaptiveRefiner.safe('Có 3 người', 'Có 2 người')
    assert AdaptiveRefiner.safe('Tôi không biết nữa.', 'Tôi không biết.')


def test_segmenter_speech_then_silence():
    s = Segmenter()
    for _ in range(6):
        assert s.feed(np.ones(1600, dtype=np.float32) * 0.1) is None
    result = None
    for _ in range(8):
        result = s.feed(np.zeros(1600, dtype=np.float32))
        if result is not None:
            break
    assert result is not None
    assert len(result) >= 16000


def test_segmenter_ignores_short_noise():
    s = Segmenter()
    for _ in range(2):
        s.feed(np.ones(1600, dtype=np.float32) * 0.1)
    outputs = [s.feed(np.zeros(1600, dtype=np.float32)) for _ in range(8)]
    assert all(x is None for x in outputs)


# 2. Centralized NLLB Language Mapping
def test_language_map_completeness():
    required = {
        "ja": "jpn_Jpan",
        "en": "eng_Latn",
        "ko": "kor_Hang",
        "zh": "zho_Hans",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "vi": "vie_Latn",
    }
    for k, v in required.items():
        assert LANGUAGE_MAP[k] == v
    assert LANGS is LANGUAGE_MAP


# 3. Language Code Validation
def test_language_code_validation():
    for lang in ['auto', 'ja', 'en', 'ko', 'zh', 'fr', 'es', 'vi', 'JA', 'En']:
        assert validate_source_language(lang) is True

    for lang in ['de', 'ru', 'it', 'invalid', '123', '', None, 12]:
        assert validate_source_language(lang) is False


# 4. Whisper Argument Generation & Safe Array Construction
def test_whisper_argument_generation():
    cli = WhisperCLI(cli='whisper-cli.exe', model='model.bin')
    wav = Path('tmp.wav')
    out = Path('result')

    # Explicit languages
    for lang in ['ja', 'en', 'ko', 'zh', 'fr', 'es', 'vi']:
        cmd = cli.build_cmd(wav, out, language=lang)
        assert isinstance(cmd, list)
        assert '-l' in cmd
        idx = cmd.index('-l')
        assert cmd[idx + 1] == lang
        assert '-m' in cmd and cmd[cmd.index('-m') + 1] == 'model.bin'
        assert '-f' in cmd and cmd[cmd.index('-f') + 1] == str(wav)

    # Auto detect
    cmd_auto = cli.build_cmd(wav, out, language='auto')
    assert cmd_auto[cmd_auto.index('-l') + 1] == 'auto'

    # Unsupported language must raise ValueError and never execute
    with pytest.raises(ValueError, match="Unsupported language for transcription"):
        cli.build_cmd(wav, out, language='unsupported_lang')


# 5. Vietnamese Translation Bypass
def test_vietnamese_translation_bypass():
    translator = NLLBTranslator()
    # Bypass must return source text directly without loading model
    assert translator._translate('Xin chào thế giới', 'vi') == 'Xin chào thế giới'


def test_vietnamese_pipeline_bypass():
    mock_stt = MagicMock()
    mock_stt.transcribe = AsyncMock(return_value=('Tôi là người Việt', 'vi'))
    mock_translator = MagicMock()
    mock_translator.translate = AsyncMock(return_value='Tôi là người Việt')
    mock_refiner = MagicMock()
    mock_refiner.refine = AsyncMock(return_value=('Tôi là người Việt', False))

    pipe = Pipeline(stt=mock_stt, translator=mock_translator, refiner=mock_refiner)
    samples = np.zeros(1600, dtype=np.float32)
    res = run_coro(pipe.process(samples, source_language='vi'))

    assert res['source'] == 'Tôi là người Việt'
    assert res['language'] == 'vi'
    assert res['raw_vi'] == 'Tôi là người Việt'
    assert res['text_vi'] == 'Tôi là người Việt'
    assert res['refined'] is False
    assert 'metrics' in res


# 6. Auto-Detection Behavior
def test_auto_detection_behavior():
    mock_stt = MagicMock()
    mock_stt.transcribe = AsyncMock(return_value=('Hello world', 'en'))
    mock_translator = MagicMock()
    mock_translator.translate = AsyncMock(return_value='Xin chào thế giới')
    mock_refiner = MagicMock()
    mock_refiner.refine = AsyncMock(return_value=('Xin chào thế giới', False))

    pipe = Pipeline(stt=mock_stt, translator=mock_translator, refiner=mock_refiner)
    samples = np.zeros(1600, dtype=np.float32)
    res = run_coro(pipe.process(samples, source_language='auto'))

    mock_stt.transcribe.assert_awaited_once_with(samples, language='auto')
    mock_translator.translate.assert_awaited_once_with('Hello world', 'en')
    assert res['language'] == 'en'
    assert res['text_vi'] == 'Xin chào thế giới'


# 7. Unsupported Language Handling (No Crashing)
def test_unsupported_language_handling():
    mock_stt = MagicMock()
    # Whisper detected a language code not in LANGUAGE_MAP
    mock_stt.transcribe = AsyncMock(return_value=('Some text', 'de'))
    mock_translator = MagicMock()
    mock_refiner = MagicMock()

    pipe = Pipeline(stt=mock_stt, translator=mock_translator, refiner=mock_refiner)
    samples = np.zeros(1600, dtype=np.float32)
    res = run_coro(pipe.process(samples, source_language='auto'))

    assert 'error' in res
    assert 'Language de is not supported' in res['error']
    assert res['language'] == 'de'
    # Translator must not have been invoked
    mock_translator.translate.assert_not_called()
    mock_refiner.refine.assert_not_called()

    # NLLBTranslator direct call with unsupported language raises ValueError
    translator = NLLBTranslator()
    with pytest.raises(ValueError, match="Unsupported language: xx"):
        translator._translate("Hello", "xx")


# 8. Adaptive Refinement Preservation & Glossary
def test_adaptive_refinement_preservation():
    refiner = AdaptiveRefiner()
    # Disabled refinement
    with patch.object(CFG, 'refine_enabled', False):
        candidate, refined = run_coro(refiner.refine("source", "raw translation", "ja", []))
        assert candidate == "raw translation"
        assert refined is False

    # Vietnamese bypass in refiner
    candidate, refined = run_coro(refiner.refine("Câu tiếng Việt", "Câu tiếng Việt", "vi", []))
    assert candidate == "Câu tiếng Việt"
    assert refined is False


# 9. Pipeline Metrics
def test_pipeline_metrics():
    mock_stt = MagicMock()
    mock_stt.transcribe = AsyncMock(return_value=('Konnichiwa', 'ja'))
    mock_translator = MagicMock()
    mock_translator.translate = AsyncMock(return_value='Xin chào')
    mock_refiner = MagicMock()
    mock_refiner.refine = AsyncMock(return_value=('Xin chào', False))

    pipe = Pipeline(stt=mock_stt, translator=mock_translator, refiner=mock_refiner)
    samples = np.zeros(1600, dtype=np.float32)
    res = run_coro(pipe.process(samples, source_language='ja'))

    metrics = res['metrics']
    assert 'stt_ms' in metrics
    assert 'translation_ms' in metrics
    assert 'refinement_ms' in metrics
    assert 'total_ms' in metrics
    assert all(isinstance(v, (int, float)) and v >= 0 for v in metrics.values())


# 10. Translation Session & Dynamic Configuration
def test_session_default_backward_compatibility():
    """Verify session defaults to 'auto' for backwards compatibility."""
    session = TranslationSession()
    assert session.source_language == 'auto'


def test_session_config_valid_and_invalid():
    """Verify session accepts valid language codes and rejects invalid ones without changing state."""
    session = TranslationSession()

    # Valid config
    err = session.handle_text_message(json.dumps({'type': 'config', 'source_language': 'ja'}))
    assert err is None
    assert session.source_language == 'ja'

    # Valid config case-insensitive
    err = session.handle_text_message(json.dumps({'type': 'config', 'source_language': 'EN'}))
    assert err is None
    assert session.source_language == 'en'

    # Invalid config should return error and preserve current setting ('en')
    err = session.handle_text_message(json.dumps({'type': 'config', 'source_language': 'unsupported'}))
    assert err is not None
    assert err['type'] == 'error'
    assert 'Invalid source_language: unsupported' in err['message']
    assert session.source_language == 'en'


def test_session_atomic_language_switch_between_segments():
    """Verify that when language is changed during active session, segments are tagged atomically."""
    mock_segmenter = MagicMock()
    seg1 = np.ones(1600, dtype=np.float32)
    seg2 = np.ones(1600, dtype=np.float32) * 2

    session = TranslationSession(segmenter=mock_segmenter)

    # 1. Configured as Japanese
    session.handle_text_message(json.dumps({'type': 'config', 'source_language': 'ja'}))
    mock_segmenter.feed.return_value = seg1

    item1 = session.handle_audio_chunk(np.zeros(1600, dtype=np.float32))
    assert item1 is not None
    segment, lang1 = item1
    assert lang1 == 'ja'
    assert np.array_equal(segment, seg1)

    # 2. Switch language to English mid-session
    session.handle_text_message(json.dumps({'type': 'config', 'source_language': 'en'}))
    mock_segmenter.feed.return_value = seg2

    item2 = session.handle_audio_chunk(np.zeros(1600, dtype=np.float32))
    assert item2 is not None
    segment, lang2 = item2
    assert lang2 == 'en'
    assert np.array_equal(segment, seg2)

    # Ensure in-flight segment 1 was not affected
    assert lang1 == 'ja'


def test_session_reset_message():
    """Verify reset message clears segmenter and context."""
    session = TranslationSession()
    session.pipeline.context.append('test => context')
    assert len(session.pipeline.context) == 1

    session.handle_text_message(json.dumps({'type': 'reset'}))
    assert len(session.pipeline.context) == 0
