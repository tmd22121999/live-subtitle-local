import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import numpy as np
from pipeline import AdaptiveRefiner, Pipeline, LANGS
from main import Segmenter

def test_languages():
    assert LANGS['ja'] == 'jpn_Jpan'
    assert LANGS['en'] == 'eng_Latn'

def test_refiner_rules():
    assert not AdaptiveRefiner.should_refine('Hi', 'Chào', [])
    assert AdaptiveRefiner.should_refine('Long sentence', 'Đây là một người mà tôi không thể giúp được', [])
    assert not AdaptiveRefiner.safe('Có 3 người', 'Có 2 người')
    assert AdaptiveRefiner.safe('Tôi không biết nữa.', 'Tôi không biết.')

def test_segmenter_speech_then_silence():
    from config import CFG
    s = Segmenter()
    for _ in range(6): assert s.feed(np.ones(1600, dtype=np.float32)*.1) is None
    result = None
    for _ in range(8):
        result = s.feed(np.zeros(1600, dtype=np.float32))
        if result is not None: break
    assert result is not None
    assert len(result) >= 16000

def test_segmenter_ignores_short_noise():
    s = Segmenter()
    for _ in range(2): s.feed(np.ones(1600, dtype=np.float32)*.1)
    outputs = [s.feed(np.zeros(1600, dtype=np.float32)) for _ in range(8)]
    assert all(x is None for x in outputs)
