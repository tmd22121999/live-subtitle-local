const ALLOWED_LANGUAGES = ['auto', 'ja', 'en', 'ko', 'zh', 'fr', 'es', 'vi'];
const LANGUAGE_NAMES = {
  auto: 'Auto Detect',
  ja: 'Japanese',
  en: 'English',
  ko: 'Korean',
  zh: 'Chinese',
  fr: 'French',
  es: 'Spanish',
  vi: 'Vietnamese',
};

const statusEl = document.getElementById('status');
const sourceSelect = document.getElementById('source-lang');
const detectedEl = document.getElementById('detected-lang');

function updateDetectedDisplay(selectedLang, detectedLang) {
  if (selectedLang === 'auto' && detectedLang && detectedLang !== 'auto') {
    const name = LANGUAGE_NAMES[detectedLang] || detectedLang;
    detectedEl.textContent = `Detected: ${name} (${detectedLang})`;
    detectedEl.style.display = 'block';
  } else {
    detectedEl.style.display = 'none';
  }
}

// 1. Initialize language selection from chrome.storage.local
chrome.storage.local.get(['source_language'], res => {
  let lang = res?.source_language;
  if (!ALLOWED_LANGUAGES.includes(lang)) {
    lang = 'auto';
  }
  sourceSelect.value = lang;
});

// 2. Query initial status from background
chrome.runtime.sendMessage({ type: 'status' }).then(r => {
  statusEl.textContent = r?.running ? 'Running' : 'Idle';
  if (r?.detected_language) {
    updateDetectedDisplay(sourceSelect.value, r.detected_language);
  }
}).catch(() => {
  statusEl.textContent = 'Idle';
});

// 3. Handle language dropdown change
sourceSelect.addEventListener('change', async () => {
  const selected = sourceSelect.value;
  if (!ALLOWED_LANGUAGES.includes(selected)) return;
  await chrome.storage.local.set({ source_language: selected });
  // Update backend / background dynamically
  chrome.runtime.sendMessage({ type: 'set-language', source_language: selected }).catch(() => {});
  // Query status to refresh display
  chrome.runtime.sendMessage({ type: 'status' }).then(r => {
    updateDetectedDisplay(selected, r?.detected_language);
  }).catch(() => {});
});

// 4. Handle start and stop buttons
for (const type of ['start', 'stop']) {
  document.getElementById(type).onclick = async () => {
    statusEl.textContent = 'Working…';
    try {
      const selected = sourceSelect.value;
      const validLang = ALLOWED_LANGUAGES.includes(selected) ? selected : 'auto';
      // Ensure storage is saved
      await chrome.storage.local.set({ source_language: validLang });

      const msg = type === 'start'
        ? { type, source_language: validLang }
        : { type };
      const result = await chrome.runtime.sendMessage(msg);
      statusEl.textContent = result.ok
        ? (type === 'start' ? 'Capturing tab audio' : 'Stopped')
        : (result.error || 'Failed');
    } catch (e) {
      statusEl.textContent = String(e.message || e);
    }
  };
}

// 5. Listen for subtitle events to update detected language in real time
chrome.runtime.onMessage.addListener(msg => {
  if (msg.type === 'subtitle') {
    const lang = msg.data?.language;
    if (lang) {
      updateDetectedDisplay(sourceSelect.value, lang);
    }
  }
});
