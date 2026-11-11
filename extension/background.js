const OFFSCREEN = 'offscreen.html';
const ALLOWED_LANGUAGES = ['auto', 'ja', 'en', 'ko', 'zh', 'fr', 'es', 'vi'];
let currentTab = null;
let currentSourceLang = 'auto';
let lastDetectedLang = null;

// Initialize language from storage
chrome.storage.local.get(['source_language'], res => {
  if (res?.source_language && ALLOWED_LANGUAGES.includes(res.source_language)) {
    currentSourceLang = res.source_language;
  }
});

async function ensureOffscreen() {
  const url = chrome.runtime.getURL(OFFSCREEN);
  const contexts = await chrome.runtime.getContexts({contextTypes:['OFFSCREEN_DOCUMENT'], documentUrls:[url]});
  if (!contexts.length) await chrome.offscreen.createDocument({url:OFFSCREEN, reasons:['USER_MEDIA'], justification:'Capture audio from selected tab for local subtitle transcription'});
}

chrome.runtime.onMessage.addListener((msg, sender, respond) => {
  (async () => {
    if (msg.type === 'start') {
      if (currentTab !== null) throw Error('Stop the existing capture first');
      const [tab] = await chrome.tabs.query({active:true, currentWindow:true});
      if (!tab?.id || !/^https?:/.test(tab.url || '')) throw Error('Open an http(s) video tab first');

      const lang = msg.source_language && ALLOWED_LANGUAGES.includes(msg.source_language)
        ? msg.source_language
        : currentSourceLang;
      currentSourceLang = lang;
      lastDetectedLang = null;

      await ensureOffscreen();
      const streamId = await chrome.tabCapture.getMediaStreamId({targetTabId:tab.id});
      await chrome.scripting.executeScript({target:{tabId:tab.id}, files:['overlay.js']});
      currentTab = tab.id;
      try {
        const started = await chrome.runtime.sendMessage({
          type: 'offscreen-start',
          streamId,
          tabId: tab.id,
          source_language: currentSourceLang
        });
        if (!started?.ok) throw Error(started?.error || 'Audio capture failed');
      } catch (error) {
        currentTab = null;
        await chrome.runtime.sendMessage({type:'offscreen-stop'}).catch(()=>{});
        await chrome.tabs.sendMessage(tab.id, {type:'subtitle-stop'}).catch(()=>{});
        throw error;
      }
      respond({ok:true});
    } else if (msg.type === 'stop') {
      await chrome.runtime.sendMessage({type:'offscreen-stop'}).catch(()=>{});
      if (currentTab !== null) chrome.tabs.sendMessage(currentTab, {type:'subtitle-stop'}).catch(()=>{});
      currentTab = null;
      lastDetectedLang = null;
      respond({ok:true});
    } else if (msg.type === 'set-language') {
      const lang = msg.source_language;
      if (ALLOWED_LANGUAGES.includes(lang)) {
        currentSourceLang = lang;
        if (currentTab !== null) {
          await chrome.runtime.sendMessage({type:'offscreen-set-language', source_language:currentSourceLang}).catch(()=>{});
        }
      }
      respond({ok:true});
    } else if (msg.type === 'subtitle') {
      if (msg.data?.language) {
        lastDetectedLang = msg.data.language;
      }
      if (currentTab !== null) {
        chrome.tabs.sendMessage(currentTab, msg).catch(()=>{});
      }
      respond({ok:true});
    } else if (msg.type === 'status') {
      respond({
        ok: true,
        running: currentTab !== null,
        source_language: currentSourceLang,
        detected_language: lastDetectedLang
      });
    }
  })().catch(e => respond({ok:false,error:String(e.message||e)}));
  return true;
});

chrome.tabs.onRemoved.addListener(id => {
  if (id === currentTab) {
    chrome.runtime.sendMessage({type:'offscreen-stop'}).catch(()=>{});
    currentTab = null;
    lastDetectedLang = null;
  }
});
