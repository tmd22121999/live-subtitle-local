const OFFSCREEN = 'offscreen.html';
let currentTab = null;

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
      await ensureOffscreen();
      const streamId = await chrome.tabCapture.getMediaStreamId({targetTabId:tab.id});
      await chrome.scripting.executeScript({target:{tabId:tab.id}, files:['overlay.js']});
      currentTab = tab.id;
      try {
        const started = await chrome.runtime.sendMessage({type:'offscreen-start', streamId, tabId:tab.id});
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
      respond({ok:true});
    } else if (msg.type === 'subtitle' && currentTab !== null) {
      chrome.tabs.sendMessage(currentTab, msg).catch(()=>{});
      respond({ok:true});
    } else if (msg.type === 'status') {
      respond({ok:true, running:currentTab !== null});
    }
  })().catch(e => respond({ok:false,error:String(e.message||e)}));
  return true;
});
chrome.tabs.onRemoved.addListener(id => {
  if (id === currentTab) {
    chrome.runtime.sendMessage({type:'offscreen-stop'}).catch(()=>{});
    currentTab = null;
  }
});
