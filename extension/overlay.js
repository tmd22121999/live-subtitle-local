(() => {
  if (window.__localViSubtitle) return;
  window.__localViSubtitle = true;
  const host = document.createElement('div');
  host.id = 'local-vi-subtitle-host';
  host.style.cssText = 'position:fixed;bottom:11%;left:50%;transform:translateX(-50%);z-index:2147483647;pointer-events:none;width:min(85vw,1050px);text-align:center;';
  const root = host.attachShadow({mode:'closed'});
  const text = document.createElement('div');
  text.style.cssText = 'display:inline-block;color:white;background:rgba(0,0,0,.76);font:600 25px/1.45 system-ui,sans-serif;padding:8px 15px;border-radius:9px;text-shadow:0 2px 3px #000;white-space:pre-wrap;overflow-wrap:anywhere;';
  text.textContent = 'Đang chờ phụ đề tiếng Việt…';
  root.append(text);
  const placeOverlay = () => (document.fullscreenElement || document.documentElement).append(host);
  placeOverlay();
  document.addEventListener('fullscreenchange', placeOverlay);
  let clearTimer;
  const onMessage = msg => {
    if (msg.type === 'subtitle-stop') {
      clearTimeout(clearTimer);
      document.removeEventListener('fullscreenchange', placeOverlay);
      chrome.runtime.onMessage.removeListener(onMessage);
      host.remove();
      window.__localViSubtitle = false;
      return;
    }
    if (msg.type !== 'subtitle') return;
    const data = msg.data;
    if (data?.type === 'subtitle' && data.text_vi) {
      text.textContent = data.text_vi;
      text.style.display = 'inline-block';
      clearTimeout(clearTimer);
      clearTimer = setTimeout(()=>{text.style.display='none';}, Math.max(2600,Math.min(7000,data.text_vi.length*85)));
    } else if (data?.type === 'error' || data?.error) {
      clearTimeout(clearTimer);
      text.textContent = 'Lỗi phụ đề: ' + (data.message || data.error);
      text.style.display = 'inline-block';
    }
  };
  chrome.runtime.onMessage.addListener(onMessage);
})();
