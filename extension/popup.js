const status = document.getElementById('status');
for (const type of ['start','stop']) document.getElementById(type).onclick = async () => {
  status.textContent = 'Working…';
  try { const result = await chrome.runtime.sendMessage({type}); status.textContent = result.ok ? (type==='start'?'Capturing tab audio':'Stopped') : result.error; }
  catch(e) {status.textContent = String(e);}
};
chrome.runtime.sendMessage({type:'status'}).then(r=>{status.textContent=r?.running?'Running':'Idle';});
