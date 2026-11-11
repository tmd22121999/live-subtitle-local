let stream = null, context = null, source = null, worklet = null, socket = null;
async function stop() {
  if (worklet) { worklet.port.onmessage = null; worklet.disconnect(); worklet = null; }
  if (source) { source.disconnect(); source = null; }
  if (stream) { stream.getTracks().forEach(t=>t.stop()); stream = null; }
  if (context) { await context.close(); context = null; }
  if (socket) { socket.close(); socket = null; }
}
async function start(id) {
  await stop();
  stream = await navigator.mediaDevices.getUserMedia({audio:{mandatory:{chromeMediaSource:'tab',chromeMediaSourceId:id}},video:false});
  context = new AudioContext();
  await context.audioWorklet.addModule('processor.js');
  source = context.createMediaStreamSource(stream);
  worklet = new AudioWorkletNode(context, 'audio-pcm');
  source.connect(worklet);
  // Restore audio playback; Chrome tabCapture otherwise mutes the captured tab.
  source.connect(context.destination);
  socket = new WebSocket('ws://127.0.0.1:8766/ws');
  socket.binaryType = 'arraybuffer';
  socket.onmessage = event => {
    try {
      const data = JSON.parse(event.data);
      chrome.runtime.sendMessage({type:'subtitle',data}).catch(()=>{});
    } catch (_) {}
  };
  socket.onerror = () => chrome.runtime.sendMessage({type:'subtitle',data:{type:'error',message:'Cannot connect to local backend on port 8766'}}).catch(()=>{});
  worklet.port.onmessage = ({data}) => {
    if (socket?.readyState === WebSocket.OPEN && socket.bufferedAmount < 1000000) socket.send(data.buffer);
  };
  await context.resume();
}
chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.type === 'offscreen-start') {start(msg.streamId).then(()=>reply({ok:true})).catch(e=>reply({ok:false,error:String(e)}));return true;}
  if (msg.type === 'offscreen-stop') {stop().then(()=>reply({ok:true}));return true;}
});
