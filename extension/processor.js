class AudioPCM extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = []; this.targetRate = 16000; this.step = sampleRate / this.targetRate; this.phase = 0; }
  process(inputs) {
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    const len = channels[0].length;
    // Lightweight downsampling; browser AudioContext may run at 48kHz.
    for (let i = 0; i < len; i++) {
      this.phase++;
      if (this.phase >= this.step) {
        this.phase -= this.step;
        let v = 0;
        for (const ch of channels) v += ch[i] / channels.length;
        this.buffer.push(v);
        if (this.buffer.length === 1600) {
          this.port.postMessage(new Float32Array(this.buffer));
          this.buffer = [];
        }
      }
    }
    return true;
  }
}
registerProcessor('audio-pcm', AudioPCM);
