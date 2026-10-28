// AudioWorkletProcessor that forwards raw mic samples to the main thread.
class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      // Copy because the underlying buffer is reused by the audio engine.
      this.port.postMessage(input[0].slice());
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
