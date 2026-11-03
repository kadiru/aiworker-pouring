import { AppStreamer, StreamType, LogLevel, EventAction, EventStatus } from '@nvidia/ov-web-rtc';
const status = document.querySelector('#status');
const button = document.querySelector('#connect');
const video = document.querySelector('#remote-video');
let requested = false;
const simState = document.querySelector('#sim-state');
async function refreshState() {
  try {
    const response = await fetch('/health', {cache: 'no-store'});
    if (!response.ok) throw new Error('Simulator unavailable');
    const state = await response.json();
    button.disabled = requested || !state.ready;
    document.querySelectorAll('[data-command]').forEach(control => {control.disabled = !state.ready;});
    simState.textContent = state.ready ? `${state.paused ? 'Paused' : 'Running'} · ${state.sim_time_s.toFixed(1)} s simulated` : 'Simulator warming up…';
    if (state.ready && !requested) status.textContent = 'Ready to connect';
  } catch {
    simState.textContent = 'Simulator offline';
    button.disabled = true;
    document.querySelectorAll('[data-command]').forEach(control => {control.disabled = true;});
  }
}
refreshState();
setInterval(refreshState, 2000);
async function connect() {
  if (requested) return;
  requested = true;
  button.disabled = true;
  status.textContent = 'Connecting to the live simulator…';
  const event = message => {
    console.info('stream', message);
    if (message.action === EventAction.START && message.status === EventStatus.SUCCESS) {
      status.textContent = 'Live · FFW-BG2';
      video.play().catch(console.warn);
      video.focus();
    }
  };
  try {
    await AppStreamer.connect({
      streamSource: StreamType.DIRECT,
      logLevel: LogLevel.INFO,
      streamConfig: {
        signalingServer: '192.168.68.60', signalingPort: 49110,
        mediaServer: '192.168.68.60', mediaPort: 48008,
        videoElementId: 'remote-video', audioElementId: 'remote-audio',
        width: 1280, height: 720, fps: 30, authenticate: false, maxReconnects: 3,
        onStart: event, onUpdate: event,
        onStop: message => { event(message); status.textContent = 'Disconnected — reload to reconnect'; },
        onTerminate: event,
      },
    });
  } catch (error) {
    console.error(error);
    status.textContent = 'Connection failed — reload to retry';
  }
}
button.addEventListener('click', connect);
video.addEventListener('loadeddata', () => { status.textContent = 'Live · FFW-BG2'; });
document.querySelectorAll('[data-command]').forEach(control => control.addEventListener('click', async () => {
  const response = await fetch('/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({command: control.dataset.command}) });
  if (!response.ok) status.textContent = 'Control request failed';
}));
