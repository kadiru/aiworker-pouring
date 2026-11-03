const status = document.querySelector('#status');
const simState = document.querySelector('#sim-state');
const preview = document.querySelector('#preview');
let ready = false;
let loading = false;
async function refreshState() {
  try {
    const response = await fetch('/health', {cache: 'no-store'});
    if (!response.ok) throw new Error('Simulator unavailable');
    const state = await response.json();
    ready = state.ready && Date.now()/1000 - state.updated_unix < 10;
    document.querySelectorAll('[data-command]').forEach(control => {control.disabled = !ready;});
    status.textContent = ready ? (state.paused ? 'Paused' : 'Live · FFW-BG2') : 'Simulator warming up…';
    const quest = state.profile === 'quest-pouring';
    document.querySelector('#quest-panel').hidden = !quest;
    document.querySelector('#motion-check').textContent = quest ? 'Test arm IK' : 'Check head motion';
    document.querySelector('#scene-note').textContent = quest ? (state.fluid_enabled ? 'Five smooth PBD liquids · No trained pouring policy' : 'Empty source and target cups') : 'Dry preview · No liquid or trained policy';
    document.querySelector('#camera-panel').hidden = !quest;
    simState.textContent = ready ? `${state.sim_time_s?.toFixed(1) ?? '0.0'} s simulated · Preview up to 5 fps` : (state.phase || 'Warming up…');
    if (quest) {
      document.querySelector('#comparison').hidden = !state.liquid_comparison;
      document.querySelector('#liquid-state').textContent = state.fluid ? `Particles: ${state.fluid.total} total · ${state.fluid.on_table ?? 0} on table · ${state.fluid.below_table} below table level` : 'Dry rehearsal';
      document.querySelectorAll('[data-command^=spill_]').forEach(button => {if (/spill_[1-5]$/.test(button.dataset.command)) button.disabled = !ready || state.recording || state.paused;});
      const spill = state.spill_test;
      const fluid = spill && state.fluid?.liquids?.[spill.id-1];
      document.querySelector('#spill-state').textContent = fluid ? `Test ${spill.id}: ${(state.sim_time_s-spill.start_sim_time).toFixed(1)} s simulated · ${fluid.on_table}/${fluid.total} particles on table · ${fluid.over_edge} past table edge · ${fluid.under_table_footprint} underneath table footprint. Robot motion held until Reset scene.` : 'Ready for teleoperation or a controlled spill test.';
      document.querySelectorAll('[data-camera]').forEach(img => {img.src = `/camera/${img.dataset.camera}.jpg?t=${Date.now()}`;});
      document.querySelector('#controllers').textContent = `Controllers: left ${state.controllers.left ? 'tracked' : 'waiting'}, right ${state.controllers.right ? 'tracked' : 'waiting'}`;
      const xrButton = document.querySelector('[data-command="start_xr"]');
      xrButton.textContent = state.xr_enabled ? 'XR enabled' : 'Start XR';
      xrButton.disabled = !ready || state.xr_enabled;
      document.querySelector('#recording').textContent = state.recording ? `Recording · ${state.samples} samples` : `Ready · ${state.saved_episodes} saved this session`;
      document.querySelector('[data-command="start"]').disabled = !ready || state.recording;
      document.querySelector('[data-command="save"]').disabled = !ready || !state.recording;
    }
  } catch {
    ready = false;
    status.textContent = 'Simulator offline';
    document.querySelectorAll('[data-command]').forEach(control => {control.disabled = true;});
  }
}
preview.addEventListener('load', () => {loading = false;});
preview.addEventListener('error', () => {loading = false;});
setInterval(() => {
  if (ready && !loading) {
    loading = true;
    preview.src = `/snapshot.jpg?t=${Date.now()}`;
  }
}, 200);
refreshState();
setInterval(refreshState, 1000);
document.querySelectorAll('[data-command]').forEach(control => control.addEventListener('click', async () => {
  try {
    const response = await fetch('/control', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({command: control.dataset.command})});
    if (!response.ok) throw new Error('Control request failed');
    await refreshState();
  } catch {status.textContent = 'Control request failed';}
}));
