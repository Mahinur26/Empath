/* Runs the occlusion map off the main thread so the camera view stays smooth. */
importScripts('empath-model.js');
let model = null;
onmessage = e => {
  const d = e.data;
  if (d.type === 'load') { model = EmpathModel.load(d.buf); postMessage({ type: 'ready' }); }
  else if (d.type === 'occ' && model) {
    const r = model.occlusion(d.gray, d.grid || 8);
    postMessage({ type: 'occ', top: r.top, base: r.base, grid: r.grid, drops: Array.from(r.drops), gray: d.gray });
  }
};
