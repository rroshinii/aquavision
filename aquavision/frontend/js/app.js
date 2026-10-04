/* AquaVision dashboard: vanilla JS, no build step. */
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = v => v == null ? '–' : (v * 100).toFixed(1) + '%';
const when = iso => new Date(iso).toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'});
const secs = ms => ms == null ? '–' : ms < 1000 ? ms + ' ms' : (ms / 1000).toFixed(1) + ' s';

async function api(path, opts = {}) {
  let r;
  try { r = await fetch(path, opts); } catch { throw new Error('Cannot reach the AquaVision server. Is it running?'); }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch {}
    throw new Error(msg);
  }
  return r.headers.get('content-type')?.includes('json') ? r.json() : r;
}
function toast(msg) {
  const t = Object.assign(document.createElement('div'), {className: 'toast', textContent: msg});
  document.body.appendChild(t); setTimeout(() => t.remove(), 3500);
}

const ICONS = {
  overview: '<path d="M3 12 12 4l9 8M5 10v10h14V10"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
  video: '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/>',
  live: '<circle cx="12" cy="12" r="3"/><path d="M6 6a8.5 8.5 0 0 0 0 12M18 6a8.5 8.5 0 0 1 0 12"/>',
  analytics: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  history: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5M12 7v5l3 2"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>',
};
const ROUTES = [['overview','Overview'],['image','Image Detection'],['video','Video Detection'],['live','Live Detection'],
  ['analytics','Analytics'],['history','Detection History'],['settings','Settings']];
$('#nav').innerHTML = ROUTES.map(([k, l]) =>
  `<a href="#/${k}" data-r="${k}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[k]}</svg>${l}</a>`).join('');

let cleanup = null;
async function route() {
  const key = (location.hash.replace('#/', '') || 'overview');
  const view = VIEWS[key] ? key : 'overview';
  if (cleanup) { cleanup(); cleanup = null; }
  document.querySelectorAll('#nav a').forEach(a => a.classList.toggle('on', a.dataset.r === view));
  $('#view').innerHTML = '<p class="muted"><span class="spin"></span> Loading…</p>';
  try { cleanup = await VIEWS[view]($('#view')) || null; }
  catch (e) { $('#view').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
  refreshFoot();
}
async function refreshFoot() {
  try { const m = await api('/api/model'); $('#foot').textContent = `Model: ${m.weights} · ${m.device.toUpperCase()}`; }
  catch { $('#foot').textContent = 'Model unavailable'; }
}
window.addEventListener('hashchange', route);

/* ---------- shared pieces ---------- */
async function modelBanner() {
  try {
    const m = await api('/api/model');
    return m.stock_coco_weights ? `<div class="alert warn">Using standard COCO weights (<b>${esc(m.weights)}</b>): only people are reported, labelled “Diver”. Add a model trained on marine classes to detect fish, coral or debris. See the README.</div>` : '';
  } catch (e) { return `<div class="alert err">${esc(e.message)}</div>`; }
}
function objectsTable(objs) {
  if (!objs?.length) return '<p class="empty">No objects were detected above the confidence threshold.</p>';
  return `<div class="table-wrap"><table><thead><tr><th>Object</th><th>Confidence</th><th>Count</th></tr></thead><tbody>${
    objs.map(o => `<tr><td>${esc(o.name)}</td><td><div style="display:flex;gap:.6rem;align-items:center"><div class="bar"><i style="width:${o.avg_confidence * 100}%"></i></div>${pct(o.avg_confidence)}</div></td><td>${o.count}</td></tr>`).join('')
  }</tbody></table></div>`;
}
function resultView(r) {
  const s = r.summary || {}, isVid = r.input_type === 'video';
  const media = isVid
    ? `<figure><figcaption>Detection result</figcaption><video src="${r.result_url}" controls preload="metadata" poster="${r.enhanced_url || ''}"></video></figure>`
    : `${r.original_url ? `<figure><figcaption>Original image</figcaption><img src="${r.original_url}" alt="Original"></figure>` : ''}
       ${r.enhanced_url ? `<figure><figcaption>Enhanced</figcaption><img src="${r.enhanced_url}" alt="Enhanced"></figure>` : ''}
       <figure><figcaption>Detection result</figcaption><img src="${r.result_url}" alt="Detection result"></figure>`;
  const timeline = isVid && s.timeline?.length ? (() => {
    const max = Math.max(1, ...s.timeline);
    return `<h3 style="font-size:1.1rem;margin:1.2rem 0 .4rem">Objects per analysed frame</h3><div class="cols" style="height:90px">${
      s.timeline.slice(0, 120).map(n => `<div title="${n}"><i style="height:${n / max * 100}%"></i></div>`).join('')}</div>`; })() : '';
  return `
    <div class="compare">${media}</div>
    <h3 style="font-size:1.2rem;margin-top:1.4rem">Detection summary</h3>
    <div class="summary">
      <div><b>${r.total_objects}</b><span>Objects detected</span></div>
      <div><b>${pct(r.max_confidence)}</b><span>Highest confidence</span></div>
      <div><b>${pct(r.avg_confidence)}</b><span>Average confidence</span></div>
      <div><b>${secs(r.processing_ms)}</b><span>Processing time</span></div>
      ${isVid ? `<div><b>${s.frames_analyzed}/${s.frames_total}</b><span>Frames analysed</span></div>` : ''}
    </div>
    ${objectsTable(s.objects)}${timeline}
    ${isVid && s.note ? `<p class="muted" style="font-size:.85rem">${esc(s.note)}</p>` : ''}
    <div class="cta-row" style="margin-top:1.2rem">
      <a class="btn primary" href="${r.result_url}?download=1" download>Download Result</a>
      <a class="btn" href="/api/history/${r.id}/report.json">Download Report (JSON)</a>
    </div>`;
}
async function openDetection(id) {
  const dlg = $('#dlg');
  dlg.innerHTML = '<p><span class="spin"></span> Loading…</p>'; dlg.showModal();
  try {
    const r = await api('/api/history/' + id);
    dlg.innerHTML = `<h2 style="display:flex;justify-content:space-between;gap:1rem;font-size:1.4rem;margin-bottom:.3rem">${r.display_id}
      <button class="btn small" id="close">Close</button></h2>
      <p class="muted" style="margin-top:0">${esc(r.filename)} · ${r.input_type} · ${when(r.created_at)}</p>
      ${r.status === 'failed' ? `<div class="alert err">${esc(r.error)}</div>` : resultView(r)}`;
    $('#close').onclick = () => dlg.close();
  } catch (e) { dlg.innerHTML = `<div class="alert err">${esc(e.message)}</div><button class="btn small" onclick="this.closest('dialog').close()">Close</button>`; }
}
$('#dlg').addEventListener('click', e => { if (e.target.id === 'dlg') e.target.close(); });
document.addEventListener('click', e => { const b = e.target.closest('[data-view]'); if (b) openDetection(b.dataset.view); });

function historyTable(rows, withDelete) {
  if (!rows.length) return '<p class="empty">No detections yet. Run an <a href="#/image">image</a> or <a href="#/video">video</a> analysis to get started.</p>';
  return `<div class="table-wrap"><table><thead><tr><th>ID</th><th>Date / time</th><th>Input</th><th>Detected objects</th><th>Confidence</th><th>Status</th><th></th></tr></thead><tbody>${
    rows.map(r => {
      const objs = (r.summary?.objects || []).map(o => `${esc(o.name)} ×${o.count}`).join(', ') || (r.status === 'failed' ? '–' : 'None');
      return `<tr><td>${r.display_id}</td><td>${when(r.created_at)}</td><td>${r.input_type}</td><td>${objs}</td><td>${pct(r.avg_confidence)}</td>
        <td><span class="badge ${r.status}">${r.status}</span></td>
        <td style="white-space:nowrap"><button class="btn small" data-view="${r.id}">View</button>${withDelete ? ` <button class="btn small danger" data-del="${r.id}">Delete</button>` : ''}</td></tr>`;
    }).join('')}</tbody></table></div>`;
}

function dropzone(el, {accept, label, onFile}) {
  el.innerHTML = `<div class="drop" tabindex="0" role="button"><strong>${label}</strong><span class="muted">Drag and drop, or click to browse</span></div><input type="file" accept="${accept}" hidden>`;
  const drop = $('.drop', el), input = $('input', el);
  const pick = f => f && onFile(f);
  drop.onclick = () => input.click();
  drop.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } };
  input.onchange = () => pick(input.files[0]);
  ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => pick(e.dataTransfer.files[0]));
}
const optionsHtml = (s, extra = '') => `<div class="opts">
  <label><input type="checkbox" id="enh" ${s.enhance_default ? 'checked' : ''}> Enhance underwater image first</label>
  <label>Confidence <input type="range" id="conf" min="5" max="95" value="${Math.round(s.confidence * 100)}"> <b id="confv">${Math.round(s.confidence * 100)}%</b></label>${extra}</div>`;

/* ---------- views ---------- */
const VIEWS = {};

VIEWS.overview = async el => {
  const [st, hist, banner] = await Promise.all([api('/api/stats'), api('/api/history?limit=8'), modelBanner()]);
  el.innerHTML = `<h1>Overview</h1><p class="page-sub">Live totals from everything analysed on this server.</p>${banner}
    <div class="stats">
      <div class="stat"><div class="k">Total detections</div><div class="v">${st.total_detections}</div></div>
      <div class="stat"><div class="k">Images analyzed</div><div class="v">${st.images_analyzed}</div></div>
      <div class="stat"><div class="k">Videos analyzed</div><div class="v">${st.videos_analyzed}</div></div>
      <div class="stat"><div class="k">Average confidence</div><div class="v">${pct(st.average_confidence)}</div></div>
      <div class="stat"><div class="k">Most detected object</div><div class="v">${st.most_detected ? esc(st.most_detected.name) : '–'}</div></div>
    </div>
    <section class="panel"><h2>Recent detections <a class="btn small" href="#/history">View all</a></h2>${historyTable(hist, false)}</section>`;
};

VIEWS.image = async el => {
  const s = await api('/api/settings'); let file = null, url = null;
  el.innerHTML = `<h1>Image Detection</h1><p class="page-sub">Upload an underwater photo to detect objects.</p>${await modelBanner()}
    <section class="panel"><div id="dz"></div><div id="prev"></div>${optionsHtml(s)}
    <button class="btn primary" id="go" disabled>Analyze</button></section><div id="out"></div>`;
  $('#conf').oninput = e => $('#confv').textContent = e.target.value + '%';
  const setFile = f => {
    if (url) URL.revokeObjectURL(url);
    file = f; $('#go').disabled = !f; $('#out').innerHTML = '';
    if (!f) { $('#prev').innerHTML = ''; return; }
    url = URL.createObjectURL(f);
    $('#prev').innerHTML = `<div class="preview"><img src="${url}" alt="Preview"><button class="btn small x" id="rm">Remove</button></div>`;
    $('#rm').onclick = () => setFile(null);
  };
  dropzone($('#dz'), {accept: 'image/*', label: 'Drop an underwater image here', onFile: setFile});
  $('#go').onclick = async () => {
    const fd = new FormData(); fd.append('file', file);
    fd.append('enhance', $('#enh').checked ? '1' : '0'); fd.append('confidence', $('#conf').value / 100);
    $('#go').disabled = true; $('#out').innerHTML = '<p><span class="spin"></span> Running the model…</p>';
    try { const r = await api('/api/detect/image', {method: 'POST', body: fd}); $('#out').innerHTML = `<section class="panel"><h2>Detection result · ${r.display_id}</h2>${resultView(r)}</section>`; }
    catch (e) { $('#out').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
    $('#go').disabled = !file;
  };
};

VIEWS.video = async el => {
  const s = await api('/api/settings'); let file = null, url = null, timer = null;
  el.innerHTML = `<h1>Video Detection</h1><p class="page-sub">Analyse a recorded dive frame by frame. The annotated video is ready to play and download.</p>${await modelBanner()}
    <section class="panel"><div id="dz"></div><div id="prev"></div>
    ${optionsHtml(s, `<label>Analyse every <input type="number" id="stride" min="1" max="30" value="${s.video_stride}" style="width:4.5rem"> frame(s)</label>`)}
    <button class="btn primary" id="go" disabled>Analyze</button></section><div id="out"></div>`;
  $('#conf').oninput = e => $('#confv').textContent = e.target.value + '%';
  const setFile = f => {
    if (url) URL.revokeObjectURL(url);
    file = f; $('#go').disabled = !f; $('#out').innerHTML = '';
    if (!f) { $('#prev').innerHTML = ''; return; }
    url = URL.createObjectURL(f);
    $('#prev').innerHTML = `<div class="preview"><video src="${url}" controls muted></video><button class="btn small x" id="rm">Remove</button></div>`;
    $('#rm').onclick = () => setFile(null);
  };
  dropzone($('#dz'), {accept: 'video/*', label: 'Drop an underwater video here', onFile: setFile});
  $('#go').onclick = async () => {
    const fd = new FormData(); fd.append('file', file); fd.append('enhance', $('#enh').checked ? '1' : '0');
    fd.append('confidence', $('#conf').value / 100); fd.append('stride', $('#stride').value);
    $('#go').disabled = true;
    $('#out').innerHTML = '<section class="panel"><p id="ptxt"><span class="spin"></span> Uploading…</p><div class="progress"><i id="pbar"></i></div></section>';
    try {
      const {job_id} = await api('/api/detect/video', {method: 'POST', body: fd});
      timer = setInterval(async () => {
        try {
          const j = await api('/api/jobs/' + job_id);
          $('#pbar').style.width = (j.progress * 100) + '%'; $('#ptxt').innerHTML = `<span class="spin"></span> Analysing video… ${Math.round(j.progress * 100)}%`;
          if (j.status === 'completed') { clearInterval(timer); $('#out').innerHTML = `<section class="panel"><h2>Detection result · ${j.record.display_id}</h2>${resultView(j.record)}</section>`; $('#go').disabled = false; }
          if (j.status === 'failed') { clearInterval(timer); $('#out').innerHTML = `<div class="alert err">${esc(j.error)}</div>`; $('#go').disabled = false; }
        } catch (e) { clearInterval(timer); $('#out').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; $('#go').disabled = false; }
      }, 1000);
    } catch (e) { $('#out').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; $('#go').disabled = !file; }
  };
  return () => { clearInterval(timer); if (url) URL.revokeObjectURL(url); };
};

VIEWS.live = async el => {
  const s = await api('/api/settings'); let stream = null, running = false, busy = false, session = {}, frames = 0;
  el.innerHTML = `<h1>Live Detection</h1><p class="page-sub">Real-time detection from your webcam. Frames are sent to this server only; nothing leaves your machine.</p>${await modelBanner()}
    <section class="panel"><div class="livebox"><video id="cam" playsinline muted></video><canvas id="ovl"></canvas><span class="muted" id="ph">Camera is off</span></div>
    ${optionsHtml(s)}<div class="cta-row"><button class="btn primary" id="start">Start camera</button><button class="btn" id="stop" disabled>Stop</button><button class="btn" id="snap" disabled>Save snapshot to history</button></div>
    <div id="msg"></div></section>
    <section class="panel"><h2>Current frame <span class="muted" id="lat" style="font-size:.9rem;font-family:var(--sans)"></span></h2><div id="cur" class="muted">Start the camera to see detections.</div></section>`;
  $('#conf').oninput = e => $('#confv').textContent = e.target.value + '%';
  const cam = $('#cam'), ovl = $('#ovl'), grab = document.createElement('canvas');
  const COLORS = ['#5fe3d4', '#ffbe5c', '#ff7a70', '#78dc8c', '#5abeff', '#dc78e6'];
  const colorOf = n => COLORS[[...n].reduce((a, c) => a + c.charCodeAt(0), 0) % COLORS.length];

  function draw(j) {
    ovl.width = cam.videoWidth; ovl.height = cam.videoHeight;
    const c = ovl.getContext('2d'), k = cam.videoWidth / j.width, lw = Math.max(2, cam.videoWidth / 320);
    c.clearRect(0, 0, ovl.width, ovl.height); c.font = `${Math.round(lw * 6)}px system-ui`; c.lineWidth = lw;
    j.detections.forEach(d => {
      const [x1, y1, x2, y2] = d.bbox.map(v => v * k), col = colorOf(d.class), t = `${d.class} ${d.confidence.toFixed(2)}`;
      c.strokeStyle = col; c.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const w = c.measureText(t).width + 8, h = lw * 7;
      c.fillStyle = col; c.fillRect(x1, Math.max(0, y1 - h), w, h); c.fillStyle = '#10201f'; c.fillText(t, x1 + 4, Math.max(h - lw * 1.5, y1 - lw * 1.5));
    });
  }
  async function tick() {
    if (!running) return;
    if (!busy && cam.videoWidth) {
      busy = true;
      const sc = Math.min(1, 640 / cam.videoWidth); grab.width = cam.videoWidth * sc; grab.height = cam.videoHeight * sc;
      grab.getContext('2d').drawImage(cam, 0, 0, grab.width, grab.height);
      const blob = await new Promise(r => grab.toBlob(r, 'image/jpeg', 0.75));
      const fd = new FormData(); fd.append('frame', blob); fd.append('enhance', $('#enh').checked ? '1' : '0'); fd.append('confidence', $('#conf').value / 100);
      try {
        const j = await api('/api/detect/frame', {method: 'POST', body: fd});
        if (running) {
          draw(j); frames++;
          const by = {}; j.detections.forEach(d => (by[d.class] ||= []).push(d.confidence));
          $('#lat').textContent = `${j.processing_ms} ms · ${frames} frames this session`;
          $('#cur').innerHTML = j.detections.length ? objectsTable(Object.entries(by).map(([name, v]) => ({name, count: v.length, avg_confidence: v.reduce((a, b) => a + b) / v.length}))) : 'No objects in this frame.';
        }
        $('#msg').innerHTML = '';
      } catch (e) { $('#msg').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
      busy = false;
    }
    setTimeout(tick, 60);
  }
  function stop() {
    running = false; stream?.getTracks().forEach(t => t.stop()); stream = null; cam.srcObject = null;
    ovl.getContext('2d').clearRect(0, 0, ovl.width, ovl.height);
    $('#ph').style.display = ''; $('#start').disabled = false; $('#stop').disabled = true; $('#snap').disabled = true;
  }
  $('#start').onclick = async () => {
    if (!navigator.mediaDevices?.getUserMedia) { $('#msg').innerHTML = '<div class="alert err">Camera access needs a secure context. Open AquaVision at http://localhost:5000 (or over HTTPS).</div>'; return; }
    try { stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 1280}}, audio: false}); }
    catch (e) { $('#msg').innerHTML = `<div class="alert err">Camera unavailable: ${esc(e.message)}. Check browser permissions and that a camera is connected.</div>`; return; }
    cam.srcObject = stream; await cam.play(); $('#ph').style.display = 'none';
    running = true; $('#start').disabled = true; $('#stop').disabled = false; $('#snap').disabled = false; tick();
  };
  $('#stop').onclick = stop;
  $('#snap').onclick = async () => {
    const c = document.createElement('canvas'); c.width = cam.videoWidth; c.height = cam.videoHeight; c.getContext('2d').drawImage(cam, 0, 0);
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92));
    const fd = new FormData(); fd.append('file', new File([blob], 'webcam-snapshot.jpg', {type: 'image/jpeg'}));
    fd.append('enhance', $('#enh').checked ? '1' : '0'); fd.append('confidence', $('#conf').value / 100);
    try { const r = await api('/api/detect/image', {method: 'POST', body: fd}); toast(`Saved as ${r.display_id} in Detection History`); }
    catch (e) { toast(e.message); }
  };
  return stop;
};

VIEWS.analytics = async el => {
  const st = await api('/api/stats');
  if (!st.total_detections && !st.images_analyzed && !st.videos_analyzed) {
    el.innerHTML = '<h1>Analytics</h1><p class="empty">No data yet. Analytics are built from your real detections. <a href="#/image">Analyse an image</a> to begin.</p>'; return;
  }
  const maxC = Math.max(1, ...st.classes.map(c => c.count)), maxH = Math.max(1, ...st.confidence_histogram), maxD = Math.max(1, ...st.daily.map(d => d.count));
  el.innerHTML = `<h1>Analytics</h1><p class="page-sub">Aggregated from ${st.images_analyzed + st.videos_analyzed} completed analyses.</p>
  <div class="charts">
    <section class="panel"><h2>Detections by class</h2>${st.classes.length ? st.classes.map(c => `<div class="hbar"><span>${esc(c.name)}</span><div class="bar"><i style="width:${c.count / maxC * 100}%"></i></div><b>${c.count}</b></div>`).join('') : '<p class="empty">No objects detected yet.</p>'}</section>
    <section class="panel"><h2>Confidence distribution</h2><div class="cols">${st.confidence_histogram.map((n, i) => `<div title="${n} detections"><span>${n || ''}</span><i style="height:${n / maxH * 100}%"></i><span>${i * 10}</span></div>`).join('')}</div><p class="muted" style="font-size:.8rem;margin:.4rem 0 0">Confidence, in % buckets of 10</p></section>
    <section class="panel"><h2>Average confidence per class</h2>${st.classes.map(c => `<div class="hbar"><span>${esc(c.name)}</span><div class="bar"><i style="width:${c.avg_confidence * 100}%"></i></div><b>${Math.round(c.avg_confidence * 100)}%</b></div>`).join('') || '<p class="empty">No data.</p>'}</section>
    <section class="panel"><h2>Detections per day</h2><div class="cols">${st.daily.map(d => `<div title="${d.date}: ${d.count}"><span>${d.count}</span><i style="height:${d.count / maxD * 100}%"></i><span>${d.date.slice(5)}</span></div>`).join('')}</div></section>
  </div>${st.failed ? `<p class="muted">${st.failed} analysis run(s) failed and are excluded. See Detection History.</p>` : ''}`;
};

VIEWS.history = async el => {
  let type = '';
  const render = async () => {
    const rows = await api('/api/history' + (type ? '?type=' + type : ''));
    $('#tbl').innerHTML = historyTable(rows, true);
  };
  el.innerHTML = `<h1>Detection History</h1><p class="page-sub">Every analysis is stored in a local SQLite database.</p>
    <section class="panel"><div class="opts" style="margin-top:0"><label>Filter <select id="ft"><option value="">All</option><option value="image">Images</option><option value="video">Videos</option></select></label>
    <a class="btn small" href="/api/history/export.csv">Export CSV</a><button class="btn small danger" id="clr">Clear all</button></div><div id="tbl"></div></section>`;
  $('#ft').onchange = e => { type = e.target.value; render(); };
  $('#clr').onclick = async () => { if (confirm('Delete all detection history and its stored files?')) { await api('/api/history', {method: 'DELETE'}); render(); } };
  el.onclick = async e => { const b = e.target.closest('[data-del]'); if (b && confirm('Delete this detection?')) { await api('/api/history/' + b.dataset.del, {method: 'DELETE'}); render(); } };
  await render();
  return () => { el.onclick = null; };
};

VIEWS.settings = async el => {
  const [s, m] = await Promise.all([api('/api/settings'), api('/api/model').catch(() => null)]);
  el.innerHTML = `<h1>Settings</h1><p class="page-sub">Defaults used for new detections.</p>
    <section class="panel"><form class="form-grid" id="f">
      <label>Default confidence threshold: <b id="cv">${Math.round(s.confidence * 100)}%</b><input type="range" name="confidence" min="5" max="95" value="${Math.round(s.confidence * 100)}"></label>
      <label><span><input type="checkbox" name="enhance_default" ${s.enhance_default ? 'checked' : ''}> Enhance images before detection by default</span></label>
      <label>Classes reported<select name="class_mode"><option value="auto" ${s.class_mode === 'auto' ? 'selected' : ''}>Auto (original behaviour: people are reported as “Diver” when using COCO weights)</option><option value="all" ${s.class_mode === 'all' ? 'selected' : ''}>All classes the model knows</option></select></label>
      <label>Video: analyse every Nth frame<input type="number" name="video_stride" min="1" max="30" value="${s.video_stride}"></label>
      <div><button class="btn primary" type="submit">Save settings</button></div></form></section>
    <section class="panel"><h2>Model</h2>${m ? `<p>Weights: <b>${esc(m.weights)}</b> · Device: <b>${m.device}</b> · Classes known to the model: <b>${m.total_model_classes}</b> · Reported now: <b>${m.classes.length}</b></p><p class="muted">To use your own model, place <code>best.pt</code> in <code>models/</code> or set <code>AQUAVISION_WEIGHTS</code>, then restart the server.</p>` : '<div class="alert err">Model unavailable.</div>'}</section>`;
  const f = $('#f'); f.confidence.oninput = () => $('#cv').textContent = f.confidence.value + '%';
  f.onsubmit = async e => {
    e.preventDefault();
    try {
      await api('/api/settings', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
        confidence: f.confidence.value / 100, enhance_default: f.enhance_default.checked, class_mode: f.class_mode.value, video_stride: +f.video_stride.value})});
      toast('Settings saved');
    } catch (err) { toast(err.message); }
  };
};

route();
