'use strict';
const $ = id => document.getElementById(id);
const state = {catalog: [], primary: null, comparison: null, index: 0, timer: null, audit: null, auditAvailable: false};
const dimensions = ['input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_input_tokens', 'provider_total_tokens'];
const labels = ['Input', 'Output', 'Reasoning', 'Cached input', 'Provider total'];
const value = x => x === null || x === undefined ? 'unknown' : Number(x).toLocaleString();
function node(tag, text, className) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (className) n.className = className; return n; }
function notice(text, warning = false) { $('notice').textContent = text; $('notice').classList.toggle('warning', warning); }
async function get(url) { const response = await fetch(url, {cache: 'no-store'}); const body = await response.json(); if (!response.ok) throw new Error(body.error || 'Artifact request failed.'); return body; }
function label(run) { const models = Object.values(run.models || {}).map(m => m.model).filter(Boolean).join(' / '); return `${run.task || 'Unknown task'} · seed ${run.seed ?? '?'} · Runtime role: ${run.role || 'not recorded'} · ${models || 'model unknown'} · ${run.label}`; }
function options(select, runs, placeholder) { select.replaceChildren(new Option(placeholder, '')); for (const run of runs) select.add(new Option(label(run), run.id)); }
function pause() { clearInterval(state.timer); state.timer = null; $('play').textContent = 'Play'; }
function clearAudit() { state.audit = null; $('audit').checked = false; $('audit-content').hidden = true; $('audit-content').replaceChildren(); $('audit-status').textContent = state.auditAvailable ? 'Privileged evidence is hidden. Explicit replay opt-in is required.' : 'Audit access is disabled unless the server was launched with --allow-audit.'; }
async function catalog() {
  pause(); clearAudit();
  try { const body = await get('/api/runs'); state.catalog = body.runs; state.auditAvailable = body.audit_available; $('audit').disabled = !body.audit_available; options($('primary'), state.catalog, 'Choose a saved run'); options($('comparison'), [], 'Single-run replay'); notice(state.catalog.length ? `${state.catalog.length} saved runs available. Replay stays on this computer and makes no model calls.` : 'No events.jsonl artifacts were found in the declared roots.', !state.catalog.length); }
  catch (error) { notice(error.message, true); }
}
function comparisonChoices() {
  const meta = state.primary?.metadata;
  const matches = state.catalog.filter(run => run.id !== meta?.id && meta?.task && Number.isInteger(meta?.seed) && run.task === meta.task && run.seed === meta.seed);
  options($('comparison'), matches, matches.length ? 'Single-run replay / choose matched run' : 'No same-task, same-seed match');
}
async function selectPrimary() {
  pause(); clearAudit(); state.comparison = null; state.index = 0;
  if (!$('primary').value) { state.primary = null; render(); return; }
  try { state.primary = await get(`/api/runs/${$('primary').value}`); comparisonChoices(); render(); }
  catch (error) { state.primary = null; render(); notice(error.message, true); }
}
async function selectComparison() {
  pause(); clearAudit();
  try { state.comparison = $('comparison').value ? await get(`/api/runs/${$('comparison').value}`) : null; render(); }
  catch (error) { state.comparison = null; render(); notice(error.message, true); }
}
function comparisonIndex() {
  const right = state.comparison?.events || [];
  if (!$('sync').checked) return Math.min(state.index, Math.max(0, right.length - 1));
  const left = state.primary?.events[state.index]; if (!left) return 0;
  const targetStep = left.step ?? 0;
  const leftSameStep = state.primary.events.slice(0, state.index + 1).filter(e => (e.step ?? 0) === targetStep).length - 1;
  const exact = right.map((event, index) => ({event, index})).filter(x => (x.event.step ?? 0) === targetStep);
  if (exact.length) return exact[Math.min(leftSameStep, exact.length - 1)].index;
  let last = 0; right.forEach((event, index) => { if ((event.step ?? 0) <= targetStep) last = index; }); return last;
}
function accumulated(run, index) {
  const replay = {text: '', actions: [], action: '', feedback: '', accepted: null, role: run.metadata.role || 'weak', calls: 0, escalations: 0, callUsage: new Map(), model: null, latency: null, outcome: 'in progress'};
  for (const event of run.events.slice(0, index + 1)) {
    const p = event.payload || {}; if (p.model_role) replay.role = p.model_role;
    if (p.observation?.text) replay.text = p.observation.text;
    if (p.available_actions) replay.actions = p.available_actions;
    if (p.request) { if (p.request.observation?.text) replay.text = p.request.observation.text; replay.actions = p.request.available_actions || replay.actions; replay.model = p.request.model; replay.calls++; replay.callUsage.set(p.call_index ?? replay.calls, null); }
    if (p.response) { replay.model = p.response.model; replay.latency = p.latency_ms; replay.callUsage.set(p.call_index ?? replay.calls, p.response.usage); }
    if (event.event_type === 'TeacherEscalated') { replay.escalations++; replay.role = 'teacher'; replay.model = p.model; }
    if (p.action?.name) { replay.action = p.action.name; if (event.event_type === 'ActionProposed') { replay.accepted = null; replay.feedback = 'Proposed; awaiting recorded outcome.'; } }
    if (p.result) { replay.accepted = p.result.accepted; replay.feedback = p.result.reason || ''; if (p.result.observation?.text) replay.text = p.result.observation.text; }
    if (event.event_type === 'ActionRejected') { replay.accepted = false; replay.action = p.proposal || replay.action; replay.feedback = p.reason || 'Rejected'; }
    if (event.event_type === 'TaskSucceeded') replay.outcome = 'succeeded';
    if (event.event_type === 'TaskFailed') replay.outcome = 'failed';
    if (event.event_type === 'RunErrored') replay.outcome = 'error';
    if (event.event_type === 'ExperimentFinished') replay.outcome = p.summary?.status || replay.outcome;
  }
  replay.usage = dimensions.map(dimension => { let known = 0, unknown = 0; for (const usage of replay.callUsage.values()) { if (typeof usage?.[dimension] === 'number') known += usage[dimension]; else unknown++; } return {known, unknown, total: unknown ? null : known}; });
  return replay;
}
function metric(parent, name, amount) { const m = node('div', undefined, 'metric'); m.append(node('small', name), node('strong', amount)); parent.append(m); }
function details(title, data) { const d = node('details'); d.append(node('summary', title), node('pre', JSON.stringify(data, null, 2), 'json')); return d; }
function panel(run, index, side) {
  const meta = run.metadata, event = run.events[index], replay = accumulated(run, index); const p = node('article', undefined, 'panel');
  const head = node('div', undefined, 'panel-head'), names = node('div'); names.append(node('div', `${side} · Runtime role: ${replay.role}`, 'role'), node('h3', replay.model || meta.models?.[replay.role]?.model || 'Model not recorded', 'model'), node('p', `${meta.task || 'Unknown task'} · seed ${meta.seed ?? '?'} · ${meta.experiment_id || meta.label}`, 'panel-meta'), node('p', Object.entries(meta.models || {}).map(([role, model]) => `Runtime ${role}: ${model.provider || '?'} / ${model.model || '?'}`).join(' · '), 'panel-meta'), node('p', 'Runtime role is an accounting annotation. In candidate diagnostics it does not identify an empirically selected Weak/Teacher hierarchy.', 'panel-meta')); head.append(names, node('span', replay.outcome, `badge ${replay.outcome}`)); p.append(head);
  const metrics = node('div', undefined, 'metrics'); metric(metrics, 'Step', event?.step ?? '—'); metric(metrics, 'Model calls so far', replay.calls); metric(metrics, 'Input tokens', value(replay.usage[0].total)); metric(metrics, 'Output tokens', value(replay.usage[1].total)); p.append(metrics);
  const body = node('div', undefined, 'panel-body');
  const sequence = event?.sequence ?? -1; const frame = (run.frames || []).filter(f => f.sequence <= sequence).at(-1);
  if (frame) { const image = node('img', undefined, 'frame'); image.src = frame.url; image.alt = `Recorded actor-visible official Alem frame at event ${frame.sequence}`; body.append(image, node('p', `Saved official Alem actor frame · event ${frame.sequence}`, 'frame-note')); }
  else body.append(node('p', 'No saved official rendering. Actor-visible text is the recorded world view.', 'frame-note'));
  body.append(node('p', 'VISIBLE OBSERVATION · INVENTORY / STATUS WHEN RECORDED', 'caption'), node('div', replay.text || 'No actor observation recorded at this event.', 'observation'));
  const action = node('div', undefined, 'action-row'); action.append(node('span', replay.action || 'No proposed action yet', 'action-name'), node('span', replay.accepted === true ? 'Accepted' : replay.accepted === false ? 'Rejected' : 'Pending / —', 'badge')); body.append(action, node('p', replay.feedback || 'Action result not recorded yet.', 'feedback'), node('p', 'ADVERTISED LEGAL ACTIONS', 'caption'));
  const legal = node('div', undefined, 'legal'); replay.actions.forEach(action => { const chip = node('span', action.name); chip.title = action.description || ''; legal.append(chip); }); if (!replay.actions.length) legal.append(node('span', 'No action list recorded')); body.append(legal);
  const usageText = replay.usage.map((u, i) => `${labels[i]}: ${value(u.total)}${u.unknown ? ` (known subtotal ${u.known}; ${u.unknown} unknown calls)` : ''}`).join(' · '); body.append(node('p', usageText, 'usage-detail'), node('p', `Teacher escalations: ${replay.escalations} · Last response latency: ${typeof replay.latency !== 'number' ? 'unknown' : `${replay.latency.toFixed(1)} ms`} · ${meta.fixture_only ? 'SCRIPTED FIXTURE · not real-model evidence' : 'Recorded model labels; capability roles require empirical evidence'}`, 'usage-detail'));
  body.append(details('Inspect selected public event', event || {}), details('Recorded final summary (outcome metadata; private evidence omitted)', run.summary)); p.append(body); return p;
}
function timeline() {
  const container = $('timeline'); container.replaceChildren(); if (!state.primary) { container.append(node('p', 'Select a saved run to begin.', 'empty')); return; }
  state.primary.events.forEach((event, index) => { const row = node('button', undefined, `event-row${index === state.index ? ' selected' : ''}`); row.append(node('span', event.sequence ?? index), node('span', `s ${event.step ?? '—'}`), node('span', event.event_type), node('span', event.payload?.model_role || '')); row.addEventListener('click', () => { pause(); state.index = index; render(false); }); container.append(row); });
}
function render(scroll = true) {
  const panels = $('panels'); panels.replaceChildren(); const left = state.primary;
  const max = Math.max(0, (left?.events.length || 1) - 1); state.index = Math.min(state.index, max); $('scrubber').max = max; $('scrubber').value = state.index;
  ['play', 'next', 'previous', 'scrubber'].forEach(id => { $(id).disabled = !left?.events.length; });
  if (left) { panels.append(panel(left, state.index, 'Primary')); if (state.comparison) panels.append(panel(state.comparison, comparisonIndex(), 'Comparison')); $('position').textContent = `Event ${state.index + 1} / ${left.events.length}`; const warnings = [...left.warnings, ...(state.comparison?.warnings || [])]; notice(warnings.length ? warnings.join(' ') : `${left.metadata.fixture_only ? 'Scripted fixture — engineering inspection only. ' : ''}Public replay: private snapshots, verifier evidence, manifests and full configurations are omitted.`, !!warnings.length); }
  else $('position').textContent = 'No run selected';
  $('pair-note').textContent = state.comparison ? 'Matched task and seed · synchronization follows step and within-step event order, not wall-clock time. Different event types may align; matching does not verify identical public conditions.' : 'Comparisons require a recorded task and seed match. Matching labels alone do not establish scientific comparability.';
  timeline(); if (scroll) { const selected = $('timeline').querySelector('.selected'); if (selected) $('timeline').scrollTop = Math.max(0, selected.offsetTop - $('timeline').offsetTop - 110); } renderAudit();
}
function advance(delta) { state.index = Math.max(0, Math.min(state.index + delta, (state.primary?.events.length || 1) - 1)); render(); if (state.index === state.primary?.events.length - 1) pause(); }
function play() { if (state.timer) { pause(); return; } if (!state.primary?.events.length) return; if (state.index >= state.primary.events.length - 1) state.index = 0; $('play').textContent = 'Pause'; state.timer = setInterval(() => advance(1), Number($('speed').value)); render(); }
function renderAudit() {
  const target = $('audit-content'); target.replaceChildren(); if (!$('audit').checked || !state.audit) { target.hidden = true; return; } target.hidden = false;
  const entry = state.audit.events.filter(e => e.index <= state.index).at(-1); target.append(node('p', 'AUDIT VIEW — latest privileged evidence at or before the selected primary event.'), node('pre', JSON.stringify(entry || {message: 'No privileged evidence recorded yet.'}, null, 2), 'json'));
}
async function audit() { if (!$('audit').checked) { clearAudit(); return; } if (!state.primary) { clearAudit(); $('audit-status').textContent = 'Select a saved run before enabling audit.'; return; } try { state.audit = await get(`/api/runs/${state.primary.metadata.id}/audit?ack=privileged-replay`); $('audit-status').textContent = 'Privileged replay evidence enabled for the primary run. Turning this off clears the audit payload from the page.'; renderAudit(); } catch (error) { clearAudit(); $('audit-status').textContent = error.message; } }
$('primary').addEventListener('change', selectPrimary); $('comparison').addEventListener('change', selectComparison); $('play').addEventListener('click', play); $('previous').addEventListener('click', () => { pause(); advance(-1); }); $('next').addEventListener('click', () => { pause(); advance(1); }); $('scrubber').addEventListener('input', () => { pause(); state.index = Number($('scrubber').value); render(); }); $('speed').addEventListener('change', () => { if (state.timer) { pause(); play(); } }); $('sync').addEventListener('change', () => render()); $('audit').addEventListener('change', audit); $('reload').addEventListener('click', async () => { const id = $('primary').value; await catalog(); if (state.catalog.some(run => run.id === id)) { $('primary').value = id; await selectPrimary(); } else { state.primary = null; state.comparison = null; render(); } });
catalog();
