'use strict';
const data = window.ARENA_SHOWCASE.games;
const el = id => document.getElementById(id);
let runIndex = 0, stepIndex = 0;
const armLabel = arm => arm === 'waypoints' ? '路点拼接' : '完整候选路线';
const resultLabel = r => r.winner === 'attacker' ? '四足到点获胜' : r.reason === 'tag' ? '四足被标记' : '四足超时失败';
const pretty = value => JSON.stringify(value, null, 2);
function showTab(id) {
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === id));
  document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === id));
  if (id !== 'replay') el('video').pause();
}
function selectRun(index) {
  runIndex = index;
  const g = data[index], folder = 'run_' + g.case.run;
  el('video').pause(); el('video').poster = g.poster; el('video').src = g.replay; el('video').load();
  el('video-link').href = g.replay; el('video-link').download = `quadruped-arena-v05-${g.case.layout}-${g.case.arm}.mp4`;
  el('final-link').href = folder + '/final.jpg';
  el('game-label').textContent = `对局 ${g.case.run} / ${g.case.layout} / ${armLabel(g.case.arm)}`;
  el('game-title').textContent = g.review.headline;
  el('game-summary').textContent = g.review.summary;
  el('result-badge').textContent = `${resultLabel(g.result)} · ${g.result.seconds.toFixed(2)} s`;
  el('result-badge').classList.toggle('win', g.result.winner === 'attacker');
  el('limitation').textContent = '事后核对：' + g.review.limitation;
  for (const [id, file] of [['decisions-link','decisions.json'],['calls-link','call_records.json'],['prompt-link','system_prompt.txt']]) el(id).href = folder + '/' + file;
  document.querySelectorAll('[data-run]').forEach(b => b.classList.toggle('active', +b.dataset.run === index));
  el('stepbar').replaceChildren();
  g.decisions.forEach((d, j) => {
    const b = document.createElement('button'); b.dataset.step = j;
    b.textContent = `${j+1} · ${d.simulation_seconds.toFixed(2)} s`;
    b.onclick = () => selectStep(j, true); el('stepbar').append(b);
  });
  selectStep(0); showTab('replay');
  history.replaceState(null, '', '#run-' + g.case.run);
}
function selectStep(index, seek = false) {
  stepIndex = index;
  const g = data[runIndex], d = g.decisions[index], s = g.steps[index];
  if (seek) {
    el('video').pause();
    // Seek to the first recorded frame at/after the decision, not the previous frame.
    el('video').currentTime = Math.ceil(s.start * g.media.fps - 1e-8) / g.media.fps + .0001;
  }
  el('decision-title').textContent = `决策 ${index + 1} / ${g.decisions.length}`;
  el('step-meta').textContent = `仿真 ${s.start.toFixed(2)}–${s.end.toFixed(2)} s · API ${s.api_seconds.toFixed(2)} s · ${s.usage.total_tokens.toLocaleString()} tokens`;
  el('rationale').replaceChildren();
  const names = { observation_evidence:'观测依据', state_evidence:'自身状态', decision_summary:'选择与简要理由', uncertainty:'不确定性' };
  for (const [key, label] of Object.entries(names)) {
    const div = document.createElement('div'); div.className = 'rationale-part';
    const h = document.createElement('h3'), p = document.createElement('p');
    h.textContent = label; p.textContent = d.assessment[key]; div.append(h, p); el('rationale').append(div);
  }
  el('action').textContent = pretty(Object.fromEntries(['observation_id','skill','params','assessment'].map(k => [k,d[k]])));
  el('shot').src = s.image;
  el('step-facts').textContent = `本步实际移动 ${s.traveled_m.toFixed(2)} m；距目标 ${s.start_goal_m.toFixed(2)} → ${s.end_goal_m.toFixed(2)} m。外部调用 UTC：${s.api_started_utc} → ${s.api_finished_utc}。`;
  el('events').textContent = pretty(s.events); el('packet').textContent = pretty(d.public_observation);
  el('portfolio').hidden = !s.routeplot;
  if (s.routeplot) {
    el('routeplot').src = s.routeplot;
    el('cards').textContent = pretty({generation:d.public_observation.route_generation, routes:d.public_observation.route_candidates});
  } else { el('routeplot').removeAttribute('src'); el('cards').textContent = ''; }
  document.querySelectorAll('[data-step]').forEach(b => b.classList.toggle('active', +b.dataset.step === index));
}
data.forEach((g, i) => {
  const button = document.createElement('button'); button.dataset.run = i;
  button.textContent = `${g.case.run}. ${g.case.layout} · ${armLabel(g.case.arm)}`;
  button.onclick = () => selectRun(i); el('runbar').append(button);
  const tr = document.createElement('tr');
  for (const text of [`${g.case.run} / ${g.case.layout}`, armLabel(g.case.arm), resultLabel(g.result), `${g.result.seconds.toFixed(2)} s`, g.decisions.length, `${g.result.api_total_seconds.toFixed(1)} s`, g.result.usage.total_tokens.toLocaleString()]) {
    const td = document.createElement('td'); td.textContent = text; tr.append(td);
  }
  el('results').append(tr);
  const block = document.createElement('div'); block.className = 'review';
  const h = document.createElement('h3'); h.textContent = `${g.case.run}. ${g.review.headline}`; block.append(h);
  for (const text of [g.review.summary, '解释限制：' + g.review.limitation]) { const p = document.createElement('p'); p.textContent = text; block.append(p); }
  const b = document.createElement('button'); b.textContent = '查看整局与决策'; b.onclick = () => {selectRun(i); window.scrollTo(0,0);}; block.append(b); el('reviews').append(block);
});
el('video').addEventListener('timeupdate', () => {
  if (!el('follow').checked) return;
  const g = data[runIndex], ds = g.decisions;
  const frame = Math.floor(el('video').currentTime * g.media.fps + 1e-5);
  const t = frame >= g.media.frames - 1 ? g.result.seconds : Math.round(frame / g.media.fps / .02) * .02;
  let j = 0; for (let i = 1; i < ds.length; i++) if (ds[i].simulation_seconds <= t + 0.001) j = i;
  if (j !== stepIndex) selectStep(j);
});
const initial = Number(location.hash.match(/^#run-(\d)$/)?.[1] || 1) - 1;
selectRun(initial >= 0 && initial < data.length ? initial : 0);
