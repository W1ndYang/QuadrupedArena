"""Run directories, replay files and a self-contained HTML viewer (no network, no build step)."""
import datetime
import gzip
import json
import os
import subprocess
from pathlib import Path

from . import GAME, PROTOCOL_VERSION, RULES_VERSION

REPO = Path(__file__).resolve().parents[1]


def workspace():
    root = os.environ.get('ARENA_WORKSPACE')
    return Path(root) if root else REPO / '.arena-workspace'


def new_run_dir(label, base=None):
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]
    path = Path(base) if base else workspace() / 'breach_arena' / 'runs' / f'{stamp}_{label}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit():
    try:
        return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_replay(match, agent, players, timing, round_index=1, decider_calls=None):
    return dict(
        format='breach-replay-v1', game=GAME, rules_version=RULES_VERSION, protocol_version=PROTOCOL_VERSION,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        commit=git_commit(), round=round_index, players=players, timing=timing,
        map=match.map.public_dict(), rules=match.rules.raw, summary=match.summary(),
        events=match.events, trace=match.trace,
        quadruped=dict(log=agent.log, skills=agent.reports, calls=decider_calls or []))


def save_replay(replay, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"round{replay['round']}"
    with gzip.open(folder / f'{name}.replay.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(replay, f, ensure_ascii=False, separators=(',', ':'))
    (folder / f'{name}.summary.json').write_text(json.dumps(
        dict(summary=replay['summary'], players=replay['players'], timing=replay['timing'],
             skills=replay['quadruped']['skills'],
             decisions=[e for e in replay['quadruped']['log'] if e['type'] == 'response']),
        ensure_ascii=False, indent=1), encoding='utf-8')
    html = folder / f'{name}.viewer.html'
    write_viewer(replay, html)
    return html


def load_replay(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def write_viewer(replay, out_path):
    data = json.dumps(replay, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    Path(out_path).write_text(VIEWER.replace('/*REPLAY*/null', data), encoding='utf-8')
    return out_path


VIEWER = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breach Arena Replay</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d2329;--muted:#66707a;--line:#d9d6cf;--high:#9aa5b1;--low:#d9a441;
--goal:#3aa76d;--quad:#178a6b;--human:#2c4fa3;--smoke:rgba(120,120,120,.35);--accent:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#15181b;--panel:#1d2125;--ink:#e6e8ea;--muted:#9aa3ab;--line:#30363c;
--high:#5d6873;--low:#a87d2c;--smoke:rgba(200,200,200,.28)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Noto Sans SC",sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;flex-wrap:wrap;align-items:baseline}
header h1{font-size:16px;margin:0}header .r{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:12px;padding:12px 16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}
canvas{width:100%;height:auto;display:block;border-radius:6px}
.ctl{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
.ctl input[type=range]{flex:1;min-width:160px}button,select{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.list{max-height:300px;overflow:auto}.list div{padding:3px 0;border-bottom:1px dashed var(--line)}
.now{background:rgba(192,57,43,.08)}h2{font-size:13px;margin:0 0 6px;color:var(--muted);font-weight:600}
.legend span{display:inline-block;margin-right:10px;color:var(--muted)}
</style></head><body>
<header><h1 id="title">Breach Arena</h1><span class="r" id="result"></span><span class="r mono" id="meta"></span></header>
<main>
<section class="card"><canvas id="cv" width="1200" height="800"></canvas>
<div class="ctl"><button id="play">▶</button><input id="slider" type="range" min="0" value="0" step="1">
<span class="mono" id="clock">0.00 s</span><select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></div>
<div class="legend" style="margin-top:6px"><span>■ 高墙</span><span style="color:var(--low)">■ 低墙 0.7 m</span>
<span style="color:var(--quad)">▲ 四足</span><span style="color:var(--human)">● 人</span><span>红线 = 正在锁定</span></div>
<div class="mono" id="status" style="margin-top:6px"></div></section>
<aside>
<div class="card"><h2>四足决策（模型/规则返回）</h2><div class="list mono" id="decisions"></div></div>
<div class="card" style="margin-top:12px"><h2>事件</h2><div class="list mono" id="events"></div></div>
</aside></main>
<script>
const R = /*REPLAY*/null;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const cv = document.getElementById('cv'), g = cv.getContext('2d');
const [W, H] = R.map.arena.size, S = Math.min(cv.width / (W + 1), cv.height / (H + 1));
const X = x => cv.width / 2 + x * S, Y = y => cv.height / 2 - y * S;
const T = R.trace, slider = document.getElementById('slider');
slider.max = T.length - 1;
const res = R.summary.result || {};
document.getElementById('title').textContent = `Breach Arena · ${R.map.id} ${R.map.title}`;
document.getElementById('result').textContent = `结果：${res.winner === 'quadruped' ? '四足到点' : res.winner === 'human' ? '人方胜（' + res.reason + '）' : res.winner} @ ${res.t} s`;
document.getElementById('meta').textContent = `${R.rules_version} · 四足=${R.players.quadruped} · 人=${R.players.human} · 时序=${R.timing} · 第${R.round}局`;
const decisions = R.quadruped.log.filter(e => e.type === 'response');
document.getElementById('decisions').innerHTML = decisions.map(d => `<div data-t="${d.t}">[${d.t.toFixed(2)}s 收到, 发出 ${d.sent?.toFixed?.(2) ?? '-'}s] ${
  (d.plan || []).map(s => s.skill + (s.params.to ? '→' + s.params.to : s.params.throw_from ? '@' + s.params.throw_from : '')).join(' / ') || ('错误: ' + d.error)}<br>${
  (d.assessment && (d.assessment.decision_summary || '')) || ''}</div>`).join('');
const evs = R.events.filter(e => !['detected', 'lost'].includes(e.type));
document.getElementById('events').innerHTML = evs.map(e => `<div data-t="${e.t}">${e.t.toFixed(2)} ${e.type} ${
  e.reason || e.winner || (e.center ? JSON.stringify(e.center) : '') || ''}</div>`).join('');
function rect(c, fill) {
  const [cx, cy] = c.center, [sx, sy] = c.size;
  g.fillStyle = fill; g.fillRect(X(cx - sx / 2), Y(cy + sy / 2), sx * S, sy * S);
}
function draw(i) {
  const s = T[i]; g.clearRect(0, 0, cv.width, cv.height);
  g.fillStyle = css('--panel'); g.fillRect(X(-W / 2), Y(H / 2), W * S, H * S);
  g.strokeStyle = css('--line'); g.lineWidth = 1;
  for (let x = -W / 2; x <= W / 2; x += 2) { g.beginPath(); g.moveTo(X(x), Y(-H / 2)); g.lineTo(X(x), Y(H / 2)); g.stroke(); }
  for (let y = -H / 2; y <= H / 2; y += 2) { g.beginPath(); g.moveTo(X(-W / 2), Y(y)); g.lineTo(X(W / 2), Y(y)); g.stroke(); }
  R.map.arena.covers.forEach(c => rect(c, c.size[2] < 1 ? css('--low') : css('--high')));
  const gl = R.map.goal; g.fillStyle = css('--goal'); g.globalAlpha = .35;
  g.beginPath(); g.arc(X(gl.center[0]), Y(gl.center[1]), gl.radius * S, 0, 7); g.fill(); g.globalAlpha = 1;
  g.fillStyle = css('--muted'); g.font = '12px system-ui';
  Object.entries(R.map.points).forEach(([k, p]) => { if (['door', 'throw', 'staging', 'standoff', 'feint'].includes(p.kind)) {
    g.fillText(k, X(p.xy[0]) + 4, Y(p.xy[1]) - 4); g.fillRect(X(p.xy[0]) - 1.5, Y(p.xy[1]) - 1.5, 3, 3); } });
  R.events.filter(e => e.type === 'smoke_released' && e.active_at - 0.4 <= s.t && s.t < e.expires_at).forEach(e => {
    g.fillStyle = css('--smoke'); g.globalAlpha = s.t >= e.active_at ? 1 : .35;
    g.beginPath(); g.arc(X(e.center[0]), Y(e.center[1]), e.radius * S, 0, 7); g.fill(); g.globalAlpha = 1; });
  [0, 1].forEach(a => { const p = s.path[a]; if (!p.length) return; g.strokeStyle = a ? css('--human') : css('--quad');
    g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(s.xy[a][0]), Y(s.xy[a][1])); p.forEach(q => g.lineTo(X(q[0]), Y(q[1]))); g.stroke(); g.setLineDash([]); });
  if (s.visible[1]) { g.strokeStyle = s.lock[1] > 0 ? css('--accent') : 'rgba(128,128,128,.6)'; g.lineWidth = s.lock[1] > 0 ? 2.5 : 1;
    g.beginPath(); g.moveTo(X(s.xy[1][0]), Y(s.xy[1][1])); g.lineTo(X(s.xy[0][0]), Y(s.xy[0][1])); g.stroke(); g.lineWidth = 1; }
  const q = s.xy[0], h = s.xy[1];
  g.save(); g.translate(X(q[0]), Y(q[1])); g.rotate(-s.yaw[0]); g.fillStyle = css('--quad');
  g.beginPath(); g.moveTo(14, 0); g.lineTo(-9, 8); g.lineTo(-9, -8); g.closePath(); g.fill(); g.restore();
  g.fillStyle = s.stunned ? css('--muted') : css('--human'); g.beginPath(); g.arc(X(h[0]), Y(h[1]), 9, 0, 7); g.fill();
  g.strokeStyle = css('--human'); g.beginPath(); g.moveTo(X(h[0]), Y(h[1]));
  g.lineTo(X(h[0] + Math.cos(s.yaw[1]) * 1.2), Y(h[1] + Math.sin(s.yaw[1]) * 1.2)); g.stroke();
  if (s.lock[1] > 0) { g.fillStyle = css('--accent'); g.fillRect(X(h[0]) - 20, Y(h[1]) - 22, 40 * Math.min(1, s.lock[1]), 4); }
  document.getElementById('clock').textContent = s.t.toFixed(2) + ' s';
  document.getElementById('status').textContent = `四足: ${s.mode[0]}  姿态 ${s.posture}${s.casting ? '  [投烟中]' : ''}   |   人: ${s.mode[1]}${s.stunned ? '  [失能]' : ''}   锁定 ${s.lock[1].toFixed(2)} s`;
  document.querySelectorAll('[data-t]').forEach(el => el.classList.toggle('now', Math.abs(+el.dataset.t - s.t) < 0.5));
}
let i = 0, playing = false, last = 0;
slider.oninput = () => { i = +slider.value; draw(i); };
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? '❚❚' : '▶'; last = performance.now(); };
function loop(now) {
  if (playing) { const sp = +document.getElementById('speed').value, dt = (now - last) / 1000 * sp; last = now;
    const target = T[i].t + dt; while (i < T.length - 1 && T[i + 1].t <= target) i++;
    if (i >= T.length - 1) { playing = false; document.getElementById('play').textContent = '▶'; }
    slider.value = i; draw(i); }
  requestAnimationFrame(loop);
}
draw(0); requestAnimationFrame(loop);
</script></body></html>
"""
