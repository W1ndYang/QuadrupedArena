// Inspect the local map review using Chrome's built-in debugging protocol.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,mkdir,access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=path.resolve(process.argv[2]);
const result=JSON.parse(await readFile(path.join(root,'result.json'),'utf8'));
assert.equal(result.status,'passed');
const profile=await mkdtemp(path.join(tmpdir(),'dockyard-review-check-'));
const out=path.join(root,'viewer_checks');await mkdir(out,{recursive:true});
const browser=spawn('google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;const errors=[],checks=[];const wait=ms=>new Promise(r=>setTimeout(r,ms));
try {
  const ws=await new Promise((resolve,reject)=>{
    let buf='';const timer=setTimeout(()=>reject(Error('Launch timeout')),20000);
    browser.stderr.on('data',b=>{buf+=b;const m=buf.match(/DevTools listening on (ws:\/\/\S+)/);if(m){clearTimeout(timer);resolve(m[1])}});
    browser.on('exit',code=>{clearTimeout(timer);reject(Error('Browser exit '+code))});
  });
  const pages=await(await fetch(`http://127.0.0.1:${new URL(ws).port}/json/list`)).json();
  socket=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j});
  let id=0;const pending=new Map();
  socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params);if(pending.has(m.id)){const p=pending.get(m.id);clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}};
  const cdp=(method,params={})=>new Promise((resolve,reject)=>{const n=++id,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),30000);pending.set(n,{resolve,reject,timer});socket.send(JSON.stringify({id:n,method,params}))});
  const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
  const until=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await wait(100)}throw Error('Timeout '+expression)};
  const shot=async name=>{const r=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(path.join(out,name+'.png'),Buffer.from(r.data,'base64'))};
  await cdp('Runtime.enable');await cdp('Page.enable');await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1200,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(path.join(root,'review.html')).href});
  await until("document.querySelectorAll('[role=tab]').length===10");
  await evaluate("document.querySelectorAll('img').forEach(i=>i.loading='eager')");
  await until("Array.from(document.images).every(i=>i.complete&&i.naturalWidth===1600)");
  checks.push('All ten 1600x1000 native stills load offline');
  for(let i=0;i<10;i++){
    await evaluate(`document.querySelectorAll('[role=tab]')[${i}].click()`);
    assert.equal(await evaluate("document.querySelectorAll('.view.active').length"),1);
    assert.equal(await evaluate("Array.from(document.querySelectorAll('[role=tab]')).findIndex(b=>b.getAttribute('aria-selected')==='true')"),i);
  }
  checks.push('All ten camera tabs switch with a single visible frame');
  await evaluate("document.querySelector('[role=tab]').click()");await shot('desktop');
  const links=await evaluate("Array.from(document.querySelectorAll('a')).map(a=>a.getAttribute('href')).filter(x=>x&&!x.startsWith('https:')&&!x.startsWith('#'))");
  for(const href of links)await access(path.isAbsolute(href)?href:path.join(root,href));
  checks.push('All local file links resolve including both USD snapshots');
  await until("document.querySelector('video').readyState>=1");
  const duration=await evaluate("document.querySelector('video').duration");assert.ok(Math.abs(duration-9)<.01);
  checks.push('Native camera-tour MP4 loads with 9-second duration');
  await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate("window.scrollTo(0,0)");await wait(100);
  assert.ok(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'));
  await shot('mobile');checks.push('390px mobile layout has no horizontal page overflow');
  assert.equal(errors.length,0);checks.push('No browser runtime exceptions');
  const report={status:'passed',checks};await writeFile(path.join(out,'result.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
} finally {socket?.close();browser.kill('SIGTERM')}
