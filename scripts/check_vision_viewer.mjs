// Local Chromium smoke/integration test; no network or npm dependencies.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=path.resolve(process.argv[2]),read=async file=>JSON.parse(await readFile(path.join(root,file),'utf8'));
const decisions=await read('decisions.json'),frames=await read('frame_index.json');
const out=path.join(root,'viewer_checks');await mkdir(out,{recursive:true});
const profile=await mkdtemp(path.join(tmpdir(),'arena-vision-check-'));
const chrome=spawn('google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;const checks=[],exceptions=[];const wait=ms=>new Promise(r=>setTimeout(r,ms));
try{
 const ws=await new Promise((resolve,reject)=>{let text='';const timeout=setTimeout(()=>reject(Error('Launch timeout')),20000);chrome.stderr.on('data',chunk=>{text+=chunk;const m=text.match(/DevTools listening on (ws:\/\/\S+)/);if(m){clearTimeout(timeout);resolve(m[1])}});chrome.on('exit',code=>{clearTimeout(timeout);reject(Error('Browser exit '+code))});});
 const pages=await (await fetch(`http://127.0.0.1:${new URL(ws).port}/json/list`)).json();socket=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j});
 let id=0;const pending=new Map();socket.onmessage=event=>{const m=JSON.parse(event.data);if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);clearTimeout(p.timeout);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}};
 const cdp=(method,params={})=>new Promise((resolve,reject)=>{const n=++id,timeout=setTimeout(()=>reject(Error('CDP timeout')),15000);pending.set(n,{resolve,reject,timeout});socket.send(JSON.stringify({id:n,method,params}))});
 const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 const until=async expr=>{for(let i=0;i<60;i++){if(await evaluate(expr))return;await wait(100)}throw Error('Timeout '+expr)};
 const shot=async name=>{const r=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(path.join(out,name),Buffer.from(r.data,'base64'))};
 await cdp('Runtime.enable');await cdp('Page.enable');await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});await cdp('Page.navigate',{url:pathToFileURL(path.join(root,'report.html')).href});
 await until(`document.querySelectorAll('#decisionSelect option').length===${decisions.length}`);await shot('overview.png');
 for(const tab of ['overview','decisions','playback','protocol']){await evaluate(`document.querySelector('[data-tab="${tab}"]').click()`);assert.deepEqual(await evaluate(`[...document.querySelectorAll('[role=tabpanel]')].filter(x=>!x.classList.contains('hidden')).map(x=>x.id)`),[tab])}checks.push('four tabs switch correctly');
 if(await evaluate("!!document.querySelector('[data-tab=thirdPerson]')")){await evaluate("document.querySelector('[data-tab=thirdPerson]').click()");await until("document.querySelector('#thirdPerson video').readyState>=1");assert.ok(await evaluate("document.querySelector('#thirdPerson video').duration>0"));await shot('third_person.png');checks.push('third-person replay tab and MP4 metadata load')}
 await evaluate("document.querySelector('[data-tab=decisions]').click()");
 for(let i=0;i<decisions.length;i++){
   await evaluate(`showStep(${i})`);assert.deepEqual(await evaluate("JSON.parse(document.getElementById('actionJSON').textContent)"),await read(`actions/${String(i).padStart(3,'0')}.json`));
   const obs=await read(`observations/${String(i).padStart(3,'0')}.json`);assert.equal(await evaluate("document.querySelectorAll('.thumb').length"),obs.frames.length);
   for(let j=0;j<obs.frames.length;j++){await evaluate(`document.querySelectorAll('.thumb')[${j}].click()`);await until("document.getElementById('mainFrame').complete && document.getElementById('mainFrame').naturalWidth===768");assert.ok((await evaluate("document.getElementById('mainFrame').src")).endsWith(obs.frames[j].file))}
   if(decisions[i].assessment){assert.equal(await evaluate("document.querySelectorAll('#summary button').length"),obs.frames.length);for(let j=0;j<obs.frames.length;j++){await evaluate(`document.querySelectorAll('#summary button')[${j}].click()`);await until("document.getElementById('mainFrame').complete && document.getElementById('mainFrame').naturalWidth===768");assert.ok((await evaluate("document.getElementById('mainFrame').src")).endsWith(obs.frames[j].file))}}
 }checks.push(`${decisions.length} original action JSON objects match; every input frame loads and switches`);
 const representative=Math.min(4,decisions.length-1),sample=await read(`observations/${String(representative).padStart(3,'0')}.json`);
 await evaluate(`showStep(${representative});selectFrame(${Math.min(3,sample.frames.length-1)})`);await until("document.getElementById('mainFrame').complete");await shot('representative_decision.png');
 await evaluate("document.getElementById('copy').click()");await until("document.getElementById('copyStatus').textContent.length>0");assert.equal(await evaluate("document.getElementById('copyStatus').textContent"),'已复制');checks.push('JSON copy button works');
 await evaluate("document.querySelector('[data-tab=playback]').click();playFrame(0);document.getElementById('play').click()");await wait(650);assert.ok(await evaluate('playIndex>=1'));await evaluate('stopPlayback()');
 await evaluate(`playFrame(${frames.length-1})`);await until("document.getElementById('playFrame').complete && document.getElementById('playFrame').naturalWidth===768");assert.ok((await evaluate("document.getElementById('playLabel').textContent")).includes(frames.at(-1).simulation_seconds.toFixed(2)));checks.push('timed frame playback and final partial tick');
 await evaluate("document.querySelector('[data-tab=decisions]').click()");await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await shot('mobile.png');assert.ok(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'));checks.push('390px mobile layout without horizontal overflow');
 assert.equal(exceptions.length,0);checks.push('no JavaScript exceptions');const result={status:'passed',checks};await writeFile(path.join(out,'result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
}finally{if(socket)socket.close();chrome.kill('SIGTERM')}
