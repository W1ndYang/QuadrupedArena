// Native offline file:// inspection; no network or API model calls.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=path.resolve(process.argv[2]);
const data=JSON.parse(await readFile(path.join(root,'data/analysis.json'),'utf8'));
const profile=await mkdtemp(path.join(tmpdir(),'arena-research-check-'));
const out=path.join(root,'viewer_checks');await mkdir(out,{recursive:true});
const chrome=spawn('google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;const errors=[];const checks=[];const wait=ms=>new Promise(r=>setTimeout(r,ms));
try{
 const ws=await new Promise((resolve,reject)=>{let buf='';const timer=setTimeout(()=>reject(Error('Launch timeout')),20000);chrome.stderr.on('data',b=>{buf+=b;const m=buf.match(/DevTools listening on (ws:\/\/\S+)/);if(m){clearTimeout(timer);resolve(m[1])}});chrome.on('exit',code=>{clearTimeout(timer);reject(Error('Browser exit '+code))})});
 const pages=await(await fetch(`http://127.0.0.1:${new URL(ws).port}/json/list`)).json();socket=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j});
 let id=0;const pending=new Map();socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params);if(pending.has(m.id)){const p=pending.get(m.id);clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}};
 const cdp=(method,params={})=>new Promise((resolve,reject)=>{const n=++id,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),45000);pending.set(n,{resolve,reject,timer});socket.send(JSON.stringify({id:n,method,params}))});
 const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 const until=async expression=>{for(let i=0;i<150;i++){if(await evaluate(expression))return;await wait(100)}throw Error('Timeout '+expression)};
 const shot=async name=>{const result=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(path.join(out,name+'.png'),Buffer.from(result.data,'base64'))};
 await cdp('Runtime.enable');await cdp('Page.enable');await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await cdp('Page.navigate',{url:pathToFileURL(path.join(root,'index.html')).href});await until("document.querySelectorAll('#run-select option').length===7");
 assert.equal(await evaluate('RUNS.reduce((n,r)=>n+r.steps.length,0)'),105);checks.push('Seven runs and all 105 archived decisions exposed');
 await evaluate("document.querySelectorAll('img').forEach(i=>i.loading='eager')");await until("Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)");await shot('desktop_overview');checks.push('All static images load offline');
 let frameCount=0;
 for(let r=0;r<data.runs.length;r++){
  await evaluate(`selectRun(${r},0)`);assert.equal(await evaluate("document.getElementById('step-select').options.length"),data.runs[r].steps.length);
  for(let s=0;s<data.runs[r].steps.length;s++){
   await evaluate(`stepIndex=${s};renderStep()`);
   assert.deepEqual(await evaluate("JSON.parse(document.getElementById('record-json').textContent)"),data.runs[r].steps[s].raw);
   for(let f=0;f<data.runs[r].steps[s].frames.length;f++){
    await evaluate(`frameIndex=${f};frame()`);await until("document.querySelector('#frame-display img').complete&&document.querySelector('#frame-display img').naturalWidth>0");frameCount++;
   }
   await until("document.querySelector('#overhead-display img').complete&&document.querySelector('#overhead-display img').naturalWidth>0");
  }
  await evaluate("document.querySelector('.replay').open=true");await wait(100);assert.ok(await evaluate("document.querySelectorAll('.overhead-strip img').length>0"));await evaluate("document.querySelector('.replay').open=false");
 }
 checks.push('All 105 decision JSON blocks equal original data; '+frameCount+' input-frame appearances load');checks.push('All nearest/exact overhead images and replay galleries switch');
 await evaluate("selectRun(2,14);frameIndex=4;frame();document.querySelector('.replay').open=true");
 assert.equal(await evaluate('RUNS[selected].metric.score_observed'),13.22);
 assert.equal(await evaluate("document.querySelectorAll('#frame-display .pixel-overlay').length"),1);
 const videoDuration=await evaluate("new Promise((resolve,reject)=>{const v=document.createElement('video');v.preload='metadata';v.onloadedmetadata=()=>resolve(v.duration);v.onerror=()=>reject(Error('New max MP4 could not load'));v.src=document.querySelector('#replay-content a[href$=\".mp4\"]').href})");
 assert.ok(Math.abs(videoDuration-47.4166667)<.1);
 await evaluate("document.getElementById('explorer').scrollIntoView({behavior:'instant'})");await wait(100);
 await shot('desktop_ds_max_updated');await evaluate("document.querySelector('.replay').open=false");
 checks.push('Replacement DS max: 20 decisions, new t=28 foot marker, third-person video metadata and 13.22 score');
 await evaluate("selectRun(6,4);frameIndex=3;frame();document.getElementById('explorer').scrollIntoView({behavior:'instant'})");await wait(150);assert.equal(await evaluate("document.querySelectorAll('#frame-display .pixel-overlay').length"),1);
 await evaluate("document.getElementById('marker-toggle').click()");assert.equal(await evaluate("document.querySelectorAll('#frame-display .pixel-overlay').length"),0);await evaluate("document.getElementById('marker-toggle').click()");
 for(const tab of ['compiled','observation','timing','decision']){await evaluate(`document.querySelector('[data-tab="${tab}"]').click()`);assert.ok(await evaluate("document.getElementById('record-json').textContent.length>50"))}
 await shot('desktop_explorer');checks.push('Frame markers, all four record tabs, and midpoint-frame selection work');
 await evaluate("selectRun(0,0)");assert.equal(await evaluate("document.getElementById('previous').disabled"),true);await evaluate("document.getElementById('next').click()");assert.equal(await evaluate('stepIndex'),1);await evaluate("document.getElementById('previous').click()");assert.equal(await evaluate('stepIndex'),0);checks.push('Previous/next controls and boundaries');
 await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await evaluate("window.scrollTo({top:0,behavior:'instant'})");await wait(100);assert.ok(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'));await shot('mobile_overview');
 await evaluate("document.getElementById('explorer').scrollIntoView({behavior:'instant'})");await shot('mobile_explorer');checks.push('390 px layout has no page-level horizontal overflow');
 await cdp('Emulation.setEmulatedMedia',{media:'print'});assert.equal(await evaluate("getComputedStyle(document.querySelector('.topnav')).display"),'none');checks.push('Print stylesheet available');
 assert.equal(errors.length,0);checks.push('No browser runtime exceptions');const result={status:'passed',checks};await writeFile(path.join(out,'result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
}finally{socket?.close();chrome.kill('SIGTERM')}
