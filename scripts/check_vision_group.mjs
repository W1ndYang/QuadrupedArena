// Offline browser check for the comparison hub and its three local report frames.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,access,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=path.resolve(process.argv[2]);
const data=JSON.parse(await readFile(path.join(root,'comparison.json'),'utf8'));
assert.equal(data.trials.length,3);
for(const row of data.trials)for(const key of ['report','onboard_gif','third_person_gif']){
  assert.ok(row[key],key);await access(path.resolve(root,row[key]));
}
for(const item of Object.values(data.shared_files_audit))assert.equal(item.identical,true);
const profile=await mkdtemp(path.join(tmpdir(),'arena-group-check-'));
const out=path.join(root,'viewer_checks');await mkdir(out,{recursive:true});
const chrome=spawn('google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;const errors=[];const wait=ms=>new Promise(r=>setTimeout(r,ms));
try{
  const ws=await new Promise((resolve,reject)=>{let buf='';const timer=setTimeout(()=>reject(Error('Launch timeout')),20000);chrome.stderr.on('data',b=>{buf+=b;const m=buf.match(/DevTools listening on (ws:\/\/\S+)/);if(m){clearTimeout(timer);resolve(m[1])}});chrome.on('exit',code=>{clearTimeout(timer);reject(Error('Browser exit '+code))})});
  const pages=await(await fetch(`http://127.0.0.1:${new URL(ws).port}/json/list`)).json();
  socket=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j});
  let id=0;const pending=new Map();
  socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params);if(pending.has(m.id)){const p=pending.get(m.id);clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}};
  const cdp=(method,params={})=>new Promise((resolve,reject)=>{const n=++id,timer=setTimeout(()=>reject(Error('CDP timeout')),15000);pending.set(n,{resolve,reject,timer});socket.send(JSON.stringify({id:n,method,params}))});
  const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
  const until=async expression=>{for(let i=0;i<80;i++){if(await evaluate(expression))return;await wait(100)}throw Error('Timeout '+expression)};
  await cdp('Runtime.enable');await cdp('Page.enable');await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(path.join(root,'index.html')).href});await until("document.querySelectorAll('#trial option').length===3");
  assert.equal(await evaluate("document.querySelectorAll('tbody tr').length"),3);
  for(let i=0;i<3;i++){
    await evaluate(`document.getElementById('trial').selectedIndex=${i};document.getElementById('trial').dispatchEvent(new Event('change'))`);
    let loaded=false;
    for(let n=0;n<80;n++){const tree=await cdp('Page.getFrameTree');if(tree.frameTree.childFrames?.some(f=>f.frame.url.endsWith(data.trials[i].report.replace('../','')))){loaded=true;break}await wait(100)}
    assert.ok(loaded,'iframe '+i+' loads expected report');
  }
  const shot=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(path.join(out,'comparison.png'),Buffer.from(shot.data,'base64'));
  await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.ok(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'));
  assert.equal(errors.length,0);
  const result={status:'passed',checks:['all report and six GIF links exist','same rules, background prompt and five frozen source files','three rows and selectable local report frames','mobile layout','no JavaScript errors']};
  await writeFile(path.join(out,'result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
}finally{socket?.close();chrome.kill('SIGTERM')}
