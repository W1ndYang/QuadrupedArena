// Local Chromium integration check; no npm dependencies, no external requests.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const directory=path.resolve(process.argv[2]);
const entries=JSON.parse(await readFile(path.join(directory,'decisions.json'),'utf8'));
const count=entries.length;assert.ok(count>0);
const selected=Math.min(17,count-1),selectedStem=String(selected).padStart(3,'0');
const selectedOutcome=JSON.parse(await readFile(path.join(directory,'decisions',selectedStem+'_outcome.json'),'utf8'));
const rowIndex=Math.min(10,count-1);
const output=path.join(directory,'viewer_checks');await mkdir(output,{recursive:true});
const profile=await mkdtemp(path.join(tmpdir(),'arena-viewer-check-'));
const chrome=spawn('google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;
const checks=[];const exceptions=[];
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
try {
  const browserWs=await new Promise((resolve,reject)=>{let text='';const timer=setTimeout(()=>reject(Error('Browser launch timeout')),20000);chrome.stderr.on('data',chunk=>{text+=chunk;const match=text.match(/DevTools listening on (ws:\/\/\S+)/);if(match){clearTimeout(timer);resolve(match[1]);}});chrome.on('exit',code=>{clearTimeout(timer);reject(Error('Browser exited '+code));});});
  const port=new URL(browserWs).port;
  const pages=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  let sequence=0;const pending=new Map();
  socket.onmessage=event=>{const item=JSON.parse(event.data);if(item.method==='Runtime.exceptionThrown')exceptions.push(item.params.exceptionDetails);if(item.id&&pending.has(item.id)){const {resolve,reject,timer}=pending.get(item.id);clearTimeout(timer);pending.delete(item.id);item.error?reject(Error(JSON.stringify(item.error))):resolve(item.result);}};
  const cdp=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method));},15000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const until=async expression=>{for(let i=0;i<60;i++){if(await evaluate(expression))return;await wait(100);}throw Error('Condition timed out '+expression);};
  const snapshot=async name=>{const r=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(path.join(output,name),Buffer.from(r.data,'base64'));};
  await cdp('Runtime.enable');await cdp('Page.enable');
  await cdp('Page.bringToFront');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1600,height:1150,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(path.join(directory,'report.html')).href});
  await until(`document.querySelectorAll('.step-button').length===${count}`);
  await until("document.getElementById('scene-image').complete && document.getElementById('scene-image').naturalWidth===960");
  assert.equal(await evaluate("document.getElementById('panel-decisions').hidden"),false);
  await snapshot('desktop_decision_01.png');checks.push('default tab and native image render');
  for(let i=0;i<count;i++){
    await evaluate(`document.querySelectorAll('.step-button')[${i}].click()`);
    await until("document.getElementById('scene-image').complete && document.getElementById('scene-image').naturalWidth===960");
    const rendered=await evaluate("JSON.parse(document.getElementById('json-content').textContent)");
    const original=JSON.parse(await readFile(path.join(directory,'requests',String(i).padStart(3,'0')+'.json'),'utf8'));
    assert.deepEqual(rendered,original);
    assert.match(await evaluate("document.getElementById('scene-image').src"),new RegExp(`/screenshots/${String(i).padStart(3,'0')}\\.png$`));
  }
  checks.push(`all ${count} original request JSON objects and screenshot paths match`);
  assert.equal(await evaluate("document.getElementById('next').disabled"),true);
  await evaluate(`document.querySelectorAll('.step-button')[${selected}].click();document.getElementById('image-after').click()`);
  await until("document.getElementById('scene-image').complete && document.getElementById('scene-image').naturalWidth===960");
  assert.ok((await evaluate("document.getElementById('image-caption').textContent")).includes(selectedOutcome.simulation_end.toFixed(2)+'s'));
  await snapshot('desktop_selected_after.png');
  const types={observation:`observations/${selectedStem}.json`,outcome:`decisions/${selectedStem}_outcome.json`,commitment:`decisions/${selectedStem}.json`,request:`requests/${selectedStem}.json`};
  for(const [kind,file] of Object.entries(types)){
    await evaluate(`document.querySelector('[data-json="${kind}"]').click()`);
    assert.deepEqual(await evaluate("JSON.parse(document.getElementById('json-content').textContent)"),JSON.parse(await readFile(path.join(directory,file),'utf8')));
  }
  checks.push('all four JSON tabs retain exact source content; before/after images switch');
  await evaluate("document.getElementById('copy-json').click()");
  await until("document.getElementById('copy-status').textContent.length>0");
  assert.equal(await evaluate("document.getElementById('copy-status').textContent"),'已复制完整 JSON');
  checks.push('copy control response: '+await evaluate("document.getElementById('copy-status').textContent"));
  await evaluate("document.getElementById('zoom').click()");assert.equal(await evaluate("document.getElementById('image-dialog').open"),true);
  await evaluate("document.getElementById('close-zoom').click()");assert.equal(await evaluate("document.getElementById('image-dialog').open"),false);
  checks.push('native image zoom dialog opens and closes');
  for(const tab of ['replay','results','method','decisions']){
    await evaluate(`document.querySelector('[data-tab="${tab}"]').click()`);
    assert.deepEqual(await evaluate("[...document.querySelectorAll('main>[role=tabpanel]')].filter(p=>!p.hidden).map(p=>p.id)"),['panel-'+tab]);
  }
  await evaluate(`document.querySelector('[data-tab=results]').click();document.querySelectorAll('.row-link')[${rowIndex}].click()`);
  assert.equal(await evaluate(`document.querySelector('.step-button[aria-current=step]').textContent.startsWith('${String(rowIndex+1).padStart(2,'0')}')`),true);
  assert.equal(await evaluate("document.getElementById('panel-decisions').hidden"),false);
  checks.push('four main tabs and timing-table navigation');
  await evaluate("document.querySelector('[data-tab=decisions]').focus()");
  await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
  assert.equal(await evaluate("document.getElementById('panel-replay').hidden"),false);checks.push('keyboard tab navigation');
  await evaluate("document.querySelector('[data-tab=decisions]').click()");
  const savedHash=await evaluate('location.hash');await cdp('Page.reload');
  await until(`document.querySelectorAll('.step-button').length===${count}`);
  assert.equal(await evaluate('location.hash'),savedHash);checks.push('step/tab selection survives refresh');
  await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await until("document.getElementById('scene-image').complete && document.getElementById('scene-image').naturalWidth===960");
  await snapshot('mobile_decision.png');
  const dimensions=await evaluate('({scroll:document.documentElement.scrollWidth,viewport:window.innerWidth,overflow:[...document.querySelectorAll("body *")].filter(e=>e.getBoundingClientRect().right>window.innerWidth+1&&!e.closest(".steps")).slice(0,8).map(e=>({tag:e.tagName,id:e.id,cls:e.className,right:e.getBoundingClientRect().right}))})');
  assert.ok(dimensions.scroll<=dimensions.viewport,JSON.stringify(dimensions));
  checks.push('390px layout has no page-wide horizontal overflow');
  assert.equal(exceptions.length,0);checks.push('no JavaScript runtime exceptions');
  const result={status:'passed',checks,screenshots:['desktop_decision_01.png','desktop_selected_after.png','mobile_decision.png']};
  await writeFile(path.join(output,'result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
} finally {if(socket)socket.close();chrome.kill('SIGTERM');}
