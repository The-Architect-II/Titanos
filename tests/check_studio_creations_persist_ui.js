// Completed creations survive later translation passes (mode/locale changes); the empty-state message returns only when
// the jobs list is really empty. DOM double whose textContent setter clears children like a real DOM; no browser, no model.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
class Element {
  constructor(id=''){this.id=id;this.value='';this.children=[];this.options=this.children;this.files=[];this.hidden=false;this._text='';this.dataset={};}
  get textContent(){return this._text||this.children.map(c=>c.textContent).join('');}
  set textContent(value){this._text=String(value??'');this.children.length=0;}   // assigning text removes element children, as in the DOM
  replaceChildren(...items){this.children.length=0;this._text='';this.append(...items);}
  append(...items){for(const item of items){if(item instanceof Element)item.parentElement=this;this.children.push(item);}}
  prepend(...items){for(const item of items)if(item instanceof Element)item.parentElement=this;this.children.unshift(...items);}
  after(node){}
  remove(){}
  setAttribute(name,value){(this.attributes??={})[String(name)]=String(value);if(String(name).startsWith('data-'))this.dataset[String(name).slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=String(value);}
  getAttribute(name){return this.attributes?.[String(name)]??null;}
  removeAttribute(name){if(this.attributes)delete this.attributes[String(name)];const key=String(name).startsWith('data-')?String(name).slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase()):null;if(key)delete this.dataset[key];}
  get classList(){const self=this;return {add(){},remove(){},toggle(){},contains(){return false;}};}
}
const nodes=new Map(),created=[],get=id=>nodes.get(id)||nodes.set(id,new Element(id)).get(id);
get('register-variant').parentElement=new Element();get('require-loaded').parentElement=new Element();get('status').parentElement=new Element();
const bound=()=>[...nodes.values(),...created].filter(n=>n.attributes&&n.attributes['data-i18n']!==undefined);
let jobs=[],catalog=[],jobsDenied=false,modelsDenied=false,revokedKey=null,modelsGate=null;const sent=[];
// refused answers shaped like the server's error envelope; the Authorization header is read as the real api() sends it
const denial=(status,code)=>({ok:false,status,json:async()=>({error:{code,message:'HTTP '+status+' '+code}})});
async function refuse(route,options){
  const auth=options?.headers?.get?.('Authorization')||'';sent.push({route,auth});
  if(route==='/v1/models'&&modelsGate){const gate=modelsGate;modelsGate=null;await gate;}
  if(revokedKey&&auth==='Bearer '+revokedKey)return denial(401,'invalid_key');
  if(modelsDenied&&route==='/v1/models')return denial(401,'authentication_required');
  if(jobsDenied&&route==='/api/jobs')return denial(403,'scope_denied');
  return null;
}
const completed={id:'060e020b073f42ff88f8357ab2699707',object:'asset.job',task:'speech',model:'vega-qwen-1.7b',status:'completed',result:{url:'http://127.0.0.1/v1/files/x.wav',format:'wav',model:'vega-qwen-1.7b'}};
const context={console,Headers,FormData,Blob,AbortController,URL,TextDecoder,setTimeout,
  location:{origin:'http://localhost'},
  document:{getElementById:get,createElement:tag=>{const n=new Element();n.tagName=tag;created.push(n);return n;},querySelectorAll:selector=>selector==='[data-i18n]'?bound():[],querySelector:get,documentElement:new Element('html'),addEventListener(){}},
  fetch:async(route,options)=>(await refuse(route,options))||({ok:true,json:async()=>{if(route==='/v1/models')return {data:catalog};if(route==='/api/info')return {models:Object.fromEntries(catalog.map(m=>[m.id,{task:m.task,backend:'qwen3-tts',device:'cuda:0'}])),settings:{},inference_contracts:{}};
    if(route==='/api/models/capabilities')return {task:'speech',backend:'qwen3-tts',source:{repository:'Qwen/Qwen3-TTS-12Hz-1.7B-Base',revision:'fd4b2543'},implemented:['speech']};if(route==='/api/jobs')return {data:jobs};return {data:[]};}})};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/titanos/web/i18n.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/titanos/web/app.js'),'utf8'),context);
const list=()=>get('jobs');
const rows=()=>list().children.filter(c=>c.className&&c.className.startsWith('job '));
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  // 1. empty list: the empty message is shown and bound for translation
  jobs=[];await context.refreshJobs();
  assert.match(list().textContent,/No jobs yet/);assert.equal(rows().length,0);
  // 2. a completed job arrives
  jobs=[completed];await context.refreshJobs();
  assert.equal(rows().length,1,'one job row expected');assert.doesNotMatch(list().textContent,/No jobs yet/);
  assert.equal(list().attributes?.['data-i18n'],undefined,'stale empty-state binding must be cleared once rows exist');
  // 3. the operator switches mode / locale (both call applyTranslations): the completed job must persist
  context.applyTranslations();
  assert.equal(rows().length,1,'translation pass wiped the creations list');assert.doesNotMatch(list().textContent,/No jobs yet/);
  context.TitanosI18n.setLocale('es');context.applyTranslations();
  assert.equal(rows().length,1,'locale translation pass wiped the creations list');
  context.TitanosI18n.setLocale('en');
  // 4. the job disappears from the API: the empty message returns and is translated
  jobs=[];await context.refreshJobs();
  assert.equal(rows().length,0);assert.match(list().textContent,/No jobs yet/);
  context.TitanosI18n.setLocale('es');context.applyTranslations();
  assert.doesNotMatch(list().textContent,/No jobs yet/);assert.ok(list().textContent.length>0,'translated empty message expected');
  context.TitanosI18n.setLocale('en');context.applyTranslations();assert.match(list().textContent,/No jobs yet/);
  // 5. and rows come back again after the empty state
  jobs=[completed];await context.refreshJobs();context.applyTranslations();
  assert.equal(rows().length,1);
  // 6. the model list has the same shape: an empty catalog binds the empty message on the container; once a model is
  //    registered and rendered, a translation pass must not wipe the model cards, and the empty message returns when empty.
  const models=()=>get('models'),cards=()=>models().children.filter(c=>String(c.className||'').split(' ').includes('model'));
  catalog=[];await context.refresh();
  assert.match(models().textContent,/No models/i);assert.equal(cards().length,0);
  catalog=[{id:'vega-qwen-1.7b',task:'speech',status:'unloaded'}];await context.refresh();
  assert.equal(cards().length,1,'one model card expected');assert.equal(models().attributes?.['data-i18n'],undefined,'stale empty-state binding must be cleared once model cards exist');
  context.applyTranslations();context.TitanosI18n.setLocale('es');context.applyTranslations();context.TitanosI18n.setLocale('en');
  assert.equal(cards().length,1,'translation pass wiped the model list');
  catalog=[];await context.refresh();assert.equal(cards().length,0);assert.match(models().textContent,/No models/i);
  // 7. the translation-pass guard alone: a bound container that (for whatever reason) still carries its text binding while
  //    holding element children is left alone, and the row-level bound nodes inside it are still translated
  const tr=vm.runInContext('tr',context),setContent=vm.runInContext('setContent',context);  // script-level bindings of app.js
  const box=get('guard-box');setContent(box,tr('jobs.empty'));
  const inner=context.document.createElement('span');setContent(inner,tr('jobs.empty'));box.append(inner);
  assert.equal(box.attributes['data-i18n'],'jobs.empty','container binding present for this check');
  context.TitanosI18n.setLocale('es');context.applyTranslations();
  assert.equal(box.children.length,1,'guard: a bound container with element children was rewritten');
  assert.doesNotMatch(inner.textContent,/No jobs yet/,'row-level bound node inside the container must translate');
  context.TitanosI18n.setLocale('en');context.applyTranslations();assert.equal(box.children.length,1);
  // 8. a key without the 'jobs' scope: GET /api/jobs answers 403. That is not a connection failure: the jobs section says
  //    the list is not available for this key (translatable), the header stays Connected and the status line is no error.
  catalog=[{id:'vega-qwen-1.7b',task:'speech',status:'unloaded'}];modelsDenied=true;await context.refresh();
  assert.match(get('connection').textContent,/Connection needed/);assert.equal(get('status').className,'error');   // a real failure first
  modelsDenied=false;jobsDenied=true;jobs=[completed];await context.refresh();
  assert.match(get('connection').textContent,/^Connected, 1 model$/,'a jobs 403 was reported as a connection failure');
  assert.notEqual(get('status').className,'error','a jobs 403 put an error on the status line');
  assert.equal(rows().length,0);assert.match(list().textContent,/not available for this key/);
  assert.equal(list().attributes?.['data-i18n'],'jobs.unavailable');
  context.TitanosI18n.setLocale('es');context.applyTranslations();assert.match(list().textContent,/no están disponibles para esta clave/);
  context.TitanosI18n.setLocale('en');context.applyTranslations();assert.match(list().textContent,/not available for this key/);
  await context.refreshJobs();assert.match(list().textContent,/not available for this key/);   // the Refresh button path
  jobsDenied=false;await context.refreshJobs();assert.equal(rows().length,1,'rows did not return once the key may list jobs');
  // other failures of /api/jobs are still errors for the caller
  vm.runInContext("key='tk_gone'",context);revokedKey='tk_gone';
  await assert.rejects(context.refreshJobs(),error=>error.status===401);revokedKey=null;vm.runInContext("key=''",context);
  // 9. a refresh in flight with a key the tab has since replaced (a Studio self-rotation adopts the new secret) is answered
  //    401 for the old key: that answer is stale, so Studio refreshes again with the current key instead of reporting it
  vm.runInContext("key='tk_old'",context);let open;modelsGate=new Promise(resolve=>{open=resolve;});
  const pending=context.refresh();vm.runInContext("key='tk_new'",context);revokedKey='tk_old';const mark=sent.length;open();await pending;
  assert.match(get('connection').textContent,/^Connected, 1 model$/,'a stale old-key 401 was reported as a connection failure');
  assert.notEqual(get('status').className,'error');
  const models401=sent.filter(r=>r.route==='/v1/models');assert.equal(models401.at(-1).auth,'Bearer tk_new','the retry did not use the current key');
  assert.deepEqual(sent.slice(mark).filter(r=>r.auth==='Bearer tk_old'&&r.route!=='/v1/models'),[],'the old key was sent after the replacement');
  revokedKey=null;vm.runInContext("key=''",context);
  // 10. a reopened job's receipt carries the job's own finish time, not the moment it was opened
  const finished=Date.UTC(2026,0,2,3,4,5)/1000,reopened=new Element();reopened.tagName='article';
  context.appendReceipt(reopened,{model:'vega-qwen-1.7b',seconds:1.5,at:finished});
  const stamp=context.clockTime(finished);assert.match(stamp,/^\d\d:\d\d:\d\d$/);
  assert(reopened.textContent.includes('Recorded '+stamp),'the receipt time is not the job time: '+reopened.textContent);
  console.log('PASS: completed creations and model cards persist across translation passes (mode/locale); the empty messages appear only for empty lists and stay translatable; a jobs 403 shows the translated not-available state without a connection failure; a stale old-key 401 after a key replacement is retried with the current key. DOM double only.');
})().catch(error=>{console.error(error);process.exitCode=1;});
