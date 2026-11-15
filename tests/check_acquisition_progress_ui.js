// Studio progress line labels totals with the record's unit; a file-count stage is never shown as bytes. DOM double only.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
class Element {
  constructor(id=''){this.id=id;this.value='';this.children=[];this.options=this.children;this.files=[];this.hidden=false;this.textContent='';}
  replaceChildren(){this.children=[];this.options=this.children;}
  append(...items){for(const item of items)if(item instanceof Element)item.parentElement=this;this.children.push(...items);}
  prepend(...items){for(const item of items)if(item instanceof Element)item.parentElement=this;this.children.unshift(...items);}
  remove(){}
  setAttribute(name,value){(this.attributes??={})[String(name)]=String(value);}getAttribute(name){return this.attributes?.[String(name)]??null;}removeAttribute(name){if(this.attributes)delete this.attributes[String(name)];}
}
const nodes=new Map(),get=id=>nodes.get(id)||nodes.set(id,new Element(id)).get(id);
get('register-variant').parentElement=new Element();get('require-loaded').parentElement=new Element();get('status').parentElement=new Element();
const root=path.join(__dirname,'../src/titanos/web');
const context={console,Headers,FormData,Blob,AbortController,URL,TextDecoder,setTimeout,
  location:{origin:'http://localhost'},document:{getElementById:get,createElement:tag=>{const node=new Element();node.tagName=tag;return node;},querySelectorAll:()=>[],querySelector:get},
  fetch:async route=>({ok:true,json:async()=>{if(route==='/v1/models')return {data:[]};if(route==='/api/info')return {models:{},settings:{},inference_contracts:{}};return {data:[]};}})};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root,'i18n.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),context);
const line=()=>get('setup-plan').children.filter(node=>node.tagName==='p').map(node=>node.textContent).join('\n');
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  const plan={registry_name:'openai/whisper-tiny.en'};
  // server-tagged file count during the per-file download stage
  context.renderAcquisition({id:'a1',status:'running',plan,progress:{stage:'downloading_files',completed:0,total:12,unit:'files'}});
  assert.match(line(),/0 \/ 12 files/);assert.doesNotMatch(line(),/12 bytes/);
  // byte stages keep bytes
  context.renderAcquisition({id:'a2',status:'running',plan,progress:{stage:'verifying',completed:155013603,total:155013603,unit:'bytes'}});
  assert.match(line(),/155013603 \/ 155013603 bytes/);
  // older records without a unit: inferred from the stage, so a file count still is not labelled bytes
  context.renderAcquisition({id:'a3',status:'running',plan,progress:{stage:'downloading_files',completed:3,total:12}});
  assert.match(line(),/3 \/ 12 files/);
  context.renderAcquisition({id:'a4',status:'running',plan,progress:{stage:'downloading',completed:10,total:100}});
  assert.match(line(),/10 \/ 100 bytes/);
  // runtime setup renderer shares the line: archive members are files, byte stages are bytes
  context.renderRuntimeSetup({id:'r1',status:'running',plan:{runtime:'llama.cpp',release:'b10809',profile:'cuda124',license:'MIT'},progress:{stage:'extracting',completed:4,total:9,unit:'files'}});
  assert.match(line(),/4 \/ 9 files/);
  context.renderRuntimeSetup({id:'r2',status:'running',plan:{runtime:'llama.cpp',release:'b10809',profile:'cuda124',license:'MIT'},progress:{stage:'downloading',completed:1,total:2048,unit:'bytes'}});
  assert.match(line(),/1 \/ 2048 bytes/);
  // no total: no unit suffix at all
  context.renderAcquisition({id:'a5',status:'queued',plan,progress:{stage:'queued',completed:0,total:null,unit:'bytes'}});
  assert.doesNotMatch(line(),/ \/ /);
  // translations carry the unit too
  context.TitanosI18n.setLocale('es');
  context.renderAcquisition({id:'a6',status:'running',plan,progress:{stage:'downloading_files',completed:2,total:12,unit:'files'}});
  assert.match(line(),/2 \/ 12 archivos/);
  context.TitanosI18n.setLocale('en');
  console.log('PASS: acquisition/runtime progress totals carry their unit (files vs bytes, server-tagged and inferred), in the fallback and translated paths. DOM double only.');
})().catch(error=>{console.error(error);process.exitCode=1;});
