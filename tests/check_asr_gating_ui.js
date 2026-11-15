// Focused Studio dispatch check with a small DOM/fetch double; no browser/model.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
class Element {
  constructor(id=''){this.id=id;this.value='';this.children=[];this.options=this.children;this.files=[];this.hidden=false;this.textContent='';}
  replaceChildren(){this.children=[];this.options=this.children;if(['model','vad-gate'].includes(this.id))this.value='';}
  append(...items){for(const item of items)if(item instanceof Element)item.parentElement=this;this.children.push(...items);if(['model','vad-gate'].includes(this.id)&&!this.value)this.value=this.children[0]?.value||'';}
  prepend(...items){for(const item of items)if(item instanceof Element)item.parentElement=this;this.children.unshift(...items);}
  remove(){}
  setAttribute(name,value){(this.attributes??={})[String(name)]=String(value);}getAttribute(name){return this.attributes?.[String(name)]??null;}removeAttribute(name){if(this.attributes)delete this.attributes[String(name)];}
}
const nodes=new Map(),get=id=>nodes.get(id)||nodes.set(id,new Element(id)).get(id),requests=[];
get('register-variant').parentElement=new Element(); // The unrelated registration input lives inside a label.
get('require-loaded').parentElement=new Element();
get('status').parentElement=new Element();
let models=[{id:'asr',task:'transcription',status:'unloaded'},{id:'silero',task:'vad',status:'unloaded'}];
const context={console,Headers,FormData,Blob,AbortController,URL,TextDecoder,setTimeout,
  location:{origin:'http://localhost'},document:{getElementById:get,createElement:()=>new Element(),querySelectorAll:()=>[],querySelector:get},
  fetch:async(route,options={})=>({ok:true,json:async()=>{
    if(route==='/v1/models')return {data:models};if(route==='/api/info')return {models:{},settings:{},inference_contracts:{}};if(route==='/api/jobs')return {data:[]};
    assert.equal(route,'/v1/audio/transcriptions');requests.push(options.body);return {text:'',vad_gate:{asr_executed:false}};
  }})};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/titanos/web/app.js'),'utf8'),context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(get('vad-gate').options.map(o=>o.value),['inherit','off','model:silero']);
  get('file').files=[new Blob(['fixture'],{type:'audio/wav'})];
  for(const value of ['model:silero','off','inherit']){get('vad-gate').value=value;await get('generate').onsubmit({preventDefault(){}});}
  assert.equal(requests.length,3);assert.equal(requests[0].get('vad_model'),'silero');assert.equal(requests[0].get('vad_enabled'),'true');
  assert.equal(requests[1].get('vad_enabled'),'false');assert.equal(requests[1].has('vad_model'),false);
  assert.equal(requests[2].has('vad_enabled'),false);assert.equal(requests[2].has('vad_model'),false);
  assert.ok(requests.every(r=>r.get('response_format')==='verbose_json'));
  console.log('PASS: Studio registered VAD selection, explicit disable, inheritance and attributed-result format. Mocked requests only.');
})().catch(error=>{console.error(error);process.exitCode=1;});
