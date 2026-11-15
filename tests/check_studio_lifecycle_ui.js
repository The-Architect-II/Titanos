/* Actual Studio lifecycle source with DOM/fetch doubles; no browser or model execution. */
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const base=path.join(__dirname,'../src/titanos/web'),nodes=new Map(),all=node=>[node,...node.children.flatMap(all)];
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.files=[];this._value=null;this._text='';this.hidden=false;this.attributes={};}
  set id(value){this._id=value;nodes.set(value,this);}get id(){return this._id;}
  set value(value){this._value=String(value);}get value(){return this._value??(this.tagName==='select'?this.options[0]?.value||'':'');}
  set maxLength(value){if(Number(value)<0)throw Error("Failed to set the 'maxLength' property: value is negative");this._maxLength=Number(value);}get maxLength(){return this._maxLength??-1;}
  get options(){return this.children.filter(node=>node.tagName==='option');}
  set textContent(value){this._text=String(value);this.replaceChildren();}get textContent(){return this._text+this.children.map(node=>node.textContent).join('');}
  set innerHTML(value){throw Error('Lifecycle/model text must not use innerHTML');}
  append(...items){for(const item of items){item.remove();item.parentElement=this;this.children.push(item);}}
  prepend(...items){for(const item of items.reverse()){item.remove();item.parentElement=this;this.children.unshift(item);}}
  replaceChildren(...items){for(const item of this.children)item.parentElement=null;this.children=[];if(this.tagName==='select')this._value=null;this.append(...items);}
  remove(){if(this.parentElement){const siblings=this.parentElement.children;siblings.splice(siblings.indexOf(this),1);this.parentElement=null;}}
  setAttribute(name,value){this.attributes[name]=String(value);}
  removeAttribute(name){delete this.attributes[name];if(name.toLowerCase()==='maxlength')this._maxLength=undefined;}
  addEventListener(){}
  focus(){document.activeElement=this;}
}
const page=new Element('document'),stack=[page];
for(const match of fs.readFileSync(path.join(base,'index.html'),'utf8').matchAll(/<\/?([\w-]+)([^>]*)>|([^<]+)/g)){
  if(match[3]){const node=new Element('text');node.textContent=match[3];stack.at(-1).append(node);continue;}
  const [token,tag,attrs]=match;if(token.startsWith('</')){while(stack.length>1&&stack.pop().tagName!==tag){}continue;}
  const node=new Element(tag);for(const attr of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)){const [,key,value]=attr;node[key]=value===undefined?true:value;}
  stack.at(-1).append(node);if(!['meta','link','input','br'].includes(tag))stack.push(node);
}
const get=id=>{assert(nodes.has(id),'Missing HTML id '+id);return nodes.get(id);};
const settings=all(page).find(node=>node.class==='settings'),details=all(get('generate')).find(node=>node.tagName==='details');
let model={id:'chat/model',task:'chat',status:'unloaded'},detailsCalls=0,confirmed=false,configuredDevice='cpu',sourceRevision='abc123',detailsReply=null,failInfo=false,authFailure=false,durableRecord=null,runtimeRecord=null;
const beta={id:'beta',task:'chat',status:'loaded'};
const requests=[],response=value=>({ok:true,json:async()=>value,blob:async()=>new Blob()});
const failure=(status,message)=>({ok:false,status,json:async()=>({error:{message}})});
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};
const stream=text=>({ok:true,body:new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({choices:[{delta:{content:text}}]})+'\n\ndata: [DONE]\n\n'));controller.close();}})});
class Form extends FormData {constructor(form){super();if(form)for(const node of all(form))if(node.name&&!node.disabled)this.set(node.name,node.value);}}
const document={activeElement:null,getElementById:get,createElement:tag=>new Element(tag),
  querySelectorAll:query=>{assert.equal(query,'.settings input, .settings select');return all(settings).filter(node=>['input','select'].includes(node.tagName));},
  querySelector:query=>{assert.equal(query,'#generate details');return details;}};
const sessionValues=new Map(),sessionStorage={getItem:key=>sessionValues.get(key)||null,setItem:(key,value)=>sessionValues.set(key,value),removeItem:key=>sessionValues.delete(key)};
const context={console,Headers,FormData:Form,Blob,AbortController,URL,TextDecoder,setTimeout,document,sessionStorage,
  confirm:()=>{confirmed=true;return true;},location:{origin:'http://localhost'},
  fetch:async(route,options={})=>{
    const request={route,options,json:typeof options.body==='string'?JSON.parse(options.body):null};requests.push(request);
    if(route==='/v1/models'){if(authFailure)return failure(401,'Invalid API key');return response({data:[model,beta]});}
    if(route==='/api/info'){if(authFailure)return failure(401,'Invalid API key');if(failInfo)return failure(503,'Transient metadata failure');return response({models:{'chat/model':{task:'chat',backend:'transformers',device:configuredDevice},beta:{task:'chat',backend:'transformers',device:'beta-device'}},settings:{},model_setup:{runtime:{configured:true,recommended_profile:'win-cpu-x64',profiles:[{id:'win-cpu-x64',title:'CPU · Windows',one_stop_supported:true}]}},
      inference_contracts:{require_loaded:{revision:1,routes:['/v1/chat/completions']}}});
    }
    if(route==='/api/models/capabilities')return response({task:'chat',backend:'transformers',source:{repository:'publisher/chat',revision:request.json.model==='beta'?'beta123':sourceRevision},implemented:['chat']});
    if(route==='/api/jobs'&&!options.method)return response({data:[
      {id:'job-running',model:'chat/model',task:'image',status:'running'},
      {id:'job-complete',model:'chat/model',task:'speech',status:'completed',result:{text:'done'}},
    ]});
    if(route==='/api/acquisitions?limit=100')return response({data:durableRecord?[durableRecord]:[]});
    if(route==='/api/acquisitions/durable-job')return response(durableRecord);
    if(route==='/api/runtime-setups')return response({data:runtimeRecord?[runtimeRecord]:[]});
    if(route==='/api/runtime-setups/runtime-job')return response(runtimeRecord);
    if(route==='/api/models/load'){model={...model,status:'loaded'};return response({model:model.id,status:'loaded'});}
    if(route==='/api/models/unload'){model={...model,status:'unloaded'};return response({model:model.id,status:'unloaded'});}
    if(route==='/api/models/details'){detailsCalls++;if(detailsReply)return await detailsReply.promise;return response({model:model.id,device:'cpu',runtime:'fixture-runtime'});}
    if(route.startsWith('/api/models/registrations/')&&options.method==='DELETE'){model={...model,status:'removed'};return response({});}
    if(route==='/v1/chat/completions')return stream('Hello from the loaded model.');
    if(route==='/api/models/register'){configuredDevice='cuda:1';sourceRevision='replacement456';model={...model,status:'unloaded'};return response({});}
    throw Error('Unexpected request '+route);
  }};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(base,'app.js'),'utf8'),context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  await tick();
  for(const id of ['load','unload','model-details','remove-registration','require-loaded','model-runtime'])get(id);
  assert.equal(detailsCalls,0,'Discovery loaded runtime details');
  for(const value of ['chat/model','chat','transformers','cpu','unloaded','publisher/chat','abc123'])assert(get('models').textContent.includes(value));
  const modelCards=all(get('models')).filter(node=>String(node.className||'').split(' ').includes('model'));assert.equal(modelCards.length,2);
  assert(String(modelCards[0].className).includes('selected'));assert.equal(modelCards[0].attributes['aria-current'],'true');
  assert(all(modelCards[0]).some(node=>node.className==='model-head'));
  assert(all(modelCards[0]).some(node=>String(node.className||'').includes('status-chip')));
  const modelSelect=modelCards[0].children.find(node=>node.className==='model-select'),technical=modelCards[0].children.find(node=>node.className==='model-technical');
  assert.equal(modelSelect.tagName,'button');assert(technical);assert(!all(modelSelect).includes(technical),'Technical details was nested inside the model selection control');
  assert.match(get('connection').parentElement.className,/connected/);
  assert.match(get('load').attributes['aria-label'],/Load chat\/model/);assert.match(get('remove-registration').attributes['aria-label'],/Remove.*chat\/model/);
  const jobButtons=all(get('jobs')).filter(node=>node.tagName==='button');assert.equal(jobButtons.length,2);
  assert(jobButtons.some(button=>/Cancel image job job-running for chat\/model/.test(button.attributes['aria-label'])));
  assert(jobButtons.some(button=>/Open speech job job-complete for chat\/model/.test(button.attributes['aria-label'])));
  assert.equal(get('load').disabled,false);assert.equal(get('unload').disabled,true);assert.equal(get('model-details').disabled,true);
  assert.equal(get('require-loaded').parentElement.hidden,false);

  await get('load').onclick();assert.deepEqual(requests.find(r=>r.route==='/api/models/load').json,{model:'chat/model'});
  assert.equal(get('load').disabled,true);assert.equal(get('unload').disabled,false);assert.equal(get('model-details').disabled,false);
  await get('model-details').onclick();assert.equal(detailsCalls,1);assert.match(get('model-runtime').textContent,/fixture-runtime/i);assert.match(get('model-runtime').textContent,/cpu/i);
  detailsReply=deferred();const staleSuccess=get('model-details').onclick();await tick();get('model').value='beta';get('model').onchange();
  detailsReply.resolve(response({model:'chat/model',device:'alpha-device',runtime:'stale-alpha-runtime'}));await staleSuccess;
  assert.equal(get('model').value,'beta');assert.equal(get('model-runtime').hidden,true);assert.doesNotMatch(get('model-runtime-output').textContent,/stale-alpha-runtime/);
  get('status').textContent='Selected beta';get('model').value='chat/model';get('model').onchange();detailsReply=deferred();
  const staleError=get('model-details').onclick();await tick();get('model').value='beta';get('model').onchange();
  detailsReply.resolve(failure(503,'stale alpha details failure'));await staleError;
  assert.equal(get('status').textContent,'Selected beta');assert.doesNotMatch(get('status').textContent,/stale alpha/i);
  detailsReply=null;get('model').value='chat/model';get('model').onchange();

  get('require-loaded').checked=true;get('prompt').value='Hello';await get('generate').onsubmit({preventDefault(){}});
  const chat=requests.find(r=>r.route==='/v1/chat/completions');assert.equal(chat.json.require_loaded,true);
  assert.equal(document.activeElement,all(get('results')).find(node=>node.className==='label'),'Completed stream did not focus its result heading');

  for(const [name,value] of Object.entries({name:'chat/model',path:'D:/Models/replacement'}))all(get('register')).find(node=>node.name===name).value=value;
  get('register-task').value='chat';get('register-backend').value='transformers';get('register-device').value='cuda:1';get('register-replace').checked=true;
  await get('register').onsubmit({preventDefault(){},target:get('register')});
  assert.match(get('models').textContent,/cuda:1.*replacement456/i);assert.equal(get('model-runtime').hidden,true,'Replacement left stale runtime details visible');
  get('require-loaded').checked=false;get('prompt').value='After replacement';await get('generate').onsubmit({preventDefault(){}});
  const replacementChat=requests.filter(r=>r.route==='/v1/chat/completions').at(-1);
  assert.equal(replacementChat.json.messages.length,1,'Replacement inherited conversation history from the prior checkpoint');

  await get('unload').onclick();assert.equal(get('load').disabled,false);assert.equal(get('model-details').disabled,true);
  await get('remove-registration').onclick();assert.equal(confirmed,true);
  assert(requests.some(r=>r.route==='/api/models/registrations/chat%2Fmodel'&&r.options.method==='DELETE'));
  assert.match(get('status').textContent,/registration removed.*files.*not deleted/i);
  failInfo=true;await get('refresh').onclick();
  assert.match(get('connection').textContent,/Connected/);assert.equal(get('prompt').maxLength,-1);assert.doesNotMatch(get('status').textContent,/maxLength|negative/i);
  failInfo=false;authFailure=true;await get('refresh').onclick();assert.match(get('status').textContent,/Invalid API key/);
  authFailure=false;durableRecord={id:'durable-job',status:'failed',error:'network interrupted',plan:{registry_name:'recovered-model'},progress:{stage:'downloading',completed:10,total:100}};
  sessionStorage.setItem('titanos.acquisition',JSON.stringify({id:'durable-job',key:'stable-key',plan:'{}'}));
  get('key').value='correct-key';await get('connect').onclick();await tick();await tick();
  assert.match(get('connection').textContent,/Connected/);assert.doesNotMatch(get('status').textContent,/Invalid API key/);
  assert.match(get('setup-plan').textContent,/recovered-model.*Retry \/ resume/s);
  runtimeRecord={id:'runtime-job',status:'interrupted',error:'restart interrupted runtime setup',created_at:99,plan:{runtime:'llama.cpp',release:'b10809',profile:'win-cpu-x64',license:'MIT'},progress:{stage:'interrupted',completed:10,total:100}};
  await get('connect').onclick();await tick();await tick();
  assert.match(get('setup-plan').textContent,/Preparing selected managed runtime.*Retry runtime setup/s);
  console.log('PASS: explicit load/details/unload/remove, configured/runtime identity, source provenance and require_loaded handshake. Mocked server only.');
})().catch(error=>{console.error(error);process.exitCode=1;});
