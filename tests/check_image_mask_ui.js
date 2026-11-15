/* Actual Studio source with DOM/fetch/deferred doubles; no browser or model inference. */
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const base=path.join(__dirname,'../src/titanos/web'),nodes=new Map();
const all=node=>[node,...node.children.flatMap(all)];
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.files=[];this._value=null;this._text='';this.hidden=false;}
  set id(value){this._id=value;nodes.set(value,this);}get id(){return this._id;}
  set value(value){this._value=String(value);if(this.type==='file'&&this._value==='')this.files=[];}get value(){return this._value??(this.tagName==='select'?this.options[0]?.value||'':'');}
  get options(){return this.children.filter(node=>node.tagName==='option');}
  set textContent(value){this._text=String(value);this.replaceChildren();}get textContent(){return this._text+this.children.map(node=>node.textContent).join('');}
  set innerHTML(value){throw Error('Result text must not use innerHTML');}
  append(...children){for(const child of children){child.remove();child.parentElement=this;this.children.push(child);}}
  prepend(...children){for(const child of children.reverse()){child.remove();child.parentElement=this;this.children.unshift(child);}}
  replaceChildren(...children){for(const child of this.children)child.parentElement=null;this.children=[];if(this.tagName==='select')this._value=null;this.append(...children);}
  remove(){if(this.parentElement){const siblings=this.parentElement.children;siblings.splice(siblings.indexOf(this),1);this.parentElement=null;}}
  setAttribute(name,value){(this.attributes??={})[String(name)]=String(value);}getAttribute(name){return this.attributes?.[String(name)]??null;}removeAttribute(name){if(this.attributes)delete this.attributes[String(name)];}
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
const standardModels=[{id:'image-a',task:'image'},{id:'chat-a',task:'chat'}];
const response=value=>({ok:true,json:async()=>value});
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const result={data:[{b64_json:'fixture-png'}]};
let models=[...standardModels],metadata,inference;
const requests=[];
const context={console,Headers,FormData,Blob,AbortController,URL,TextDecoder,setTimeout,
  location:{origin:'http://localhost'},document:{getElementById:get,createElement:tag=>new Element(tag),
    querySelectorAll:query=>{assert.equal(query,'.settings input, .settings select');return all(settings).filter(node=>['input','select'].includes(node.tagName));},
    querySelector:query=>{assert.equal(query,'#generate details');return details;}},
  fetch:async(route,options={})=>{
    const request={route,options,json:typeof options.body==='string'?JSON.parse(options.body):null};requests.push(request);
    if(route==='/v1/models')return metadata?metadata():response({data:models});
    if(route==='/api/info')return response({models:{},settings:{},inference_contracts:{}});
    if(route==='/api/models/capabilities')return response({backend:'diffusers',task:'image',instruction_edit:null});
    if(route==='/api/jobs'&&!options.method)return response({data:[]});
    assert(inference,'Unexpected request '+route);return inference(request);
  }};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(base,'app.js'),'utf8'),context);
const choose=id=>{get('model').value=id;get('model').onchange();};
const submit=()=>get('generate').onsubmit({preventDefault(){}});
const posts=()=>requests.filter(request=>request.options.method==='POST'&&request.route!=='/api/models/capabilities');
const bytes=async file=>Buffer.from(await file.arrayBuffer()).toString();
(async()=>{
  await tick();assert.equal(get('model').value,'image-a');assert.equal(get('image-edit-fields').hidden,false);
  assert.equal(get('image-source').accept,'image/*');assert.equal(get('image-mask').accept,'image/png,.png');
  assert.equal(get('preserve-unmasked').value,'true');assert.match(get('image-edit-fields').textContent,/exactly the source image/);
  choose('chat-a');assert.equal(get('image-edit-fields').hidden,true);choose('image-a');
  get('prompt').value='Original prompt';get('file').files=[new Blob(['stale other modality'])];
  // No source retains durable generation jobs and omits all edit-only fields.
  inference=async request=>request.route==='/api/jobs'?response({id:'generation'}):response({status:'completed',result});
  await submit();assert.equal(posts().at(-1).route,'/api/jobs');
  assert.deepEqual(posts().at(-1).json,{model:'image-a',task:'image',prompt:'Original prompt'});
  const beforeInvalid=posts().length,resultsBeforeInvalid=get('results').children.length;
  get('image-mask').files=[new Blob(['mask'],{type:'image/png'})];await submit();
  assert.equal(posts().length,beforeInvalid);assert.match(get('status').textContent,/source image/);
  assert.equal(get('results').children.length,resultsBeforeInvalid);
  // Source-only editing omits preservation, even if the next mask setting is off.
  get('image-source').files=[new Blob(['source'],{type:'image/png'})];get('image-mask').files=[];
  get('preserve-unmasked').value='false';inference=async()=>response(result);await submit();
  let request=posts().at(-1),form=request.options.body;
  assert.equal(request.route,'/v1/images/edits');assert(request.options.signal);
  assert.deepEqual([...form.keys()],['model','prompt','image']);assert.equal(await bytes(form.get('image')),'source');
  // Masked edits retain true/false exactly and show safe metadata after the image.
  const masking={preserve_unmasked:true,compositing:'copy_unmasked_rgb',input:'<img onerror=bad>'};
  get('image-mask').files=[new Blob(['mask'],{type:'image/png'})];
  get('size').value='32x32';get('steps').value='2';get('seed').value='0';
  for(const preserve of ['true','false']){
    get('preserve-unmasked').value=preserve;inference=async()=>response({...result,masking:{...masking,preserve_unmasked:preserve==='true'}});
    await submit();request=posts().at(-1);form=request.options.body;
    assert.equal(request.json,null);assert.equal(form.get('preserve_unmasked'),preserve);
    assert.equal(await bytes(form.get('mask')),'mask');assert.equal(form.get('size'),'32x32');
    assert.equal(form.get('steps'),'2');assert.equal(form.get('seed'),'0');
    const card=get('results').children[0],imageIndex=card.children.findIndex(node=>node.tagName==='img');
    const trace=card.children.find(node=>node.tagName==='details');assert(card.children.indexOf(trace)>imageIndex);
    assert.equal(all(card).filter(node=>node.tagName==='img').length,1);
    assert.match(trace.textContent,/Mask and pixel preservation details/);assert.match(trace.textContent,/<img onerror=bad>/);
    assert.equal(JSON.parse(trace.children.find(node=>node.tagName==='pre').textContent).preserve_unmasked,preserve==='true');
  }
  get('clear-image-inputs').onclick();assert.equal(get('image-source').files.length,0);assert.equal(get('image-mask').files.length,0);
  get('image-source').files=[new Blob(['source'],{type:'image/png'})];get('image-mask').files=[new Blob(['mask'],{type:'image/png'})];
  const beforeError=get('results').children.length;
  inference=async()=>({ok:false,status:400,json:async()=>({error:{message:'Mask dimensions must match the source'}})});
  await submit();assert.equal(get('results').children.length,beforeError);assert.match(get('status').textContent,/dimensions/);
  // Pending edit owns its original model/files/options, including through Refresh and Stop cleanup.
  const wait=deferred();inference=async()=>wait.promise;
  get('preserve-unmasked').value='false';const pending=submit();await tick();
  const owned=posts().at(-1),signal=owned.options.signal,count=posts().length,beforeLate=get('results').children.length;
  get('image-source').files=[new Blob(['next source'])];get('image-mask').files=[];
  get('preserve-unmasked').value='true';get('prompt').value='Next prompt';get('seed').value='9';
  assert.equal(await bytes(owned.options.body.get('image')),'source');assert.equal(await bytes(owned.options.body.get('mask')),'mask');
  assert.equal(owned.options.body.get('preserve_unmasked'),'false');assert.equal(owned.options.body.get('prompt'),'Original prompt');
  assert.equal(owned.options.body.get('seed'),'0');
  models=standardModels.filter(model=>model.id!=='image-a');await get('refresh').onclick();
  assert.equal(get('model').value,'image-a');assert(get('submit').disabled&&get('model').disabled&&!get('stop').hidden);
  assert.equal(get('image-edit-fields').hidden,false);choose('chat-a');assert.equal(get('model').value,'image-a');
  await submit();assert.equal(posts().length,count);
  const finalRefresh=deferred();metadata=()=>finalRefresh.promise;
  await get('stop').onclick();assert(signal.aborted);wait.resolve(response({...result,masking}));await tick();
  assert.equal(get('results').children.length,beforeLate,'Late aborted edit rendered');
  await submit();assert.equal(posts().length,count);assert.equal(get('submit').disabled,true);
  finalRefresh.resolve(response({data:models}));await pending;metadata=null;
  assert.equal(get('model').value,'chat-a');assert.equal(get('image-edit-fields').hidden,true);
  assert.equal(get('stop').hidden,true);assert.equal(get('submit').disabled,false);assert.equal(get('status').textContent,'Stopped.');
  assert.equal(get('prompt').value,'Next prompt');assert.equal(get('preserve-unmasked').value,'true');
  console.log('PASS: generation/edit routing, mask/source validation, omission/false, multipart snapshots, safe masking metadata, and deferred Refresh/Stop ownership. No browser or trained model.');
})().catch(error=>{console.error(error);process.exitCode=1;});
