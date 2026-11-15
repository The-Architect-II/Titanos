/* Actual Studio HTML/app with the existing small DOM/fetch doubles; no browser or model. */
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const base=path.join(__dirname,'../src/titanos/web'),nodes=new Map(),all=node=>[node,...node.children.flatMap(all)];
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.files=[];this._value=null;this._text='';this.hidden=false;this.attributes={};}
  set id(value){this._id=value;nodes.set(value,this);}get id(){return this._id;}
  set list(value){throw TypeError('HTMLInputElement.list is read-only');}get list(){return this.attributes.list;}
  set value(value){this._value=String(value);if(this.type==='file'&&this._value==='')this.files=[];}
  get value(){return this._value??(this.tagName==='select'?this.options[0]?.value||'':this.tagName==='option'?this.textContent:'');}
  get options(){return this.children.filter(node=>node.tagName==='option');}
  set textContent(value){this._text=String(value);this.replaceChildren();}get textContent(){return this._text+this.children.map(node=>node.textContent).join('');}
  set innerHTML(value){throw Error('Untrusted text must not use innerHTML');}
  append(...children){for(const child of children){child.remove();child.parentElement=this;this.children.push(child);}}
  prepend(...children){for(const child of children.reverse()){child.remove();child.parentElement=this;this.children.unshift(child);}}
  replaceChildren(...children){for(const child of this.children)child.parentElement=null;this.children=[];if(this.tagName==='select')this._value=null;this.append(...children);}
  remove(){if(this.parentElement){const siblings=this.parentElement.children;siblings.splice(siblings.indexOf(this),1);this.parentElement=null;}}
  setAttribute(name,value){this.attributes[name]=String(value);}
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
const standardModels=[{id:'photo-tools',task:'image'},{id:'document-reader',task:'image-to-text'},
  {id:'tesseract-instructpix2pix-looking',task:'image'},{id:'tesseract-looking-ocr',task:'image-to-text'},{id:'chat-a',task:'chat'}];
const instruction={task:'image',backend:'diffusers',instruction_edit:{family:'stable-diffusion-instruct-pix2pix',instruction_edit:true,requires_input_image:true}};
const caps=new Map([['photo-tools',instruction],['document-reader',{task:'image-to-text',backend:'tesseract'}],
  ['tesseract-instructpix2pix-looking',{task:'image',backend:'diffusers',instruction_edit:null}],['tesseract-looking-ocr',{task:'image-to-text',backend:'transformers'}],
  ['chat-a',{task:'chat',backend:'transformers'}]]);
const response=value=>({ok:true,json:async()=>value});
const failure=message=>({ok:false,status:503,json:async()=>({error:{message}})});
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
class Form extends FormData {constructor(form){super();if(form)for(const node of all(form))if(node.name&&!node.disabled)this.set(node.name,node.value);}}
class Reader {readAsDataURL(file){reads.push({reader:this,file});}}
const result={data:[{b64_json:'fixture-png'}]},requests=[],reads=[];
let models=[...standardModels],inference,metadata,capability;
const context={console,Headers,FormData:Form,Blob,AbortController,URL,TextDecoder,setTimeout,FileReader:Reader,
  location:{origin:'http://localhost'},document:{getElementById:get,createElement:tag=>new Element(tag),
    querySelectorAll:query=>{assert.equal(query,'.settings input, .settings select');return all(settings).filter(node=>['input','select'].includes(node.tagName));},
    querySelector:query=>{assert.equal(query,'#generate details');return details;}},
  fetch:async(route,options={})=>{
    const request={route,options,json:typeof options.body==='string'?JSON.parse(options.body):null};requests.push(request);
    if(route==='/v1/models')return metadata?metadata():response({data:models});
    if(route==='/api/info')return response({models:{},settings:{},inference_contracts:{}});
    if(route==='/api/models/capabilities'){
      assert.equal(options.method,'POST');assert.deepEqual(Object.keys(request.json),['model']);
      assert(caps.has(request.json.model),'Capability request for an unrelated task');
      return capability?capability(request):response(caps.get(request.json.model));
    }
    if(route==='/api/devices?backend=openvino')return response({devices:[
      {id:'openvino:CPU',name:'CPU',runtime_address:'CPU',identity_kind:'runtime_address'},
      {id:'openvino:GPU',name:'Singleton GPU',runtime_address:'GPU',identity_kind:'runtime_address'},
      {id:'OV-gpu-uuid',uuid:'OV-gpu-uuid',name:'Pinned GPU',runtime_address:'GPU.1',identity_kind:'vendor_uuid'},
      {id:'OV-npu-uuid',uuid:'OV-npu-uuid',name:'Pinned NPU',runtime_address:'NPU',identity_kind:'vendor_uuid'},
      {id:'openvino:NPU',name:'NPU',runtime_address:'NPU',identity_kind:'runtime_address'},
    ]});
    if(route.startsWith('/api/devices'))return response({devices:[
      {id:'onnx:CPUExecutionProvider',name:'CPU provider',runtime_address:'CPUExecutionProvider',identity_kind:'runtime_address'},
      {id:'onnx:CUDAExecutionProvider',name:'Unsupported generic CUDA provider',runtime_address:'CUDAExecutionProvider',identity_kind:'runtime_address'},
      {id:'DML-00000000:12345678',name:'Arc test adapter',runtime_address:'dml:2',identity_kind:'dxgi_luid',
        device:'dml',device_id:2,luid:'00000000:12345678',software:false,provider_options:{device_id:'2'}},
    ]});
    if(route==='/api/jobs'&&!options.method)return response({data:[]});
    if(route==='/api/models/register'||route==='/api/models/unload')return response({});
    assert(inference,'Unexpected request '+route);return inference(request);
  }};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(base,'app.js'),'utf8'),context);
const choose=id=>{get('model').value=id;get('model').onchange();};
const submit=()=>get('generate').onsubmit({preventDefault(){}});
const posts=()=>requests.filter(request=>request.options.method==='POST'&&!request.route.startsWith('/api/models'));
const visible=id=>!get(id).parentElement.hidden;
const bytes=async file=>Buffer.from(await file.arrayBuffer()).toString();
const finishRead=value=>{reads.at(-1).reader.result=value;reads.at(-1).reader.onload();};
const upload=new Blob(['source image'],{type:'image/png'}),mask=new Blob(['stale mask'],{type:'image/png'});
async function ocr(){const pending=submit();await tick();finishRead('data:image/png;base64,AA==');await pending;return posts().at(-1);}
(async()=>{
  await tick();assert.equal(get('model').value,'photo-tools');
  assert.deepEqual([...new Set(requests.filter(r=>r.route==='/api/models/capabilities').map(r=>r.json.model))].sort(),[...caps.keys()].sort());
  assert.equal(get('image-source').required,true);assert.equal(get('mask-fields').hidden,true);
  assert.equal(get('image-mask').disabled,true);assert.equal(get('preserve-unmasked').disabled,true);
  assert.match(get('prompt-caption').textContent,/Instruction/i);
  for(const id of ['n','steps','seed','guidance_scale','image_guidance_scale'])assert(visible(id),'Missing instruction control '+id);
  assert.equal(visible('size'),false);
  // Registration keeps automatic task defaults while exposing explicit backend selection.
  assert.deepEqual(get('register-backend').options.map(option=>option.value),[
    'auto','transformers','diffusers','qwen3-tts','sherpa-onnx','ace-step','onnx','openvino','openvino-genai','tesseract','onnx-diffusers','llama-cpp','xtts'
  ]);
  assert(all(get('register-task')).some(option=>option.value==='tensor'));
  assert(all(get('register-task')).some(option=>option.value==='vad'));
  assert(all(get('register-task')).some(option=>option.value==='rerank'));
  assert(all(get('register-task')).some(option=>option.value==='depth-estimation'));
  assert(all(get('register-task')).some(option=>option.value==='image-segmentation'));
  for(const id of ['register-dtype','register-graph','register-voices','register-dml-adapter','register-disable-metacommands','register-replace','load-devices','register-submit','first-run'])get(id);
  assert.equal(get('register-backend').value,'auto');
  const registrations=()=>requests.filter(r=>r.route==='/api/models/register');
  const register=async()=>{const before=registrations().length;await get('register').onsubmit({preventDefault(){},target:get('register')});assert.equal(registrations().length,before+1,get('status').textContent);return registrations().at(-1).json;};
  const registrationTask=value=>{get('register-task').value=value;get('register-task').onchange();};
  for(const [name,value] of Object.entries({name:'registered',path:'D:/Models/local'}))all(get('register')).find(node=>node.name===name).value=value;
  registrationTask('chat');get('register-device').value='auto';
  assert.equal(get('register-variant').name,'variant');assert.equal(visible('register-variant'),false);assert.equal(get('register-variant').disabled,true);
  assert.deepEqual(await register(),{name:'registered',spec:{path:'D:/Models/local',task:'chat',device:'auto',backend:'transformers'}});
  // Automatic image/video registration exposes the suffix, while an empty suffix preserves the old payload.
  for(const task of ['image','video']){
    registrationTask(task);assert.equal(visible('register-variant'),true);assert.equal(get('register-variant').disabled,false);get('register-variant').value='';
    assert.deepEqual((await register()).spec,{path:'D:/Models/local',task,device:'auto',backend:'diffusers'});
    for(const variant of ['fp16','non_ema','fp16-fp32']){
      get('register-variant').value=variant;
      assert.deepEqual((await register()).spec,{path:'D:/Models/local',task,device:'auto',backend:'diffusers',load_options:{variant}});
    }
  }
  registrationTask('chat');assert.equal(visible('register-variant'),false);
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'chat',device:'auto',backend:'transformers'});
  get('register-backend').value='transformers';get('register-backend').onchange();
  assert(!['image','video'].includes(get('register-task').value));assert.equal(visible('register-variant'),false);
  get('register-variant').value='../hidden-stale-value';
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'chat',device:'auto',backend:'transformers'});
  get('register-backend').value='diffusers';get('register-backend').onchange();
  assert(['image','video'].includes(get('register-task').value));assert.equal(visible('register-variant'),true);
  get('register-variant').value='';assert.equal(Object.hasOwn((await register()).spec,'load_options'),false);
  for(const variant of ['bf16','non_ema','fp16-fp32']){
    get('register-variant').value=variant;assert.deepEqual((await register()).spec.load_options,{variant});
  }
  // Invoke the actual submit handler directly: HTML pattern checks alone must not admit malformed suffixes.
  for(const value of ['../fp16','fp16/fp32','fp16\\fp32','fp16:fp32','fp16.safetensors',' fp16','fp16 ','non ema','fp16\t','fp16\n','fp16\r','fp16\r\n','fπ16']){
    get('register-variant').value=value;const before=registrations().length;
    await get('register').onsubmit({preventDefault(){},target:get('register')});
    assert.equal(registrations().length,before,'Malformed variant was registered: '+JSON.stringify(value));
    assert.match(get('status').textContent,/variant|suffix|alphanumeric/i);
  }
  get('register-backend').value='tesseract';get('register-backend').onchange();
  assert.equal(get('register-task').value,'image-to-text');assert.equal(get('register-device').value,'cpu');
  assert.equal(visible('register-variant'),false);
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'image-to-text',device:'cpu',backend:'tesseract'});
  // Every safe backend exposes its required fields and serializes only active values.
  get('register-backend').value='transformers';get('register-backend').onchange();registrationTask('depth-estimation');
  get('register-device').value='cpu';get('register-dtype').value='float32';
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'depth-estimation',device:'cpu',backend:'transformers',dtype:'float32'});
  get('register-backend').value='onnx';get('register-backend').onchange();registrationTask('vad');
  assert.equal(get('register-device').value,'cpu');get('register-graph').value='silero.onnx';
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'vad',device:'cpu',backend:'onnx',graph:'silero.onnx'});
  get('register-backend').value='openvino';get('register-backend').onchange();registrationTask('tensor');
  get('register-device').value='NPU';get('register-graph').value='model.xml';
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'tensor',device:'NPU',backend:'openvino',graph:'model.xml'});
  get('register-backend').value='openvino-genai';get('register-backend').onchange();registrationTask('image');await get('load-devices').onclick();
  const genaiDevices=get('register-devices').options.map(item=>item.value);
  assert(genaiDevices.includes('CPU'));assert(genaiDevices.includes('GPU.0'));assert(genaiDevices.includes('OV-gpu-uuid'));
  assert(!genaiDevices.includes('GPU'));assert(!genaiDevices.includes('NPU'));assert(!genaiDevices.includes('OV-npu-uuid'));
  get('register-device').value='GPU.0';
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'image',device:'GPU.0',backend:'openvino-genai'});
  get('register-backend').value='qwen3-tts';get('register-backend').onchange();
  assert.equal(get('register-task').value,'speech');get('register-device').value='cuda:0';get('register-dtype').value='bfloat16';
  get('register-voices').value='{"vega":"D:/Voices/vega.wav"}';
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'speech',device:'cuda:0',backend:'qwen3-tts',dtype:'bfloat16',voices:{vega:'D:/Voices/vega.wav'}});
  get('register-backend').value='sherpa-onnx';get('register-backend').onchange();assert.equal(get('register-task').value,'speech');assert.equal(get('register-device').value,'cpu');
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'speech',device:'cpu',backend:'sherpa-onnx'});
  get('register-backend').value='ace-step';get('register-backend').onchange();assert.equal(get('register-task').value,'music');get('register-device').value='cpu';
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'music',device:'cpu',backend:'ace-step'});
  get('register-replace').checked=true;assert.equal((await register()).replace,true);get('register-replace').checked=false;
  // DirectML registration binds the current adapter index and LUID returned by inventory.
  get('register-backend').value='onnx-diffusers';get('register-backend').onchange();
  get('register-device').value='dml';await get('load-devices').onclick();get('register-dml-adapter').value='DML-00000000:12345678';
  assert(get('register-devices').options.some(item=>item.value==='cpu'),'ONNX CPU inventory did not offer the accepted device value');
  assert(!get('register-devices').options.some(item=>['CUDAExecutionProvider','dml:2'].includes(item.value)||/Arc test adapter/.test(item.textContent)),'Unsupported or adapter-specific ONNX address leaked into generic device suggestions');
  get('register-dml-adapter').onchange();
  assert.deepEqual((await register()).spec,{path:'D:/Models/local',task:'image',device:'dml',backend:'onnx-diffusers',
    load_options:{device_id:2,adapter_luid:'00000000:12345678'}});
  // Native/pickle registration remains visible but cannot be weakened into an HTTP request.
  for(const backend of ['llama-cpp','xtts']){
    const before=registrations().length;get('register-backend').value=backend;get('register-backend').onchange();
    assert.equal(get('register-submit').disabled,true);await get('register').onsubmit({preventDefault(){},target:get('register')});
    assert.equal(registrations().length,before);assert.match(get('registration-help').textContent,/CLI|titanos add/i);
  }
  // An instruction model requires a source even when form submission is invoked directly.
  const noSource=posts().length;get('prompt').value='  Paint the car blue.  ';get('image-mask').files=[mask];await submit();
  assert.equal(posts().length,noSource);assert.match(get('status').textContent,/source image/i);
  get('image-source').files=[upload];get('size').value='32x32';get('preserve-unmasked').value='false';
  get('steps').value='4';get('seed').value='0';get('n').value='2';get('guidance_scale').value='7.5';get('image_guidance_scale').value='1.5';
  get('temperature').value='1.2';get('max_tokens').value='128';inference=async()=>response(result);
  await submit();let request=posts().at(-1),form=request.options.body;
  assert.equal(request.route,'/v1/images/edits');assert(request.options.signal);assert.equal(request.json,null);
  assert.deepEqual([...form.keys()].sort(),['model','prompt','image','steps','seed','n','guidance_scale','image_guidance_scale'].sort());
  for(const [name,value] of Object.entries({model:'photo-tools',prompt:'  Paint the car blue.  ',steps:'4',seed:'0',n:'2',guidance_scale:'7.5',image_guidance_scale:'1.5'}))assert.equal(form.get(name),value);
  assert.equal(await bytes(form.get('image')),'source image');
  // Names do not infer backend/family; ordinary generation and masked editing retain their payloads.
  choose('tesseract-instructpix2pix-looking');assert.equal(get('image-source').required,false);
  assert.equal(get('mask-fields').hidden,false);assert.equal(get('image-mask').disabled,false);assert.equal(get('preserve-unmasked').disabled,false);
  assert.equal(visible('size'),true);assert.equal(visible('image_guidance_scale'),false);
  await submit();form=posts().at(-1).options.body;
  assert.deepEqual([...form.keys()].sort(),['model','prompt','image','size','steps','seed','mask','preserve_unmasked'].sort());
  assert.equal(form.get('preserve_unmasked'),'false');assert.equal(await bytes(form.get('mask')),'stale mask');
  get('clear-image-inputs').onclick();inference=async r=>r.route==='/api/jobs'?response({id:'ordinary-image'}):response({status:'completed',result});
  await submit();assert.deepEqual(posts().at(-1).json,{model:'tesseract-instructpix2pix-looking',task:'image',prompt:'  Paint the car blue.  ',size:'32x32',steps:4,seed:0});
  // Tesseract reads only its own options and freezes them before the asynchronous file read.
  choose('document-reader');get('file').files=[upload];assert.equal(get('prompt').required,false);
  for(const id of ['ocr_language','ocr_psm','ocr_geometry'])assert(visible(id),'Missing OCR control '+id);
  for(const id of ['max_tokens','temperature','top_p','image_guidance_scale'])assert.equal(visible(id),false);
  get('ocr_language').value='eng+deu';get('ocr_psm').value='6';get('ocr_geometry').value='false';get('language').value='fr';
  const reading=submit();await tick();get('ocr_language').value='fra';get('ocr_psm').value='3';get('ocr_geometry').value='true';
  inference=async()=>response({text:'Example OCR'});finishRead('data:image/png;base64,AA==');await reading;
  assert.equal(posts().at(-1).route,'/v1/images/ocr');assert.deepEqual(posts().at(-1).json,{model:'document-reader',image:'data:image/png;base64,AA==',language:'eng+deu',psm:6,include_geometry:false});
  get('ocr_language').value='';get('ocr_psm').value='';get('ocr_geometry').value='';
  assert.deepEqual((await ocr()).json,{model:'document-reader',image:'data:image/png;base64,AA=='});
  get('ocr_geometry').value='true';assert.equal((await ocr()).json.include_geometry,true);
  choose('tesseract-looking-ocr');assert.equal(visible('ocr_language'),false);assert.equal(visible('max_tokens'),true);
  assert.deepEqual((await ocr()).json,{model:'tesseract-looking-ocr',image:'data:image/png;base64,AA==',max_tokens:128,temperature:1.2});
  // Both positive capability flags are necessary, and a late old refresh cannot undo a newer result.
  for(const flags of [{instruction_edit:true,requires_input_image:false},{instruction_edit:false,requires_input_image:true},{family:'unknown',instruction_edit:true,requires_input_image:true}]){
    caps.set('photo-tools',{...instruction,instruction_edit:flags});await get('refresh').onclick();choose('photo-tools');
    assert.equal(get('image-source').required,false);assert.equal(get('mask-fields').hidden,false);
    assert.equal(get('submit').disabled,true);assert.equal(get('image-edit-fields').hidden,true);
  }
  caps.set('photo-tools',instruction);await get('refresh').onclick();
  const oldCaps=deferred();capability=r=>r.json.model==='photo-tools'?oldCaps.promise:response(caps.get(r.json.model));
  const olderRefresh=get('refresh').onclick();await tick();
  capability=null;caps.set('photo-tools',{...instruction,instruction_edit:null});await get('refresh').onclick();
  assert.equal(get('image-source').required,false);oldCaps.resolve(response(instruction));await olderRefresh;
  assert.equal(get('image-source').required,false,'Stale capability reply overwrote the newer refresh');
  caps.set('photo-tools',instruction);await get('refresh').onclick();
  // Capability lookup failure must not fall back to the generic image/transformer request.
  capability=async r=>r.json.model==='photo-tools'||r.json.model==='document-reader'?failure('Capability lookup failed for '+r.json.model):response(caps.get(r.json.model));
  await get('refresh').onclick();
  for(const id of ['photo-tools','document-reader']){
    choose(id);assert.equal(get('submit').disabled,true);const count=posts().length,readCount=reads.length;await submit();
    assert.equal(posts().length,count);assert.equal(reads.length,readCount);
  }
  choose('tesseract-looking-ocr');assert.equal(get('submit').disabled,false);
  capability=async r=>r.json.model==='photo-tools'?response({task:'image',instruction_edit:null}):response(caps.get(r.json.model));
  await get('refresh').onclick();choose('photo-tools');assert.equal(get('submit').disabled,true);
  const malformedCount=posts().length;await submit();assert.equal(posts().length,malformedCount,'Malformed capabilities enabled generic inference');
  capability=null;await get('refresh').onclick();choose('photo-tools');assert.equal(get('submit').disabled,false);
  const beforeCatalogFailure=posts().length;metadata=async()=>{throw Error('Catalog unavailable');};
  await get('refresh').onclick();assert.equal(get('submit').disabled,true);await submit();
  assert.equal(posts().length,beforeCatalogFailure,'Stale image controls allowed inference after catalog failure');
  metadata=null;await get('refresh').onclick();assert.equal(get('submit').disabled,false);
  // An instruction reply owns its original inputs through capability refresh and Stop cleanup.
  get('image-source').files=[upload];get('image-mask').files=[mask];get('prompt').value='Original instruction';
  const reply=deferred();inference=async()=>reply.promise;const pending=submit();await tick();
  const owned=posts().at(-1),count=posts().length,beforeLate=get('results').children.length;
  get('image-source').files=[new Blob(['next image'])];get('prompt').value='Next instruction';get('image_guidance_scale').value='9';
  assert.equal(await bytes(owned.options.body.get('image')),'source image');assert.equal(owned.options.body.get('prompt'),'Original instruction');
  assert.equal(owned.options.body.get('image_guidance_scale'),'1.5');
  const waitingCaps=deferred();capability=r=>r.json.model==='photo-tools'?waitingCaps.promise:response(caps.get(r.json.model));
  const refreshing=get('refresh').onclick();await tick();
  assert(get('submit').disabled&&get('model').disabled&&!get('stop').hidden);choose('document-reader');assert.equal(get('model').value,'photo-tools');
  await submit();assert.equal(posts().length,count);await get('stop').onclick();assert(owned.options.signal.aborted);
  waitingCaps.resolve(response(instruction));await refreshing;capability=null;
  const finalRefresh=deferred();metadata=()=>finalRefresh.promise;reply.resolve(response(result));await tick();
  assert.equal(get('results').children.length,beforeLate,'Aborted instruction reply rendered');
  assert.equal(get('submit').disabled,true);await submit();assert.equal(posts().length,count);
  finalRefresh.resolve(response({data:models}));await pending;metadata=null;
  assert.equal(get('stop').hidden,true);assert.equal(get('submit').disabled,false);assert.equal(get('status').textContent,'Stopped.');
  assert.equal(get('prompt').value,'Next instruction');
  // Stop during an OCR read prevents its later POST and releases ownership only after cleanup.
  choose('document-reader');const beforeStop=posts().length,stopped=submit();await tick();await get('stop').onclick();
  finishRead('data:image/png;base64,AA==');await stopped;assert.equal(posts().length,beforeStop);assert.equal(get('status').textContent,'Stopped.');
  // Generation-only ONNX models remain usable and ignore stale hidden edit inputs.
  models.push({id:'onnx-image',task:'image'});caps.set('onnx-image',{backend:'onnx-diffusers',task:'image',instruction_edit:null,image_edit:false,image_mask:null});
  await get('refresh').onclick();choose('onnx-image');assert.equal(get('submit').disabled,false);
  assert.equal(get('image-edit-fields').hidden,true);get('image-source').files=[upload];get('image-mask').files=[mask];get('prompt').value='One red mug';
  const beforeOnnx=posts().length;inference=async request=>{
    if(request.route==='/api/jobs'&&request.options.method==='POST')return response({id:'f'.repeat(32)});
    assert.equal(request.route,'/api/jobs/'+'f'.repeat(32));return response({status:'completed',result});
  };
  await submit();const onnxPosts=posts().slice(beforeOnnx);assert.equal(onnxPosts.length,1);assert.equal(onnxPosts[0].route,'/api/jobs');
  assert.equal(onnxPosts[0].json.model,'onnx-image');assert.equal(onnxPosts[0].json.prompt,'One red mug');assert(!Object.hasOwn(onnxPosts[0].json,'image'));
  console.log('PASS: capability-based forms, ONNX generation-only hidden-input isolation, registration variants, exact payloads, capability blocking and deferred Refresh/Stop ownership. No browser or trained model.');
})().catch(error=>{console.error(error);process.exitCode=1;});
