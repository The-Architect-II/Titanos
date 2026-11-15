/* Real HTML/app source with small DOM/fetch/deferred doubles; no browser or model. */
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const base=path.join(__dirname,'../src/titanos/web');
const all=node=>[node,...node.children.flatMap(all)],nodes=new Map();
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.files=[];this._value=null;this._text='';this.hidden=false;}
  set id(value){this._id=value;nodes.set(value,this);}get id(){return this._id;}
  set value(value){this._value=String(value);}get value(){return this._value??(this.tagName==='select'?this.options[0]?.value||'':'');}
  get options(){return this.children.filter(node=>node.tagName==='option');}
  set textContent(value){this._text=String(value);this.replaceChildren();}get textContent(){return this._text+this.children.map(node=>node.textContent).join('');}
  set innerHTML(value){throw Error('Untrusted text must not use innerHTML');}
  append(...children){for(const child of children){child.remove();child.parentElement=this;this.children.push(child);}}
  prepend(...children){for(const child of children.reverse()){child.remove();child.parentElement=this;this.children.unshift(child);}}
  replaceChildren(...children){for(const child of this.children)child.parentElement=null;this.children=[];if(this.tagName==='select')this._value=null;this.append(...children);}
  remove(){if(this.parentElement){const siblings=this.parentElement.children;siblings.splice(siblings.indexOf(this),1);this.parentElement=null;}}
  setAttribute(name,value){(this.attributes??={})[String(name)]=String(value);}getAttribute(name){return this.attributes?.[String(name)]??null;}removeAttribute(name){if(this.attributes)delete this.attributes[String(name)];}
}
// Parse the actual page's tags/attributes, so missing IDs or hidden required fields fail.
const page=new Element('document'),stack=[page];
for(const match of fs.readFileSync(path.join(base,'index.html'),'utf8').matchAll(/<\/?([\w-]+)([^>]*)>|([^<]+)/g)){
  if(match[3]){const text=new Element('text');text.textContent=match[3];stack.at(-1).append(text);continue;}
  const [token,tag,attrs]=match;if(token.startsWith('</')){while(stack.length>1&&stack.pop().tagName!==tag){}continue;}
  const node=new Element(tag);for(const attr of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)){const [,key,value]=attr;node[key]=value===undefined?true:value;}
  stack.at(-1).append(node);if(!['meta','link','input','br'].includes(tag))stack.push(node);
}
const get=id=>{assert(nodes.has(id),'Missing HTML id '+id);return nodes.get(id);};
const settings=all(page).find(node=>node.class==='settings');
const details=all(get('generate')).find(node=>node.tagName==='details');
const standardModels=[{id:'pairs',task:'pair-classification'},{id:'chat-a',task:'chat'},{id:'chat-b',task:'chat'},{id:'vision',task:'vision'},
  {id:'embedding',task:'embeddings'},{id:'sentiment',task:'sentiment'},{id:'asr',task:'transcription'},{id:'image',task:'image'},
  {id:'translation',task:'translation'},{id:'ocr',task:'image-to-text'},{id:'detect',task:'object-detection'}];
let models=[...standardModels],inference,metadata,reads=[],requests=[];
const response=value=>({ok:true,json:async()=>value});
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
class Form extends FormData {constructor(form){super();if(form)for(const node of all(form))if(node.name)this.set(node.name,node.value);}}
class Reader {readAsDataURL(file){reads.push({reader:this,file});}}
const context={console,Headers,FormData:Form,Blob,AbortController,URL,TextDecoder,setTimeout,FileReader:Reader,
  location:{origin:'http://localhost'},document:{getElementById:get,createElement:tag=>new Element(tag),
    querySelectorAll:query=>{assert.equal(query,'.settings input, .settings select');return all(settings).filter(node=>['input','select'].includes(node.tagName));},
    querySelector:query=>{assert.equal(query,'#generate details');return details;}},
  fetch:async(route,options={})=>{
    const request={route,options,json:typeof options.body==='string'?JSON.parse(options.body):null};requests.push(request);
    if(route==='/v1/models')return metadata?metadata():response({data:models});
    if(route==='/api/info')return response({models:{},settings:{},inference_contracts:{require_loaded:{revision:1,routes:['/v1/chat/completions']}}});
    if(route==='/api/models/capabilities'){const model=models.find(item=>item.id===request.json.model);return response({backend:model.task==='image'?'diffusers':'transformers',task:model.task,instruction_edit:null});}
    if(route==='/api/jobs'&&!options.method)return response({data:[]});
    if(route==='/api/models/register'||route==='/api/models/unload')return response({});
    assert(inference,'Unexpected request '+route);return inference(request);
  }};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(base,'app.js'),'utf8'),context);
const submit=()=>get('generate').onsubmit({preventDefault(){}});
const choose=id=>{get('model').value=id;get('model').onchange();};
const posts=()=>requests.filter(r=>r.options.method==='POST'&&!r.route.startsWith('/api/models'));
const prediction={object:'text_pair.classification',schema_version:'1',model:'pairs',score_kind:'softmax_uncalibrated',calibrated:false,
  labels:[{index:0,label:'<img onerror=bad>'},{index:1,label:'LABEL_1'},{index:2,label:'LABEL_2'}],
  results:[{index:0,id:'pair-0',id_source:'index_default',label_index:1,label:'LABEL_1',logits:[-1000,0,0],scores:[0,0.5,0.5],
    input:{input_tokens:17,processed_tokens:12,truncated:true,original_token_ids_sha256:'a'.repeat(64),processed_token_ids_sha256:'b'.repeat(64)}}],
  truncation:{enabled:true,strategy:'only_first',preserved_sequence:'hypothesis'},provenance:{device:'cpu',checkpoint_digest:'c'.repeat(64)}};
const stream=text=>({ok:true,body:new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({choices:[{delta:{content:text}}]})+'\n\ndata: [DONE]\n\n'));controller.close();}})});
(async()=>{
  await tick();assert.equal(get('model').value,'pairs');assert.equal(get('pair-fields').hidden,false);
  assert.equal(get('prompt-label').hidden,true);assert.equal(get('prompt').required,false);assert.equal(get('file-label').hidden,true);
  assert.equal(get('premise').required,true);assert.equal(get('hypothesis').required,true);
  assert.equal(get('max_input_tokens').parentElement.hidden,false);assert.equal(get('truncate').parentElement.hidden,false);
  assert.equal(get('temperature').parentElement.hidden,true);
  const task=all(get('register')).find(node=>node.name==='task');assert(task.options.some(option=>option.textContent==='pair-classification'));
  task.value='pair-classification';for(const [name,value] of Object.entries({name:'new-pair',path:'D:/Models/pair',device:'cpu'}))all(get('register')).find(node=>node.name===name).value=value;
  await get('register').onsubmit({preventDefault(){},target:get('register')});
  assert.deepEqual(requests.find(r=>r.route==='/api/models/register').json,{name:'new-pair',spec:{path:'D:/Models/pair',task:'pair-classification',device:'cpu',backend:'transformers'}});
  // Stale media/chat settings must never enter a pair request; preserve exact text and omission.
  get('file').files=[new Blob(['stale image'])];get('prompt').value='stale prompt';get('temperature').value='0.7';get('max_tokens').value='99';
  get('premise').value='  A source\nwith <script>literal text</script>.  ';get('hypothesis').value=' A statement. ';
  inference=async()=>response(prediction);
  await submit();assert.deepEqual(posts().at(-1).json,{model:'pairs',pairs:[{premise:get('premise').value,hypothesis:get('hypothesis').value}]});
  assert.equal(posts().at(-1).route,'/v1/classifications/text-pairs');assert.equal(reads.length,0);
  for(const value of ['false','true']){get('truncate').value=value;get('max_input_tokens').value='128';await submit();assert.equal(posts().at(-1).json.truncate,value==='true');assert.equal(posts().at(-1).json.max_input_tokens,128);}
  const table=all(get('results')).find(node=>node.tagName==='table');
  assert.deepEqual(table.children.filter(node=>node.tagName==='tr').slice(1).map(row=>row.children.map(cell=>cell.textContent)),[
    ['0','<img onerror=bad>','-1000','0'],['1','LABEL_1','0','0.5'],['2','LABEL_2','0','0.5']]);
  assert(!all(get('results')).some(node=>node.tagName==='img'));assert(get('results').textContent.includes('uncalibrated softmax'));
  assert(get('results').textContent.includes('Input tokens: 17; processed: 12; premise truncated: yes. Hypothesis preserved.'));
  assert(get('results').textContent.includes('"id_source": "index_default"'));assert(get('results').textContent.includes('"checkpoint_digest"'));
  for(const value of ['0','1.5','Infinity','NaN','9007199254740992']){const before=posts().length;get('max_input_tokens').value=value;await submit();assert.equal(posts().length,before);assert.match(get('status').textContent,/positive safe integer/);}
  get('max_input_tokens').value='';get('truncate').value='';
  const beforeBlank=posts().length;get('hypothesis').value=' ';await submit();assert.equal(posts().length,beforeBlank);get('hypothesis').value='Statement';
  const priorResults=get('results').children.length;
  inference=async()=>({ok:false,status:422,json:async()=>({error:{message:'<img onerror=bad> Pair too long'}})});
  await submit();assert.match(get('status').textContent,/^<img onerror=bad> Pair too long.*correct the named fields/i);assert.equal(get('status').className,'error');
  assert.equal(get('results').children.length,priorResults,'Error reused an old result as a new classification');
  // Defer a result past Stop, force refresh/selection, then defer the owner's final refresh.
  const wait=deferred();inference=async()=>wait.promise;
  const before=posts().length,resultsBefore=get('results').children.length;
  const pending=submit();await tick();const owned=posts().at(-1).options.signal;
  assert(get('submit').disabled&&get('model').disabled&&!get('stop').hidden);
  metadata=async()=>{throw Error('Simulated refresh failure');};await get('refresh').onclick();metadata=null;
  assert.equal(get('status').textContent,'Generating…');assert.equal(get('submit').disabled,true);
  models=standardModels.filter(model=>model.id!=='pairs');await get('refresh').onclick();
  assert.equal(get('model').value,'pairs');assert.equal(get('submit').disabled,true);assert.equal(get('pair-fields').hidden,false);
  choose('chat-b');assert.equal(get('model').value,'pairs');await get('clear').onclick();await get('unload').onclick();
  get('premise').value='Next draft';await submit();assert.equal(posts().length,before+1);assert.equal(posts().at(-1).options.signal,owned);
  const finalRefresh=deferred();metadata=()=>finalRefresh.promise;
  await get('stop').onclick();assert(owned.aborted);wait.resolve(response(prediction));await tick();
  assert.equal(get('results').children.length,resultsBefore,'Aborted late result rendered');
  await submit();assert.equal(posts().length,before+1,'Cleanup released ownership before refresh settled');assert.equal(get('submit').disabled,true);
  finalRefresh.resolve(response({data:models}));await pending;metadata=null;
  assert.equal(get('stop').hidden,true);assert.equal(get('model').value,'chat-a');assert(!get('model').options.some(option=>option.value==='pairs'));
  assert.equal(get('status').textContent,'Stopped.');assert.equal(get('submit').disabled,false);
  // Chat history remains with its model, and a late older refresh cannot unlock a newer owner.
  get('file').files=[];get('prompt').value='First turn';inference=async()=>stream('First answer');await submit();
  const background=deferred();metadata=()=>background.promise;const refreshing=get('refresh').onclick();metadata=null;
  const chat=deferred();inference=async()=>chat.promise;get('prompt').value='Second turn';const chatPending=submit();await tick();
  const chatRequest=posts().at(-1),chatSignal=chatRequest.options.signal;
  assert.equal(chatRequest.json.model,'chat-a');assert.equal(chatRequest.json.messages.length,3);assert.equal(chatRequest.json.messages[1].content,'First answer');
  background.resolve(response({data:models}));await refreshing;assert.equal(get('submit').disabled,true);
  choose('chat-b');get('prompt').value='Edited next draft';get('temperature').value='1.2';await submit();await get('clear').onclick();
  assert.equal(posts().at(-1),chatRequest);assert.equal(get('model').value,'chat-a');
  chat.resolve(stream('Second answer'));await chatPending;assert.equal(chatSignal.aborted,false);
  get('prompt').value='Third turn';inference=async()=>stream('Third answer');await submit();assert.equal(posts().at(-1).json.messages.length,5);
  choose('chat-b');await submit();assert.equal(posts().at(-1).json.messages.length,1);
  // Stop targets a streaming reader; its partial response never becomes completed history.
  let blockedRead;inference=async({options})=>({ok:true,body:{getReader:()=>({read:()=>new Promise((resolve,reject)=>{blockedRead=resolve;options.signal.addEventListener('abort',()=>reject(Object.assign(Error('Stopped'),{name:'AbortError'})),{once:true});}),releaseLock(){}})}});
  const stopped=submit();await tick();await get('stop').onclick();await stopped;
  assert(get('results').textContent.includes('Stopped (partial)'));inference=async()=>stream('After stop');await submit();assert.equal(posts().at(-1).json.messages.length,3);
  // Snapshot vision settings/history before awaited file read; Stop prevents the eventual POST.
  get('require-loaded').checked=true;choose('vision');get('file').files=[new Blob(['image'])];get('prompt').value='Describe the original';get('temperature').value='0.2';
  const visionPending=submit();await tick();get('prompt').value='Edited';get('temperature').value='1.8';choose('chat-a');
  reads.at(-1).reader.result='data:image/png;base64,AA==';reads.at(-1).reader.onload();await visionPending;
  assert.equal(posts().at(-1).json.model,'vision');assert.equal(posts().at(-1).json.messages[0].content[0].text,'Describe the original');assert.equal(posts().at(-1).json.temperature,0.2);
  assert.equal(Object.hasOwn(posts().at(-1).json,'require_loaded'),false,'Hidden chat-only require_loaded leaked into vision');
  const visionCount=posts().length,stoppedFile=submit();await tick();await get('stop').onclick();reads.at(-1).reader.result='data:image/png;base64,AA==';reads.at(-1).reader.onload();await stoppedFile;assert.equal(posts().length,visionCount);
  // Existing direct branches keep controls and now all use their owned cancellation signal.
  for(const [id,route] of [['embedding','/v1/embeddings'],['sentiment','/v1/sentiment'],['translation','/v1/translations'],['asr','/v1/audio/transcriptions']]){
    choose(id);inference=async()=>response({text:'legacy'});get('language').value='en';await submit();const last=posts().at(-1);assert.equal(last.route,route);assert(last.options.signal);
    if(id==='asr'){assert.equal(last.options.body.get('response_format'),'verbose_json');assert.equal(get('prompt').required,false);}else assert.equal(last.json.model,id);
  }
  // Detection options preserve omission/false/zero and snapshot before the file read.
  choose('detect');get('file').files=[new Blob(['image'])];inference=async()=>response({objects:[]});
  assert.equal(get('nms').parentElement.hidden,false);assert.equal(get('nms_iou').parentElement.hidden,false);
  for(const [mode,iou,expected] of [['','',undefined],['false','0',false],['true','0.7',true]]){
    get('nms').value=mode;get('nms_iou').value=iou;
    const detectionPending=submit();await tick();get('nms').value=mode==='true'?'false':'true';get('nms_iou').value='0.9';
    reads.at(-1).reader.result='data:image/png;base64,AA==';reads.at(-1).reader.onload();await detectionPending;
    const detection=posts().at(-1);assert.equal(detection.route,'/v1/images/detect');assert.equal(detection.json.nms,expected);
    assert.equal(detection.json.nms_iou,iou===''?undefined:Number(iou));
  }
  get('nms').value='';get('nms_iou').value='';
  // Stop during job creation must cancel exactly the returned owned ID, once, before allowing a new request.
  choose('image');const creation=deferred(),cancel=deferred();let cancels=0;
  inference=async request=>{if(request.route==='/api/jobs')return creation.promise;assert.equal(request.route,'/api/jobs/owned-job');assert.equal(request.options.method,'DELETE');cancels++;return cancel.promise;};
  const jobPending=submit();await tick();await get('stop').onclick();creation.resolve(response({id:'owned-job'}));await tick();
  const cancelling=get('stop').onclick();await submit();assert.equal(cancels,1);assert.equal(get('submit').disabled,true);
  cancel.resolve(response({}));await cancelling;await jobPending;assert.equal(cancels,1);assert.equal(get('status').textContent,'Stopped.');
  models=[];await get('refresh').onclick();assert.equal(get('submit').disabled,true);const count=posts().length;await submit();assert.equal(posts().length,count);
  console.log('PASS: pair form/registration/omission/native scores, invalid controls, deferred refresh/select/Stop/cleanup ownership, chat history, file snapshots and legacy direct/job branches. No browser or trained model.');
})().catch(error=>{console.error(error);process.exitCode=1;});
