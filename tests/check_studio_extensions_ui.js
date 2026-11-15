/* Actual Studio source with DOM/fetch doubles; no browser or model inference. */
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const base=path.join(__dirname,'../src/titanos/web'),nodes=new Map();
const all=node=>[node,...node.children.flatMap(all)];
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.files=[];this._value=null;this._text='';this.hidden=false;this.attributes={};}
  set id(value){this._id=value;nodes.set(value,this);}get id(){return this._id;}
  set value(value){this._value=String(value);if(this.type==='file'&&this._value==='')this.files=[];}
  get value(){return this._value??(this.tagName==='select'?this.options[0]?.value||'':'');}
  get options(){return this.children.filter(node=>node.tagName==='option');}
  set textContent(value){this._text=String(value);this.replaceChildren();}
  get textContent(){return this._text+this.children.map(node=>node.textContent).join('');}
  set innerHTML(value){throw Error('Studio text must not use innerHTML');}
  append(...children){for(const child of children){child.remove();child.parentElement=this;this.children.push(child);}}
  prepend(...children){for(const child of children.reverse()){child.remove();child.parentElement=this;this.children.unshift(child);}}
  replaceChildren(...children){for(const child of this.children)child.parentElement=null;this.children=[];if(this.tagName==='select')this._value=null;this.append(...children);}
  remove(){if(this.parentElement){const siblings=this.parentElement.children;siblings.splice(siblings.indexOf(this),1);this.parentElement=null;}}
  setAttribute(name,value){this.attributes[name]=String(value);}
  addEventListener(){}
  focus(){context.document.activeElement=this;}
  getContext(){assert.equal(this.tagName,'canvas');return {createImageData:(width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)}),putImageData:image=>{this.imageData=image;}};}
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
const models=[
  {id:'depth',task:'depth-estimation',status:'unloaded'},
  {id:'segment',task:'image-segmentation',status:'unloaded'},
  {id:'rerank',task:'rerank',status:'unloaded'},
  {id:'vad',task:'vad',status:'unloaded'},
  {id:'asr',task:'transcription',status:'unloaded'},
  {id:'translator',task:'translation',status:'unloaded'},
  {id:'tensor',task:'tensor',status:'unloaded'},
  {id:'mystery',task:'future-task',status:'unloaded'},
];
const response=value=>({ok:true,json:async()=>value,blob:async()=>new Blob()});
const requests=[],reads=[];
class Reader {readAsDataURL(file){reads.push({reader:this,file});}}
class Form extends FormData {constructor(form){super();if(form)for(const node of all(form))if(node.name&&!node.disabled)this.set(node.name,node.value);}}
const context={console,Headers,FormData:Form,Blob,AbortController,URL,TextDecoder,setTimeout,FileReader:Reader,DataView,Uint8Array,Float32Array,atob,
  location:{origin:'http://localhost'},document:{activeElement:null,getElementById:get,createElement:tag=>new Element(tag),
    querySelectorAll:query=>{assert.equal(query,'.settings input, .settings select');return all(settings).filter(node=>['input','select'].includes(node.tagName));},
    querySelector:query=>{assert.equal(query,'#generate details');return details;}},
  fetch:async(route,options={})=>{
    const request={route,options,json:typeof options.body==='string'?JSON.parse(options.body):null};requests.push(request);
    if(route==='/api/info')return response({models:Object.fromEntries(models.map(m=>[m.id,{task:m.task,backend:'transformers',device:'cpu'}])),settings:{
      max_tokens:{value:4096},max_steps:{value:77},max_video_frames:{value:99},max_video_fps:{value:30},max_generated_images:{value:3},
      max_prompt_chars:{value:1234},max_text_chars:{value:100000},max_speech_chars:{value:5000},
      rerank_max_input_tokens:{value:512},pair_max_input_tokens:{value:256},sentiment_max_input_tokens:{value:384},
      segmentation_max_points:{value:64},max_image_dimension:{value:2048},max_image_pixels:{value:4194304}
    },inference_contracts:{}});
    if(route==='/v1/models')return response({data:models});
    if(route==='/api/models/capabilities')return response({task:request.json.model==='mystery'?'future-task':models.find(m=>m.id===request.json.model).task,backend:'transformers',instruction_edit:null});
    if(route==='/api/jobs'&&!options.method)return response({data:[]});
    if(route==='/v1/files/audio.wav')return {ok:true,blob:async()=>new Blob(['audio'],{type:'audio/wav'})};
    if(route==='/v1/files/video.mp4')return {ok:true,blob:async()=>new Blob(['video'],{type:'video/mp4'})};
    if(route==='/fixture/not-loaded')return {ok:false,status:409,json:async()=>({error:{code:'model_not_loaded',message:'Model is not loaded'}})};
    if(route==='/fixture/validation')return {ok:false,status:422,json:async()=>({detail:[{loc:['body','max_tokens'],msg:'Input should be less than or equal to 4096'}]})};
    return response({model:request.json?.model||'',width:1,height:1,data:'AAAAAA==',masks:[],results:[],segments:[]});
  }};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(base,'app.js'),'utf8'),context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const choose=id=>{get('model').value=id;get('model').onchange();};
const submit=()=>get('generate').onsubmit({preventDefault(){}});
const inference=()=>requests.filter(r=>r.options.method==='POST'&&!r.route.startsWith('/api/models'));
const finishRead=value=>{reads.at(-1).reader.result=value;reads.at(-1).reader.onload();};
(async()=>{
  await tick();
  assert.equal(get('generate').attributes['aria-busy'],'false');
  assert.equal(get('max_tokens').max,'4096');assert.equal(get('steps').max,'77');assert.equal(get('frames').max,'99');assert.equal(get('fps').max,'30');assert.equal(get('n').max,'3');
  assert.match(get('size').placeholder,/2048/);assert.equal(get('vad-threshold').step,'any');
  for(const id of ['rerank-fields','rerank-documents','rerank-truncate','rerank-return-documents','top_n',
    'vad-sample-rate','vad-threshold','vad-min-speech','vad-min-silence','vad-speech-pad','vad-return-frames',
    'segmentation-fields','segment-mode','segment-x0','segment-y0','segment-x1','segment-y1','segment-points','segment-point-labels','segment-multimask','segment-threshold',
    'audio-action','audio-translation-fields','translation-model','source-language','target-language',
    'model-setup','setup-task','setup-provider','setup-runtime-profile','setup-query','setup-search','setup-link','setup-import','setup-results','setup-plan','setup-message'])get(id);
  choose('depth');get('file').files=[new Blob(['image'],{type:'image/png'})];
  const pending=submit();await tick();finishRead('data:image/png;base64,AA==');await pending;
  assert.equal(inference().at(-1).route,'/v1/images/depth','Depth must never fall through to object detection');

  choose('segment');get('file').files=[new Blob(['segment'],{type:'image/png'})];
  for(const [id,value] of Object.entries({'segment-x0':'1','segment-y0':'2','segment-x1':'30','segment-y1':'40','segment-multimask':'true','segment-threshold':'0'}))get(id).value=value;
  let request=submit();await tick();finishRead('data:image/png;base64,AQ==');await request;
  assert.equal(inference().at(-1).route,'/v1/images/segment');
  assert.deepEqual(inference().at(-1).json,{model:'segment',image:'data:image/png;base64,AQ==',box:[1,2,30,40],multimask:true,mask_threshold:0});
  get('segment-mode').value='points';get('segment-mode').onchange();
  get('segment-points').value='[[2.5,1.25],[12,5]]';get('segment-point-labels').value='[1,0]';get('segment-multimask').value='';
  request=submit();await tick();finishRead('data:image/png;base64,Ag==');await request;
  assert.deepEqual(inference().at(-1).json,{model:'segment',image:'data:image/png;base64,Ag==',points:[[2.5,1.25],[12,5]],point_labels:[1,0],mask_threshold:0});
  get('segment-points').value='not JSON';const beforeBad=inference().length;await submit();assert.equal(inference().length,beforeBad);
  get('segment-mode').value='box';get('segment-mode').onchange();get('segment-x0').value='5';get('segment-y0').value='1';get('segment-x1').value='2';get('segment-y1').value='8';
  await submit();assert.equal(inference().length,beforeBad);assert.match(get('status').textContent,/x0.*x1|ordered/i);

  choose('rerank');get('prompt').value='Which passage is relevant?';get('rerank-documents').value='First passage\nSecond passage';
  assert.equal(get('prompt').maxLength,1234);assert.equal(get('max_input_tokens').max,'512');
  get('top_n').value='1';get('max_input_tokens').value='128';get('rerank-truncate').value='false';get('rerank-return-documents').value='true';
  await submit();assert.equal(inference().at(-1).route,'/v1/rerank');
  assert.deepEqual(inference().at(-1).json,{model:'rerank',query:'Which passage is relevant?',documents:['First passage','Second passage'],top_n:1,max_input_tokens:128,truncate:false,return_documents:true});

  choose('vad');get('file').files=[new Blob(['audio'],{type:'audio/wav'})];
  for(const [id,value] of Object.entries({'vad-sample-rate':'16000','vad-threshold':'0.5','vad-min-speech':'0','vad-min-silence':'100','vad-speech-pad':'30','vad-return-frames':'false'}))get(id).value=value;
  request=submit();await tick();finishRead('data:audio/wav;base64,Aw==');await request;
  assert.equal(inference().at(-1).route,'/v1/audio/vad');
  assert.deepEqual(inference().at(-1).json,{model:'vad',audio_base64:'data:audio/wav;base64,Aw==',sample_rate:16000,threshold:0.5,min_speech_ms:0,min_silence_ms:100,speech_pad_ms:30,return_frames:false});

  choose('translator');assert.equal(get('prompt').maxLength,100000);
  choose('asr');get('audio-action').value='translate';get('audio-action').onchange();get('file').files=[new Blob(['french'],{type:'audio/wav'})];
  get('language').value='fr';get('translation-model').value='translator';get('source-language').value='fra_Latn';
  await submit();const audio=inference().at(-1);assert.equal(audio.route,'/v1/audio/translations');
  assert.equal(audio.options.body.get('model'),'asr');assert.equal(audio.options.body.get('translation_model'),'translator');
  assert.equal(audio.options.body.get('source_language'),'fra_Latn');assert.equal(audio.options.body.has('target_language'),false);
  assert.equal(audio.options.body.get('language'),'fr');
  get('target-language').value='en';await submit();assert.equal(inference().at(-1).options.body.get('target_language'),'en');

  const before=inference().length;choose('tensor');
  assert.equal(get('submit').disabled,true);
  assert.match(get('task').textContent,/API|tensor/i);
  await submit();assert.equal(inference().length,before,'Tensor Studio state sent an inference request');

  choose('mystery');assert.equal(get('submit').disabled,true);
  assert.match(get('task').textContent,/unsupported|unknown/i);
  await submit();assert.equal(inference().length,before,'Unknown task fell through to an unrelated route');
  await assert.rejects(()=>context.api('/fixture/not-loaded'),error=>error.status===409&&error.code==='model_not_loaded'&&/Load/i.test(error.message));
  await assert.rejects(()=>context.api('/fixture/validation'),error=>error.status===422&&/max_tokens.*4096/i.test(error.message));

  const results=get('results'),render=async value=>{const before=results.children.length;await context.showResult(value,'FIXTURE');assert.equal(results.children.length,before+1);return results.children[0];};
  let card=await render({object:'list',model:'embed',data:[{index:0,object:'embedding',embedding:[0.25,-0.5,1]}],usage:{prompt_tokens:3,total_tokens:3}});
  assert.match(card.textContent,/1 vector.*3 dimensions/i);assert.match(card.textContent,/0\.25/);
  card=await render({object:'text.sentiment',model:'sentiment',status:'ok',label:'positive',scores:{positive:.8,neutral:.1,negative:.1},score_kind:'classifier_softmax_uncalibrated',calibrated:false});
  assert.match(card.textContent,/positive[\s\S]*uncalibrated/i);
  card=await render({model:'translate',text:'Bonjour',source_language:'eng_Latn',target_language:'fra_Latn',translated:true});
  assert.match(card.textContent,/Bonjour[\s\S]*eng_Latn.*fra_Latn/i);
  card=await render({model:'detect',task:'object-detection',width:100,height:80,objects:[{label:'<img onerror=bad>',score:.75,box:{xmin:1,ymin:2,xmax:30,ymax:40}}]});
  assert.match(card.textContent,/<img onerror=bad>.*0\.75.*1.*2.*30.*40/i);assert(!all(card).some(node=>node.tagName==='img'));
  card=await render({model:'asr',task:'transcribe',text:'Spoken result',segments:null,words:null});
  assert.equal(card.children.find(node=>node.tagName==='pre').textContent,'Spoken result');
  card=await render({object:'list',model:'rerank',score_kind:'cross_encoder_logit',results:[{index:1,relevance_score:2.5,document:{text:'<script>literal</script>'}}],usage:{prompt_tokens:4,total_tokens:4},input_details:[],truncation:{enabled:false}});
  assert.match(card.textContent,/uncalibrated/i);assert.match(card.textContent,/document 1.*2\.5.*<script>literal<\/script>/i);assert(!all(card).some(node=>node.tagName==='script'));
  card=await render({model:'vad',task:'vad',sample_rate:16000,duration:2,score_kind:'speech_probability_uncalibrated',segments:[{start:0.25,end:1,start_sample:4000,end_sample:16000}]});
  assert.match(card.textContent,/uncalibrated/i);assert.match(card.textContent,/0\.25.*1 seconds/i);
  const depthBytes=Buffer.alloc(8);depthBytes.writeFloatLE(0,0);depthBytes.writeFloatLE(2,4);
  card=await render({model:'depth',task:'depth-estimation',semantics:'relative_inverse_depth',larger_values:'nearer',units:null,calibrated:false,width:2,height:1,encoding:'base64_float32_le',data:depthBytes.toString('base64')});
  const canvas=all(card).find(node=>node.tagName==='canvas');assert(canvas&&canvas.imageData);assert.match(card.textContent,/relative inverse depth.*nearer.*not metric/i);
  const largeDepth=Buffer.alloc(512*512*4);
  card=await render({model:'depth-large',task:'depth-estimation',semantics:'relative_inverse_depth',larger_values:'nearer',units:null,calibrated:false,width:512,height:512,encoding:'base64_float32_le',data:largeDepth.toString('base64')});
  assert(all(card).some(node=>node.tagName==='canvas'),'A normal 512x512 depth map did not render');
  await assert.rejects(()=>context.showResult({task:'depth-estimation',width:2,height:1,encoding:'base64_float32_le',data:'AAAAAA=='},'BAD'),/byte count|depth/i);
  card=await render({model:'segment',task:'image-segmentation',mask_contract:'alpha-binary-v1',score_kind:'predicted_iou_uncalibrated',calibrated:false,prompt:{box:[1,2,3,4]},masks:[{index:0,png_base64:'fixture-png',selected_pixels:12,predicted_iou:.6,native_mask_token_index:0}]});
  assert(all(card).some(node=>node.tagName==='img'));assert.match(card.textContent,/Mask 0.*12 selected pixels.*0\.6.*uncalibrated/i);
  const heading=card.children.find(node=>node.className==='label');assert.equal(heading.tagName,'h2');assert.equal(context.document.activeElement,heading);
  context.status('Fixture failure',true);assert.equal(get('status').attributes.role,'alert');assert.equal(get('status').attributes['aria-live'],'assertive');assert.equal(context.document.activeElement,get('status'));
  context.status('Recovered');assert.equal(get('status').attributes.role,'status');assert.equal(get('status').attributes['aria-live'],'polite');
  const mediaBox=new Element('div');await context.asset('/v1/files/audio.wav',mediaBox);await context.asset('/v1/files/video.mp4',mediaBox);
  for(const media of mediaBox.children.filter(node=>['audio','video'].includes(node.tagName)))assert.match(media.attributes['aria-label'],/Generated (audio|video) asset/);
  console.log('PASS: explicit task dispatch, modality payloads/results, server limits/errors and truthful unsupported states. DOM/fetch doubles only.');
})().catch(error=>{console.error(error);process.exitCode=1;});
