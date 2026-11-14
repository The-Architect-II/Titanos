'use strict';
const $=id=>document.getElementById(id);let key='',catalog=[],history=[],active=null,urls=[],refreshEpoch=0,refreshing=false,refreshFailed=false,serverInfo=null,infoAttempted=false,runtimeModelId=null,lifecycleEpoch=0,lastConnectionError=false,activeAcquisition=null,setupReconciled=false;
const fallbackText={'connection.connected':'Connected, {{count}} models','connection.needed':'Connection needed','status.connected':'Connected. Select a model.','status.ready':'Ready','status.done':'Done','status.queued':'Queued','status.running':'Running…','status.cancelling':'Cancelling','status.completed':'Completed','status.failed':'Failed','status.cancelled':'Cancelled','status.interrupted':'Interrupted','jobs.open':'Open','jobs.cancel':'Cancel','jobs.empty':'No jobs yet. Image, audio, music and video runs are listed here while they run and after they finish.','jobs.unavailable':'Jobs are not available for this key. Connect with a key that has the jobs scope to list and run jobs.','models.empty':'No models yet. Use Add a model, or run titanos pull.','a11y.openJob':'Open {{task}} job {{id}} for {{model}}','a11y.cancelJob':'Cancel {{task}} job {{id}} for {{model}}','error.model_not_loaded':'Use Load for this model or allow lazy loading.','error.validation':'Correct the named fields and active limits.','error.unavailable':'Check the complete model, runtime and selected device.'};
const fallbackTranslate=(name,variables={})=>(fallbackText[name]||name).replace(/\{\{(\w+)\}\}/g,(_,field)=>String(variables[field]??''));
const i18n=globalThis.TitanosI18n||{initialize:()=>({mode:'simple',locale:'en'}),state:()=>({mode:'simple',locale:'en'}),setMode:mode=>({mode,locale:'en'}),setLocale:locale=>({mode:'simple',locale}),t:fallbackTranslate,formatBytes:value=>`${value} bytes`};
let preferences=i18n.initialize();
const tr=(name,variables)=>ui(name,fallbackText[name]||name,variables);
// Only explicitly constructed UI messages are translated. Raw content has no marker.
const UI_MESSAGE=Symbol('titanos.ui.message'),textBindings=new WeakMap(),attributeBindings=new WeakMap();
function ui(name,fallback,variables={}){
  return {[UI_MESSAGE]:true,name,fallback,variables,toString(){
    const text=i18n.t(name,this.variables);
    return text===name?String(this.fallback??name).replace(/\{\{(\w+)\}\}/g,(_,field)=>String(this.variables[field]??'')):text;
  }};
}
function setContent(node,value){
  if(value?.[UI_MESSAGE]){textBindings.set(node,value);node.setAttribute?.('data-i18n',value.name);}
  else{textBindings.delete(node);node.removeAttribute?.('data-i18n');}
  node.textContent=value==null?'':String(value);return node;
}
function setUiAttribute(node,name,value){
  const kind=name==='aria-label'?'aria':name;
  if(value?.[UI_MESSAGE]){
    const bindings=attributeBindings.get(node)||{};bindings[name]=value;attributeBindings.set(node,bindings);
    node.setAttribute?.('data-i18n-'+kind,value.name);
  }else{const bindings=attributeBindings.get(node);if(bindings)delete bindings[name];node.removeAttribute?.('data-i18n-'+kind);}
  if(name==='placeholder'||name==='alt')node[name]=String(value??'');
  else node.setAttribute?.(name,String(value??''));return node;
}
// counted messages: a *_one key for exactly one, the plain key otherwise (en/es plural; ja/zh-CN carry the same text)
function uiCount(name,fallback,one,variables={}){return Number(variables.count)===1?ui(name+'_one',one,variables):ui(name,fallback,variables);}
function taskLabel(value){return ui('task.'+value,value);}
function stateLabel(value){return ui('status.'+value,value);}
function stageLabel(value){return ui('stage.'+value,value);}
// Progress totals carry their unit from the server (bytes, files, steps); older records without a unit are inferred from the stage so a file count is never labelled as bytes.
function progressUnit(progress){return progress.unit||(['downloading_files','extracting'].includes(progress.stage)?'files':'bytes');}
function progressTotal(progress){return progress.total?ui('ui.progress_total',' / {{total}} {{unit}}',{total:progress.total,unit:ui('unit.'+progressUnit(progress),progressUnit(progress))}):'';}
function uiJoin(parts,separator='',empty=''){return ui('ui.join','{{text}}',{get text(){return parts.map(value=>String(value??'')).join(separator)||String(empty);}});}
function assignmentModeLabel(value){return tr({single:'advanced.single',model_sharding:'advanced.sharding',tensor_parallel:'advanced.tensor'}[value]||value);}

function bindStaticTranslations(){
  for(const id of ['setup-task','register-task'])if($(id))for(const item of $(id).options)setContent(item,taskLabel(item.value));
  if($('model'))setUiAttribute($('model'),'aria-label',tr('composer.model'));
  if($('vad-gate'))setUiAttribute($('vad-gate'),'aria-label',ui('ui.vad_check','Speech activity check'));
}
function applyTranslations(){
  if(!document.documentElement)return;
  bindStaticTranslations();
  document.title=String(ui('ui.page_title','Titanos · Local inference'));
  for(const node of document.querySelectorAll('[data-i18n]')){if(node.children?.length)continue;setContent(node,textBindings.get(node)||tr(node.dataset.i18n));}
  for(const [suffix,attribute] of [['placeholder','placeholder'],['aria','aria-label'],['alt','alt']]){
    for(const node of document.querySelectorAll('[data-i18n-'+suffix+']')){
      const value=attributeBindings.get(node)?.[attribute]||tr(node.getAttribute('data-i18n-'+suffix));
      setUiAttribute(node,attribute,value);
    }
  }
  for(const [id,name] of Object.entries({'setup-task-label':'setup.task','setup-provider-label':'setup.source','setup-query-label':'setup.search','setup-link-label':'setup.link'}))if($(id))setContent($(id),tr(name));
  if($('setup-summary'))setContent($('setup-summary'),tr('setup.title'));if($('setup-intro'))setContent($('setup-intro'),tr('setup.intro'));
  if($('setup-search'))setContent($('setup-search'),tr('setup.find'));if($('setup-import'))setContent($('setup-import'),tr('setup.inspect'));
  for(const [id,name] of [['load','composer.load'],['unload','composer.unload'],['model-details','composer.details'],['remove-registration','composer.remove'],['clear','composer.clear'],['assignment-load','advanced.loadDevices']])if($(id))setContent($(id),tr(name));
  if($('first-run-summary'))setContent($('first-run-summary'),tr('firstRun.title'));if($('first-run-copy'))setContent($('first-run-copy'),tr('firstRun.copy'));
  $('ui-mode').value=preferences.mode;$('ui-locale').value=preferences.locale;
}

function setupPreferences(){
  if(!$('ui-mode')||!$('ui-locale'))return;
  document.addEventListener?.('titanos:preferences',event=>{preferences=event.detail;applyTranslations();select(false);});
  $('ui-mode').value=preferences.mode;$('ui-locale').value=preferences.locale;applyTranslations();
  $('ui-mode').onchange=()=>{preferences=i18n.setMode($('ui-mode').value);applyTranslations();select(false);if(preferences.mode==='advanced')loadAssignment().catch(()=>{});};
  $('ui-locale').onchange=()=>{preferences=i18n.setLocale($('ui-locale').value);applyTranslations();select(false);refreshJobs().catch(error=>status(error.message,true));};
}
const taskPolicy={
  chat:'studio',vision:'studio',embeddings:'studio',transcription:'studio',speech:'studio',image:'studio',video:'studio',
  translation:'studio','image-to-text':'studio','object-detection':'studio',sentiment:'studio',rerank:'studio',vad:'studio',diarization:'studio',
  'pair-classification':'studio','depth-estimation':'studio','image-segmentation':'studio',music:'studio',tensor:'api-only'
};
function status(text,bad=false,kind=bad?'error':'info'){
  const node=$('status');setContent(node,text);node.className=bad?'error':'';
  if(!bad&&text)node.className=kind==='success'?'success':'info';
  node.setAttribute?.('role',bad?'alert':'status');node.setAttribute?.('aria-live',bad?'assertive':'polite');
  if(bad){node.tabIndex=-1;node.focus?.();}
}
class ApiError extends Error{constructor(message,status,code,details){super(message);this.name='ApiError';this.status=status;this.code=code;this.details=details;}}
function apiAction(code,httpStatus){
  const actions={
    model_not_loaded:'error.model_not_loaded',registry_busy:'error.registry_busy',registry_read_only:'error.registry_read_only',
    model_not_found:'error.model_not_found',unsupported_capability:'error.unsupported_capability',request_too_large:'error.request_too_large',
    device_unavailable:'error.device_unavailable',runtime_missing:'error.runtime_missing',acquisition_not_prepared:'error.acquisition_not_prepared',provider_error:'error.provider_error',invalid_backend_response:'error.invalid_backend_response'
  };
  return actions[code]?tr(actions[code]):httpStatus===401?tr('error.authentication'):httpStatus===403?tr('error.permission'):httpStatus===422?tr('error.validation'):httpStatus===503?tr('error.unavailable'):'';
}
function validationMessage(detail){
  if(typeof detail==='string')return detail;
  if(!Array.isArray(detail))return null;
  return detail.map(item=>`${Array.isArray(item.loc)?item.loc.filter(part=>part!=='body').join('.')+': ':''}${item.msg||ui("ui.invalid_value","Invalid value")}`).join('; ');
}
async function api(path,options={}){
  const headers=new Headers(options.headers||{});if(key)headers.set('Authorization','Bearer '+key);
  // R29: Studio marks its broker and key requests; a custom header forces a CORS preflight, which Titanos never approves
  if(/^\/api\/(broker|keys)(\/|\?|$)/.test(path))headers.set('X-Titanos-Studio','1');
  const hint=typeof routingLocation==='function'?routingLocation():'';
  if(hint&&String(options.method||'GET').toUpperCase()!=='GET'&&/^\/(v1\/|v1beta\/|api\/jobs)/.test(path))headers.set('X-Titanos-Location',hint);
  if(options.json){headers.set('Content-Type','application/json');options.body=JSON.stringify(options.json);delete options.json;}
  let response;
  try{response=await fetch(path,{...options,headers});}
  catch(error){
    // a network failure (the browser's TypeError) is said in Titanos words; an abort stays an abort
    if(error?.name!=='TypeError')throw error;
    throw new ApiError(String(ui('error.unreachable','Titanos is not reachable at {{host}}. Start it or press Refresh.',{host:String(globalThis.location?.host||'this address')})),0,'unreachable',null);
  }
  if(response.ok){if(typeof rememberProvenance==='function')rememberProvenance(response);return response;}
  let body=null;try{body=await response.json();}catch{}
  const code=body?.error?.code||(response.status===422?'validation_error':null);
  const original=body?.error?.message||validationMessage(body?.detail)||ui("ui.server_status","Server returned {{status}}",{status:response.status});
  // the next-step hints are written for local model and job routes; broker and key answers carry their own reason
  const action=/^\/api\/(broker|keys)(\/|\?|$)/.test(path)&&![401,403].includes(response.status)?'':apiAction(code,response.status);
  const text=String(original).trim();
  throw new ApiError(action?`${text}${/[.!?。！？]$/.test(text)?'':'.'} ${action}`:original,response.status,code,body);
}
function chosen(){return active?.model||catalog.find(x=>x.id===$('model').value);}
function taskIssue(model=chosen()){if(!model)return ui("ui.select_model","No model selected.");if(taskPolicy[model.task]==='api-only')return ui("ui.tensor_api","Tensor models use the typed /api/infer API; Studio does not guess tensor shapes or data.");if(taskPolicy[model.task]!=='studio')return ui("ui.unsupported_task","Unknown or unsupported Studio task: {{task}}",{task:model.task});return null;}
function needsCapabilities(model){return ['image','image-to-text'].includes(model?.task);}
function isTesseract(model=chosen()){return model?.capabilities?.backend==='tesseract';}
function generationOnly(model=chosen()){return model?.task==='image'&&model?.capabilities?.image_edit===false;}
function isInstruction(model=chosen()){const value=model?.capabilities?.instruction_edit;return model?.capabilities?.backend==='diffusers'&&value?.family==='stable-diffusion-instruct-pix2pix'&&value.instruction_edit===true&&value.requires_input_image===true;}
function controlIssue(model=chosen()){return needsCapabilities(model)&&(model.capability_error||!model.capabilities)?ui("ui.controls_unavailable","Model controls unavailable. Refresh to retry. ")+(model.capability_error||''):null;}
function busy(){
  const running=Boolean(active),model=chosen(),state=model?.status;
  $('generate').setAttribute?.('aria-busy',String(running));$('results').setAttribute?.('aria-busy',String(running));
  $('submit').disabled=running||devicePolicyPending||cabinetConnected===false||((refreshing||refreshFailed)&&needsCapabilities(model))||!model||Boolean(controlIssue())||Boolean(taskIssue());
  for(const id of ['model','clear','connect'])$(id).disabled=running||devicePolicyPending;
  for(const id of ['assignment-apply','assignment-load','assignment-mode','assignment-devices','assignment-ranks','assignment-replicas','assignment-reset','device-policy','device-refresh'])if($(id))$(id).disabled=running||devicePolicyPending||(id==='device-policy'&&(!assignmentSaved.has(model?.id)||assignmentDrafts.get(model?.id)?.dirty));
  const external=Boolean(model?.external);
  if($('load'))$('load').disabled=running||devicePolicyPending||!model||external||state==='loaded'||state==='loading';
  $('unload').disabled=running||devicePolicyPending||!model||external||!['loaded','loading','error'].includes(state);
  if($('model-details'))$('model-details').disabled=running||external||state!=='loaded';
  if($('remove-registration'))$('remove-registration').disabled=running||external||!model;
  $('stop').hidden=!running;
}
function refreshVAD(){const value=$('vad-gate').value;$('vad-gate').replaceChildren();for(const [id,label] of [['inherit',ui("ui.default_model_server","Model / server default")],['off',ui("ui.disable_request","Disable for this request")],...catalog.filter(m=>m.task==='vad').map(m=>['model:'+m.id,m.id])]){const option=document.createElement('option');option.value=id;setContent(option,label);$('vad-gate').append(option);}if([...$('vad-gate').options].some(o=>o.value===value))$('vad-gate').value=value;}
function refreshTaskModels(){
  const select=$('translation-model'),value=select.value;select.replaceChildren(option('',ui("ui.whisper_direct","Whisper direct translation")));
  for(const model of catalog.filter(item=>item.task==='translation'))select.append(option(model.id));
  if([...select.options].some(item=>item.value===value))select.value=value;
  for(const [id,task,label] of [['diarization-model','diarization',ui('audio.choose_speakers','Choose a speaker model')],['audio-text-model','sentiment',ui('audio.no_text_analysis','No transcript analysis')]]){
    const node=$(id),saved=node.value;node.replaceChildren(option('',label));
    for(const model of catalog.filter(item=>item.task===task||task==='sentiment'&&item.task==='chat'))node.append(option(model.id));
    if([...node.options].some(item=>item.value===saved))node.value=saved;
    else if(task==='diarization'&&node.options.length===2)node.value=node.options[1].value;
  }
}
function applyServerInfo(){
  const value=name=>serverInfo?.settings?.[name]?.value;
  for(const [id,name] of [['max_tokens','max_tokens'],['steps','max_steps'],['frames','max_video_frames'],['fps','max_video_fps'],['n','max_generated_images'],['top_n','rerank_max_documents']]){
    const limit=value(name);$(id).max=limit===undefined?'':String(limit);
  }
  const dimension=value('max_image_dimension'),pixels=value('max_image_pixels');
  setUiAttribute($('size'),"placeholder",dimension===undefined?ui("ui.image_size_example","e.g. 1024x1024"):ui("ui.image_size_limit","WIDTHxHEIGHT · up to {{dimension}}px per side{{pixels}}",{dimension,pixels:pixels!==undefined?ui("ui.pixel_limit"," and {{count}} pixels",{count:pixels}):''}));
  updateRuntimeProfiles();
}
function updateRuntimeProfiles(){
  const node=$('setup-runtime-profile');if(!node)return;const runtime=serverInfo?.model_setup?.runtime,rows=runtime?.profiles||[];
  const choice=runtimeProfileChoice||(runtime?.metadata_status==='unverified'?'':runtime?.recommended_profile||rows.find(row=>row.recommended&&row.one_stop_supported)?.id||rows.find(row=>row.one_stop_supported)?.id||'');
  node.replaceChildren(...rows.map(row=>{const selected=row.recommended||row.id===runtime?.recommended_profile;const item=option(row.id,uiJoin([row.title||row.id,selected?ui('ui.recommended_suffix',' · recommended'):'',!row.one_stop_supported?uiJoin([' · ',tr('device.runtimeUnavailable')]):'']));item.disabled=!row.one_stop_supported;return item;}));
  if(choice&&!rows.some(row=>row.id===choice)){const item=option(choice,uiJoin([choice,' · ',tr('device.runtimeUnavailable')]));item.disabled=true;node.append(item);}
  node.value=choice;if(node.parentElement)node.parentElement.hidden=!rows.length&&!choice;
}
function selectedRuntimeProfile(){const id=$('setup-runtime-profile')?.value;return (serverInfo?.model_setup?.runtime?.profiles||[]).find(row=>row.id===id);}
function applyMaxLength(node,limit){if(Number.isSafeInteger(limit)&&limit>=0)node.maxLength=limit;else node.removeAttribute?.('maxlength');}
function renderModelSetup(model){
  const panel=$('model-setup-check');if(!panel)return;
  const setup=model?.capabilities?.setup;panel.hidden=!setup;panel.replaceChildren();if(!setup)return;
  const labels={blocked:ui('setup.check_blocked','Setup needs attention'),needs_review:ui('setup.check_unknown','Compatibility unverified'),ready_for_load_check:ui('setup.check_ready','Files ready for a load check'),verified_at_load:ui('setup.check_verified','Verified at load')};
  add(panel,'p',ui('setup.check_title','{{model}} · {{state}}',{model:model.id,state:labels[setup.status]||labels.needs_review}));
  add(panel,'p',setup.observed_at_load?ui('setup.check_observed','Runtime details of the loaded model confirm the items marked Loaded. Local file checks remain advisory.'):ui('setup.check_scope','Local file checks are advisory. Loading verifies runtime compatibility.')).className=setup.observed_at_load?'hint advanced-only':'hint';
  for(const issue of setup.issues||[]){
    const item=document.createElement('div');item.className='setup-issue'+(issue.severity==='error'?'':' advanced-only')+(issue.severity==='info'?' observed':'');
    add(item,'p',issue.message);
    if(issue.files?.length)add(item,'p',ui('setup.check_files','Files: {{files}}',{files:issue.files.join(', ')}));
    if(issue.next_action)add(item,'p',ui('setup.check_next','Next: {{action}}',{action:issue.next_action}));
    panel.append(item);
  }
}
function select(reset=true){
  if(active){$('model').value=active.model.id;reset=false;}if(reset)history=[];
  const model=chosen(),task=model?.task||'',pair=task==='pair-classification',nativeOCR=isTesseract(model),instruction=isInstruction(model),issue=controlIssue(model),unsupported=taskIssue(model);
  renderModelSetup(model);
  for(const row of [...$('models').children]){const id=row.dataset?.modelId||row._modelId;if(!id)continue;row.className='model'+(row.dataset?.external?' external':'')+(id===model?.id?' selected':'');row.setAttribute?.('aria-current',id===model?.id?'true':'false');}
  const setting=name=>serverInfo?.settings?.[name]?.value;
  const promptSetting=task==='speech'?'max_speech_chars':['image','video','rerank','image-to-text'].includes(task)?'max_prompt_chars':'max_text_chars';
  applyMaxLength($('prompt'),setting(promptSetting));
  for(const id of ['premise','hypothesis'])applyMaxLength($(id),setting('max_text_chars'));
  const inputLimit=setting(task==='pair-classification'?'pair_max_input_tokens':task==='sentiment'?'sentiment_max_input_tokens':task==='rerank'?'rerank_max_input_tokens':'embedding_max_tokens');
  $('max_input_tokens').max=inputLimit===undefined?'':String(inputLimit);
  $('pair-fields').hidden=!pair;$('rerank-fields').hidden=task!=='rerank';$('segmentation-fields').hidden=task!=='image-segmentation';
  $('music-fields').hidden=task!=='music';
  if(task==='music'){
    const controls=model?.capabilities?.music?.controls||{},song=$('music-mode').value==='song';
    if($('music-lyrics').parentElement)$('music-lyrics').parentElement.hidden=!(song&&controls.lyrics);
    for(const [id,supported] of [['music-duration',controls.duration_max],['music-language',controls.vocal_languages?.length],['music-bpm',controls.bpm],['music-key',controls.key_scale],['music-reference',controls.reference_audio]]){
      const node=$(id);if(node.parentElement)node.parentElement.hidden=!supported;node.disabled=!supported;
    }
  }
  if($('audio-action').parentElement)$('audio-action').parentElement.hidden=task!=='transcription';$('audio-translation-fields').hidden=task!=='transcription'||$('audio-action').value!=='translate';
  $('image-edit-fields').hidden=task!=='image'||Boolean(issue)||generationOnly(model);$('prompt-label').hidden=pair||nativeOCR||Boolean(issue)||['transcription','vad','depth-estimation','image-segmentation','tensor'].includes(task);
  setContent($('prompt-caption'),instruction?ui("ui.instruction","Instruction"):task==='rerank'?ui("ui.query","Query"):tr('composer.prompt'));
  setContent($('settings-caption'),nativeOCR?ui("ui.ocr_settings","OCR settings"):instruction?ui("ui.edit_settings","Edit settings"):tr('composer.settings'));
  for(const id of ['premise','hypothesis'])$(id).required=pair;
  const runState=model?.external||model?.status==='loaded'?ui('run.ready','Ready. Press Run.'):ui('run.unloaded','Not loaded. Load it first, or Run loads it if this server allows.');
  setContent($('task'),unsupported|| (nativeOCR?ui("ui.native_ocr","Native OCR · Tesseract"):instruction?ui("ui.instruction_edit","Instruction image edit"):runState));
  if($('assignment-panel'))$('assignment-panel').hidden=Boolean(model?.external);
  $('audio-analysis-fields').hidden=task!=='transcription'||$('audio-action').value!=='analyze';
  $('vad-gate-label').hidden=task!=='transcription';$('file-label').hidden=!['vision','transcription','image-to-text','object-detection','video','depth-estimation','image-segmentation','vad','diarization'].includes(task);
  $('file').accept=['transcription','vad','diarization'].includes(task)?'audio/*':'image/*';$('file').required=nativeOCR||['transcription','vad','diarization','depth-estimation','image-segmentation'].includes(task);
  $('prompt').required=!['transcription','image-to-text','object-detection','pair-classification','depth-estimation','image-segmentation','vad','diarization','tensor'].includes(task);
  if(task==='diarization')$('prompt-label').hidden=true;
  $('image-source').required=instruction;
  setContent($('image-source-caption'),instruction?ui("ui.source_required","Source image (required)"):ui("ui.source_optional","Source image (optional)"));
  setContent($('image-source-help'),instruction?ui("ui.source_instruction_help","Choose the source image and describe the change. Its dimensions are kept."):ui("ui.source_edit_help","Add a source image to edit it. Leave it empty to generate a new image."));
  $('mask-fields').hidden=instruction;$('image-mask').disabled=instruction;$('preserve-unmasked').disabled=instruction;
  setContent($('clear-image-inputs'),instruction?ui("ui.clear_source","Clear source image"):ui("ui.clear_mask","Clear source and mask"));
  $('clear').hidden=!['chat','vision'].includes(task);
  const requireLoaded=$('require-loaded');if(requireLoaded.parentElement)requireLoaded.parentElement.hidden=!(task==='chat'&&!model?.external&&serverInfo?.inference_contracts?.require_loaded?.revision===1&&serverInfo.inference_contracts.require_loaded.routes?.includes('/v1/chat/completions'));
  if(task!=='chat'||model?.external)requireLoaded.checked=false;
  for(const [id,verb] of [['load',ui("ui.load","Load")],['unload',ui("ui.unload","Unload")],['model-details',ui("ui.show_runtime_for","Show runtime details for")],['remove-registration',ui("ui.remove_registration_for","Remove registration for")]])setUiAttribute($(id),'aria-label',ui("ui.lifecycle_aria","{{action}} {{model}}",{action:verb,model:model?.id||ui("ui.selected_model","selected model")}));
  if(runtimeModelId!==model?.id||model?.status!=='loaded')$('model-runtime').hidden=true;
  renderDeviceSummary(model);renderModelReadout(model);revealSelected();
  const fields={chat:['max_tokens','temperature','top_p','top_k','min_p','repeat_penalty','seed'],vision:['max_tokens','temperature','top_p','top_k','min_p','repeat_penalty','seed'],sentiment:['max_input_tokens','min_score','min_margin'],'pair-classification':['max_input_tokens','truncate'],rerank:['max_input_tokens','top_n','rerank-truncate','rerank-return-documents'],vad:['vad-sample-rate','vad-threshold','vad-min-speech','vad-min-silence','vad-speech-pad','vad-return-frames'],'image-segmentation':['segment-multimask','segment-threshold'],image:['size','steps','seed'],video:['size','steps','seed','frames','fps'],speech:['voice'],music:['seed','guidance_scale','steps'],transcription:['language'],translation:['max_tokens','temperature','top_p','language'],'image-to-text':['max_tokens','temperature','top_p'],'object-detection':['threshold','nms','nms_iou']};
  const visible=issue?[]:nativeOCR?['ocr_language','ocr_psm','ocr_geometry']:instruction?['guidance_scale','image_guidance_scale','steps','n','seed']:(fields[task]||[]).filter(name=>task!=='music'||model?.capabilities?.music?.controls?.[name]===true);
  if(!issue&&(task==='diarization'||task==='transcription'&&$('audio-action').value==='analyze'))visible.push('diarization-count','diarization-threshold','diarization-on','diarization-off','audio-timeout');
  if(!issue&&task==='transcription'&&$('audio-action').value==='analyze')visible.push('max_tokens');
  document.querySelectorAll('.settings input, .settings select').forEach(input=>{if(input.id==='require-loaded'||input.id.startsWith('music-'))return;input.parentElement.hidden=!visible.includes(input.id);input.disabled=!visible.includes(input.id);});
  ($('settings-details')||document.querySelector('#generate details')).hidden=!visible.length;
  if(nativeOCR)setUiAttribute($('ocr_language'),"placeholder",ui("ui.default_model_server","Model / server default")+(model.capabilities.defaults?.language?': '+model.capabilities.defaults.language:''));
  if((controlIssue(model)||unsupported)&&!active&&model)status(controlIssue(model)||unsupported,Boolean(model));busy();
  loadAssignment().catch(()=>{});renderCabinet();
}
async function modelControls(model){
  try{
    const capabilities=await(await api('/api/models/capabilities',{method:'POST',json:{model:model.id}})).json();
    if(!capabilities||typeof capabilities.backend!=='string'||capabilities.task!==model.task||(model.task==='image'&&!Object.hasOwn(capabilities,'instruction_edit')))throw Error(ui("ui.model_missing_controls","Incomplete model capability response"));
    if(model.task==='image'&&capabilities.instruction_edit!==null&&!isInstruction({...model,capabilities}))throw Error(ui("ui.unsupported_instruction","Unsupported instruction-edit capability profile"));
    return {...model,capabilities};
  }catch(error){return {...model,capability_error:error.message};}
}
async function refresh(forceInfo=false){
  const epoch=++refreshEpoch,usedKey=key;++lifecycleEpoch;let controlsUpdated=false,brokerRefresh=null;refreshing=true;busy();
  try{
    const requestInfo=Boolean(forceInfo)||!infoAttempted;
    const infoRequest=requestInfo?(infoAttempted=true,api('/api/info').then(response=>response.json()).catch(()=>null)):Promise.resolve(serverInfo);
    const [resultResponse,info]=await Promise.all([api('/v1/models'),infoRequest]);
    const result=await resultResponse.json();serverInfo=info;applyServerInfo();
    const rows=Array.isArray(result.data)&&typeof catalogEntry==='function'?result.data.map(catalogEntry).filter(model=>model!==null):result.data;
    if(!Array.isArray(rows)||rows.some(model=>!model||typeof model.id!=='string'||typeof model.task!=='string'))throw Error(ui("ui.invalid_catalog","Invalid model catalog"));
    const identified=rows.map(model=>({...model,configured:serverInfo?.models?.[model.id]||null}));
    const updated=await Promise.all(identified.map(model=>model.external?externalControls(model):modelControls(model)));if(epoch!==refreshEpoch)return;
    catalog=updated;controlsUpdated=true;refreshFailed=false;refreshVAD();refreshTaskModels();const requested=active?.model.id||$('model').value;
    const ordered=[...catalog.filter(model=>!model.external),...catalog.filter(model=>model.external)];let divided=false;
    const selected=catalog.some(model=>model.id===requested)?requested:(ordered[0]?.id||'');
    $('model').replaceChildren();setContent($('models'),null);$('models').replaceChildren();   // drop a previous empty-state binding before rows are appended (same class as the jobs list)
    for(const m of ordered){
      const op=document.createElement('option');op.value=m.id;if(m.external)op.className='external';
      setContent(op,m.external?ui("ui.model_title_cloud","{{model}} · {{task}} · {{provider}}",{model:m.id,task:taskLabel(m.task),provider:upstreamLabel(m.external.upstream)}):ui("ui.model_title","{{model}} · {{task}}",{model:m.id,task:taskLabel(m.task)}));$('model').append(op);
      if(m.external&&!divided){divided=true;const divider=document.createElement('div');divider.className='rail-divider';divider.setAttribute?.('aria-hidden','true');setContent(divider,ui('rail.cloud_routes','Cloud routes'));$('models').append(divider);}
      const row=document.createElement('div');row.className='model'+(m.external?' external':'')+(m.id===selected?' selected':'');if(row.dataset){row.dataset.modelId=m.id;row.dataset.state=String(m.status||'unknown').toLowerCase();if(m.external)row.dataset.external='1';}else row._modelId=m.id;row.setAttribute?.('aria-current',m.id===selected?'true':'false');
      const choose=document.createElement('button');choose.type='button';choose.className='model-select';setUiAttribute(choose,'aria-label',ui('a11y.selectModel','Select {{model}}, {{task}}, {{status}}',{model:m.id,task:taskLabel(m.task),status:stateLabel(m.status)}));
      const head=document.createElement('div');head.className='model-head';const name=document.createElement('strong');name.title=m.id;
      const at=m.external?m.id.lastIndexOf('@'):-1;if(at>0){name.append(m.id.slice(0,at));const suffix=document.createElement('span');suffix.className='at';suffix.textContent=m.id.slice(at);name.append(suffix);}else setContent(name,m.id);
      const residency=document.createElement('span');residency.className='status-chip '+String(m.status||'unknown').toLowerCase().replace(/[^a-z0-9_-]/g,'-');setContent(residency,stateLabel(m.status||'unknown'));head.append(name,residency);
      const configured=m.configured||{},source=m.capabilities?.source||{};
      const implemented=[...(Array.isArray(m.capabilities?.implemented)?m.capabilities.implemented:[]),...Object.entries(m.capabilities||{}).filter(([,value])=>value&&typeof value==='object'&&value.implemented===true).map(([name])=>name)];
      const meta=document.createElement('div');meta.className='model-meta';const chips=document.createElement('span');chips.className='model-chips';meta.append(chips);
      for(const [value,advanced] of [[taskLabel(m.task),false],[m.capabilities?.backend||configured.backend,true],[configured.device==='auto'?tr('advanced.auto'):configured.device,true]].filter(([value])=>value)){const chip=document.createElement('span');chip.className='model-chip'+(advanced?' advanced-only':'');setContent(chip,value);chips.append(chip);}
      const badge=m.external?externalBadge(m):null;if(badge)chips.append(badge);
      const technical=document.createElement('details');technical.className='model-technical';const summary=document.createElement('summary');summary.setAttribute?.('data-i18n','models.technical');setContent(summary,tr('models.technical'));setUiAttribute(summary,'aria-label',ui('models.technical_for','Details for {{model}}',{model:m.id}));
      const trace=document.createElement('p');setContent(trace,uiJoin([source.repository,source.revision,implemented.length?ui("ui.capabilities","Capabilities: {{items}}",{items:[...new Set(implemented)].join(', ')}):null,m.runtime_fault,m.capability_error?ui("ui.capability_error","Capabilities unavailable: {{error}}",{error:m.capability_error}):null].filter(Boolean),' · ',ui("ui.source_metadata_empty","No additional source or runtime metadata reported.")));
      technical.append(summary,trace);const glyph=document.createElement('span');glyph.className='cell-glyph '+(m.external?'external':String(m.status||'unknown').toLowerCase().replace(/[^a-z0-9_-]/g,'-'));glyph.setAttribute?.('aria-hidden','true');
      choose.append(glyph,head,meta);row.append(choose,technical);
      choose.onclick=()=>{if(active)return;$('model').value=m.id;$('model').onchange();};
      $('models').append(row);
    }
    if(active&&!catalog.some(m=>m.id===active.model.id)){const op=document.createElement('option');op.value=active.model.id;setContent(op,active.model.id+ui("ui.request_in_progress"," · request in progress"));$('model').append(op);}
    if(active||catalog.some(m=>m.id===selected))$('model').value=selected;
    if(!catalog.length)setContent($('models'),tr('models.empty'));cabinetSetupPanel();
    setContent($('connection'),uiCount('connection.connected','Connected, {{count}} models','Connected, 1 model',{count:catalog.length}));if($('connection').parentElement)$('connection').parentElement.className='connection connected';cabinetConnection(true);$('connection-panel')?.classList?.remove('needed');if(lastConnectionError&&!active)status(tr('status.connected'),false,'success');lastConnectionError=false;select(!catalog.some(m=>m.id===requested));
    // the Cloud and routing panel is gated only by /api/keys/self: it refreshes independently of the jobs and acquisition reads
    brokerRefresh=typeof refreshBroker==='function'?refreshBroker().catch(()=>{}):Promise.resolve();await refreshJobs();await reconcileAcquisitions();await brokerRefresh;
  }catch(e){if(epoch!==refreshEpoch)return;
    // a 401/403 for a key this tab has since replaced (a Studio self-rotation) is stale: refresh again with the current key
    if(key!==usedKey&&(e.status===401||e.status===403))return refresh();
    if(!brokerRefresh&&(e.status===401||e.status===403)&&typeof brokerReset==='function')brokerReset();if(!controlsUpdated)refreshFailed=true;setContent($('connection'),tr('connection.needed'));if($('connection').parentElement)$('connection').parentElement.className='connection offline';cabinetConnection(false);if(e.status===401&&$('connection-panel')){$('connection-panel').classList?.add('needed');$('connection-panel').open=true;}if(!active){lastConnectionError=true;status(e.message,true);}}
  finally{if(epoch===refreshEpoch){refreshing=false;busy();}}
}
function card(label,text=''){const box=document.createElement('article'),kind=label?.[UI_MESSAGE]?label.name.split('.').at(-1):String(label).split('·')[0].trim().toLowerCase().replace(/[^a-z0-9]+/g,'-');box.className='result result-'+kind;const title=document.createElement('h2');title.className='label';title.tabIndex=-1;setContent(title,label);const pre=document.createElement('pre');setContent(pre,text);const spine=document.createElement('span');spine.className='spine';spine.setAttribute?.('aria-hidden','true');box.append(spine,title,pre);$('results').prepend(box);renderCabinet();return {box,pre,title};}
async function asset(url,box,signal){const parsed=new URL(url,location.origin);if(parsed.origin!==location.origin||!parsed.pathname.startsWith('/v1/files/'))throw Error(ui("ui.unexpected_asset","Unexpected asset location"));const blob=await(await api(parsed.pathname,{signal})).blob();signal?.throwIfAborted();box._bytes=(box._bytes||0)+(blob.size||0);const local=URL.createObjectURL(blob);urls.push(local);let view;if(blob.type.startsWith('image/')){view=document.createElement('img');setUiAttribute(view,"alt",ui("ui.asset_image","Generated image asset"));}else if(blob.type.startsWith('video/')){view=document.createElement('video');view.controls=true;setUiAttribute(view,'aria-label',ui("ui.asset_video","Generated video asset"));}else if(blob.type.startsWith('audio/')){view=document.createElement('audio');view.controls=true;setUiAttribute(view,'aria-label',ui("ui.asset_audio","Generated audio asset"));}if(view){view.src=local;box.append(view);if(view.tagName?.toLowerCase?.()==='img')mediaSize(view,box);}const link=document.createElement('a');link.href=local;link.download=parsed.pathname.split('/').pop();setContent(link,ui("ui.save_asset","Save asset"));box.append(link);}
function mediaSize(image,box){
  // the decoded size under an image: a tiny image still reads as a result, not an empty card
  const size=document.createElement('span');size.className='media-size';box.append(size);
  const show=()=>{if(image.naturalWidth)size.textContent=`${image.naturalWidth}×${image.naturalHeight} px`;};
  if(image.complete)show();else image.addEventListener?.('load',show,{once:true});return size;
}
function add(parent,tag,text){const node=document.createElement(tag);setContent(node,text);parent.append(node);return node;}
function resultDetails(box,result,summaryText=ui("ui.full_result","Full result and provenance")){
  const details=document.createElement('details');add(details,'summary',summaryText);add(details,'pre',JSON.stringify(result,null,2));box.append(details);return details;
}
function depthValues(result){
  if(result.encoding!=='base64_float32_le'||!Number.isSafeInteger(result.width)||!Number.isSafeInteger(result.height)||result.width<1||result.height<1||typeof result.data!=='string')throw Error(ui("ui.depth_invalid","Invalid relative-depth result metadata"));
  let binary;try{binary=atob(result.data);}catch{throw Error(ui("ui.depth_base64_invalid","Invalid relative-depth base64 data"));}
  const expected=result.width*result.height*4;if(binary.length!==expected)throw Error(ui("ui.depth_bytes_invalid","Relative-depth byte count does not match its dimensions"));
  const bytes=new Uint8Array(expected);for(let index=0;index<expected;index++)bytes[index]=binary.charCodeAt(index);
  const view=new DataView(bytes.buffer),values=[];for(let offset=0;offset<expected;offset+=4){const value=view.getFloat32(offset,true);if(!Number.isFinite(value))throw Error(ui("ui.depth_finite","Relative-depth values must be finite"));values.push(value);}
  return {bytes,values};
}
async function showResult(result,label,signal){
  signal?.throwIfAborted();if(result.object==='text_pair.classification')return showPairs(result,label);
  const depth=result?.task==='depth-estimation'?depthValues(result):null;
  const {box,pre,title}=card(label);
  if(Array.isArray(result?.data)&&result.data.length&&result.data.every(row=>Array.isArray(row?.embedding))){
    const dimensions=result.data[0].embedding.length;
    setContent(pre,uiCount("ui.vector_preview","{{count}} vectors · {{dimensions}} dimensions\nPreview: {{preview}}","{{count}} vector · {{dimensions}} dimensions\nPreview: {{preview}}",{count:result.data.length,dimensions,preview:result.data[0].embedding.slice(0,12).join(', ')+(dimensions>12?' …':'')}));
    resultDetails(box,result,ui("ui.full_vectors","Full vectors, usage and input accounting"));
  }else if(result?.object==='text.sentiment'){
    const scores=result.scores?Object.entries(result.scores).map(([name,value])=>`${name}: ${value}`).join(' · '):ui("ui.no_scores","No numeric scores returned");
    setContent(pre,uiJoin([result.label,' · ',result.status,'\n',scores,'\n',result.score_kind,result.calibrated===false?ui("ui.uncalibrated_suffix"," · uncalibrated"):'']));
    resultDetails(box,result);
  }else if(result?.task==='object-detection'||Array.isArray(result?.objects)){
    pre.remove();add(box,'p',uiCount("ui.detection_summary","{{count}} detected objects{{size}}. Native scores are not verified truth.","{{count}} detected object{{size}}. Native scores are not verified truth.",{count:result.objects.length,size:result.width&&result.height?ui("ui.input_size"," · {{width}}×{{height}} input",{width:result.width,height:result.height}):''}));
    for(const item of result.objects)add(box,'p',ui("ui.detection_item","{{label}} · {{score}} · box {{box}}",{label:item.label,score:item.score,box:[item.box.xmin,item.box.ymin,item.box.xmax,item.box.ymax].join(', ')}));
    resultDetails(box,result);
  }else if(result?.score_kind==='cross_encoder_logit'&&Array.isArray(result.results)){
    pre.remove();add(box,'p',ui("ui.rerank_note","Reranker order and raw cross-encoder logits; scores are uncalibrated and are not probabilities."));
    for(const item of result.results)add(box,'p',ui("ui.rerank_item","Document {{index}} · raw score {{score}}{{text}}",{index:item.index,score:item.relevance_score,text:item.document?.text!==undefined?' · '+item.document.text:''}));
    resultDetails(box,result,ui("ui.full_rerank","Full rerank result, input accounting and provenance"));
  }else if(result?.task==='diarization'||result?.task==='audio_analysis'){
    const speakers=result.diarization||result;
    if(result.task==='audio_analysis')setContent(pre,result.text||ui('ui.no_text','No text returned.'));else pre.remove();
    add(box,'p',ui('audio.result_summary','{{count}} anonymous speakers · {{overlaps}} overlapping intervals',{count:speakers.speakers.length,overlaps:speakers.overlaps.length}));
    add(box,'p',ui('audio.anonymous_note','Labels apply only to this recording. Overlap does not separate voices or identify people.'));
    for(const item of speakers.segments)add(box,'p',ui('audio.speaker_interval','{{speaker}} · {{start}}–{{end}} seconds',{speaker:item.speaker,start:item.start.toFixed(2),end:item.end.toFixed(2)}));
    for(const item of speakers.overlaps)add(box,'p',ui('audio.overlap_interval','Overlap: {{speakers}} · {{start}}–{{end}} seconds',{speakers:item.speakers.join(', '),start:item.start.toFixed(2),end:item.end.toFixed(2)}));
    for(const item of result.speaker_segments||[]){
      const labels=item.speakers.length?item.speakers.join(', '):ui('audio.unattributed','Unattributed');
      add(box,'p',ui('audio.speaker_text','{{speakers}}: {{text}}',{speakers:uiJoin([labels,item.overlap?uiJoin([' · ',ui('audio.overlap','Overlap')]):'']),text:item.text}));
      if(item.attribution==='timestamps_outside_audio')add(box,'p',ui('audio.outside_timing','The transcript timing extends outside this recording; its speaker is not assigned.'));
    }
    if(result.text_analysis)resultDetails(box,result.text_analysis,ui('audio.transcript_analysis','Transcript text analysis'));
    resultDetails(box,result,ui('audio.result_details','Speaker timing, alignment, limits and provenance'));
  }else if(result?.task==='vad'&&Array.isArray(result.segments)){
    pre.remove();add(box,'p',uiCount("ui.vad_result","{{count}} speech intervals · {{kind}} · uncalibrated.","{{count}} speech interval · {{kind}} · uncalibrated.",{count:result.segments.length,kind:result.score_kind||ui("ui.speech_activity","speech activity")}));
    for(const item of result.segments)add(box,'p',ui("ui.vad_interval","{{start}}–{{end}} seconds · samples {{first}}–{{last}}",{start:item.start,end:item.end,first:item.start_sample,last:item.end_sample}));
    resultDetails(box,result);
  }else if(depth){
    pre.remove();add(box,'p',ui("ui.depth_note","Relative inverse depth preview: brighter values are nearer. Display normalization is not metric depth and values are not physically comparable across scenes."));
    const canvas=document.createElement('canvas');canvas.width=result.width;canvas.height=result.height;canvas.setAttribute('role','img');setUiAttribute(canvas,'aria-label',ui("ui.depth_alt","Display-normalized relative inverse depth; brighter is nearer"));
    const context=canvas.getContext('2d'),image=context.createImageData(result.width,result.height);
    let minimum=Infinity,maximum=-Infinity;for(const value of depth.values){if(value<minimum)minimum=value;if(value>maximum)maximum=value;}const span=maximum-minimum;
    for(let index=0;index<depth.values.length;index++){const shade=Math.round(255*(span?(depth.values[index]-minimum)/span:.5)),offset=index*4;image.data[offset]=shade;image.data[offset+1]=shade;image.data[offset+2]=shade;image.data[offset+3]=255;}
    context.putImageData(image,0,0);box.append(canvas);
    const raw=URL.createObjectURL(new Blob([depth.bytes],{type:'application/octet-stream'}));urls.push(raw);const link=add(box,'a',ui("ui.save_depth","Save raw little-endian float32 depth"));link.href=raw;link.download=`${result.model||'depth'}-${result.width}x${result.height}.f32`;
    resultDetails(box,{...result,data:'[raw float32 available from download]'},ui("ui.depth_details","Depth metadata and provenance"));
  }else if(result?.task==='image-segmentation'&&Array.isArray(result.masks)){
    pre.remove();add(box,'p',ui("ui.mask_note","Image-aligned RGBA edit masks in native order. Predicted IoU values are uncalibrated; Studio does not choose a “best” mask or invent class labels."));
    for(const mask of result.masks){
      const group=document.createElement('section');add(group,'p',ui("ui.mask_item","Mask {{index}} · {{pixels}} selected pixels · predicted IoU {{iou}} · uncalibrated",{index:mask.index,pixels:mask.selected_pixels,iou:mask.predicted_iou}));
      const image=document.createElement('img');image.src='data:image/png;base64,'+mask.png_base64;setUiAttribute(image,"alt",ui("ui.mask_alt","Segmentation mask {{index}}; transparent pixels are selected for editing",{index:mask.index}));group.append(image);
      const link=add(group,'a',ui("ui.save_mask","Save mask PNG"));link.href=image.src;link.download=`${result.model||'segmentation'}-mask-${mask.index}.png`;box.append(group);
    }
    resultDetails(box,result,ui("ui.mask_details","Mask prompt, native order and provenance"));
  }else if(result?.text!==undefined&&(['transcription','transcribe','translate'].includes(result.task)||Array.isArray(result.segments))){
    setContent(pre,result.text||ui("ui.no_text","No text returned."));
    if(Array.isArray(result.segments)){const list=document.createElement('div');for(const item of result.segments)add(list,'p',`${item.start}–${item.end}: ${item.text||''}`);box.append(list);}
    resultDetails(box,result,ui("ui.transcript_details","Transcript segments, quality fields and provenance"));
  }else if(result?.text!==undefined&&(result.source_language!==undefined||result.target_language!==undefined||result.translated!==undefined)){
    setContent(pre,uiJoin([result.text,'\n',result.source_language||ui("ui.source_unspecified","source unspecified"),' → ',result.target_language||ui("ui.target_unspecified","target unspecified"),result.translated===false?ui("ui.unchanged_suffix"," · unchanged"):ui("ui.translated_suffix"," · translated")]));
    resultDetails(box,result);
  }else if(result.ocr?.backend==='tesseract'){
    setContent(pre,result.text||ui("ui.no_text","No text returned."));resultDetails(box,result,ui("ui.ocr_details","OCR details and geometry (native confidence is uncalibrated)"));
  }else if(result?.text!==undefined&&(result.task==='image-to-text'||result.lines)){
    setContent(pre,result.text||ui("ui.no_text","No text returned."));resultDetails(box,result);
  }else if(result.url){pre.remove();await asset(result.url,box,signal);if(result.task==='music')resultDetails(box,result,ui("ui.music_details","Music format, effective controls and provenance"));
  }else if(Array.isArray(result.data)&&result.data.some(item=>item.url||item.b64_json)){
    pre.remove();for(const item of result.data){if(item.url)await asset(item.url,box,signal);else{const image=document.createElement('img');image.src='data:image/png;base64,'+item.b64_json;setUiAttribute(image,"alt",ui("ui.generated_image","Generated image"));box.append(image);mediaSize(image,box);}}
  }else setContent(pre,typeof result==='string'?result:JSON.stringify(result,null,2));
  if(result.masking)resultDetails(box,result.masking,ui("ui.preserve_details","Mask and pixel preservation details"));
  title.focus?.();return box;
}
const fileData=file=>new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(file);});
function controls(names){const result={};for(const name of names){const value=$(name).value;if(value!=='')result[name]=['size','language','voice','nms'].includes(name)?value:Number(value);}return result;}
function showPairs(result,label){
  const {box,pre,title}=card(label);pre.remove();
  const add=(parent,tag,text)=>{const node=document.createElement(tag);setContent(node,text);parent.append(node);return node;};
  add(box,'p',ui("ui.pair_note","Model-native predictions. Scores are uncalibrated softmax values, not verified truth or calibrated probabilities."));
  for(const row of result.results){
    add(box,'p',ui("ui.pair_prediction","Pair {{index}} · {{id}} · Predicted native label: {{label}} (class {{class}})",{index:row.index,id:row.id,label:row.label,class:row.label_index}));
    const table=document.createElement('table');table.className='pair-scores';box.append(table);
    add(table,'caption',ui("ui.class_outputs","Native class outputs"));const header=document.createElement('tr');table.append(header);
    for(const text of [ui("ui.class_index","Class index"),ui("ui.native_label","Native label"),ui("ui.raw_logit","Raw logit"),ui("ui.raw_score","Uncalibrated score")])add(header,'th',text).scope='col';
    for(const item of result.labels){const tr=document.createElement('tr');table.append(tr);for(const value of [item.index,item.label,row.logits[item.index],row.scores[item.index]])add(tr,'td',String(value));}
    add(box,'p',ui("ui.pair_accounting","Input tokens: {{input}}; processed: {{processed}}; premise truncated: {{truncated}}. Hypothesis preserved.",{input:row.input.input_tokens,processed:row.input.processed_tokens,truncated:row.input.truncated?ui("ui.yes","yes"):ui("ui.no","no")}));
  }
  const trace=document.createElement('details');box.append(trace);add(trace,'summary',ui("ui.pair_details","Full result, input accounting and model provenance"));add(trace,'pre',JSON.stringify(result,null,2));title.focus?.();
}
async function streamChat(data,signal){
  const {box,pre,title}=card(ui("ui.assistant","Assistant · {{model}}",{model:data.model}));let text='',pending='',reader,chunks=0,usage=null;
  const started=nowMs();box.classList?.add('streaming');
  const counter=document.createElement('p'),count=document.createElement('b'),unit=document.createElement('span');counter.className='counter';counter.setAttribute?.('aria-hidden','true');count.textContent='0';setContent(unit,ui('receipt.chunks_unit','chunks'));counter.append(count,unit);box.append(counter);
  try{const response=await api('/v1/chat/completions',{method:'POST',json:{...data,stream:true,stream_options:{include_usage:true}},signal});reader=response.body.getReader();const decoder=new TextDecoder();
    if(typeof appendProvenance==='function')appendProvenance(box);while(true){const {done,value}=await reader.read();signal.throwIfAborted();if(done)break;pending+=decoder.decode(value,{stream:true});let boundary;while((boundary=pending.indexOf('\n'))>=0){const line=pending.slice(0,boundary).trim();pending=pending.slice(boundary+1);if(!line.startsWith('data:'))continue;const raw=line.slice(5).trim();if(raw==='[DONE]')continue;const item=JSON.parse(raw);if(item.error)throw Error(item.error.message);if(item.usage)usage=item.usage;const delta=item.choices?.[0]?.delta?.content||'';if(delta){chunks++;count.textContent=String(chunks);}text+=delta;setContent(pre,text);}}
  }catch(error){box.classList?.remove('streaming');setContent(title,uiJoin([textBindings.get(title)||title.textContent,' · ',signal.aborted?ui("ui.partial_stopped","Stopped (partial)"):ui("ui.incomplete","Incomplete")]));appendReceipt(box,{model:data.model,started,usage,chunks,text,failed:true});throw error;}finally{reader?.releaseLock();}
  if(Number.isFinite(usage?.completion_tokens)){count.textContent=String(usage.completion_tokens);setContent(unit,ui('receipt.tokens_unit','tokens'));}
  appendReceipt(box,{model:data.model,started,usage,chunks,text});title.focus?.();return text;
}
async function refreshJobs(){
  let response;
  try{response=await api('/api/jobs');}
  catch(error){
    // a key without the 'jobs' scope is refused here (403): the list is not available to this key, which is not a
    // connection failure, so the section says so and the connection header and status line are left alone
    if(error?.status!==403)throw error;
    setContent($('jobs'),null);$('jobs').replaceChildren();setContent($('jobs'),tr('jobs.unavailable'));return;
  }
  const result=await response.json();
  // The empty-state message is a translation binding on the container; drop it before rows are appended so a later
  // translation pass (mode or locale change) cannot rewrite a populated list back to "No creations yet".
  setContent($('jobs'),null);$('jobs').replaceChildren();liveJobTimes.clear();
  jobsSummary={running:0,queued:0,done:0,failed:0,cancelled:0};
  if(result.data.length){const head=document.createElement('div');head.className='job-head';head.setAttribute?.('aria-hidden','true');
    for(const [name,text] of [['jobs.col_state','State'],['jobs.col_started','Started'],['jobs.col_job','Job'],['jobs.col_task','Task'],['jobs.col_model','Model'],['jobs.col_time','Time'],['jobs.col_actions','Actions']])add(head,'span',ui(name,text));$('jobs').append(head);}
  for(const j of result.data){
    if(['running','cancelling'].includes(j.status))jobsSummary.running++;else if(j.status==='queued')jobsSummary.queued++;else if(j.status==='completed')jobsSummary.done++;else if(['failed','interrupted'].includes(j.status))jobsSummary.failed++;else if(j.status==='cancelled')jobsSummary.cancelled++;
    const live=['queued','running','cancelling'].includes(j.status),row=document.createElement('div');row.className='job '+j.status;
    const state=document.createElement('span');state.className='status-chip '+j.status;
    const glyph=document.createElement('span');glyph.className='cell-glyph '+j.status;glyph.setAttribute?.('aria-hidden','true');
    const word=document.createElement('span');setContent(word,ui("ui.job_phase","{{status}}{{phase}}",{status:stateLabel(j.status),phase:j.native_generation?.stage?' · '+stageLabel(j.native_generation.stage):''}));state.append(glyph,word);
    if(j.status==='running'){const lattice=document.createElement('span');lattice.className='job-lattice';lattice.setAttribute?.('aria-hidden','true');for(let index=0;index<6;index++)lattice.append(document.createElement('i'));state.append(lattice);}
    const start=document.createElement('span');start.className='job-start';setContent(start,clockTime(j.created_at));
    const id=document.createElement('span');id.className='job-id';id.title=String(j.id||'');setContent(id,String(j.id||'').slice(0,8));
    const name=document.createElement('span');name.className='job-task';setContent(name,taskLabel(j.task||'video'));
    const model=document.createElement('span');model.className='job-model';setContent(model,j.model);
    const time=document.createElement('span');time.className='job-time';setContent(time,jobElapsed(j));if(live)liveJobTimes.set(time,j.created_at);
    const actions=document.createElement('span');actions.className='job-actions';row.append(state,start,id,name,model,time,actions);
    if(j.status==='completed'){const b=document.createElement('button');b.className='quiet';setContent(b,tr('jobs.open'));setUiAttribute(b,'aria-label',tr('a11y.openJob',{task:taskLabel(j.task||'video'),id:j.id,model:j.model}));b.onclick=()=>showResult(j.result,ui("ui.asset","Asset")).then(box=>{if(box)appendReceipt(box,{model:j.model,seconds:jobSeconds(j),result:j.result,decision:j.decision_id,at:j.finished_at??j.updated_at??j.created_at});}).catch(e=>status(e.message,true));actions.append(b);}
    if(live){const b=document.createElement('button');b.className='quiet';setContent(b,tr('jobs.cancel'));setUiAttribute(b,'aria-label',tr('a11y.cancelJob',{task:taskLabel(j.task||'video'),id:j.id,model:j.model}));b.onclick=async()=>{try{await api('/api/jobs/'+j.id,{method:'DELETE'});await refreshJobs();}catch(e){status(e.message,true);}};actions.append(b);}
    if(j.error){const error=document.createElement('small');error.className='job-error';setContent(error,j.error);row.append(error);}$('jobs').append(row);
  }
  if(!result.data.length)setContent($('jobs'),tr('jobs.empty'));
  startJobClock();renderCabinet();
}
// --- bay readouts, the model readout, receipts, the progress lattice and the boot readout. Display only: the one request
// made here is the cost lookup of a receipt (GET /api/broker/decisions, only with broker:read). ---
let jobsSummary=null,cabinetConnected=null,setupToggled=false,setupSettled=false,jobClock=null;const liveJobTimes=new Map();
function nowMs(){return globalThis.performance?.now?.()??Date.now();}
function upstreamLabel(id){try{const row=(brokerState.upstreams||[]).find(item=>item.id===id);return row?.label||brokerProfileLabel(row?.profile||id)||id||'';}catch{return id||'';}}
function clockTime(seconds){const value=Number(seconds);if(!Number.isFinite(value)||value<=0)return '';const date=new Date(value*1000);return [date.getHours(),date.getMinutes(),date.getSeconds()].map(part=>String(part).padStart(2,'0')).join(':');}
function jobSeconds(job){
  const live=['queued','running','cancelling'].includes(job?.status),start=Number(job?.created_at),end=live?Date.now()/1000:Number(job?.updated_at??job?.finished_at??job?.created_at);
  return Number.isFinite(start)&&Number.isFinite(end)?Math.max(0,end-start):null;
}
function secondsText(value){return Number.isFinite(value)?ui('receipt.seconds','{{value}} s',{value:value.toFixed(value<10?2:1)}):'';}
function jobElapsed(job){return secondsText(jobSeconds(job));}
// running and queued rows keep counting between list refreshes; the timer never keeps a test process alive
function startJobClock(){
  if(!liveJobTimes.size){if(jobClock)globalThis.clearInterval?.(jobClock);jobClock=null;return;}
  if(jobClock||typeof globalThis.setInterval!=='function')return;
  jobClock=globalThis.setInterval(()=>{for(const [node,start] of liveJobTimes)if(node.isConnected===false)liveJobTimes.delete(node);else setContent(node,jobElapsed({created_at:start,status:'running'}));if(!liveJobTimes.size)startJobClock();},1000);jobClock?.unref?.();
}
function readoutParts(node,parts){
  if(!node)return;node.replaceChildren?.();
  for(const part of parts.filter(Boolean)){const span=document.createElement('span');if(part.cls)span.className=part.cls;setContent(span,part.text);node.append(span);}
}
function readoutLine(key,value,cls=''){
  const line=document.createElement('p');line.className='readout-line'+(cls.includes('advanced-only')?' advanced-only':'');
  const name=document.createElement('span');name.className='readout-key';setContent(name,key);const lead=document.createElement('span');lead.className='readout-lead';
  const text=document.createElement('span');text.className='readout-value '+cls.replace('advanced-only','').trim();setContent(text,value);text.title=String(value);line.append(name,lead,text);return line;
}
function renderModelReadout(model){
  const list=$('model-readout'),route=$('route-note'),form=$('generate');form?.classList?.toggle?.('external',Boolean(model?.external));
  if(route){
    route.hidden=!model?.external;route.replaceChildren?.();
    if(model?.external){const where=document.createElement('b');setContent(where,upstreamLabel(model.external.upstream)||ui('readout.cloud','Cloud'));const text=document.createElement('span');setContent(text,ui('ui.route_note','This route leaves your machine. Titanos sends only what you run here.'));route.append(where,text);}
  }
  if(!list)return;list.replaceChildren?.();list.hidden=!model;if(!model)return;
  const source=model.capabilities?.source||{},configured=model.configured||{};
  const rows=model.external?[[ui('readout.route','Route'),ui('readout.route_cloud','Cloud · {{provider}}',{provider:upstreamLabel(model.external.upstream)}),'cloud'],
      [ui('readout.class','Class'),model.external.kind==='virtual'?ui('broker.badge_virtual','Virtual'):ui('readout.cloud_model','Cloud model')],[ui('readout.task','Task'),taskLabel(model.task)],[ui('readout.state','State'),stateLabel(model.status||'unknown')]]
    :[[ui('readout.route','Route'),ui('readout.route_local','This computer')],[ui('readout.state','State'),stateLabel(model.status||'unknown'),model.status==='loaded'?'live':''],
      [ui('readout.device','Device'),configured.device==='auto'?tr('advanced.auto'):configured.device],[ui('readout.task','Task'),taskLabel(model.task)],
      [ui('readout.backend','Backend'),model.capabilities?.backend||configured.backend,'advanced-only'],[ui('readout.mode','Mode'),model.capabilities?.configured_mode,'advanced-only'],
      [ui('readout.source','Source'),source.repository?source.repository+(source.revision?'@'+String(source.revision).slice(0,7):''):null,'advanced-only']];
  for(const [name,value,cls] of rows)if(value!==null&&value!==undefined&&String(value)!=='')list.append(readoutLine(name,value,cls||''));
}
function revealSelected(){
  // keep the selected cell in view inside the rail (the vertical list, or the strip at 640px and below) without moving the page
  const list=$('models'),body=list?.parentElement,row=[...(list?.children||[])].find(node=>String(node.className||'').includes('selected'));
  if(!body||!row||typeof body.getBoundingClientRect!=='function'||typeof row.getBoundingClientRect!=='function')return;
  const outer=body.getBoundingClientRect(),inner=row.getBoundingClientRect();
  if(body.scrollWidth>body.clientWidth+1&&(inner.left<outer.left||inner.right>outer.right))body.scrollLeft+=inner.left-outer.left-12;
  // the row lands at the top edge with its own margin, so the row above is out of view rather than sliced
  const margin=typeof globalThis.getComputedStyle==='function'?parseFloat(getComputedStyle(row).marginTop)||0:0;
  if(body.scrollHeight>body.clientHeight+1&&(inner.top<outer.top||inner.bottom>outer.bottom))body.scrollTop+=inner.top-outer.top-(body.clientTop||0)-margin;
  railMore();
}
function renderLattice(host,completed,total){
  // ten pointy-top cells fill in steps; the native progress element beside it stays the accessible value
  const ratio=total?Math.max(0,Math.min(1,completed/total)):0,filled=Math.floor(ratio*10),lattice=document.createElement('div');lattice.className='lattice';lattice.setAttribute?.('aria-hidden','true');
  for(let index=0;index<10;index++){const cell=document.createElement('i');if(index<filled)cell.className='on';else if(index===filled&&ratio<1)cell.className='now';lattice.append(cell);}
  const percent=document.createElement('span');percent.textContent=Math.round(ratio*100)+'%';lattice.append(percent);host.append(lattice);return lattice;
}
function resultAmount(result,box){
  const usage=result?.usage;
  if(Number.isFinite(usage?.total_tokens))return ui('receipt.tokens','{{count}} tokens',{count:usage.total_tokens});
  if(Number.isFinite(usage?.prompt_tokens))return ui('receipt.tokens','{{count}} tokens',{count:usage.prompt_tokens});
  let bytes=Number(box?._bytes)||0;
  for(const item of Array.isArray(result?.data)?result.data:[])if(typeof item?.b64_json==='string')bytes+=Math.floor(item.b64_json.length*3/4);
  if(bytes)return i18n.formatBytes(bytes);
  if(typeof result?.text==='string')return ui('receipt.characters','{{count}} characters',{count:result.text.length});
  return null;
}
function appendReceipt(box,{model,started=null,seconds=null,usage=null,chunks=null,text=null,result=null,decision=null,failed=false,at=null}={}){
  // one footer per result under a dashed tear line: tags, then provenance or model and route, time, size, cost and the clock
  if(!box||box._receipt)return null;
  const entry=catalog.find(row=>row.id===model)||null,served=box._provenance||null,external=Boolean(entry?.external)||/^(cloud|router):/.test(String(served?.servedBy||''));
  const foot=document.createElement('footer');foot.className='receipt';box._receipt=foot;
  const tag=(cls,label)=>{const node=document.createElement('span');node.className='tag '+cls;setContent(node,label);foot.append(node);};
  tag(failed?'failed':'recorded',failed?ui('receipt.failed','Failed'):ui('receipt.recorded','Recorded result'));
  if(external)tag('external',ui('receipt.external','External'));
  const parts=document.createElement('span'),inner=document.createElement('span');parts.className='parts';parts.append(inner);foot.append(parts);
  const part=(value,cls='')=>{if(value===null||value===undefined||String(value)==='')return null;const node=document.createElement('span');if(cls)node.className=cls;setContent(node,value);inner.append(node);return node;};
  if(served?.line){part(textBindings.get(served.line)||served.line.textContent,'provenance');served.line.remove?.();}
  if(!served){part(model);part(entry?.external?upstreamLabel(entry.external.upstream):entry?.configured?.device&&entry.configured.device!=='auto'?entry.configured.device:entry?ui('readout.route_local','This computer'):null);}
  const elapsed=Number.isFinite(seconds)?seconds:started!==null?(nowMs()-started)/1000:null;part(secondsText(elapsed));
  if(Number.isFinite(usage?.completion_tokens))part(ui('receipt.tokens','{{count}} tokens',{count:usage.completion_tokens}));
  else if(Number.isFinite(chunks)&&chunks>0)part(ui('receipt.chunks','{{count}} chunks',{count:chunks}));
  if(typeof text==='string')part(i18n.formatBytes(typeof TextEncoder==='function'?new TextEncoder().encode(text).length:text.length));
  else part(resultAmount(result,box));
  part(ui('receipt.recorded_at','Recorded {{time}}',{time:clockTime(Number.isFinite(Number(at))&&Number(at)>0?Number(at):Date.now()/1000)}));
  box.classList?.remove('streaming');box.classList?.add(failed?'failed':'done');if(external)box.classList?.add('external');box.append(foot);
  receiptCost(inner,decision||served?.decision||null).catch(()=>{});renderCabinet();return foot;
}
async function receiptCost(host,decision){
  // a priced cloud call shows its cost; the decision log is read only by a principal that may read it
  if(!decision||typeof brokerAllows!=='function'||!brokerAllows('broker:read'))return;
  const rows=(await(await api('/api/broker/decisions?limit=50')).json())?.data||[];const row=rows.find(item=>item?.decision_id===decision);
  if(row?.cost_usd===null||row?.cost_usd===undefined)return;const node=document.createElement('span');node.className='cost';setContent(node,ui('receipt.cost','${{amount}}',{amount:String(row.cost_usd)}));host.insertBefore?host.insertBefore(node,host.lastChild||null):host.append(node);
}
function refreshExternalLabels(){
  // provider labels arrive with the broker panel, after the rail was drawn: restate them where they are shown
  const find=(node,cls)=>{for(const child of [...(node?.children||[])]){if(String(child.className||'').includes(cls))return child;const found=find(child,cls);if(found)return found;}return null;};
  for(const row of [...($('models')?.children||[])]){const model=catalog.find(item=>item.id===row.dataset?.modelId);const badge=model?.external?find(row,'model-badge'):null;if(badge)badge.replaceWith?.(externalBadge(model));}
  for(const option of [...($('model')?.options||[])]){const model=catalog.find(item=>item.id===option.value);if(model?.external)setContent(option,ui("ui.model_title_cloud","{{model}} · {{task}} · {{provider}}",{model:model.id,task:taskLabel(model.task),provider:upstreamLabel(model.external.upstream)}));}
  renderModelReadout(chosen());
}
let railFrame=0,railAlignNext=false;
function railSoon(align=true){
  // one rail pass per frame; a pass that follows a scroll in progress never moves the list under the pointer
  if(align!==false)railAlignNext=true;if(railFrame)return;
  const later=typeof globalThis.requestAnimationFrame==='function'?globalThis.requestAnimationFrame.bind(globalThis):fn=>setTimeout(fn,16);
  railFrame=later(()=>{railFrame=0;const align=railAlignNext;railAlignNext=false;railMore(align);});
}
function railMore(align=true){
  // The rail shows whole rows only. Its scroller is cut back to the last row that fits (never through a row's text) and
  // ends on a cue counting the rows below that edge (nothing when the list fits); once a scroll settles, a row sliced by
  // the leading edge is brought fully into view (or passed when most of it is already out). The room comes from the window
  // around the scroller, which the cut never changes, so a pass is stable. A pass runs on render, scroll, scroll end,
  // resize, a disclosure toggled in the rail (a row's Details, Register a local path, Connection) and any size change of
  // the window or the list. The desktop rail scrolls down; the strip at 900px and below scrolls sideways.
  const cue=$('rail-more'),list=$('models'),body=list?.parentElement,frame=body?.parentElement;
  if(!cue||!frame||typeof body.getBoundingClientRect!=='function'||typeof frame.getBoundingClientRect!=='function'||typeof globalThis.getComputedStyle!=='function')return;
  if(!body._railCue){body._railCue=true;
    body.addEventListener?.('scroll',()=>railSoon(false),{passive:true});body.addEventListener?.('scrollend',()=>railSoon(true),{passive:true});
    globalThis.addEventListener?.('resize',()=>railSoon(true),{passive:true});
    (frame.closest?.('aside')||frame).addEventListener?.('toggle',()=>railSoon(true),true);
    if(typeof globalThis.ResizeObserver==='function'){const watch=new globalThis.ResizeObserver(()=>railSoon(true));watch.observe(frame);watch.observe(list);}
  }
  const px=value=>parseFloat(value)||0,style=getComputedStyle(body),frameStyle=getComputedStyle(frame),cueStyle=getComputedStyle(cue);
  const across=style.overflowY==='hidden'&&style.overflowX!=='hidden';
  const [lead,trail,limit,other,scroll,client]=across?['left','right','maxWidth','maxHeight','scrollLeft','clientWidth']:['top','bottom','maxHeight','maxWidth','scrollTop','clientHeight'];
  body.style[other]='';
  const borders=across?px(style.borderLeftWidth)+px(style.borderRightWidth):px(style.borderTopWidth)+px(style.borderBottomWidth);
  const pad=across?px(style.paddingRight):px(style.paddingBottom);
  // the border box the frame offers the scroller, and the line the cue takes below it
  const room=across?frame.clientWidth-px(frameStyle.paddingLeft)-px(frameStyle.paddingRight)-px(style.marginLeft)-px(style.marginRight)
    :frame.clientHeight-px(frameStyle.paddingTop)-px(frameStyle.paddingBottom)-px(style.marginTop)-px(style.marginBottom);
  const cueLine=across?0:px(cueStyle.lineHeight)+px(cueStyle.paddingTop)+px(cueStyle.paddingBottom)+px(cueStyle.marginTop)+px(cueStyle.marginBottom);
  const box=body.getBoundingClientRect(),inner=box[lead]+(across?body.clientLeft:body.clientTop);
  const spans=[...(list.children||[])].filter(row=>typeof row.getBoundingClientRect==='function').map(row=>{const r=row.getBoundingClientRect();return {row,a:r[lead]-inner,b:r[trail]-inner};});
  const extent=across?body.scrollWidth:body.scrollHeight;
  if(align){
    const first=spans.findIndex(({b})=>b>0.5),sliced=spans[first];
    if(sliced&&sliced.a<-0.5){
      const next=spans[first+1],delta=!next||sliced.b>(sliced.b-sliced.a)/2?sliced.a:next.a,want=Math.max(0,body[scroll]+delta);
      // near the end the window shrinks so the aligned position can be reached (the rows from there to the end fill it)
      if(want>extent-body[client]+0.5)body.style[limit]=Math.floor(extent-want+borders)+'px';
      body[scroll]=want;return railMore(false);
    }
  }
  // a list longer than the room keeps the cue's line reserved (also at its end), so the cut never flips back and forth
  const scrolls=extent>room-borders+0.5,space=room-borders-(scrolls?cueLine:0),last=spans[spans.length-1];
  let end=0,endsList=false;
  for(const span of spans){if(span.b<=0.5||span.a>=space)continue;if(span.b<=space+0.5){if(span.b>end){end=span.b;endsList=span===last;}}}
  body.style[limit]=scrolls&&end>0?Math.round(Math.min(space,end+(endsList?pad:0))+borders)+'px':'';
  const edge=scrolls&&end>0?end:space,below=spans.filter(({row,b})=>row.dataset?.modelId&&b>edge+0.5).length;
  setContent(cue,below&&!across?ui('rail.more','{{count}} more below',{count:below}):'');
}
function cabinetConnection(ok){
  cabinetConnected=ok;const light=$('connection-light');if(light)light.className='connection-light cell-glyph '+(ok?'connected':'offline');
  const host=$('connection-host');if(host)host.textContent=ok?String(globalThis.location?.host||''):'';renderCabinet();
}
function cabinetSetupPanel(){
  // "Add a model" is open on the first catalog load only when no local model is registered; after that it follows the user
  // (a download that completes inside it never closes it under them)
  const panel=$('model-setup');if(!panel||setupToggled||setupSettled)return;setupSettled=true;const wanted=!catalog.some(model=>!model.external);
  if(panel.open!==wanted)panel.open=wanted;
}
function bootLine(key,value,cls=''){
  const line=document.createElement('p');line.className='boot-line';const name=document.createElement('span');name.className='boot-key';setContent(name,key);
  const lead=document.createElement('span');lead.className='boot-lead';const text=document.createElement('span');text.className='boot-value'+(cls?' '+cls:'');setContent(text,value);line.append(name,lead,text);return line;
}
function renderCabinet(){
  if(typeof document==='undefined'||!$('boot'))return;
  const local=catalog.filter(model=>!model.external),cloud=catalog.filter(model=>model.external),loaded=local.filter(model=>model.status==='loaded'),model=chosen(),jobs=jobsSummary||{running:0,queued:0,done:0,failed:0};
  const ordered=[...local,...cloud],position=model?ordered.findIndex(row=>row.id===model.id)+1:0;
  readoutParts($('rail-readout'),[{text:ui('rail.local','{{count}} local',{count:local.length})},loaded.length?{text:ui('rail.loaded','{{count}} loaded',{count:loaded.length}),cls:'lit'}:null,cloud.length?{text:ui('rail.cloud','{{count}} cloud',{count:cloud.length}),cls:'ext'}:null,position?{text:ui('rail.position','{{index}} of {{count}}',{index:position,count:ordered.length}),cls:'rail-pos'}:null]);
  // the composer head stays empty: the readout grid below the select and the step-3 line already say what is chosen
  readoutParts($('composer-readout'),[]);
  const cards=[...($('results')?.children||[])].filter(node=>!String(node.className||'').includes('result-you')).length;
  readoutParts($('results-readout'),cards?[{text:uiCount('results.count','{{count}} results','{{count}} result',{count:cards}),cls:'count'}]:[]);
  readoutParts($('jobs-readout'),jobsSummary?[jobs.running?{text:ui('jobs.running_count','{{count}} running',{count:jobs.running}),cls:'count'}:null,jobs.queued?{text:ui('jobs.queued_count','{{count}} queued',{count:jobs.queued})}:null,jobs.done?{text:ui('jobs.done_count','{{count}} done',{count:jobs.done}),cls:'lit'}:null,jobs.failed?{text:ui('jobs.failed_count','{{count}} failed',{count:jobs.failed}),cls:'bad'}:null,jobs.cancelled?{text:ui('jobs.cancelled_count','{{count}} cancelled',{count:jobs.cancelled})}:null]:[]);
  railMore();
  const boot=$('boot');boot.replaceChildren?.();
  const lines=document.createElement('div');lines.className='boot-lines';
  const title=document.createElement('p');title.className='boot-title'+(cabinetConnected?'':' waiting');
  setContent(title,cabinetConnected?ui('boot.ready','> Titanos_ready'):cabinetConnected===false?ui('boot.offline','> Connection needed'):ui('boot.waiting','> Booting'));
  const summary=document.createElement('span');summary.className='boot-summary';
  setContent(summary,uiJoin([ui('rail.local','{{count}} local',{count:local.length}),' · ',ui('rail.loaded','{{count}} loaded',{count:loaded.length}),cloud.length?' · ':'',cloud.length?ui('rail.cloud','{{count}} cloud',{count:cloud.length}):'']));
  const broker=serverInfo?.broker&&typeof serverInfo.broker==='object'?serverInfo.broker:null;
  lines.append(title,
    bootLine(ui('boot.engine','Engine'),cabinetConnected?(serverInfo?.version?ui('boot.engine_ok','OK · v{{version}}',{version:serverInfo.version}):ui('boot.ok','OK')):'…',cabinetConnected?'ok':''),
    bootLine(ui('boot.models','Models'),uiJoin([ui('rail.local','{{count}} local',{count:local.length}),' · ',ui('rail.loaded','{{count}} loaded',{count:loaded.length})])),
    bootLine(ui('boot.cloud','Cloud'),broker?.configured?uiJoin([ui('rail.cloud','{{count}} cloud',{count:cloud.length}),' · ',ui('boot.policy','policy {{policy}}',{policy:broker.cloud_policy_effective||'off'})]):ui('boot.cloud_off','Off · nothing leaves this computer'),cloud.length?'ext':''),
    bootLine(ui('boot.jobs','Jobs'),uiJoin([ui('jobs.running_count','{{count}} running',{count:jobs.running}),' · ',ui('jobs.done_count','{{count}} done',{count:jobs.done})])),
    bootLine(ui('boot.selected','Selected'),model?model.id:ui('ui.select_model','No model selected.'),model?.external?'ext':''));
  title.append(summary);
  const map=document.createElement('div');map.className='hive-map';map.setAttribute?.('role','img');
  setUiAttribute(map,'aria-label',ui('boot.map_label','Model map: {{loaded}} loaded of {{count}} local, {{cloud}} cloud',{loaded:loaded.length,count:local.length,cloud:cloud.length}));
  // one cell per model, nothing padded: every cell on the map is a real model
  for(const item of ordered.slice(0,20)){const cell=document.createElement('span');
    cell.className=item.external?'cell-external':item.status==='loaded'?'cell-loaded':item.id===model?.id?'cell-selected':'cell-idle';cell.title=item.id;map.append(cell);}
  const legend=document.createElement('p');legend.className='hive-legend';setContent(legend,ui('boot.legend','Cube loaded · filled selected · orbit cloud · hollow idle'));
  const mapBox=document.createElement('div');mapBox.className='boot-map';mapBox.append(map,legend);
  // the empty state: a sentence, then the empty cells below it (the lattice never sits behind text)
  const empty=document.createElement('div');empty.className='results-empty';const note=document.createElement('p');note.className='results-empty-text';
  setContent(note,local.length||cloud.length?ui('results.empty','Nothing recorded yet. Each result lands here with a receipt: model, route, time and size.'):ui('results.empty_first','No cells filled yet. Add a model in 02, or connect a provider in 07.'));
  const field=document.createElement('span');field.className='lattice-field';field.setAttribute?.('aria-hidden','true');empty.append(note,field);
  boot.append(...(ordered.length?[lines,mapBox,empty]:[lines,empty]));
}
// One owner holds the form through response handling, job cancellation and refresh.
async function cancelJob(request){
  if(request.job&&!request.cancellation)request.cancellation=api('/api/jobs/'+request.job,{method:'DELETE'}).then(()=>refreshJobs());
  return request.cancellation;
}
$('generate').onsubmit=async event=>{
  event.preventDefault();if(active)return;const m=chosen();if(!m||((refreshing||refreshFailed)&&needsCapabilities(m)))return;if(controlIssue(m)){status(controlIssue(m),true);return;}if(taskIssue(m)){status(taskIssue(m),true);return;}
  const request={controller:new AbortController(),model:m,job:null,cancellation:null,started:nowMs()};active=request;lastProvenance=null;
  const signal=request.controller.signal;busy();status(ui("ui.generating","Generating…"));
  // Snapshot inputs before file reads or network waits; edits become the next draft.
  const prompt=$('prompt').value,file=$('file').files[0],conversation=[...history],nativeOCR=isTesseract(m),instruction=isInstruction(m);
  const source=generationOnly(m)?null:$('image-source').files[0],mask=generationOnly(m)?null:$('image-mask').files[0],preserve=$('preserve-unmasked').value;
  const values=controls(['max_tokens','temperature','top_p','top_k','min_p','repeat_penalty','seed','size','steps','frames','fps','voice','max_input_tokens','min_score','min_margin','language','threshold','nms','nms_iou','guidance_scale','image_guidance_scale','n','top_n','vad-sample-rate','vad-threshold','vad-min-speech','vad-min-silence','vad-speech-pad','segment-threshold']);
  const take=names=>Object.fromEntries(names.filter(name=>Object.hasOwn(values,name)).map(name=>[name,values[name]]));
  const pair={premise:$('premise').value,hypothesis:$('hypothesis').value},truncate=$('truncate').value,gate=$('vad-gate').value;
  const ocr={language:$('ocr_language').value,psm:$('ocr_psm').value,geometry:$('ocr_geometry').value};
  const rerank={documents:$('rerank-documents').value,truncate:$('rerank-truncate').value,returnDocuments:$('rerank-return-documents').value};
  const vad={returnFrames:$('vad-return-frames').value};
  const segment={mode:$('segment-mode').value,box:['segment-x0','segment-y0','segment-x1','segment-y1'].map(id=>$(id).value),points:$('segment-points').value,labels:$('segment-point-labels').value,multimask:$('segment-multimask').value};
  const audio={action:$('audio-action').value,translationModel:$('translation-model').value,sourceLanguage:$('source-language').value,targetLanguage:$('target-language').value,
    diarizationModel:$('diarization-model').value,textModel:$('audio-text-model').value,
    options:Object.fromEntries([['diarization-count','num_speakers'],['diarization-threshold','cluster_threshold'],['diarization-on','min_duration_on'],['diarization-off','min_duration_off']].filter(([id])=>$(id).value!=='').map(([id,name])=>[name,Number($(id).value)])),timeout:$('audio-timeout').value};
  const music={mode:$('music-mode').value,lyrics:$('music-lyrics').value,duration:$('music-duration').value,
    vocalLanguage:$('music-language').value,bpm:$('music-bpm').value,keyScale:$('music-key').value,reference:$('music-reference').files[0]};
  const show=async(result,decision=null)=>{signal.throwIfAborted();if(active===request){const box=await showResult(result,ui("ui.result_title","{{task}} · {{model}}",{task:taskLabel(m.task),model:m.id}),signal);if(typeof appendProvenance==='function')appendProvenance(box);appendReceipt(box,{model:m.id,started:request.started,result,decision});}};
  try{
    if(instruction&&!source)throw Error(ui("ui.choose_source","Choose a source image for this instruction edit"));
    if(m.task==='image'&&!instruction&&mask&&!source)throw Error(ui("ui.mask_needs_source","Choose a source image before adding a mask"));
    if(['chat','vision'].includes(m.task)){
      let content=prompt;if(m.task==='vision'&&file)content=[{type:'text',text:prompt},{type:'image_url',image_url:{url:await fileData(file)}}];signal.throwIfAborted();
      card(ui("ui.you","You · {{model}}",{model:m.id}),prompt).box.classList?.add('done');const messages=[...conversation,{role:'user',content}];
      const chat={model:m.id,messages,...take(['max_tokens','temperature','top_p','top_k','min_p','repeat_penalty','seed'])};
      if(m.task==='chat'&&$('require-loaded')?.checked&&serverInfo?.inference_contracts?.require_loaded?.revision===1)chat.require_loaded=true;
      const answer=await streamChat(chat,signal);
      signal.throwIfAborted();if(active===request)history=[...messages,{role:'assistant',content:answer}];
    }else if(m.task==='image'&&source){
      const data=new FormData();data.set('model',m.id);data.set('prompt',prompt);data.set('image',source);
      for(const [name,value] of Object.entries(take(instruction?['steps','seed','n','guidance_scale','image_guidance_scale']:['size','steps','seed'])))data.set(name,String(value));
      if(mask&&!instruction){if(!['true','false'].includes(preserve))throw Error(ui("ui.choose_preservation","Choose whether to keep pixels outside the mask"));data.set('mask',mask);data.set('preserve_unmasked',preserve);}
      await show(await(await api('/v1/images/edits',{method:'POST',body:data,signal})).json());
    }else if(['image','speech','video','music'].includes(m.task)){
      const data={model:m.id,task:m.task,...(m.task==='speech'?{input:prompt,...take(['voice'])}:m.task==='music'?
        {prompt,mode:music.mode,...(music.lyrics?{lyrics:music.lyrics}:{}),...(music.duration?{duration:Number(music.duration)}:{}),
         ...(music.vocalLanguage?{vocal_language:music.vocalLanguage}:{}),...(music.bpm?{bpm:Number(music.bpm)}:{}),
         ...(music.keyScale?{key_scale:music.keyScale}:{}),...take(['seed','guidance_scale','steps'].filter(name=>m.capabilities?.music?.controls?.[name]===true))}:
        {prompt,...take(['size','steps','seed',...(m.task==='video'?['frames','fps']:[])])})};
      if(m.task==='video'&&file)data.input_reference=await fileData(file);signal.throwIfAborted();
      if(m.task==='music'&&music.reference)data.reference_audio_base64=await fileData(music.reference);signal.throwIfAborted();
      // Retain the returned job ID even if Stop arrives during creation, then cancel it.
      const jobHeaders=m.external?{'Idempotency-Key':globalThis.crypto?.randomUUID?.()||('studio-'+Date.now()+'-'+Math.random().toString(16).slice(2))}:undefined;
      const j=await(await api('/api/jobs',{method:'POST',json:data,...(jobHeaders?{headers:jobHeaders}:{})})).json();request.job=j.id;
      if(signal.aborted){await cancelJob(request);signal.throwIfAborted();}
      status(ui("ui.job_queued","Queued. The job is listed under Jobs and keeps running if you leave this page."));await refreshJobs();
      while(true){signal.throwIfAborted();const state=await(await api('/api/jobs/'+j.id,{signal})).json();if(state.status==='completed'){await show(state.result,state.decision_id||null);break;}if(['failed','cancelled','interrupted'].includes(state.status))throw Error(state.error||state.status);if(m.task==='music'&&state.native_generation?.stage)status(`${tr('status.running')} · ${stageLabel(state.native_generation.stage)}`);else if(['running','cancelling'].includes(state.status))status(tr('status.'+state.status));await new Promise(resolve=>setTimeout(resolve,1000));}
      await refreshJobs();
    }else if(m.task==='rerank'){
      const documents=rerank.documents.split(/\r?\n/).filter(line=>line.trim());
      if(!prompt.trim())throw Error(ui("ui.enter_query","Enter a reranking query"));
      if(!documents.length)throw Error(ui("ui.enter_documents","Enter at least one nonempty document line"));
      const data={model:m.id,query:prompt,documents,...take(['top_n','max_input_tokens'])};
      if(!['','true','false'].includes(rerank.truncate))throw Error(ui("ui.choose_rerank_truncation","Choose a supported rerank truncation policy"));
      if(rerank.truncate!=='')data.truncate=rerank.truncate==='true';
      if(!['','true','false'].includes(rerank.returnDocuments))throw Error(ui("ui.choose_return_documents","Choose whether rerank results include documents"));
      if(rerank.returnDocuments!=='')data.return_documents=rerank.returnDocuments==='true';
      await show(await(await api('/v1/rerank',{method:'POST',json:data,signal})).json());
    }else if(m.task==='diarization'||m.task==='transcription'&&audio.action==='analyze'){
      if(!file)throw Error(ui('ui.choose_audio','Choose an audio file'));
      if(m.task==='transcription'&&!audio.diarizationModel)throw Error(ui('audio.choose_speakers','Choose a speaker model'));
      const data={model:m.id,audio_base64:await fileData(file)};signal.throwIfAborted();
      if(audio.timeout!=='')data.timeout=Number(audio.timeout);
      if(m.task==='diarization')Object.assign(data,audio.options);
      else{
        Object.assign(data,{diarization_model:audio.diarizationModel,diarization_options:audio.options,...take(['language','max_tokens'])});
        if(audio.textModel)data.text_analysis_model=audio.textModel;
        if(gate==='off')data.vad_enabled=false;else if(gate.startsWith('model:')){data.vad_model=gate.slice(6);data.vad_enabled=true;}
      }
      await show(await(await api(m.task==='diarization'?'/v1/audio/diarizations':'/v1/audio/analysis',{method:'POST',json:data,signal})).json());
    }else if(m.task==='vad'){
      if(!file)throw Error(ui("ui.choose_audio","Choose an audio file"));
      const data={model:m.id,audio_base64:await fileData(file)};signal.throwIfAborted();
      for(const [source,target] of [['vad-sample-rate','sample_rate'],['vad-threshold','threshold'],['vad-min-speech','min_speech_ms'],['vad-min-silence','min_silence_ms'],['vad-speech-pad','speech_pad_ms']])if(Object.hasOwn(values,source))data[target]=values[source];
      if(!['','true','false'].includes(vad.returnFrames))throw Error(ui("ui.choose_vad_probabilities","Choose whether VAD returns frame probabilities"));
      if(vad.returnFrames!=='')data.return_frames=vad.returnFrames==='true';
      await show(await(await api('/v1/audio/vad',{method:'POST',json:data,signal})).json());
    }else if(m.task==='pair-classification'){
      if(!pair.premise.trim()||!pair.hypothesis.trim())throw Error(ui("ui.enter_pair","Enter both the premise and hypothesis"));
      const data={model:m.id,pairs:[pair],...take(['max_input_tokens'])};
      if(Object.hasOwn(data,'max_input_tokens')&&(!Number.isSafeInteger(data.max_input_tokens)||data.max_input_tokens<1))throw Error(ui("ui.positive_token_limit","Input token limit must be a positive safe integer"));
      if(!['','true','false'].includes(truncate))throw Error(ui("ui.choose_pair_truncation","Choose a supported pair truncation policy"));
      if(truncate!=='')data.truncate=truncate==='true';
      await show(await(await api('/v1/classifications/text-pairs',{method:'POST',json:data,signal})).json());
    }else if(nativeOCR){
      if(!file)throw Error(ui("ui.choose_image","Choose an image"));
      const data={model:m.id,image:await fileData(file)};signal.throwIfAborted();
      if(ocr.language!==''){if(!/^[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*$/.test(ocr.language))throw Error(ui("ui.ocr_language_invalid","Use a Tesseract language code such as eng or eng+deu"));data.language=ocr.language;}
      if(ocr.psm!==''){if(!['3','4','5','6','7','8','9','10','11','13'].includes(ocr.psm))throw Error(ui("ui.choose_ocr_page","Choose a supported OCR page mode"));data.psm=Number(ocr.psm);}
      if(ocr.geometry!==''){if(!['true','false'].includes(ocr.geometry))throw Error(ui("ui.choose_ocr_geometry","Choose a supported OCR geometry option"));data.include_geometry=ocr.geometry==='true';}
      await show(await(await api('/v1/images/ocr',{method:'POST',json:data,signal})).json());
    }else if(m.task==='sentiment'){
      await show(await(await api('/v1/sentiment',{method:'POST',json:{model:m.id,text:prompt,...take(['max_input_tokens','min_score','min_margin'])},signal})).json());
    }else if(m.task==='embeddings'){
      await show(await(await api('/v1/embeddings',{method:'POST',json:{model:m.id,input:prompt},signal})).json());
    }else if(m.task==='transcription'){
      if(!file)throw Error(ui("ui.choose_audio","Choose an audio file"));const data=new FormData();data.set('model',m.id);data.set('file',file);if(values.language)data.set('language',values.language);
      if(gate==='off')data.set('vad_enabled','false');else if(gate.startsWith('model:')){data.set('vad_model',gate.slice(6));data.set('vad_enabled','true');}data.set('response_format','verbose_json');
      let route='/v1/audio/transcriptions';
      if(audio.action==='translate'){
        route='/v1/audio/translations';if(audio.targetLanguage.trim())data.set('target_language',audio.targetLanguage);
        if(audio.translationModel){if(!audio.sourceLanguage.trim())throw Error(ui("ui.choose_nmt_source","Choose the NMT source language ID for the translation model"));data.set('translation_model',audio.translationModel);data.set('source_language',audio.sourceLanguage);}
      }
      await show(await(await api(route,{method:'POST',body:data,signal})).json());
    }else if(m.task==='image-segmentation'){
      if(!file)throw Error(ui("ui.choose_image","Choose an image"));const data={model:m.id};
      if(segment.mode==='box'){
        if(segment.box.some(value=>value===''))throw Error(ui("ui.enter_box","Enter all four box coordinates"));
        data.box=segment.box.map(Number);if(data.box.some(value=>!Number.isFinite(value)))throw Error(ui("ui.box_finite","Box coordinates must be finite numbers"));
        if(data.box.some(value=>value<0)||data.box[0]>=data.box[2]||data.box[1]>=data.box[3])throw Error(ui("ui.box_order","Segmentation box must be nonnegative and ordered with x0 < x1 and y0 < y1"));
      }else if(segment.mode==='points'){
        try{data.points=JSON.parse(segment.points);data.point_labels=JSON.parse(segment.labels);}catch{throw Error(ui("ui.points_valid_json","Points and point labels must be valid JSON arrays"));}
        const pointLimit=serverInfo?.settings?.segmentation_max_points?.value;
        if(!Array.isArray(data.points)||!data.points.length||(pointLimit!==undefined&&data.points.length>pointLimit)||!data.points.every(point=>Array.isArray(point)&&point.length===2&&point.every(value=>Number.isFinite(value)&&value>=0))||!Array.isArray(data.point_labels)||data.point_labels.length!==data.points.length||!data.point_labels.every(value=>value===0||value===1))throw Error(ui("ui.point_rules","Points require nonnegative finite [x,y] pairs and one 0/1 label per point{{limit}}",{limit:pointLimit!==undefined?ui("ui.point_limit","; at most {{count}} points are configured",{count:pointLimit}):''}));
      }else throw Error(ui("ui.choose_segmentation","Choose a segmentation prompt type"));
      if(!['','true','false'].includes(segment.multimask))throw Error(ui("ui.choose_multimask","Choose a supported segmentation alternatives option"));
      if(segment.multimask!=='')data.multimask=segment.multimask==='true';
      if(Object.hasOwn(values,'segment-threshold'))data.mask_threshold=values['segment-threshold'];
      data.image=await fileData(file);signal.throwIfAborted();
      await show(await(await api('/v1/images/segment',{method:'POST',json:data,signal})).json());
    }else{
      const data={model:m.id};let route;
      if(m.task==='translation'){route='/v1/translations';Object.assign(data,{text:prompt,target_language:values.language||'',...take(['max_tokens','temperature','top_p'])});}
      else if(['image-to-text','object-detection','depth-estimation','image-segmentation'].includes(m.task)){if(!file)throw Error(ui("ui.choose_image","Choose an image"));data.image=await fileData(file);signal.throwIfAborted();route={'image-to-text':'/v1/images/ocr','object-detection':'/v1/images/detect','depth-estimation':'/v1/images/depth','image-segmentation':'/v1/images/segment'}[m.task];Object.assign(data,take(m.task==='image-to-text'?['max_tokens','temperature','top_p']:m.task==='object-detection'?['threshold','nms_iou']:[]));if(m.task==='object-detection'&&values.nms!==undefined)data.nms=values.nms==='true';}
      else throw Error(ui("ui.no_request_adapter","Studio has no request adapter for {{task}}",{task:m.task}));
      await show(await(await api(route,{method:'POST',json:data,signal})).json());
    }
    signal.throwIfAborted();if(active===request)status(tr('status.done'),false,'success');
  }catch(e){if(active===request)status(e.name==='AbortError'?ui("ui.stopped","Stopped."):e.message,e.name!=='AbortError');}
  finally{
    if(active===request){
      if(request.cancellation)try{await request.cancellation;}catch(e){status(ui("ui.cancel_failed","Job cancellation failed: ")+e.message,true);}
      await refresh();
      if(active===request){
        active=null;const unchanged=catalog.some(model=>model.id===m.id&&model.task===m.task);
        for(const option of [...$('model').options])if(!catalog.some(model=>model.id===option.value))option.remove();
        if(!catalog.some(model=>model.id===$('model').value))$('model').value=catalog[0]?.id||'';
        select(!unchanged);busy();
      }
    }
  }
};
$('stop').onclick=async()=>{const request=active;if(!request)return;request.controller.abort();try{await cancelJob(request);}catch(e){if(active===request)status(ui("ui.cancel_failed","Job cancellation failed: ")+e.message,true);}};
$('clear-image-inputs').onclick=()=>{$('image-source').value='';if(!isInstruction())$('image-mask').value='';};
const registrationTaskOrder=['chat','vision','embeddings','transcription','speech','image','video','music','tensor','translation','image-to-text','object-detection','sentiment','rerank','vad','diarization','pair-classification','depth-estimation','image-segmentation'];
const registrationPolicies={
  transformers:{tasks:['chat','vision','embeddings','transcription','speech','translation','image-to-text','object-detection','sentiment','rerank','pair-classification','depth-estimation','image-segmentation'],inventory:'pytorch',dtype:true},
  diffusers:{tasks:['image','video'],inventory:'pytorch',dtype:true,variant:true},
  'qwen3-tts':{tasks:['speech'],inventory:'pytorch',dtype:true,voices:true},
  'sherpa-onnx':{tasks:['speech','diarization'],inventory:'system',cpu:true},
  'ace-step':{tasks:['music'],inventory:'pytorch'},
  onnx:{tasks:['embeddings','tensor','vad'],inventory:'onnx',graph:true},
  openvino:{tasks:['embeddings','tensor'],inventory:'openvino',graph:true},
  'openvino-genai':{tasks:['chat','image'],inventory:'openvino'},
  tesseract:{tasks:['image-to-text'],inventory:'system',cpu:true},
  'onnx-diffusers':{tasks:['image'],inventory:'onnx',onnxImage:true},
  'llama-cpp':{tasks:['chat','vision'],cliOnly:true,help:ui("ui.native_registration_help","llama-cpp needs native GGUF options. Register it with titanos add --backend llama-cpp.")},
  xtts:{tasks:['speech'],cliOnly:true,help:ui("ui.xtts_registration_help","XTTS requires explicit trusted pickle permission. Register it with titanos add --backend xtts --allow-pickle.")}
};
const registrationBackends=['auto',...Object.keys(registrationPolicies)],dmlAdapters=new Map();
function option(value,label=value){const node=document.createElement('option');node.value=value;setContent(node,label);return node;}
function registrationField(id,label,tag='input'){
  const box=document.createElement('label'),caption=document.createElement('span'),control=document.createElement(tag);
  setContent(caption,label);control.id=id;box.append(caption,control);$('register').append(box);return control;
}
function setupRegistration(){
  $('register-task').replaceChildren(...registrationTaskOrder.map(value=>option(value)));
  $('register-backend').replaceChildren(...registrationBackends.map(value=>option(value,value==='auto'?ui("ui.default_for_task","Default for task"):value)));
  const dtype=registrationField('register-dtype',ui("ui.dtype","Compute dtype"),'select');dtype.name='dtype';
  dtype.append(...['auto','float32','float16','bfloat16'].map(value=>option(value,value==='auto'?ui("ui.default_runtime","Runtime default"):value)));
  const graph=registrationField('register-graph',ui("ui.graph","Prepared graph filename (relative, optional)"));graph.name='graph';setUiAttribute(graph,"placeholder",ui("ui.graph_example","e.g. model.onnx or model.xml"));
  const voices=registrationField('register-voices',ui("ui.voice_map","Voice map JSON (optional)"),'textarea');voices.name='voices';voices.rows=3;setUiAttribute(voices,"placeholder",'{"voice-name":"D:/Voices/reference.wav"}');
  const adapter=registrationField('register-dml-adapter',ui("ui.dml_adapter","DirectML adapter"),'select');adapter.append(option('',ui("ui.load_devices_first","Load known devices first")));
  const metacommands=registrationField('register-disable-metacommands',ui("ui.dml_metacommands","DirectML metacommands"),'select');metacommands.append(option('',ui("ui.default_runtime","Runtime default")),option('false',ui("ui.enabled_false","Enabled (explicit false)")),option('true',ui("ui.disabled_diagnostic","Disabled (diagnostic only)")));
  const replace=registrationField('register-replace',ui("ui.replace_registration","Replace an existing registration"));replace.name='replace';replace.type='checkbox';replace.value='true';replace.checked=false;
  if(replace.parentElement){replace.parentElement.className='check';replace.parentElement.prepend?.(replace);}
  const devices=document.createElement('datalist');devices.id='register-devices';$('register').append(devices);$('register-device').setAttribute?.('list','register-devices');
  const load=document.createElement('button');load.id='load-devices';load.type='button';load.className='quiet';setContent(load,ui("ui.load_devices","Load known devices"));$('register').append(load);
  const descendants=node=>[node,...[...node.children].flatMap(descendants)];
  const submit=descendants($('register')).find(node=>String(node.tagName).toLowerCase()==='button'&&node!==load)||document.createElement('button');
  if(!submit.textContent)setContent(submit,ui("ui.add_model","Add model"));
  submit.id='register-submit';submit.remove();$('register').append(submit);
  const first=document.createElement('details');first.id='first-run';first.className='panel advanced-only';
  const summary=document.createElement('summary');summary.id='first-run-summary';setContent(summary,tr('firstRun.title'));
  const copy=document.createElement('p');copy.id='first-run-copy';copy.className='hint';setContent(copy,tr('firstRun.copy'));
  first.append(summary,copy);($('register').parentElement?.parentElement||$('register')).append(first);
}
function setupLifecycle(){
  const setup=document.createElement('section');setup.id='model-setup-check';setup.className='panel';setup.hidden=true;
  const form=$('generate');if(form.insertBefore)form.insertBefore(setup,$('step-input')||$('prompt-label'));else form.append(setup);
  const row=$('unload').parentElement||$('generate');
  const button=(id,text)=>{const node=document.createElement('button');node.id=id;node.type='button';node.className='quiet';setContent(node,text);return node;};
  const load=button('load',tr('composer.load')),details=button('model-details',tr('composer.details')),remove=button('remove-registration',tr('composer.remove'));
  details.className+=' advanced-only';remove.className+=' advanced-only';
  const unload=$('unload'),clear=$('clear');unload.remove();clear.remove();row.append(load,unload,details,remove,clear);
  const runtime=document.createElement('details');runtime.id='model-runtime';runtime.className='panel advanced-only';runtime.hidden=true;
  const summary=document.createElement('summary');setContent(summary,ui("ui.observed_runtime","Observed loaded runtime details"));
  const output=document.createElement('pre');output.id='model-runtime-output';runtime.append(summary,output);$('generate').after?$('generate').after(runtime):$('generate').parentElement?.append(runtime);
  const label=document.createElement('label'),check=document.createElement('input');check.id='require-loaded';check.type='checkbox';check.checked=false;
  const caption=document.createElement('span');setContent(caption,ui("ui.require_loaded","Require this chat model to be loaded; do not start it implicitly"));label.className='check';label.append(check,caption);
  const settings=$('max_tokens').parentElement?.parentElement||$('generate');settings.append(label);
}
const assignmentDrafts=new Map(),assignmentSaved=new Map(),assignmentInventories=new Map(),acquisitionRuntimeProfiles=new Map();let assignmentModel=null,assignmentEpoch=0,assignmentDeviceEpoch=0,devicePolicyPending=false,runtimeProfileChoice=null;
function deviceReason(value){
  const code=typeof value==='string'?value:value?.code,keys={visibility_masked:'device.masked',masked:'device.masked',runtime_missing:'device.runtimeMissing',missing_runtime:'device.runtimeMissing',cpu_only:'device.cpuOnly',backend_cpu_only:'device.cpuOnly',weak_identity:'device.weak',identity_unverified:'device.weak',device_unavailable:'device.unavailable',unavailable:'device.unavailable',unsupported_backend:'device.incompatible',incompatible:'device.incompatible',unsupported_task:'device.incompatible',nccl_unavailable:'device.collective',collective_unavailable:'device.collective',unknown_architecture:'device.unknownGroup',heterogeneous_devices:'device.homogeneous'};
  if(code==='visibility_unverified'||code==='runtime_unverified')return tr('device.unverified');
  return tr(keys[code]||'device.incompatible');
}
function unavailableAssignmentDevices(value){return [...new Set([...(value?.missing_devices||[]),...(value?.unavailable_devices||[])])];}
function rowSelectable(row){return row&&row.assignable!==false&&row.compatible!==false&&row.present!==false&&row.visibility?.status!=='masked'&&row.stable!==false;}
function deviceLabel(identity,model=chosen()){
  const row=assignmentInventories.get(model?.id)?.data?.find(item=>item.id===identity);
  return row?uiJoin([row.label||row.id,row.label&&row.label!==row.id?' · '+(preferences.mode==='advanced'?row.id:row.runtime_address||row.id.slice(-8)):'']):identity;
}
function deviceConstraints(inventory){return Object.values(inventory?.visibility?.constraints||{}).some(value=>value!==null&&value!==undefined&&value!=='unset')||(inventory?.data||[]).some(row=>row.visibility?.status==='masked');}
function renderDeviceSummary(model=chosen()){
  const panel=$('device-selection');if(!panel)return;panel.hidden=!model||Boolean(model.external);if(!model||model.external)return;
  const value=assignmentSaved.get(model.id),inventory=assignmentInventories.get(model.id),draft=assignmentDrafts.get(model.id),selection=value?.selection;
  const policy=$('device-policy'),saved=value?.policy||(value?.groups?.length?'pinned':model.configured?.device==='cpu'?'explicit_cpu':'automatic');
  policy.replaceChildren(option('auto',tr('device.auto')),option('cpu',tr('device.cpu')));if(saved==='pinned')policy.append(option('pinned',tr('device.keepPinned')));
  policy.value=saved==='explicit_cpu'?'cpu':saved==='pinned'?'pinned':'auto';policy.disabled=Boolean(active||devicePolicyPending||draft?.dirty||!value);
  let message=tr('device.checking');
  if(value){
    const ids=(value.groups||[]).flat(),id=selection?.device||value.configured_device||model.configured?.device,unavailable=unavailableAssignmentDevices(value);
    message=unavailable.length?ui('device.missingSaved','Saved device is unavailable: {{devices}}. No automatic CPU fallback.',{devices:unavailable.join(', ')}):ids.length?ui('device.saved','Saved devices: {{devices}}',{devices:ids.map(id=>String(deviceLabel(id,model))).join(', ')}):
      ['unavailable','unknown'].includes(selection?.status)?tr('device.selectionUnknown'):
      selection?.status==='recommended'?ui('device.recommended','Automatic recommendation: {{device}}',{device:deviceLabel(id,model)}):
      ui('device.selected','Selected device: {{device}}',{device:id==='auto'?tr('device.auto'):deviceLabel(id||'cpu',model)});
  }
  setContent($('device-selection-text'),message);const observed=value?.observed;if($('device-line'))setContent($('device-line'),uiJoin([tr('device.policy'),' · ',policy.options?.[policy.selectedIndex]?.textContent||tr('device.checking')]));
  setContent($('device-observed-text'),observed&&observed.observed!==false&&observed.device?ui('device.observed','Observed loaded device: {{device}}',{device:observed.label||deviceLabel(observed.device,model)}):tr('device.notObserved'));
  $('device-visibility').hidden=!deviceConstraints(inventory);setContent($('device-visibility'),tr('device.maskNotice'));
  setContent($('device-policy-note'),draft?.dirty?tr('device.unsavedNote'):tr('device.selectionNote'));
}
function assignmentLayout(){
  if($('assignment-ranks').parentElement)$('assignment-ranks').parentElement.hidden=$('assignment-mode').value!=='tensor_parallel';
  if($('assignment-replicas').parentElement)$('assignment-replicas').parentElement.hidden=$('assignment-mode').value!=='single';
}
function modeIssue(mode,model=chosen()){
  if(mode==='single')return null;const support=(assignmentSaved.get(model?.id)?.modes||model?.capabilities?.modes)?.[mode];
  if(!support?.implemented)return tr('device.modeUnsupported');
  if(support.available===false||support.runtime_available===false||support.platform_supported===false)return deviceReason(support.reason||support.reasons?.[0]||'collective_unavailable');
  if(mode==='tensor_parallel'&&support.collectives&&model?.capabilities?.backend!=='llama-cpp'){
    const inventory=assignmentInventories.get(model?.id),selected=[...$('assignment-devices').selectedOptions].map(item=>item.value);
    const ids=selected.length?selected:[assignmentSaved.get(model?.id)?.selection?.device].filter(Boolean);
    for(const id of ids){const row=inventory?.data?.find(item=>item.id===id),family=row?.device_type==='cpu'?'cpu':row?.signature?.runtime_vendor;
      const collective=support.collectives[family];if(collective?.available===false)return deviceReason(collective.reason||'collective_unavailable');}
  }
  return null;
}
function assignmentIssue(model=chosen()){
  const mode=$('assignment-mode').value,problem=modeIssue(mode,model);if(problem)return problem;
  const ids=[...$('assignment-devices').selectedOptions].map(item=>item.value),inventory=assignmentInventories.get(model?.id),replicas=Number($('assignment-replicas').value||1),ranks=Number($('assignment-ranks').value||2);
  if(ids.length&&inventory)for(const id of ids){const row=inventory.data.find(item=>item.id===id);if(!rowSelectable(row))return row?.visibility?.status==='masked'?tr('device.masked'):row?.reasons?.length?deviceReason(row.reasons[0]):tr('device.unavailable');}
  if(!Number.isSafeInteger(replicas)||replicas<1||replicas>16)return tr('device.replicaCount');
  if(mode==='single'){
    if(ids.length>1&&replicas===1)return tr('device.singleCount');
    if(replicas>1&&ids.length!==replicas)return tr('ui.replica_device');
  }else{
    const rows=ids.map(id=>inventory?.data?.find(row=>row.id===id)),cpuRanks=mode==='tensor_parallel'&&ids.length===1&&rows[0]?.device_type==='cpu';
    if(ids.length<2&&!cpuRanks)return tr('device.twoDevices');
    if(!Number.isSafeInteger(ranks)||ranks<2||ranks>16)return tr('device.rankCount');
    if(mode==='tensor_parallel'&&!cpuRanks&&ranks!==ids.length)return tr('device.rankMatch');
    if(inventory&&!cpuRanks){
      if(rows.some(row=>!row?.signature?.runtime_vendor||!row.signature.architecture))return tr('device.unknownGroup');
      if(new Set(rows.map(row=>JSON.stringify(row.signature))).size!==1)return tr('device.homogeneous');
    }
  }
  return null;
}
function renderAssignmentChoices(model=chosen()){
  const draft=assignmentDrafts.get(model?.id),inventory=assignmentInventories.get(model?.id),list=$('assignment-devices');if(!list)return;
  const ids=new Set(draft?.devices||[]);
  if(inventory)list.replaceChildren(...(inventory.data||[]).map(row=>{
    const reason=row.visibility?.status==='masked'?tr('device.masked'):row.reasons?.length?deviceReason(row.reasons[0]):tr('device.incompatible');
    const item=option(row.id,uiJoin([row.label||row.id,' · '+row.id,row.memory_mib?` · ${row.memory_mib} MiB`:'',!rowSelectable(row)?uiJoin([' · ',reason]):'']));item.title=String(item.textContent||'');item.disabled=!rowSelectable(row);return item;
  }));
  for(const id of ids)if(![...list.options].some(item=>item.value===id)){const item=option(id,uiJoin([id,' · ',tr('device.unavailable')]));item.disabled=true;list.append(item);}
  for(const item of list.options)item.selected=ids.has(item.value);
  const native=model?.capabilities?.backend==='llama-cpp';
  for(const item of $('assignment-mode').options){setContent(item,item.value==='tensor_parallel'&&native?tr('device.nativeTensor'):item.value==='model_sharding'&&native?tr('device.nativeShard'):assignmentModeLabel(item.value));item.disabled=Boolean(modeIssue(item.value,model));}
  if(draft)$('assignment-mode').value=draft.mode;
  if($('assignment-mode-limits'))setContent($('assignment-mode-limits'),uiJoin(['model_sharding','tensor_parallel'].filter(mode=>modeIssue(mode,model)).map(mode=>ui('device.modeReason','{{mode}}: {{reason}}',{mode:mode==='tensor_parallel'&&native?tr('device.nativeTensor'):assignmentModeLabel(mode),reason:modeIssue(mode,model)})),' · '));
  if($('assignment-compatibility')){const problem=assignmentIssue(model);setContent($('assignment-compatibility'),problem||tr('device.compatibilityNote'));$('assignment-compatibility').className=problem?'hint warning':'hint';}
  if($('device-inventory-details'))setContent($('device-inventory-details'),JSON.stringify({visibility:inventory?.visibility,probe_errors:inventory?.probe_errors,selection:assignmentSaved.get(model?.id)?.selection,modes:assignmentSaved.get(model?.id)?.modes||model?.capabilities?.modes,devices:inventory?.data},null,2));
  renderDeviceSummary(model);
}
function renderAssignment(value){
  $('assignment-mode').value=value.mode;assignmentLayout();$('assignment-ranks').value=String(value.tensor_parallel_size||2);$('assignment-replicas').value=String(value.replicas||1);renderAssignmentChoices();
  const unavailable=unavailableAssignmentDevices(value);
  setContent($('assignment-status'),value.dirty?tr('advanced.unsaved'):unavailable.length?ui('device.missingSaved','Saved device is unavailable: {{devices}}. No automatic CPU fallback.',{devices:unavailable.join(', ')}):value.automatic?tr('advanced.auto'):uiCount('ui.replica_summary','{{mode}} · {{count}} replicas','{{mode}} · {{count}} replica',{mode:assignmentModeLabel(value.mode),count:value.replicas}));
}
function rememberAssignment(){
  if(!assignmentModel||chosen()?.id!==assignmentModel)return;++assignmentEpoch;
  assignmentDrafts.set(assignmentModel,{mode:$('assignment-mode').value,tensor_parallel_size:Number($('assignment-ranks').value||2),replicas:Number($('assignment-replicas').value||1),devices:[...$('assignment-devices').selectedOptions].map(item=>item.value),dirty:true});
  $('assignment-apply').disabled=false;setContent($('assignment-status'),tr('advanced.unsaved'));renderAssignmentChoices();
}
async function loadAssignmentDevices(refresh=false){
  const model=chosen();if(!model||model.external)return;const id=model.id,epoch=++assignmentDeviceEpoch;
  try{const value=await(await api('/api/assignments/devices?model='+encodeURIComponent(id)+(refresh?'&refresh=true':''))).json();
    if(epoch!==assignmentDeviceEpoch||chosen()?.id!==id)return;if(!Array.isArray(value.data))throw Error(String(tr('device.invalidInventory')));
    assignmentInventories.set(id,value);renderAssignmentChoices();
  }catch(error){if(epoch===assignmentDeviceEpoch&&chosen()?.id===id)setContent($('assignment-status'),error.message);}
}
async function loadAssignment(){
  const model=chosen();if(!model||model.external)return;const id=model.id,draft=assignmentDrafts.get(id);
  if(assignmentModel!==id){assignmentModel=id;++assignmentEpoch;renderAssignment(draft||{mode:'single',devices:[],automatic:true});}
  if(!assignmentInventories.has(id))loadAssignmentDevices().catch(()=>{});
  if(draft?.dirty){renderAssignment(draft);$('assignment-apply').disabled=false;return;}
  const epoch=++assignmentEpoch;$('assignment-apply').disabled=!draft;
  const current=()=>epoch===assignmentEpoch&&chosen()?.id===id&&assignmentModel===id&&!assignmentDrafts.get(id)?.dirty;
  try{const value=await(await api('/api/assignments/'+encodeURIComponent(id))).json();if(!current())return;
    if(value.model!==id)throw Error(ui('ui.wrong_assignment','Server returned an assignment for a different model.'));
    assignmentSaved.set(id,value);const saved={...value,devices:(value.groups||[]).flat(),dirty:false};assignmentDrafts.set(id,saved);renderAssignment(saved);$('assignment-apply').disabled=false;
  }catch(error){if(current())setContent($('assignment-status'),error.message);}
}
async function saveDevicePolicy(){
  const model=chosen(),policy=$('device-policy').value;if(active||devicePolicyPending||!model||policy==='pinned')return;
  if(assignmentDrafts.get(model.id)?.dirty){renderDeviceSummary();return;}devicePolicyPending=true;busy();renderDeviceSummary();
  try{const result=await(await api('/api/assignments',{method:'POST',json:{model:model.id,mode:'single',groups:[],policy}})).json();
    if(result.model!==model.id)throw Error(ui('ui.wrong_assignment','Server returned an assignment for a different model.'));
    assignmentSaved.set(model.id,result);assignmentDrafts.set(model.id,{...result,devices:(result.groups||[]).flat(),dirty:false});await refresh(true);
  }catch(error){if(chosen()?.id===model.id)status(error.message,true);}finally{devicePolicyPending=false;busy();renderDeviceSummary();}
}
function updateSetupDevices(inventory){
  const select=$('setup-device');if(!select)return;const saved=select.value||'auto';select.replaceChildren(option('auto',tr('device.auto')),option('cpu',tr('device.cpu')));
  for(const row of inventory?.data||[])if(row.id!=='cpu'){const item=option(row.id,`${row.label||row.id} · ${row.id}`);item.disabled=row.present===false||row.stable===false||row.visibility?.status==='masked';select.append(item);}
  if(![...select.options].some(item=>item.value===saved)){const item=option(saved,uiJoin([saved,' · ',tr('device.unavailable')]));item.disabled=true;select.append(item);}select.value=saved;
}
function setupAssignments(){
  if(!$('assignment-panel'))return;
  if(typeof document!=='undefined'&&$('prompt-label')){
    const summary=document.createElement('details');summary.id='device-selection';summary.className='device-selection';const line=document.createElement('summary');line.id='device-line';summary.append(line);
    const label=document.createElement('label');label.htmlFor='device-policy';add(label,'span',tr('device.policy'));const policy=document.createElement('select');policy.id='device-policy';policy.disabled=true;label.append(policy);summary.append(label);
    for(const id of ['device-selection-text','device-observed-text','device-visibility','device-policy-note']){const p=add(summary,'p','');p.id=id;p.className=id==='device-policy-note'?'hint advanced-only':'hint';p.setAttribute('aria-live','polite');}
    const refresh=document.createElement('button');refresh.id='device-refresh';refresh.type='button';refresh.className='quiet';setContent(refresh,tr('device.refresh'));refresh.onclick=()=>Promise.all([loadAssignmentDevices(true),loadAssignment()]);summary.append(refresh);
    if($('generate').insertBefore)$('generate').insertBefore(summary,$('step-input')||$('prompt-label'));else $('generate').append(summary);policy.onchange=saveDevicePolicy;
    const reason=add($('assignment-panel'),'p',tr('device.compatibilityNote'));reason.id='assignment-compatibility';reason.className='hint';const modeNotes=add($('assignment-panel'),'p','');modeNotes.id='assignment-mode-limits';modeNotes.className='hint warning';
    const details=document.createElement('details');add(details,'summary',tr('device.inventoryDetails'));const pre=add(details,'pre','');pre.id='device-inventory-details';$('assignment-panel').append(details);
    const reset=document.createElement('button');reset.id='assignment-reset';reset.type='button';reset.className='quiet';setContent(reset,tr('device.discard'));reset.onclick=()=>{const id=chosen()?.id,value=assignmentSaved.get(id);if(active||!value)return;++assignmentEpoch;assignmentDrafts.set(id,{...value,devices:(value.groups||[]).flat(),dirty:false});renderAssignment(assignmentDrafts.get(id));};$('assignment-panel').append(reset);
  }
  $('assignment-mode').onchange=()=>{assignmentLayout();rememberAssignment();};
  for(const id of ['assignment-ranks','assignment-replicas','assignment-devices'])$(id).onchange=rememberAssignment;
  for(const id of ['assignment-ranks','assignment-replicas'])$(id).oninput=rememberAssignment;
  assignmentLayout();$('assignment-load').onclick=()=>loadAssignmentDevices(true);
  $('assignment-apply').onclick=async()=>{
    const model=chosen();if(active||devicePolicyPending||!model||assignmentModel!==model.id||!assignmentDrafts.has(model.id))return;
    const problem=assignmentIssue(model);if(problem){setContent($('assignment-status'),problem);return;}
    const submitted=assignmentDrafts.get(model.id),mode=$('assignment-mode').value,ids=[...$('assignment-devices').selectedOptions].map(item=>item.value),replicas=Number($('assignment-replicas').value||1),groups=!ids.length?[]:mode==='single'&&replicas>1?ids.map(id=>[id]):[ids];
    try{const result=await(await api('/api/assignments',{method:'POST',json:{model:model.id,mode,groups,tensor_parallel_size:Number($('assignment-ranks').value||2)}})).json();
      if(result.model!==model.id)throw Error(ui('ui.wrong_assignment','Server returned an assignment for a different model.'));
      assignmentSaved.set(model.id,result);if(assignmentDrafts.get(model.id)===submitted)assignmentDrafts.set(model.id,{...result,devices:(result.groups||[]).flat(),dirty:false});
      if(chosen()?.id===model.id)setContent($('assignment-status'),uiCount('ui.replica_saved','{{mode}} · {{count}} replicas saved','{{mode}} · {{count}} replica saved',{mode:assignmentModeLabel(result.mode),count:result.replicas}));await refresh(true);
    }catch(error){if(chosen()?.id===model.id)setContent($('assignment-status'),error.message);}
  };
}
function setupTaskFields(){
  const form=$('generate'),settings=$('max_tokens').parentElement?.parentElement||form,taskHost=document.createElement('div');taskHost.id='task-fields';taskHost.className='task-fields';
  const anchor=document.querySelector('#generate details');if(form.insertBefore)form.insertBefore(taskHost,anchor);else form.append(taskHost);
  const fieldHost=taskHost;
  const field=(host,id,label,tag='input')=>{
    const box=document.createElement('label'),caption=document.createElement('span'),control=document.createElement(tag);
    setContent(caption,label);control.id=id;box.append(caption,control);host.append(box);return control;
  };
  const rerank=document.createElement('div');rerank.id='rerank-fields';rerank.hidden=true;fieldHost.append(rerank);
  const documents=field(rerank,'rerank-documents',ui("ui.documents","Documents (one nonempty passage per line)"),'textarea');documents.rows=6;
  field(settings,'top_n',ui("ui.top_n","Rerank result count")).type='number';
  const rerankTruncate=field(settings,'rerank-truncate',ui("ui.rerank_truncate","Rerank input truncation"),'select');
  rerankTruncate.append(option('',ui("ui.default_model_server","Model / server default")),option('false',ui("ui.reject_oversized","Reject oversized inputs")),option('true',ui("ui.allow_truncate","Allow truncation")));
  const returnDocuments=field(settings,'rerank-return-documents',ui("ui.return_documents","Return document text"),'select');
  returnDocuments.append(option('',ui("ui.default_off","Default (off)")),option('false',ui("ui.off","Off")),option('true',ui("ui.on","On")));

  for(const [id,label,type] of [
    ['vad-sample-rate',ui("ui.vad_rate","VAD sample rate"),'select'],['vad-threshold',ui("ui.vad_threshold","VAD threshold"),'input'],['vad-min-speech',ui("ui.vad_min_speech","Minimum speech (ms)"),'input'],
    ['vad-min-silence',ui("ui.vad_min_silence","Minimum silence (ms)"),'input'],['vad-speech-pad',ui("ui.vad_padding","Speech padding (ms)"),'input'],['vad-return-frames',ui("ui.vad_probabilities","Return frame probabilities"),'select']
  ]){
    const control=field(settings,id,label,type);if(type==='input'){control.type='number';control.step='any';}
    if(id==='vad-sample-rate')control.append(option('',ui("ui.default_model_server","Model / server default")),option('8000','8000 Hz'),option('16000','16000 Hz'));
    if(id==='vad-return-frames')control.append(option('',ui("ui.default_off","Default (off)")),option('false',ui("ui.off","Off")),option('true',ui("ui.on","On")));
  }

  const segmentation=document.createElement('div');segmentation.id='segmentation-fields';segmentation.hidden=true;fieldHost.append(segmentation);
  const mode=field(segmentation,'segment-mode',ui("ui.segmentation_prompt","Segmentation prompt"),'select');mode.append(option('box',ui("ui.box_coordinates","Box coordinates")),option('points',ui("ui.positive_negative_points","Positive / negative points")));
  for(const [id,label] of [['segment-x0',ui("ui.box_x0","Box x0")],['segment-y0',ui("ui.box_y0","Box y0")],['segment-x1',ui("ui.box_x1","Box x1")],['segment-y1',ui("ui.box_y1","Box y1")]]){const control=field(segmentation,id,label);control.type='number';control.step='any';}
  const points=field(segmentation,'segment-points',ui("ui.points_json","Points JSON, e.g. [[60.5,80],[100,120]]"),'textarea');points.rows=3;
  const labels=field(segmentation,'segment-point-labels',ui("ui.point_labels_json","Point labels JSON, e.g. [1,0]"),'textarea');labels.rows=2;
  const multimask=field(settings,'segment-multimask',ui("ui.segmentation_alternatives","Segmentation alternatives"),'select');multimask.append(option('',ui("ui.default_model_server","Model / server default")),option('false',ui("ui.single_mask","Single mask")),option('true',ui("ui.native_multimask","Native multimask alternatives")));
  const threshold=field(settings,'segment-threshold',ui("ui.mask_threshold","Mask threshold"));threshold.type='number';threshold.step='any';

  const audioAction=field(fieldHost,'audio-action',ui("ui.audio_action","Audio action"),'select');audioAction.append(option('transcribe',ui("ui.transcribe","Transcribe")),option('analyze',ui('audio.analyze','Transcribe with speakers and overlap')),option('translate',ui("ui.translate_english","Translate speech to English")));
  const analysis=document.createElement('div');analysis.id='audio-analysis-fields';analysis.hidden=true;fieldHost.append(analysis);
  field(analysis,'diarization-model',ui('audio.speaker_model','Speaker model'),'select').append(option('',ui('audio.choose_speakers','Choose a speaker model')));
  const textAnalysis=field(analysis,'audio-text-model',ui('audio.transcript_analysis','Transcript text analysis'),'select');if(textAnalysis.parentElement)textAnalysis.parentElement.className='advanced-only';textAnalysis.append(option('',ui('audio.no_text_analysis','No transcript analysis')));
  add(analysis,'p',ui('audio.anonymous_note','Labels apply only to this recording. Overlap does not separate voices or identify people.'));
  for(const [id,label] of [['diarization-count',ui('audio.speaker_count','Speaker count (0: automatic)')],['diarization-threshold',ui('audio.cluster_threshold','Clustering threshold')],['diarization-on',ui('audio.minimum_speech','Minimum speech interval (seconds)')],['diarization-off',ui('audio.minimum_gap','Fill speech gaps shorter than (seconds)')],['audio-timeout',ui('audio.time_budget','Total time budget (milliseconds)')]]){
    const control=field(settings,id,label);control.type='number';control.step=['diarization-count','audio-timeout'].includes(id)?'1':'any';control.min=id==='audio-timeout'?'1':'0';setUiAttribute(control,'placeholder',ui('ui.default_model_server','Model / server default'));
  }
  const audio=document.createElement('div');audio.id='audio-translation-fields';audio.hidden=true;fieldHost.append(audio);
  field(audio,'translation-model',ui("ui.translation_model","Translation model (optional composed ASR + NMT)"),'select').append(option('',ui("ui.whisper_direct","Whisper direct translation")));
  field(audio,'source-language',ui("ui.nmt_source","NMT source language ID (required with translation model)")).placeholder='e.g. fra_Latn';
  field(audio,'target-language',ui("ui.nmt_target","English NMT target language ID (optional)")).placeholder=ui("ui.nmt_target_placeholder","Translation model default, then eng_Latn; e.g. en");

  const segmentMode=()=>{
    const pointsMode=$('segment-mode').value==='points';
    for(const id of ['segment-x0','segment-y0','segment-x1','segment-y1'])if($(id).parentElement)$(id).parentElement.hidden=pointsMode;
    for(const id of ['segment-points','segment-point-labels'])if($(id).parentElement)$(id).parentElement.hidden=!pointsMode;
  };
  mode.onchange=segmentMode;segmentMode();
  audioAction.onchange=()=>select(false);
}
const setupTasks=[
  ['chat','Chat and write'],['vision','Understand an image'],['image','Make or edit an image'],
  ['video','Make video or image-to-video'],['transcription','Transcribe audio'],['speech','Speak text'],
  ['embeddings','Create embeddings'],['translation','Translate text'],['image-to-text','Read or caption an image'],
  ['object-detection','Detect objects'],['depth-estimation','Estimate relative depth'],
  ['image-segmentation','Create an edit mask'],['sentiment','Classify sentiment'],
  ['rerank','Rerank passages'],['pair-classification','Compare two texts'],['vad','Detect speech activity'],['diarization','Find speakers and overlap'],['music','Create music or a song']
];
function setupMessage(text,bad=false,kind=bad?'error':'info'){const node=$('setup-message');setContent(node,text);node.className='setup-message '+(bad?'error':kind);}
function setupPending(value){
  try{
    if(value===undefined)return JSON.parse(globalThis.sessionStorage?.getItem('titanos.acquisition')||'null');
    if(value===null)globalThis.sessionStorage?.removeItem('titanos.acquisition');
    else globalThis.sessionStorage?.setItem('titanos.acquisition',JSON.stringify(value));
  }catch{return null;}
  return value;
}
function runtimePending(value){
  try{
    if(value===undefined)return JSON.parse(globalThis.sessionStorage?.getItem('titanos.runtime-setup')||'null');
    if(value===null)globalThis.sessionStorage?.removeItem('titanos.runtime-setup');
    else globalThis.sessionStorage?.setItem('titanos.runtime-setup',JSON.stringify(value));
  }catch{return null;}
  return value;
}
function setupOneStop(){
  const host=$('generate').parentElement||$('generate'),panel=document.createElement('details');panel.id='model-setup';panel.className='setup-panel bay';panel.open=!catalog.length;
  const summary=document.createElement('summary'),plate=document.createElement('span'),number=document.createElement('span'),separator=document.createElement('span'),title=document.createElement('span');summary.className='bay-summary';
  plate.className='bay-plate';number.className='bay-no';number.textContent='02';separator.className='bay-sep';separator.textContent='//';separator.setAttribute?.('aria-hidden','true');title.id='setup-summary';setContent(title,tr('setup.title'));plate.append(number,separator,title);
  const intro=document.createElement('span');intro.id='setup-intro';intro.className='bay-line';setContent(intro,tr('setup.intro'));summary.append(plate,intro);summary.addEventListener?.('click',()=>{setupToggled=true;});
  const controls=document.createElement('div');controls.className='setup-controls';
  const field=(id,label,tag='input')=>{const box=document.createElement('label'),caption=document.createElement('span'),control=document.createElement(tag);caption.id=id+'-label';setContent(caption,label);control.id=id;box.append(caption,control);controls.append(box);return control;};
  const task=field('setup-task',tr('setup.task'),'select');for(const [value] of setupTasks)task.append(option(value,tr('task.'+value)));
  const provider=field('setup-provider',tr('setup.source'),'select');provider.append(option('huggingface','Hugging Face'),option('github-sherpa','GitHub · Sherpa ONNX TTS'),option('ace-step',ui("ui.provider_ace","ACE-Step 1.5 · music")));
  const runtimeProfile=field('setup-runtime-profile',ui("ui.runtime_profile","Managed native runtime (GGUF chat)"),'select');if(runtimeProfile.parentElement)runtimeProfile.parentElement.hidden=true;
  runtimeProfile.parentElement.className='advanced-only';runtimeProfile.onchange=()=>{runtimeProfileChoice=runtimeProfile.value;};
  const setupDevice=field('setup-device',tr('device.policy'),'select');setupDevice.parentElement.className='advanced-only';setupDevice.append(option('auto',tr('device.auto')),option('cpu',tr('device.cpu')));
  const setupRefresh=document.createElement('button');setupRefresh.id='setup-refresh-devices';setupRefresh.type='button';setupRefresh.className='quiet advanced-only';setContent(setupRefresh,tr('device.refresh'));controls.append(setupRefresh);
  setupRefresh.onclick=async()=>{setupRefresh.disabled=true;try{
    updateSetupDevices(await(await api('/api/assignments/devices?refresh=true')).json());
    const runtime=serverInfo?.model_setup?.runtime;if(runtime?.profiles_route){
      const fresh=await(await api('/api/runtime-profiles?refresh=true')).json();
      if(!Array.isArray(fresh.data))throw Error(String(tr('device.invalidInventory')));
      Object.assign(runtime,{profiles:fresh.data,recommended_profile:fresh.recommended,metadata_status:fresh.metadata_status});updateRuntimeProfiles();
      if(fresh.metadata_status==='unverified')setupMessage(tr('device.unverified'),true);
    }
  }catch(error){setupMessage(error.message,true);}finally{setupRefresh.disabled=false;}};
  const query=field('setup-query',tr('setup.search'));setUiAttribute(query,"placeholder",ui("ui.search_example","small English embedding model"));
  const actions=document.createElement('div');actions.className='setup-actions';
  const search=document.createElement('button');search.id='setup-search';search.type='button';search.className='go';setContent(search,tr('setup.find'));
  actions.append(search);controls.append(actions);
  const linkField=field('setup-link',tr('setup.link'));linkField.type='url';setUiAttribute(linkField,"placeholder",'https://huggingface.co/owner/model');
  const importButton=document.createElement('button');importButton.id='setup-import';importButton.type='button';importButton.className='quiet';setContent(importButton,tr('setup.inspect'));controls.append(importButton);
  const message=document.createElement('p');message.id='setup-message';message.className='setup-message';message.setAttribute?.('role','status');message.setAttribute?.('aria-live','polite');
  const results=document.createElement('div');results.id='setup-results';results.className='setup-results';
  const plan=document.createElement('div');plan.id='setup-plan';plan.className='setup-plan';plan.hidden=true;
  panel.append(summary,controls,message,results,plan);
  if(host.insertBefore)host.insertBefore(panel,$('generate'));else host.append(panel);
  provider.onchange=()=>{if(provider.value==='github-sherpa'){task.value='speech';task.disabled=true;}else if(provider.value==='ace-step'){task.value='music';task.disabled=true;}else task.disabled=false;};
  task.onchange=()=>{if(task.value==='music'){provider.value='ace-step';provider.onchange();}};
  search.onclick=oneStopSearch;importButton.onclick=oneStopImport;
}
function setupRegistryName(candidate){return String(candidate.id||'model').toLowerCase().replace(/[^a-z0-9._/-]+/g,'-').replace(/^[-/]+|[-/]+$/g,'').slice(0,80)||'model';}
function renderSetupCandidate(candidate){
  const article=document.createElement('article');article.className='setup-candidate';
  const heading=document.createElement('h3');setContent(heading,candidate.name||candidate.id);
  const source=document.createElement('p');source.className='setup-source';setContent(source,[candidate.provider,candidate.author,candidate.library,candidate.license].filter(Boolean).join(' · ')||candidate.provider);
  const facts=document.createElement('p');facts.className='hint';setContent(facts,uiJoin([taskLabel(candidate.titanos_task||candidate.task),candidate.recommended_backend||candidate.backend,candidate.size?`${Math.ceil(candidate.size/1024/1024)} MiB`:null,candidate.gated?ui("ui.gated_access","Gated access"):ui("ui.public_access","Public access")].filter(Boolean),' · '));
  const note=document.createElement('p');note.className='hint';setContent(note,ui("ui.compatibility_note","Compatibility is verified during planning and load."));
  const inspect=document.createElement('button');inspect.type='button';inspect.className='quiet';inspect.setAttribute?.('data-i18n','setup.review');setContent(inspect,tr('setup.review'));inspect.onclick=()=>oneStopPlan(candidate);
  article.append(heading,source,facts,note,inspect);if(candidate.compatibility?.note){const detail=document.createElement('details');detail.className='advanced-only';add(detail,'summary',ui("ui.provider_notes","Provider notes (original)"));add(detail,'pre',candidate.compatibility.note);article.append(detail);}$('setup-results').append(article);
}
async function oneStopSearch(){
  const provider=$('setup-provider').value,task=$('setup-task').value,query=$('setup-query').value.trim();
  $('setup-search').disabled=true;$('setup-results').replaceChildren();$('setup-plan').hidden=true;setupMessage(ui("ui.searching","Searching provider metadata…"));
  try{
    const route=`/api/model-providers/${encodeURIComponent(provider)}/search?task=${encodeURIComponent(task)}&query=${encodeURIComponent(query)}&limit=20`;
    const result=await(await api(route)).json();
    for(const candidate of result.data)renderSetupCandidate({...candidate,provider,titanos_task:task});
    setupMessage(result.data.length?uiCount("ui.found_candidates","Found {{count}} candidates. Review compatibility before downloading.","Found {{count}} candidate. Review compatibility before downloading.",{count:result.data.length}):ui("ui.no_candidates","No matching candidates. Try a broader query or another source."));
  }catch(error){setupMessage(error.message,true);}
  finally{$('setup-search').disabled=false;}
}
async function oneStopImport(){
  const url=$('setup-link').value.trim();if(!url){setupMessage(ui("ui.enter_link","Enter an HTTPS model page or artifact link."),true);return;}
  $('setup-import').disabled=true;$('setup-results').replaceChildren();$('setup-plan').hidden=true;setupMessage(ui("ui.inspecting_link","Inspecting the model link…"));
  try{
    const result=await(await api('/api/model-providers/import',{method:'POST',json:{url}})).json();
    if(result.imported.provider&&result.metadata){renderSetupCandidate({...result.metadata,provider:result.imported.provider,titanos_task:$('setup-task').value});setupMessage(ui("ui.recognized_provider","Recognized provider metadata. Compatibility still requires a concrete plan."));}
    else if(result.metadata){
      const page=result.metadata,card=document.createElement('article');card.className='setup-candidate';
      const pageTitle=(page.title||ui("ui.public_page","Public model page")).replace(/ContentsMenuExpand.*$/,'').trim().slice(0,160)||ui("ui.public_page","Public model page");
      add(card,'h3',pageTitle);add(card,'p',page.description||ui("ui.no_description","No page description was declared."));
      const source=document.createElement('a');source.href=page.final_url;source.target='_blank';source.rel='noreferrer';setContent(source,page.final_url);card.append(source);
      add(card,'p',ui("ui.page_license","Page-declared license: {{license}} · Compatibility: unverified",{license:page.license_claim||ui("ui.not_declared","not declared")}));
      const relevance=row=>{const value=(row.url+' '+row.label).toLowerCase();return (row.provider_hint==='huggingface'?20:0)+(/gguf|safetensors|checkpoint|weights|model|download/.test(value)?10:0)+(/ollama|llama\.cpp/.test(value)?5:0);};
      const modelLinks=[...(page.model_links||[])].sort((left,right)=>relevance(right)-relevance(left)),visibleLinks=[...modelLinks.slice(0,8),...(page.license_links||[])];
      for(const row of visibleLinks){
        const line=document.createElement('div'),link=document.createElement('a');link.href=row.url;link.target='_blank';link.rel='noreferrer';setContent(link,row.label||row.url);line.append(link);
        if(modelLinks.includes(row)){const inspect=document.createElement('button');inspect.type='button';inspect.className='quiet';setContent(inspect,ui("ui.inspect_provider_link","Inspect provider link"));inspect.onclick=()=>{$('setup-link').value=row.url;oneStopImport();};line.append(inspect);}
        card.append(line);
      }
      const technical=document.createElement('details');add(technical,'summary',ui("ui.page_evidence","Technical page evidence"));add(technical,'pre',`Source: ${page.source_url}\nFinal: ${page.final_url}\nBytes: ${page.content_bytes}\nSHA-256: ${page.content_sha256}\nRedirects: ${(page.redirects||[]).join(' → ')||'none'}\nAll extracted links:\n${modelLinks.map(row=>row.url).join('\n')}`);card.append(technical);
      add(card,'p',page.note);$('setup-results').append(card);setupMessage(ui("ui.page_imported","Public page metadata imported. Linked providers still require inspect and planning."));
    }else{
      const card=document.createElement('article');card.className='setup-candidate';add(card,'h3',ui("ui.metadata_link","Metadata-only model link"));add(card,'p',result.imported.canonical_url);add(card,'p',ui("ui.metadata_only_note","No supported provider/format was resolved. Keep this link as provenance; Titanos will not execute page instructions or claim compatibility."));$('setup-results').append(card);setupMessage(ui("ui.metadata_link_imported","Link imported as metadata only."));
    }
  }catch(error){setupMessage(error.message,true);}
  finally{$('setup-import').disabled=false;}
}
function musicSetupIssue(plan){
  if(plan.backend!=='ace-step'||plan.runtime?.status==='available')return null;
  return plan.runtime?.status==='unsupported_adapter'?ui("ui.music_variant_unsupported","This music runtime does not support the selected variant. Choose XL SFT or a runtime that explicitly supports this variant before downloading."):ui("ui.music_runtime_missing","This package cannot run this music model. Use a complete Titanos music distribution with the matching ACE-Step runtime, then inspect the model again. No model download was started.");
}
async function oneStopPlan(candidate){
  const task=candidate.titanos_task||$('setup-task').value,provider=candidate.provider||$('setup-provider').value;
  const backend=candidate.recommended_backend||candidate.backend||(task==='image'||task==='video'?'diffusers':'transformers');
  const request={provider,identifier:candidate.id,revision:candidate.revision||null,registry_name:setupRegistryName(candidate),task,backend,device:$('setup-device')?.value||'auto'};
  const requestedDevice=request.device,profile=backend==='llama-cpp'?selectedRuntimeProfile():null,plannedProfile=profile?.id||null;
  if(backend==='llama-cpp'&&runtimeProfileChoice&&profile?.runtime_kind==='cpu'&&request.device==='auto')request.device='cpu';
  $('setup-plan').hidden=false;$('setup-plan').replaceChildren();setupMessage(ui("ui.resolving_plan","Resolving an immutable acquisition plan…"));
  try{
    if(backend==='llama-cpp'){
      if(!profile?.one_stop_supported)throw Error(String(tr('device.runtimeUnavailable')));
      if(profile.runtime_kind==='cpu'&&!['auto','cpu'].includes(requestedDevice))throw Error(String(tr('device.profileMismatch')));
      const metadata=await(await api('/api/model-providers/inspect',{method:'POST',json:{provider,identifier:candidate.id,revision:candidate.revision||null}})).json();
      const weights=(metadata.files||[]).filter(row=>/\.gguf$/i.test(row.path)&&!/mmproj|projector/i.test(row.path));
      weights.sort((left,right)=>{const l=/q4_k_m/i.test(left.path)?0:1,r=/q4_k_m/i.test(right.path)?0:1;return l-r||(left.size??Number.MAX_SAFE_INTEGER)-(right.size??Number.MAX_SAFE_INTEGER)||left.path.localeCompare(right.path);});
      if(!weights.length)throw Error(ui("ui.no_gguf","No compatible GGUF model file was found in this exact revision."));
      request.model_file=weights[0].path;
      if(task==='vision'){
        const projector=(metadata.files||[]).find(row=>/\.gguf$/i.test(row.path)&&/mmproj|projector/i.test(row.path));
        if(!projector)throw Error(ui("ui.missing_projector","Native vision requires a matching GGUF projector in the same revision."));
        request.mmproj=projector.path;
      }
    }
    const plan=await(await api('/api/acquisitions/plan',{method:'POST',json:request})).json();
    const heading=add($('setup-plan'),'h3',ui("ui.get_ready","Get {{model}} ready",{model:candidate.name||candidate.id}));
    const facts=add($('setup-plan'),'p',uiJoin([plan.provider_title,taskLabel(plan.task),plan.backend,plan.device==='auto'?tr('advanced.auto'):plan.device,plan.expected_bytes!==null?`${Math.ceil(plan.expected_bytes/1024/1024)} MiB`:ui("ui.size_unavailable","Size unavailable")],' · '));facts.className='setup-facts';
    add($('setup-plan'),'p',ui("ui.pinned_revision","Pinned revision: {{revision}}",{revision:plan.resolved_revision}));
    add($('setup-plan'),'p',ui("ui.license_access","License: {{license}} · Access: {{access}}",{license:plan.license||ui("ui.license_review_needed","Inspect acquired license before activation"),access:plan.gated?ui("ui.gated_access","Gated access"):ui("ui.public_access","Public access")}));
    add($('setup-plan'),'p',ui("ui.runtime_qualification","Runtime: {{runtime}} · Qualification: {{qualification}}",{runtime:stateLabel(plan.runtime.status||'unknown'),qualification:plan.compatibility.qualification||ui("ui.unmeasured","not measured for this exact model/device")}));
    const files=document.createElement('details');add(files,'summary',uiCount("ui.selected_files","{{count}} selected files","{{count}} selected file",{count:plan.files.length}));add(files,'pre',uiJoin(plan.files.map(row=>uiJoin([row.path,row.size!==null&&row.size!==undefined?uiJoin([' · ',ui("ui.byte_count","{{count}} bytes",{count:row.size})]):''])),'\n'));$('setup-plan').append(files);
    const needsRuntime=plan.backend==='llama-cpp'&&(plan.runtime.status==='missing_runtime'||plan.runtime.managed_profile!==plannedProfile),musicIssue=musicSetupIssue(plan);
    const button=document.createElement('button');button.type='button';button.className='structural';if(!needsRuntime)button.setAttribute?.('data-i18n','setup.download');setContent(button,musicIssue?ui("ui.music_runtime_required","Music runtime required"):needsRuntime?tr('device.prepareRuntime'):tr('setup.download'));button.disabled=Boolean(musicIssue);button.onclick=async()=>{if(musicIssue){setupMessage(musicIssue,true);return;}try{if((($('setup-device')?.value||'auto')!==requestedDevice)||(plan.backend==='llama-cpp'&&$('setup-runtime-profile')?.value!==plannedProfile))throw Error(String(tr('device.replan')));if(needsRuntime)await prepareManagedRuntime(plan,plannedProfile);await oneStopAcquire(plan,plannedProfile);}catch(error){setupMessage(error.message,true);}};$('setup-plan').append(button);
    if(musicIssue){add($('setup-plan'),'p',musicIssue);setupMessage(musicIssue,true);}else setupMessage(ui("ui.plan_ready","Plan resolved. Downloading will not activate a partial or failed model."));
    heading.tabIndex=-1;heading.focus?.();
  }catch(error){setupMessage(error.message,true);}
}
function renderAcquisition(record){
  const plan=$('setup-plan');plan.hidden=false;plan.replaceChildren();
  add(plan,'h3',ui("ui.acquisition_heading","{{state}} {{model}}",{state:record.status==='active'?ui("ui.acquisition_complete_prefix","Acquisition complete:"):ui("ui.preparing","Preparing"),model:record.plan.registry_name}));
  const progress=document.createElement('progress');progress.id='setup-progress';progress.max=record.progress.total||1;progress.value=Math.min(record.progress.completed||0,progress.max);plan.append(progress);renderLattice(plan,record.progress.completed||0,record.progress.total||0);
  add(plan,'p',ui("ui.progress","{{status}} · {{stage}} · {{completed}}{{total}}",{status:stateLabel(record.status),stage:stageLabel(record.progress.stage),completed:record.progress.completed||0,total:progressTotal(record.progress)}));
  if(['queued','running','cancelling'].includes(record.status)){const cancel=document.createElement('button');cancel.type='button';cancel.className='quiet';cancel.setAttribute?.('data-i18n','setup.cancel');setContent(cancel,tr('setup.cancel'));cancel.onclick=async()=>{try{await api('/api/acquisitions/'+record.id,{method:'DELETE'});setupMessage(ui("ui.cancel_requested","Cancellation requested. Partial data remains inactive."));}catch(error){setupMessage(error.message,true);}};plan.append(cancel);}
}
function renderReadyAcquisition(record){
  const panel=$('setup-plan');panel.hidden=false;panel.replaceChildren();add(panel,'h3',tr('status.ready'));
  add(panel,'p',ui("ui.job_name","{{model}} · {{task}}",{model:record.activation.model,task:stateLabel(record.activation.status)}));
  setupMessage(tr('setup.ready'),false,'success');
}
function renderRuntimeSetup(record){
  const panel=$('setup-plan');panel.hidden=false;panel.replaceChildren();add(panel,'h3',ui('device.preparingRuntime','Preparing selected managed runtime'));
  const progress=document.createElement('progress');progress.max=record.progress.total||1;progress.value=Math.min(record.progress.completed||0,progress.max);panel.append(progress);renderLattice(panel,record.progress.completed||0,record.progress.total||0);
  add(panel,'p',ui("ui.progress","{{status}} · {{stage}} · {{completed}}{{total}}",{status:stateLabel(record.status),stage:stageLabel(record.progress.stage),completed:record.progress.completed||0,total:progressTotal(record.progress)}));
  add(panel,'p',ui("ui.runtime_summary","{{runtime}} {{release}} · {{profile}} · {{license}} · {{note}}",{runtime:record.plan.runtime,release:record.plan.release,profile:record.plan.profile,license:record.plan.license,note:ui("ui.no_python","no user Python required")}));
  if(['queued','running','cancelling'].includes(record.status)){const cancel=document.createElement('button');cancel.type='button';cancel.className='quiet';setContent(cancel,ui("ui.cancel_runtime","Cancel runtime setup"));cancel.onclick=()=>api('/api/runtime-setups/'+record.id,{method:'DELETE'}).catch(error=>setupMessage(error.message,true));panel.append(cancel);}
}
async function pollRuntimeSetup(identifier){
  while(true){
    const record=await(await api('/api/runtime-setups/'+identifier)).json();renderRuntimeSetup(record);
    if(record.status==='prepared'){runtimePending(null);setupMessage(ui('device.runtimeReady','Selected managed runtime is verified and ready.'));return record;}
    if(['failed','cancelled','interrupted'].includes(record.status)){
      const retry=document.createElement('button');retry.type='button';retry.className='quiet';setContent(retry,ui("ui.retry_runtime","Retry runtime setup"));retry.onclick=async()=>{const pending=runtimePending(),next=await(await api('/api/runtime-setups/'+identifier+'/retry',{method:'POST'})).json();runtimePending({...pending,id:next.id});await pollRuntimeSetup(next.id);if(pending?.model_plan)await oneStopAcquire(pending.model_plan,pending.profile);};$('setup-plan').append(retry);
      throw Error(record.error||ui("ui.runtime_state","Runtime setup {{status}}.",{status:stateLabel(record.status)}));
    }
    await new Promise(resolve=>setTimeout(resolve,500));
  }
}
async function prepareManagedRuntime(modelPlan,selectedProfile=null){
  const runtime=serverInfo?.model_setup?.runtime,profile=selectedProfile||$('setup-runtime-profile')?.value||runtime?.recommended_profile;
  if(!runtime?.configured||!profile)throw Error(ui("ui.runtime_missing","This server has no compatible operator-configured native runtime profile."));
  const existing=runtimePending();
  if(existing?.id){const record=await(await api('/api/runtime-setups/'+existing.id)).json();if((record.plan?.profile||existing.profile)!==profile)throw Error(String(tr('device.pendingRuntime')));await pollRuntimeSetup(existing.id);return;}
  setupMessage(ui('device.planningRuntime','Planning the selected server-owned runtime…'));
  const runtimePlan=await(await api('/api/runtime-setups/plan',{method:'POST',json:{profile}})).json();
  if(runtimePlan.runtime!=='llama.cpp'||runtimePlan.profile!==profile)throw Error(ui("ui.unexpected_runtime","Server returned an unexpected runtime plan."));
  const record=await(await api('/api/runtime-setups',{method:'POST',json:{profile}})).json();
  runtimePending({id:record.id,profile,model_plan:modelPlan});renderRuntimeSetup(record);await pollRuntimeSetup(record.id);
}
async function activateAcquisition(record){
  let accepted=null;
  if(!record._accepted&&!record.license&&record.license_evidence?.length){
    const plan=$('setup-plan');add(plan,'p',ui("ui.review_all_licenses","Review all acquired model license evidence before activation."));
    for(const evidence of record.license_evidence){const details=document.createElement('details');add(details,'summary',evidence.path);add(details,'pre',evidence.text||evidence.sha256);plan.append(details);}
    const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';const caption=document.createElement('span');setContent(caption,ui("ui.licenses_ack","I reviewed every listed license/provenance document"));label.className='check';label.append(check,caption);plan.append(label);
    if(!check.checked){const button=document.createElement('button');button.type='button';button.className='structural';setContent(button,ui("ui.activate_reviewed","Activate after review"));button.onclick=async()=>{if(!check.checked){setupMessage(ui("ui.ack_required","Review and check the complete license acknowledgement first."),true);return;}await activateAcquisition({...record,_accepted:record.license_evidence_digest});};plan.append(button);return;}
  }
  accepted=record._accepted||accepted;
  setupMessage(ui("ui.registering_loading","Registering and loading the verified model…"));
  try{
    const pending=setupPending(),runtimeProfile=acquisitionRuntimeProfiles.get(record.id)||(pending?.id===record.id?pending.runtime_profile:null);
    const payload={accepted_license_sha256:accepted,load:true};if(record.plan?.backend==='llama-cpp'&&runtimeProfile)payload.runtime_profile=runtimeProfile;
    const activeRecord=await(await api(`/api/acquisitions/${record.id}/activate`,{method:'POST',json:payload})).json();
    acquisitionRuntimeProfiles.delete(record.id);activeAcquisition=null;setupPending(null);await refresh(true);$('model').value=activeRecord.activation.model;$('model').onchange();renderReadyAcquisition(activeRecord);status(tr('status.ready'),false,'success');
  }catch(error){setupMessage(error.message,true);}
}
async function pollAcquisition(identifier){
  while(activeAcquisition===identifier){
    const record=await(await api('/api/acquisitions/'+identifier)).json();renderAcquisition(record);
    if(record.status==='prepared'){await activateAcquisition(record);return;}
    if(record.status==='active'){
      if(activeAcquisition!==identifier)return;
      const pending=setupPending(),expectedProfile=acquisitionRuntimeProfiles.get(identifier)||(pending?.id===identifier?pending.runtime_profile:null);
      acquisitionRuntimeProfiles.delete(identifier);activeAcquisition=null;if(pending?.id===identifier)setupPending(null);
      await refresh(true);
      if(active||activeAcquisition)return;
      if(refreshFailed){setupMessage(ui("ui.active_catalog_failed","This acquisition is complete, but the model catalog could not be refreshed. Refresh Models before selecting it."),true);return;}
      const plan=record.plan||{},name=plan.registry_name,model=catalog.find(row=>row.id===name),source=model?.capabilities?.source;
      const repository=plan.provider==='huggingface'?plan.candidate_id:plan.source_url;
      if(plan.backend==='llama-cpp'&&expectedProfile&&record.activation?.runtime_profile!==expectedProfile){
        setupMessage(tr(record.activation?.runtime_profile?'device.activeProfileMismatch':'device.activeProfileUnknown'),true);return;
      }
      if(!model||model.task!==plan.task||(plan.backend&&(model.capabilities?.backend||model.configured?.backend||model.backend)!==plan.backend)
          ||(plan.device&&plan.device!=='auto'&&(model.configured?.device||model.device)!==plan.device)
          ||(repository&&source?.repository!==repository)||(plan.resolved_revision&&source?.revision!==plan.resolved_revision)
          ||(record.activation?.model&&record.activation.model!==name)){
        setupMessage(ui("ui.active_registration_changed","This acquisition is complete, but its model registration is missing or has changed. Review Models before running it."),true);return;
      }
      $('model').value=model.id;$('model').onchange();
      setupMessage(ui("ui.already_registered","This model is already registered. Continue in the task workspace below."));return;
    }
    if(['failed','cancelled','interrupted'].includes(record.status)){
      activeAcquisition=null;setupMessage(record.error||ui("ui.acquisition_state","Acquisition {{status}}.",{status:stateLabel(record.status)}),true);
      const retry=document.createElement('button');retry.type='button';retry.className='quiet';setContent(retry,ui("ui.retry_acquisition","Retry / resume"));retry.onclick=async()=>{const pending=setupPending(),profile=acquisitionRuntimeProfiles.get(identifier)||(pending?.id===identifier?pending.runtime_profile:null);const value=await(await api(`/api/acquisitions/${identifier}/retry`,{method:'POST'})).json();if(profile)acquisitionRuntimeProfiles.set(value.id,profile);if(pending?.id===identifier)setupPending({...pending,id:value.id});activeAcquisition=value.id;pollAcquisition(value.id).catch(error=>setupMessage(error.message,true));};$('setup-plan').append(retry);return;
    }
    await new Promise(resolve=>setTimeout(resolve,500));
  }
}
async function reconcileAcquisitions(){
  if(setupReconciled||!$('model-setup'))return;setupReconciled=true;
  try{
    const runtime=runtimePending();
    if(runtime?.id){
      pollRuntimeSetup(runtime.id).then(()=>runtime.model_plan?oneStopAcquire(runtime.model_plan,runtime.profile):null).catch(error=>setupMessage(error.message,true));
      return;
    }
    if(serverInfo?.model_setup?.runtime?.configured){
      try{
        const setups=await(await api('/api/runtime-setups')).json();
        const recoverable=(setups.data||[]).filter(row=>['queued','running','cancelling','interrupted'].includes(row.status)).sort((left,right)=>(right.created_at||0)-(left.created_at||0));
        if(recoverable[0]){runtimePending({id:recoverable[0].id});pollRuntimeSetup(recoverable[0].id).catch(error=>setupMessage(error.message,true));return;}
      }catch{}
    }
    const result=await(await api('/api/acquisitions?limit=100')).json(),pending=setupPending();
    const unresolved=(result.data||[]).filter(row=>['queued','running','cancelling','prepared','failed','cancelled','interrupted'].includes(row.status));
    const record=(pending?.id&&(result.data||[]).find(row=>row.id===pending.id))||unresolved[0];
    if(!record){if(pending)setupPending(null);return;}
    activeAcquisition=record.id;renderAcquisition(record);setupMessage(ui("ui.acquisition_recovered","Recovered {{status}} acquisition {{model}}.",{status:stateLabel(record.status),model:record.plan.registry_name}));
    pollAcquisition(record.id).catch(error=>{activeAcquisition=null;setupMessage(error.message,true);});
  }catch{setupReconciled=false;}
}
async function oneStopAcquire(plan,runtimeProfile=null){
  const musicIssue=musicSetupIssue(plan);if(musicIssue){setupMessage(musicIssue,true);return;}
  if(activeAcquisition){setupMessage(ui("ui.acquisition_busy","Another model acquisition is already active in this tab."),true);return;}
  setupMessage(ui("ui.acquisition_starting","Starting durable acquisition…"));
  try{
    const serialized=JSON.stringify(plan),previous=setupPending();
    const pending=previous?.plan===serialized?previous:{key:globalThis.crypto?.randomUUID?.()||String(Date.now()),plan:serialized};
    if(plan.backend==='llama-cpp'&&runtimeProfile)pending.runtime_profile=runtimeProfile;
    setupPending(pending);
    const record=await(await api('/api/acquisitions',{method:'POST',headers:{'Idempotency-Key':pending.key},json:plan})).json();
    if(pending.runtime_profile)acquisitionRuntimeProfiles.set(record.id,pending.runtime_profile);
    setupPending({...pending,id:record.id});
    activeAcquisition=record.id;renderAcquisition(record);await pollAcquisition(record.id);
  }catch(error){activeAcquisition=null;setupMessage(error.message,true);}
}
function automaticBackend(task){return task==='diarization'?'sherpa-onnx':task==='music'?'ace-step':['image','video'].includes(task)?'diffusers':['tensor','vad'].includes(task)?'onnx':'transformers';}
function selectedRegistrationPolicy(){
  const selection=registrationBackends.includes($('register-backend').value)?$('register-backend').value:'auto';
  const backend=selection==='auto'?automaticBackend($('register-task').value||'chat'):selection;
  return {selection,backend,policy:registrationPolicies[backend]};
}
function registrationBackend(){
  let {selection,backend,policy}=selectedRegistrationPolicy();const task=$('register-task');
  for(const item of task.options)item.disabled=Boolean(selection!=='auto'&&!policy.tasks.includes(item.value));
  if(selection!=='auto'&&!policy.tasks.includes(task.value)){task.value=policy.tasks[0];({selection,backend,policy}=selectedRegistrationPolicy());}
  if(policy.cpu||task.value==='vad')$('register-device').value='cpu';
  if(policy.onnxImage&&!['cpu','dml'].includes($('register-device').value))$('register-device').value='cpu';
  const show=(id,visible)=>{const node=$(id);node.disabled=!visible;if(node.parentElement)node.parentElement.hidden=!visible;};
  show('register-variant',Boolean(policy.variant));
  show('register-dtype',Boolean(policy.dtype));
  show('register-graph',Boolean(policy.graph));
  show('register-voices',Boolean(policy.voices));
  const dml=Boolean(policy.onnxImage&&$('register-device').value==='dml');
  show('register-dml-adapter',dml);show('register-disable-metacommands',dml);
  $('register-submit').disabled=Boolean(policy.cliOnly);$('load-devices').disabled=Boolean(policy.cliOnly);
  setContent($('registration-help'),policy.cliOnly?policy.help:
    backend==='tesseract'?ui("ui.tesseract_help","Choose the folder containing LSTM .traineddata files. Tesseract uses the CPU."):
    task.value==='vad'?ui("ui.vad_help","Choose a Silero-compatible ONNX graph. VAD is CPU-only."):
    policy.onnxImage?ui("ui.onnx_image_help","Choose a prepared SD1 ONNX component directory. DirectML needs a current adapter index and LUID; CPU needs neither."):
    ui("ui.checkpoint_help","Point to a complete local checkpoint with its tokenizer, processor, components or prepared graph. Model weights stay outside Titanos."));
}
// --- broker panel (Cloud and routing) ---
// Rendered only when /api/info carries a broker block and GET /api/keys/self lists broker:read. Every call goes through
// api() to this Titanos server (never to a provider host). Secrets are write-only: the credential input is never
// prefilled, is emptied and replaced right after submit, and a minted key secret lives only in the closure of its
// one-time box (brokerSecretBox) until the box is dismissed or the principal changes; nothing is written to
// localStorage or sessionStorage. A key change (Connect) or a 401/403 resets the panel so no earlier principal's
// rendering survives. When the tab's own key is rotated here, the new secret also becomes the tab's in-memory key
// (exactly as Connect holds it) before the panel refreshes, so the old secret is never sent again.
const BROKER_EMPTY={self:null,status:null,upstreams:[],health:{},credentials:[],aliases:[],virtuals:[],policy:null,keys:[],usage:[],decisions:[],profiles:[],envCandidates:null,envStatus:null,connectResult:null,connectDraft:null,message:null,credentialResult:null};
const brokerState={...BROKER_EMPTY,epoch:0};
let brokerSecretBox=null,lastProvenance=null;const brokerOpenFolds=new Set();
const BROKER_LOCATIONS=['local_only','prefer_local','auto','cloud_only'];
const BROKER_TRIGGERS=['queue_full','rate_limit_error','model_not_loaded','worker_unavailable','model_load_failed','device_unavailable','upstream_rate_limited','upstream_unavailable','upstream_timeout','network','breaker_open'];
const BROKER_SCOPE_CHOICES=['inference','jobs','mcp','models:read','cloud','route:explicit','broker:read','broker:admin','settings:read','audit:read'];
const BROKER_UPSTREAM_EXTRA=['id','ready','readiness','credential_validation','catalog_status','credentials_reset'];
function brokerAllows(scope,self=brokerState.self){const held=self?.scopes||[];return held.includes('admin')||held.includes(scope)||(scope==='broker:read'&&held.includes('broker:admin'));}
function brokerKeyless(){return brokerState.self?.kind==='anonymous_loopback';}
// R29: the keyless Studio principal (standalone role, loopback bind, the X-Titanos-Studio header api() sends) may write
// credentials and connect providers; a keyless server without R29 answers as the plain anonymous principal
function brokerStudio(){return brokerKeyless()&&brokerState.self?.id==='keyless-studio';}
function brokerCanConnect(){return brokerAllows('broker:admin')&&(brokerStudio()||(!brokerKeyless()&&brokerAllows('credentials:write')));}
function brokerSecureOrigin(){const where=globalThis.location||{};const host=String(where.hostname||'').replace(/^\[|\]$/g,'');return where.protocol==='https:'||host==='localhost'||host==='::1'||/^127\./.test(host);}
function routingLocation(){const node=$('location'),label=$('location-label');return node&&label&&!label.hidden?String(node.value||''):'';}
function rememberProvenance(response){
  const get=name=>response?.headers?.get?.(name)||null;const servedBy=get('x-titanos-served-by');if(!servedBy)return;
  lastProvenance={servedBy,servedModel:get('x-titanos-served-model')||'',fallbacks:get('x-titanos-fallback-count')||'0',decision:get('x-titanos-decision-id')||null};
}
function appendProvenance(box){
  if(!box||!lastProvenance)return null;const line=document.createElement('p');line.className='provenance hint';
  setContent(line,ui('broker.provenance','Served by {{served}} · {{model}} · fallbacks {{fallbacks}}',{served:lastProvenance.servedBy,model:lastProvenance.servedModel,fallbacks:lastProvenance.fallbacks}));
  box._provenance={line,...lastProvenance};box.append(line);return line;
}
const EXTERNAL_TASKS=[['chat','chat'],['vision','vision'],['embeddings','embeddings'],['speech','speech'],['transcription','transcription'],['images','image']];
// Model-id hints, checked in order; moderation has no Studio task, so such a row is not listed.
const EXTERNAL_HINTS=[['moderation',/moderation/i],['transcription',/whisper|transcri|scribe|speech-to-text|(^|[-_/])stt([-_.]|$)/i],['speech',/(^|[-_/])tts([-_.]|$)|text-to-speech|^eleven_/i],
  ['embeddings',/embed/i],['images',/dall-e|gpt-image|imagen|flux|stable-diffusion|sdxl/i]];
function catalogEntry(model){
  // Broker catalog rows (cloud, router, virtual) carry no task; Studio derives one from their capabilities.
  if(!model||typeof model!=='object'||typeof model.task==='string'||typeof model.id!=='string')return model;
  const info=model.titanos;if(!info||typeof info!=='object')return model;
  const capabilities=info.class==='virtual'?[info.capability]:(Array.isArray(info.capabilities)?info.capabilities:[]);
  // a static-catalog row carries its profile's whole capability list; a model id that names one modality wins over chat
  const hinted=info.class==='virtual'?null:EXTERNAL_HINTS.find(([capability,pattern])=>capabilities.includes(capability)&&pattern.test(model.id));
  if(hinted&&!EXTERNAL_TASKS.some(([capability])=>capability===hinted[0]))return null;
  const found=hinted?EXTERNAL_TASKS.find(([capability])=>capability===hinted[0]):EXTERNAL_TASKS.find(([capability])=>capabilities.includes(capability));if(!found)return null;
  return {...model,task:found[1],status:info.class==='virtual'||info.status==='available'?'ready':String(info.status||'unknown'),external:{kind:info.class==='virtual'?'virtual':'external',upstream:info.upstream||null}};
}
function externalControls(model){return {...model,capabilities:{backend:'cloud',task:model.task,implemented:[model.task],instruction_edit:null,image_edit:false}};}
function externalBadge(model){
  if(!model?.external)return null;const chip=document.createElement('span');chip.className='model-chip model-badge';
  const upstream=brokerState.upstreams.find(row=>row.id===model.external.upstream),label=upstream?.label||brokerProfileLabel(upstream?.profile||model.external.upstream)||'';
  setContent(chip,model.external.kind==='virtual'?ui('broker.badge_virtual','Virtual'):label||ui('readout.cloud','Cloud'));return chip;
}
function bn(tag,props={},...children){
  const node=document.createElement(tag);
  for(const [name,value] of Object.entries(props)){
    if(value===undefined||value===null)continue;
    if(name==='text')setContent(node,value);else if(name==='cls')node.className=value;
    else if(['type','name','autocomplete','spellcheck','autocapitalize','role','for'].includes(name)){node.setAttribute?.(name,String(value));if(name==='type'||name==='name')node[name]=value;}
    else node[name]=value;
  }
  for(const child of children.flat())if(child!==null&&child!==undefined&&child!==false)node.append(child);
  return node;
}
function brokerField(label,control){return bn('label',{},bn('span',{text:label}),control);}
function brokerSelect(id,values,labeller,value){const node=bn('select',{id});for(const item of values){const choice=bn('option',{value:item,text:labeller?labeller(item):item});node.append(choice);}if(value!==undefined)node.value=value;return node;}
function brokerButton(text,onclick,cls='quiet'){return bn('button',{type:'button',cls,text,onclick});}
function brokerMessage(text,bad=false){brokerState.message={text,bad};const node=$('broker-message');if(node){setContent(node,text);node.className='hint'+(bad?' error':'');}}
function brokerReset(){
  brokerState.epoch++;Object.assign(brokerState,BROKER_EMPTY,{upstreams:[],health:{},credentials:[],aliases:[],virtuals:[],keys:[],usage:[],decisions:[],profiles:[]});brokerSecretBox=null;
  const panel=$('broker-panel'),body=$('broker-body');if(panel)panel.hidden=true;body?.replaceChildren?.();
}
function locationLabel(value){return ui('broker.location_'+value,value);}
async function brokerCall(path,options,done){
  try{const answer=await(await api(path,options)).json();if(done)done(answer);await refreshBroker();return answer;}
  catch(error){brokerMessage(error.message,true);return null;}
}
async function refreshBroker(){
  const panel=$('broker-panel');if(!panel)return;const epoch=++brokerState.epoch;
  const info=serverInfo&&typeof serverInfo.broker==='object'?serverInfo.broker:null,label=$('location-label');
  if(!info){brokerReset();if(label)label.hidden=true;return;}
  if(label)label.hidden=!info.configured;
  let self=null,failure=null;try{self=await(await api('/api/keys/self')).json();}catch(error){failure=error;}
  if(epoch!==brokerState.epoch)return;
  // a transient failure keeps the current rendering; a refused or missing principal clears it
  if(failure&&![401,403,404].includes(failure.status))return;
  if(!self||!brokerAllows('broker:read',self)){brokerReset();return;}
  if(brokerState.self&&brokerState.self.id!==self.id){brokerReset();return refreshBroker();}
  brokerState.self=self;
  const loads=[['status','/api/broker'],['upstreams','/api/broker/upstreams'],['aliases','/api/broker/models'],['virtuals','/api/broker/virtual-models'],['policy','/api/broker/policy'],['usage','/api/broker/usage?group_by=upstream,model'],['decisions','/api/broker/decisions?limit=10']];
  if(brokerAllows('credentials:write')||brokerAllows('broker:admin'))loads.push(['credentials','/api/broker/credentials']);
  if(brokerAllows('keys:admin'))loads.push(['keys','/api/keys']);
  if(brokerAllows('broker:admin'))loads.push(['profiles','/api/broker/profiles']);
  // R30: which well-known credential variable NAMES are set where Titanos runs (names and booleans only)
  let envStatus=null;
  const envLoad=brokerCanConnect()?api('/api/broker/env-candidates').then(response=>response.json()).catch(error=>{envStatus=error?.status||0;return null;}):Promise.resolve(null);
  const [values,env]=await Promise.all([Promise.all(loads.map(([,path])=>api(path).then(response=>response.json()).catch(()=>null))),envLoad]);
  if(epoch!==brokerState.epoch)return;
  loads.forEach(([name],index)=>{const value=values[index];brokerState[name]=Array.isArray(value?.data)?value.data:Array.isArray(value?.rows)?value.rows:value;});
  brokerState.envCandidates=Array.isArray(env?.data)?env.data:null;brokerState.envStatus=envStatus;
  const health=await Promise.all((brokerState.upstreams||[]).map(row=>api('/api/broker/upstreams/'+encodeURIComponent(row.id)+'/health').then(response=>response.json()).catch(()=>null)));
  if(epoch!==brokerState.epoch)return;
  brokerState.health=Object.fromEntries((brokerState.upstreams||[]).map((row,index)=>[row.id,health[index]]));
  panel.hidden=false;renderBroker();
}
// A re-render replaces every node of the panel: keyboard focus inside it is carried to the matching new node, so an Arrow
// on the Provider select (which fires change in Chromium) or a Connect by Enter leaves focus where it was (WCAG 2.4.3, 3.2.2).
// A node is matched by its id, else by its place among the focusable nodes of its nearest ancestor with an id.
const BROKER_FOCUSABLE=new Set(['input','select','textarea','button','summary','a']);
function brokerFocusables(node,out=[]){for(const child of node?.children||[]){const tag=String(child.tagName||'').toLowerCase();if(BROKER_FOCUSABLE.has(tag)||child.getAttribute?.('tabindex')!=null)out.push(child);brokerFocusables(child,out);}return out;}
function brokerWithin(node,body){for(let at=node;at;at=at.parentElement)if(at===body)return true;return false;}
function brokerFocusMark(body){
  const node=typeof document!=='undefined'?document.activeElement:null;if(!node||node===body||!brokerWithin(node,body))return null;
  const mark={id:node.id||'',tag:String(node.tagName||'').toLowerCase(),type:String(node.type||''),start:node.selectionStart,end:node.selectionEnd};
  if(!mark.id){let owner=node.parentElement;while(owner&&owner!==body&&!owner.id)owner=owner.parentElement;const scope=owner&&owner!==body?owner:body;mark.scope=scope===body?'':scope.id;mark.index=brokerFocusables(scope).indexOf(node);}
  return mark;
}
function brokerFocusRestore(body,mark){
  if(!mark)return;const scope=mark.id?null:mark.scope?$(mark.scope):body,node=mark.id?$(mark.id):scope&&mark.index>=0?brokerFocusables(scope)[mark.index]:null;
  if(!node||!brokerWithin(node,body)||String(node.tagName||'').toLowerCase()!==mark.tag||String(node.type||'')!==mark.type)return;
  if(typeof document!=='undefined'&&document.activeElement===node)return;
  node.focus?.({preventScroll:true});
  if(typeof mark.start==='number'&&typeof node.setSelectionRange==='function'){try{node.setSelectionRange(mark.start,mark.end);}catch{}}
}
function renderBroker(){
  const body=$('broker-body');if(!body)return;const admin=brokerAllows('broker:admin'),focusMark=brokerFocusMark(body);
  if(!brokerAllows('keys:admin'))brokerSecretBox=null;
  const message=bn('p',{id:'broker-message',cls:'hint'+(brokerState.message?.bad?' error':''),role:'status'});if(brokerState.message)setContent(message,brokerState.message.text);
  // numbered sub-sections: Connect and the routes stay open; the expert sections fold away and only appear in Advanced
  const sections=[[brokerConnect(),true,false],[brokerUpstreams(admin),true,false],[brokerCredentials(),false,true],[admin?brokerRouting():null,false,true],[admin?brokerPolicy():null,false,true],[brokerActivity(),false,true],[brokerAllows('keys:admin')?brokerKeys():null,false,true]].filter(([section])=>section);
  body.replaceChildren(...[bn('p',{cls:'cloud-lede',text:ui('broker.lede','These routes leave this computer. Titanos sends a request to a provider only when you run one of its models, and only through an upstream enabled here.')}),message,brokerSummary(),
    ...sections.map(([section,open,expert],index)=>fold(section,'07.'+(index+1),open,expert))].filter(Boolean));
  brokerFocusRestore(body,focusMark);
  if(typeof refreshExternalLabels==='function')refreshExternalLabels();
  if(typeof renderCabinet==='function')renderCabinet();
}
function readoutBlock(rows){
  return bn('div',{cls:'readout'},...rows.filter(Boolean).map(([label,value,cls,id])=>bn('p',{cls:'readout-line'},bn('span',{cls:'readout-key',text:label}),bn('span',{cls:'readout-lead'}),bn('span',{cls:'readout-value'+(cls?' '+cls:''),id,text:value}))));
}
function fold(section,index,open=false,expert=false){
  // a numbered, collapsible sub-section; its open state survives the panel's re-renders
  if(!section)return null;const children=[...(section.childNodes||section.children||[])],head=children.find(node=>String(node.tagName).toLowerCase()==='h3');
  const key=section.id||index,box=bn('details',{cls:'broker-section'+(expert?' advanced-only':''),id:section.id||undefined,open:brokerOpenFolds.has(key)||(open&&!brokerOpenFolds.has('closed:'+key))});
  box.ontoggle=()=>{if(box.open){brokerOpenFolds.add(key);brokerOpenFolds.delete('closed:'+key);}else{brokerOpenFolds.delete(key);brokerOpenFolds.add('closed:'+key);}};
  const title=bn('span');setContent(title,head?(textBindings.get(head)||head.textContent):'');
  box.append(bn('summary',{},bn('span',{cls:'sub-index',text:index}),title),...children.filter(node=>node!==head));return box;
}
function brokerSummary(){
  const status=brokerState.status||{},policy=status.cloud_policy||{},vault=status.vault||{},counts=status.upstreams||{};
  return bn('section',{id:'broker-summary',cls:'broker-section'},
    readoutBlock([[ui('broker.k_role','Role'),String(status.role||'standalone'),'','broker-role'],
      [ui('broker.k_policy','Cloud policy'),ui('broker.policy_value','{{effective}} (file {{file}}, ceiling {{ceiling}})',{effective:policy.effective||'off',file:policy.file||'off',ceiling:policy.ceiling||'off'}),(policy.effective||'off')==='off'?'':'ext','broker-policy'],
      [ui('broker.k_vault','Vault'),vault.sealed?ui('broker.vault_value_sealed','{{backend}} · sealed',{backend:vault.backend||'-'}):ui('broker.vault_value_open','{{backend}} · open',{backend:vault.backend||'-'}),vault.sealed?'warn':'','broker-vault'],
      [ui('broker.k_upstreams','Upstreams'),ui('broker.upstream_counts','{{enabled}} of {{total}} enabled · {{healthy}} healthy',{enabled:counts.enabled??0,total:counts.total??0,healthy:counts.healthy??0}),counts.healthy?'ok':''],
      [ui('broker.k_egress','Egress'),String(status.egress||'-')],
      [ui('broker.k_keyless','Keyless use'),brokerState.policy?.anonymous_cloud?ui('broker.keyless_on','Allowed on this computer'):ui('broker.keyless_off','Needs a key with cloud access'),brokerState.policy?.anonymous_cloud?'ext':'']]));
}
function brokerCredentialOf(row){return (brokerState.credentials||[]).find(item=>item.id===row.credential)||null;}
const BROKER_PROFILES={openai:['OpenAI','OA'],anthropic:['Anthropic','AN'],gemini:['Gemini','GE'],elevenlabs:['ElevenLabs','EL'],nvidia_api_catalog:['NVIDIA API catalog','NV'],hf_router:['Hugging Face router','HF'],
  azure_openai:['Azure OpenAI','AZ'],ollama_cloud:['Ollama Cloud','OL'],xai:['xAI','XA'],groq:['Groq','GQ'],mistral:['Mistral','MI'],together:['Together','TG'],deepseek:['DeepSeek','DS'],openrouter:['OpenRouter','OR'],
  fireworks:['Fireworks','FW'],perplexity:['Perplexity','PX'],cerebras:['Cerebras','CB'],sambanova:['SambaNova','SN'],custom_openai:['OpenAI-compatible server','CO'],nim_custom:['NVIDIA NIM (self-hosted)','NM'],
  litellm:['LiteLLM','LL'],hivemind_gateway:['HiveMind gateway','HM'],pair_facade:['PAIR facade','PF']};
function brokerProfileLabel(id){return BROKER_PROFILES[id]?.[0]||String(id||'');}
function hexcap(profile,dim=false){return bn('span',{cls:'hexcap'+(dim?' dim':''),role:'presentation'},bn('span',{text:BROKER_PROFILES[profile]?.[1]||String(profile||'?').slice(0,2).toUpperCase()}));}
function brokerUpstreams(admin){
  const list=bn('div',{id:'broker-upstreams',cls:'tiles',role:'list'});
  if(!(brokerState.upstreams||[]).length)list.append(bn('p',{cls:'hint',text:ui('broker.no_upstreams','No upstreams are configured.')}));
  for(const row of brokerState.upstreams||[]){
    const credential=brokerCredentialOf(row),health=brokerState.health?.[row.id]||{},breaker=health.breaker?.state||health.breaker||'-';
    const open=breaker==='open',online=Boolean(row.enabled)&&row.ready!==false&&!open,validated=credential?.validation==='ok'||credential?.validation==='valid';
    const chip=bn('span',{cls:'chip '+(!row.enabled?'offline':open?'open':online?'online':'offline'),text:!row.enabled?ui('broker.disabled','Disabled'):open?ui('broker.breaker_open','Breaker open'):online?ui('broker.online','Online'):ui('broker.not_ready','Not ready')});
    // a live catalog lists its models at run time: count the ones Studio lists for this upstream
    const listed=(typeof catalog!=='undefined'?catalog:[]).filter(model=>model?.external&&String(model.id).endsWith('@'+row.id)).length,models=Math.max(Array.isArray(row.models)?row.models.length:0,listed);
    const item=bn('div',{cls:'tile broker-item'+(online?' online':''),role:'listitem'},hexcap(row.profile,!online),
      bn('div',{cls:'tile-name'},bn('strong',{text:row.label||(BROKER_PROFILES[row.profile]?brokerProfileLabel(row.profile):row.id)}),chip),
      bn('p',{cls:'tile-meta',text:ui('broker.tile_meta','upstream {{id}} · profile {{profile}}',{id:row.id,profile:row.profile})}),
      readoutBlock([[ui('broker.k_credential','Credential'),credential?(credential.source==='env'&&credential.env_var?ui('broker.credential_env','{{id}} · env {{name}}',{id:credential.id,name:credential.env_var}):ui('broker.credential_kind','{{id}} · {{source}}',{id:credential.id,source:credential.source||'-'})):String(row.credential||'-')],
        [ui('broker.k_fingerprint','Fingerprint'),String(credential?.fingerprint||'-')],
        [ui('broker.k_validation','Validation'),credential?ui('broker.validated_value','{{state}} · {{at}}',{state:String(credential.validation||'-'),at:credential.validated_at?String(credential.validated_at).slice(0,16).replace('T',' '):'-'}):String(row.credential_validation||'-'),validated?'ok':''],
        [ui('broker.k_state','State'),ui('broker.state_value','{{enabled}} · breaker {{breaker}}',{enabled:row.enabled?ui('broker.enabled','Enabled'):ui('broker.disabled','Disabled'),breaker:typeof breaker==='string'?breaker:'-'}),open?'warn':row.enabled?'ok':''],
        [ui('broker.k_catalog','Catalog'),uiCount('broker.catalog_value','{{source}} · {{count}} models','{{source}} · {{count}} model',{source:row.catalog_status?.source||row.catalog||'-',count:models})]]));
    if(admin){
      const toggle=bn('input',{type:'checkbox',checked:Boolean(row.enabled)});
      toggle.onchange=()=>{const spec=brokerUpstreamSpec(row);spec.enabled=Boolean(toggle.checked);return brokerCall('/api/broker/upstreams/'+encodeURIComponent(row.id),{method:'PUT',json:spec});};
      // the label states what the switch is, not what it would do
      item.append(bn('div',{cls:'tile-actions'},bn('label',{cls:'broker-toggle'},toggle,bn('span',{cls:'state-word',text:row.enabled?ui('broker.enabled','Enabled'):ui('broker.disabled','Disabled')})),
        brokerButton(ui('broker.test','Test'),()=>brokerCall('/api/broker/upstreams/'+encodeURIComponent(row.id)+'/test',{method:'POST'},answer=>brokerMessage(answer.ok?ui('broker.test_ok','Test passed in {{ms}} ms',{ms:answer.latency_ms??'-'}):ui('broker.test_failed','Test failed: {{code}}',{code:answer.code||answer.status}),!answer.ok)))));
    }
    list.append(item);
  }
  return bn('section',{cls:'broker-section'},bn('h3',{text:ui('broker.routes','Cloud routes')}),list);
}
// --- R30/R31: Connect a provider ---
function brokerEnvFor(profile){
  const row=(brokerState.envCandidates||[]).find(item=>item&&item.provider===profile);
  return (Array.isArray(row?.variables)?row.variables:[]).filter(item=>item&&typeof item.name==='string').map(item=>({name:item.name,set:Boolean(item.set)}));
}
function brokerProfiles(){
  const listed=Array.isArray(brokerState.profiles)&&brokerState.profiles.length?brokerState.profiles:Object.keys(BROKER_PROFILES).map(id=>({id,kind:'cloud',auth:'bearer'}));
  const order=Object.keys(BROKER_PROFILES),rank=id=>{const at=order.indexOf(id);return at<0?order.length:at;};
  return listed.filter(row=>row&&typeof row.id==='string'&&!['slot'].includes(row.kind)&&row.adapter!=='none').sort((a,b)=>rank(a.id)-rank(b.id)||a.id.localeCompare(b.id));
}
// a write-only field is emptied and swapped for a fresh one on submit; when it held focus (Enter pressed in it), the fresh
// field takes focus before the browser can drop it to <body>, so the re-render's focus restore finds it again (WCAG 2.4.3)
function brokerSwapSecretField(field,fresh){
  const held=typeof document!=='undefined'&&document.activeElement===field;field.value='';field.replaceWith?.(fresh);
  if(held&&fresh.parentElement)fresh.focus?.({preventScroll:true});return fresh;
}
function brokerConnectSecretInput(){const input=bn('input',{id:'broker-connect-secret',type:'password',autocomplete:'off',spellcheck:'false',autocapitalize:'off'});input.setAttribute?.('aria-labelledby','broker-connect-secret-label');input.value='';return input;}
function brokerConnect(){
  if(!brokerAllows('broker:admin'))return null;
  const section=bn('section',{id:'broker-connect',cls:'broker-section'},bn('h3',{text:ui('broker.connect_title','Connect a provider')}));
  if(!brokerCanConnect()){
    // a keyless server without the R29 Studio principal, or a key without credentials:write
    section.append(bn('p',{id:'broker-connect-cli',cls:'cli-note',text:brokerKeyless()?ui('broker.connect_cli_keyless','This server does not accept provider setup from Studio without a key. On this computer run: titanos broker connect <provider> --from-env NAME (or --from-stdin to paste a key).'):ui('broker.connect_cli_scope','Connecting a provider needs a key with credentials:write. On this computer run: titanos broker connect <provider> --from-env NAME (or --from-stdin to paste a key).')}));
    return section;
  }
  const secure=brokerSecureOrigin(),profiles=brokerProfiles(),configured=new Map((brokerState.upstreams||[]).map(row=>[row.profile,row]));
  const usable=id=>{const row=configured.get(id),profile=profiles.find(item=>item.id===id)||{};return Boolean(row&&row.enabled&&row.ready!==false&&(row.credential||profile.auth==='none'));};
  // enabled but not ready: the key is not validated, or the server cannot read it (an env: variable it does not hold)
  const unready=id=>{const row=configured.get(id);return Boolean(row&&row.enabled&&!usable(id));};
  const draft=brokerState.connectDraft||{},found=id=>brokerEnvFor(id).find(item=>item.set);
  const preferred=profiles.find(row=>found(row.id)&&!configured.has(row.id))||profiles.find(row=>!configured.has(row.id))||profiles[0];
  const chosenId=profiles.some(row=>row.id===draft.provider)?draft.provider:preferred?.id;const profile=profiles.find(row=>row.id===chosenId)||{};
  const provider=bn('select',{id:'broker-connect-provider'});
  for(const row of profiles){const key=found(row.id),done=usable(row.id),idle=configured.has(row.id)&&!done;
    provider.append(bn('option',{value:row.id,text:done?ui('broker.connect_option_done','{{name}} · connected',{name:brokerProfileLabel(row.id)}):key?ui('broker.connect_option_env','{{name}} · {{variable}} found',{name:brokerProfileLabel(row.id),variable:key.name}):idle&&unready(row.id)?ui('broker.connect_option_unready','{{name}} · configured, not ready',{name:brokerProfileLabel(row.id)}):idle?ui('broker.connect_option_idle','{{name}} · configured, not enabled',{name:brokerProfileLabel(row.id)}):brokerProfileLabel(row.id)}));}
  provider.value=chosenId||'';
  provider.onchange=()=>{brokerState.connectDraft={provider:provider.value};renderBroker();};
  const variables=brokerEnvFor(chosenId),discovery=Array.isArray(brokerState.envCandidates),keyFound=variables.find(item=>item.set);
  const needsKey=!(profile.auth==='none'),optional=Boolean(profile.auth_optional)||profile.auth==='none';
  const sourceDefault=draft.source||(keyFound?'env':needsKey?'value':'none');
  const radios=[],radio=value=>{const node=bn('input',{type:'radio',name:'broker-connect-source',value,checked:sourceDefault===value});node.onchange=()=>{brokerState.connectDraft={provider:chosenId,source:value};};radios.push(node);return node;};
  // a variable known to the server is chosen by NAME; the page never sees its value
  let env;
  if(discovery&&variables.length){env=bn('select',{id:'broker-connect-env'});for(const item of variables)env.append(bn('option',{value:item.name,disabled:!item.set,text:item.set?ui('broker.env_set','{{name}} · set',{name:item.name}):ui('broker.env_unset','{{name}} · not set',{name:item.name})}));env.value=(keyFound||variables[0]).name;}
  else{env=bn('input',{id:'broker-connect-env',autocomplete:'off',spellcheck:'false',autocapitalize:'off'});setUiAttribute(env,'placeholder',ui('broker.env_placeholder','Variable name, e.g. OPENAI_API_KEY'));}
  const envHint=discovery?(variables.length?(keyFound?ui('broker.env_hint_found','Titanos stores a reference to {{name}}, never its value.',{name:keyFound.name}):ui('broker.env_hint_missing','None of these is set where Titanos runs. Set one and restart, or paste a key.')):ui('broker.env_hint_none','This provider has no well-known variable; type the name of one that is set where Titanos runs.'))
    :brokerState.envStatus===404||brokerState.envStatus===405?ui('broker.env_missing_route','This server does not report environment keys (GET /api/broker/env-candidates answered {{status}}). Type a variable name, or paste a key.',{status:brokerState.envStatus}):ui('broker.env_unavailable','Environment keys could not be listed. Type a variable name, or paste a key.');
  // each option: the radio in its own label; the select or key field beside it is named by the option text (aria-labelledby).
  // Focus alone never changes the chosen source (WCAG 2.1.1, 3.2.1): Tab passes through every field on the way to Connect.
  // Only a deliberate edit of an option's own field (typing a key, picking a variable) selects that option.
  const choice=(value,id,text,...rest)=>{const node=radio(value);const caption=bn('span',{id,text});
    for(const control of rest)if(control&&['select','input'].includes(String(control.tagName).toLowerCase()))control.setAttribute?.('aria-labelledby',id);
    const box=bn('div',{cls:'connect-option'},bn('label',{cls:'connect-choice'},node,caption),...rest);
    // delegated on the option, so a key field replaced after a submit still selects its option when typed into
    const pick=event=>{if(event?.target!==node&&!node.checked){node.checked=true;node.onchange?.();}};
    box.addEventListener?.('input',pick);box.addEventListener?.('change',pick);
    return box;};
  const options=[choice('env','broker-connect-env-label',ui('broker.source_env_choice','Use a key from this server\'s environment'),env,bn('p',{cls:'hint',text:envHint})),
    choice('value','broker-connect-secret-label',ui('broker.source_value_choice','Paste a key (sent once, never shown again)'),brokerConnectSecretInput())];
  if(optional)options.push(choice('none','broker-connect-none-label',ui('broker.source_none_choice','No key'),bn('p',{cls:'hint',text:ui('broker.source_none_hint','This profile can run without a credential.')})));
  const base=bn('input',{id:'broker-connect-base-url',type:'url',autocomplete:'off',spellcheck:'false'});setUiAttribute(base,'placeholder',profile.default_base_url?ui('broker.base_default','Default: {{url}}',{url:profile.default_base_url}):ui('broker.base_required','Required for this profile, e.g. https://host/v1'));
  const anonymous=bn('input',{id:'broker-connect-anonymous',type:'checkbox',checked:!brokerState.policy?.anonymous_cloud});
  const aliases=bn('input',{id:'broker-connect-aliases',type:'checkbox',checked:false});
  const form=bn('form',{id:'broker-connect-form',cls:'connect-form'},
    bn('label',{},bn('span',{text:ui('broker.connect_provider','Provider')}),provider,bn('span',{cls:'hint',text:usable(chosenId)?ui('broker.connect_update','Connected as upstream {{id}}. Connecting again replaces its credential and refreshes its models.',{id:configured.get(chosenId).id}):unready(chosenId)?ui('broker.connect_unready','Upstream {{id}} is enabled but not ready: its key is not validated, or the server cannot read it from its own environment. Connecting checks the key from the server, binds it and refreshes its models.',{id:configured.get(chosenId).id}):configured.has(chosenId)?ui('broker.connect_idle','Upstream {{id}} is configured but not enabled. Connecting checks the key, binds it and enables the upstream.',{id:configured.get(chosenId).id}):ui('broker.connect_new','Titanos creates the upstream from the provider profile, checks the key, lists its models and enables it.')})),
    bn('fieldset',{cls:'connect-source'},bn('legend',{text:ui('broker.connect_source','Key')}),...options),
    bn('label',{cls:'connect-full'+(profile.default_base_url?' advanced-only':'')},bn('span',{text:profile.default_base_url?ui('broker.base_optional','Base URL (optional)'):ui('broker.base_url','Base URL')}),base),
    brokerStudio()&&!brokerState.policy?.anonymous_cloud?bn('label',{cls:'check connect-full'},anonymous,bn('span',{text:ui('broker.connect_anonymous','Let Studio and other keyless programs on this computer use this provider (otherwise a key with cloud access is needed)')})):null,
    bn('label',{cls:'check connect-full advanced-only'},aliases,bn('span',{text:ui('broker.connect_aliases','Also publish the provider\'s model ids as plain aliases')})),
    bn('button',{type:'submit',cls:'go',text:ui('broker.connect','Connect')}));
  if(!secure){form.append(bn('p',{id:'broker-connect-insecure',cls:'hint warning',text:ui('broker.insecure','Credential writes need https or a loopback address; this page is neither, so the form is disabled.')}));for(const node of [provider,env,base,anonymous,aliases])node.disabled=true;}
  form.onsubmit=event=>{event?.preventDefault?.();return brokerConnectSubmit({secure,provider,env,base,anonymous,aliases,radios});};
  section.append(bn('p',{cls:'hint',text:ui('broker.connect_hint','Pick a provider and give Titanos its key. A key found in this server\'s environment is used by name; its value never reaches this page.')}),form);
  const result=brokerConnectResult();if(result)section.append(result);
  return section;
}
async function brokerConnectSubmit({secure,provider,env,base,anonymous,aliases,radios=[]}){
  const field=$('broker-connect-secret');let value=field?String(field.value||''):'';
  if(field)brokerSwapSecretField(field,brokerConnectSecretInput());
  if(!secure){value='';return brokerMessage(ui('broker.insecure','Credential writes need https or a loopback address; this page is neither, so the form is disabled.'),true);}
  const source=radios.find(node=>node.checked)?.value||brokerState.connectDraft?.source||'value';
  const body={provider:String(provider.value||''),enable:true};
  if(!body.provider){value='';return brokerMessage(ui('broker.connect_pick','Pick a provider first.'),true);}
  if(source==='env'){const name=String(env.value||'').trim();value='';if(!name)return brokerMessage(ui('broker.blank_unchanged','The value was blank; nothing changed.'));body.credential={env:name};}
  else if(source==='value'){if(!value)return brokerMessage(ui('broker.blank_unchanged','The value was blank; nothing changed.'));body.credential={value};}
  value='';
  if(String(base.value||'').trim())body.base_url=String(base.value).trim();
  if(brokerStudio()&&!brokerState.policy?.anonymous_cloud&&anonymous.checked)body.anonymous_cloud=true;
  if(aliases.checked)body.aliases=true;
  brokerMessage(ui('broker.connecting','Connecting {{provider}}: checking the key and listing models…',{provider:brokerProfileLabel(body.provider)}));
  let response=null;
  try{response=await api('/api/broker/connect',{method:'POST',json:body});}
  catch(error){
    delete body.credential;const missing=error?.status===404||error?.status===405;
    brokerState.connectResult={provider:body.provider,ok:false,text:missing?ui('broker.connect_missing','This server has no one-step connect (POST /api/broker/connect answered {{status}}). Update Titanos, or on this computer run: titanos broker connect {{provider}} --from-env NAME',{status:error.status,provider:body.provider}):String(error?.message||error)};
    brokerMessage(brokerState.connectResult.text,true);await refreshBroker().catch(()=>{});return null;
  }
  delete body.credential;
  let answer=null;try{answer=await response.json();}catch{}
  if(!answer||typeof answer!=='object'||!answer.upstream){brokerState.connectResult={provider:body.provider,ok:false,text:ui('broker.connect_unexpected','The server answered, but not with a connect result; nothing is shown as connected.')};brokerMessage(brokerState.connectResult.text,true);await refreshBroker().catch(()=>{});return null;}
  // only the summary is kept: the answer never carries a secret, and the pasted value is already gone
  const models=Array.isArray(answer.models)?answer.models.filter(item=>typeof item==='string'):[];
  brokerState.connectResult={provider:body.provider,ok:true,upstream:{id:String(answer.upstream.id||body.provider),enabled:Boolean(answer.upstream.enabled)},validation:String(answer.validation||'-'),
    policy:answer.cloud_policy&&typeof answer.cloud_policy==='object'?{effective:String(answer.cloud_policy.effective||'off'),ceiling:String(answer.cloud_policy.ceiling||'off')}:null,anonymous:Boolean(answer.anonymous_cloud),
    models:models.slice(0,200),total:Number.isFinite(answer.models_total)?answer.models_total:models.length,aliases:Array.isArray(answer.aliases)?answer.aliases.length:0,
    credential:answer.credential&&typeof answer.credential==='object'?{fingerprint:String(answer.credential.fingerprint||''),source:String(answer.credential.source||''),env:answer.credential.env_var?String(answer.credential.env_var):null}:null};
  const result=brokerState.connectResult,upstreamId=result.upstream.id;
  result.steps=[result.policy?.effective==='off'?ui('broker.step_ceiling','Cloud calls are refused while the cloud policy or its ceiling is off. Raise them to use {{upstream}}.',{upstream:upstreamId}):null,
    !result.anonymous&&brokerKeyless()?ui('broker.step_keyless','Keyless programs on this computer, Studio included, reach {{upstream}} only when keyless cloud use is allowed (policy, Advanced).',{upstream:upstreamId}):null,
    result.total&&!result.aliases?ui('broker.step_call','Its models are listed under Models with their full ids, for example {{example}}.',{example:`${result.models[0]||'model'}@${upstreamId}`}):null].filter(Boolean);
  brokerState.connectDraft=null;
  brokerMessage(uiCount('broker.connected','Connected {{provider}}: {{count}} models found. They are listed under Models, marked External.','Connected {{provider}}: {{count}} model found. It is listed under Models, marked External.',{provider:brokerProfileLabel(body.provider),count:brokerState.connectResult.total}));
  await refreshBroker().catch(()=>{});if(typeof refresh==='function')await refresh(true).catch(()=>{});
  return answer;
}
function brokerConnectResult(){
  const result=brokerState.connectResult;if(!result)return null;
  if(!result.ok)return bn('div',{id:'broker-connect-result',cls:'connect-result failed',role:'status'},bn('h4',{text:ui('broker.connect_failed','{{provider}} is not connected',{provider:brokerProfileLabel(result.provider)})}),bn('p',{cls:'hint error',text:result.text}));
  const shown=result.models.slice(0,24),more=result.total-shown.length;
  return bn('div',{id:'broker-connect-result',cls:'connect-result',role:'status'},bn('h4',{text:ui('broker.connect_done','{{provider}} connected',{provider:brokerProfileLabel(result.provider)})}),
    readoutBlock([[ui('broker.k_upstream','Upstream'),ui('broker.upstream_enabled','{{id}} · {{state}}',{id:result.upstream.id,state:result.upstream.enabled?ui('broker.enabled','Enabled'):ui('broker.disabled','Disabled')}),result.upstream.enabled?'ok':'warn'],
      [ui('broker.k_validation','Validation'),result.validation,result.validation==='ok'?'ok':''],
      result.credential?[ui('broker.k_credential','Credential'),result.credential.env?ui('broker.credential_ref','env {{name}} · {{fingerprint}}',{name:result.credential.env,fingerprint:result.credential.fingerprint}):ui('broker.credential_vault','{{source}} · {{fingerprint}}',{source:result.credential.source||'vault',fingerprint:result.credential.fingerprint})]:null,
      result.policy?[ui('broker.k_policy','Cloud policy'),ui('broker.policy_effective','{{effective}} (ceiling {{ceiling}})',{effective:result.policy.effective,ceiling:result.policy.ceiling}),result.policy.effective==='off'?'warn':'ext']:null,
      [ui('broker.k_keyless','Keyless use'),result.anonymous?ui('broker.keyless_on','Allowed on this computer'):ui('broker.keyless_off','Needs a key with cloud access')],
      [ui('broker.k_models','Models'),ui('broker.models_found','{{count}} found',{count:result.total}),result.total?'ext':'']]),
    shown.length?bn('ul',{cls:'connect-models'},...shown.map(name=>bn('li',{text:name})),more>0?bn('li',{text:ui('broker.models_more','+{{count}} more',{count:more})}):null):null,
    ...result.steps.map(step=>bn('p',{cls:'hint',text:step})));
}
function brokerCredentials(){
  const section=bn('section',{cls:'broker-section'},bn('h3',{text:ui('broker.credentials','Add or replace credential')}));
  if(brokerKeyless()&&!brokerStudio()){
    section.append(bn('p',{id:'broker-keyless-note',cls:'hint',text:ui('broker.keyless_note','Keyless mode cannot write credentials from Studio. On this computer run: titanos broker credential set <id> --upstream <upstream> (the value is read from a hidden prompt), or create a key with titanos keys create to enable this form.')}));
    return section;
  }
  if(!brokerAllows('credentials:write'))return null;
  const secure=brokerSecureOrigin(),upstreams=(brokerState.upstreams||[]).map(row=>row.id);
  const upstream=brokerSelect('broker-credential-upstream',upstreams,null,upstreams.includes(brokerState.credentialResult?.upstream)?brokerState.credentialResult.upstream:upstreams[0]);
  const id=bn('input',{id:'broker-credential-id',autocomplete:'off',spellcheck:'false'});setUiAttribute(id,'placeholder',ui('broker.credential_id_hint','credential id, e.g. openai-key'));
  const expected=bn('p',{id:'broker-credential-expected',cls:'hint'});
  const showExpected=()=>setContent(expected,upstream.value?ui('broker.credential_expected','Leave the id blank to use {{id}}.',{id:brokerExpectedCredential(upstream.value)}):'');showExpected();upstream.onchange=showExpected;
  const source=brokerSelect('broker-credential-source',['secret','env','file'],value=>ui('broker.source_'+value,value),'secret');
  const env=bn('input',{id:'broker-credential-env',autocomplete:'off',spellcheck:'false'}),file=bn('input',{id:'broker-credential-file',autocomplete:'off',spellcheck:'false'});
  const result=bn('pre',{id:'broker-credential-result'}),secret=brokerSecretInput();
  if(brokerState.credentialResult)setContent(result,JSON.stringify(brokerState.credentialResult.meta,null,2));
  const form=bn('form',{id:'broker-credential-form',cls:'broker-form'},brokerField(ui('broker.upstream','Upstream'),upstream),brokerField(ui('broker.credential_id','Credential id'),id),expected,
    brokerField(ui('broker.source','Source'),source),brokerField(ui('broker.secret','Secret value (write-only)'),secret),
    brokerField(ui('broker.env','Environment variable on the server host'),env),brokerField(ui('broker.file','Owner-only file path on the server host'),file),
    bn('div',{cls:'form-actions'},bn('button',{type:'submit',cls:'structural',text:ui('broker.save','Save credential')}),brokerButton(ui('broker.remove_credential','Remove credential'),()=>brokerRemoveCredential(id.value.trim()),'quiet danger')),result);
  if(!secure){section.append(bn('p',{id:'broker-insecure',cls:'hint warning',text:ui('broker.insecure','Credential writes need https or a loopback address; this page is neither, so the form is disabled.')}));for(const node of [upstream,id,source,env,file,secret])node.disabled=true;}
  form.onsubmit=event=>{event?.preventDefault?.();return brokerSaveCredential({secure,upstream,id,source,env,file,result});};
  section.append(form);return section;
}
function brokerUpstream(id){return (brokerState.upstreams||[]).find(row=>row.id===id)||null;}
function brokerExpectedCredential(upstreamId){return brokerUpstream(upstreamId)?.credential||(upstreamId+'-key');}
function brokerUpstreamSpec(row){return Object.fromEntries(Object.entries(row).filter(([name])=>!BROKER_UPSTREAM_EXTRA.includes(name)));}
async function brokerBindCredential(upstreamId,name){
  // a saved credential serves nothing until its upstream references it; bind it when this principal may edit upstreams
  const row=brokerUpstream(upstreamId);if(!row||row.credential===name)return null;
  if(!brokerAllows('broker:admin'))return ui('broker.bind_needed','Saved. Upstream {{upstream}} does not use {{id}} yet; a key with broker:admin must bind it.',{upstream:upstreamId,id:name});
  if(row.credential&&!globalThis.confirm?.(String(ui('broker.confirm_bind','Upstream {{upstream}} uses credential {{current}}. Switch it to {{id}}?',{upstream:upstreamId,current:row.credential,id:name}))))
    return ui('broker.bind_kept','Saved. Upstream {{upstream}} still uses {{current}}.',{upstream:upstreamId,current:row.credential});
  await api('/api/broker/upstreams/'+encodeURIComponent(upstreamId),{method:'PUT',json:{...brokerUpstreamSpec(row),credential:name}});
  return ui('broker.bound','Saved and bound to {{upstream}}. The credential value is not shown again.',{upstream:upstreamId});
}
function brokerSecretInput(){
  const input=bn('input',{id:'broker-secret',type:'password',autocomplete:'off',spellcheck:'false',autocapitalize:'off'});input.value='';return input;
}
async function brokerSaveCredential({secure,upstream,id,source,env,file,result}){
  const field=$('broker-secret');let value=field?String(field.value||''):'';
  if(field)brokerSwapSecretField(field,brokerSecretInput());
  if(!secure){value='';return brokerMessage(ui('broker.insecure','Credential writes need https or a loopback address; this page is neither, so the form is disabled.'),true);}
  const kind=source.value,body={upstream:upstream.value},name=id.value.trim()||brokerExpectedCredential(upstream.value);
  if(kind==='secret'){if(!value){return brokerMessage(ui('broker.blank_unchanged','The value was blank; nothing changed.'));}body.secret=value;}
  else if(kind==='env'){if(!env.value.trim())return brokerMessage(ui('broker.blank_unchanged','The value was blank; nothing changed.'));body.env=env.value.trim();}
  else{if(!file.value.trim())return brokerMessage(ui('broker.blank_unchanged','The value was blank; nothing changed.'));body.file=file.value.trim();}
  value='';
  try{
    const response=await api('/api/broker/credentials/'+encodeURIComponent(name),{method:'PUT',json:body});
    delete body.secret;const meta=await response.json(),shown={id:meta.id,fingerprint:meta.fingerprint,validated_at:meta.validated_at,validation:meta.validation};
    brokerState.credentialResult={upstream:body.upstream,meta:shown};setContent(result,JSON.stringify(shown,null,2));
    brokerMessage(ui('broker.saved','Saved. The credential value is not shown again.'));
    try{const bound=await brokerBindCredential(body.upstream,name);if(bound)brokerMessage(bound);}catch(error){brokerMessage(error.message,true);}
  }catch(error){delete body.secret;brokerMessage(error.message,true);}
  await refreshBroker().catch(()=>{});
}
async function brokerRemoveCredential(name){
  if(!name)return brokerMessage(ui('broker.credential_id_needed','Enter the credential id to remove.'),true);
  if(!globalThis.confirm?.(String(ui('broker.confirm_remove','Remove credential {{id}}? Upstreams using it stop working.',{id:name}))))return;
  return brokerCall('/api/broker/credentials/'+encodeURIComponent(name),{method:'DELETE'},()=>brokerMessage(ui('broker.removed','Removed.')));
}
function brokerRouting(){
  const aliases=bn('div',{id:'broker-aliases',cls:'broker-list'});
  for(const row of brokerState.aliases||[])aliases.append(bn('div',{cls:'broker-item'},bn('span',{text:ui('broker.alias_row','{{name}} → {{target}}',{name:row.name,target:row.target})}),brokerButton(ui('broker.remove','Remove'),()=>globalThis.confirm?.(String(ui('broker.confirm_alias','Remove {{name}}?',{name:row.name})))&&brokerCall('/api/broker/models/'+encodeURIComponent(row.name),{method:'DELETE'}),'quiet danger')));
  const aliasName=bn('input',{id:'broker-alias-name'}),aliasTarget=bn('input',{id:'broker-alias-target'});setUiAttribute(aliasTarget,'placeholder',ui('broker.target_hint','model@upstream'));
  const aliasForm=bn('form',{id:'broker-alias-form',cls:'broker-form'},brokerField(ui('broker.alias_name','Published name'),aliasName),brokerField(ui('broker.alias_target','Target'),aliasTarget),bn('button',{type:'submit',cls:'quiet',text:ui('broker.save_alias','Save alias')}));
  aliasForm.onsubmit=event=>{event?.preventDefault?.();return brokerCall('/api/broker/models/'+encodeURIComponent(aliasName.value.trim()),{method:'PUT',json:{target:aliasTarget.value.trim()}});};
  const virtuals=bn('div',{id:'broker-virtuals',cls:'broker-list'});
  for(const row of brokerState.virtuals||[])virtuals.append(bn('div',{cls:'broker-item'},bn('span',{text:ui('broker.virtual_row','{{name}} · {{strategy}} · {{targets}}',{name:row.name,strategy:row.strategy,targets:(row.targets||[]).map(t=>t.ref).join(', ')})}),brokerButton(ui('broker.remove','Remove'),()=>globalThis.confirm?.(String(ui('broker.confirm_alias','Remove {{name}}?',{name:row.name})))&&brokerCall('/api/broker/virtual-models/'+encodeURIComponent(row.name),{method:'DELETE'}),'quiet danger')));
  const name=bn('input',{id:'broker-virtual-name'}),capability=brokerSelect('broker-virtual-capability',['chat','embeddings','speech','transcription','images','moderation'],null,'chat');
  const strategy=brokerSelect('broker-virtual-strategy',['ordered','weighted','cheapest','lowest_latency'],value=>ui('broker.strategy_'+value,value),'ordered');
  const targets=bn('textarea',{id:'broker-virtual-targets',rows:3});setUiAttribute(targets,'placeholder',ui('broker.targets_hint','One target per line in order, e.g. qwen3-8b then gpt-4o-mini@openai'));
  const attempts=bn('input',{id:'broker-virtual-attempts',type:'number',min:1,max:6});
  const triggers=bn('fieldset',{id:'broker-virtual-fallback',cls:'broker-checks'},bn('legend',{text:ui('broker.fallback_on','Fall back on')}),...BROKER_TRIGGERS.map(code=>bn('label',{},bn('input',{type:'checkbox',value:code,checked:true}),bn('span',{text:code}))));
  const virtualForm=bn('form',{id:'broker-virtual-form',cls:'broker-form'},brokerField(ui('broker.virtual_name','Virtual model name'),name),brokerField(ui('broker.capability','Capability'),capability),brokerField(ui('broker.strategy','Strategy'),strategy),brokerField(ui('broker.targets','Ordered targets'),targets),triggers,brokerField(ui('broker.max_attempts','Maximum attempts'),attempts),bn('button',{type:'submit',cls:'quiet',text:ui('broker.save_virtual','Save virtual model')}));
  virtualForm.onsubmit=event=>{
    event?.preventDefault?.();const refs=String(targets.value||'').split(/\r?\n/).map(line=>line.trim()).filter(Boolean);
    const spec={capability:capability.value,strategy:strategy.value,targets:refs.map(ref=>({ref})),fallback_on:[...(triggers.children||[])].map(node=>node.children?.[0]).filter(box=>box?.checked&&box.value).map(box=>box.value)};
    if(attempts.value)spec.max_attempts=Number(attempts.value);
    return brokerCall('/api/broker/virtual-models/'+encodeURIComponent(name.value.trim()),{method:'PUT',json:spec});
  };
  return bn('section',{cls:'broker-section'},bn('h3',{text:ui('broker.routing','Aliases and virtual models')}),aliases,aliasForm,virtuals,virtualForm);
}
function brokerPolicy(){
  const policy=brokerState.policy||{},order=['off','explicit','fallback'],ceiling=order.indexOf(policy.ceiling||'fallback');
  const cloud=brokerSelect('broker-cloud-policy',order.slice(0,Math.max(ceiling,0)+1),value=>ui('broker.cloud_'+value,value),policy.cloud_policy||'off');
  const placement=brokerSelect('broker-default-location',BROKER_LOCATIONS,locationLabel,policy.default_location||'auto');
  const anonymous=bn('input',{id:'broker-anonymous-cloud',type:'checkbox',checked:Boolean(policy.anonymous_cloud),disabled:brokerKeyless()&&!brokerStudio()});
  const lan=bn('input',{id:'broker-allow-http-lan',type:'checkbox',checked:Boolean(policy.allow_http_lan),disabled:brokerKeyless()});
  const form=bn('form',{id:'broker-policy-form',cls:'broker-form'},brokerField(ui('broker.cloud_policy','Cloud policy'),cloud),brokerField(ui('broker.default_location','Default placement'),placement),
    bn('label',{cls:'broker-toggle'},anonymous,bn('span',{text:ui('broker.anonymous_cloud','Allow cloud for keyless loopback callers')})),
    bn('label',{cls:'broker-toggle'},lan,bn('span',{text:ui('broker.allow_http_lan','Allow plain http to LAN upstreams')})),
    brokerKeyless()?bn('p',{cls:'hint',text:brokerStudio()?ui('broker.studio_policy','Plain http to LAN upstreams is changed only from the CLI or with a key; edit broker.json on the host.'):ui('broker.keyless_policy','Keyless callers cannot change these two switches; edit broker.json on the host.')}):null,
    bn('button',{type:'submit',cls:'quiet',text:ui('broker.save_policy','Save policy')}));
  form.onsubmit=event=>{event?.preventDefault?.();const body={cloud_policy:cloud.value,default_location:placement.value,anonymous_cloud:Boolean(anonymous.checked),allow_http_lan:Boolean(lan.checked),max_fallbacks:policy.max_fallbacks??2,auto_model:policy.auto_model??null};return brokerCall('/api/broker/policy',{method:'PUT',json:body});};
  return bn('section',{cls:'broker-section'},bn('h3',{text:ui('broker.policy','Policy')}),form);
}
function brokerTable(id,columns,rows,empty=null){
  const table=bn('table',{id,cls:'broker-table'}),head=bn('tr');for(const [,label] of columns)head.append(bn('th',{text:label}));table.append(head);
  if(!rows.length&&empty){const cell=bn('td',{cls:'empty',text:empty,colSpan:columns.length});table.append(bn('tr',{},cell));}
  for(const row of rows){const line=bn('tr');for(const [name] of columns)line.append(bn('td',{text:String(typeof name==='function'?name(row):row?.[name]??'-')}));table.append(line);}
  return bn('div',{cls:'broker-scroll'},table);
}
function brokerActivity(){
  const usage=brokerTable('broker-usage',[['upstream',ui('broker.col_upstream','Upstream')],['model',ui('broker.col_model','Model')],['requests',ui('broker.col_requests','Requests')],['cost_usd',ui('broker.col_cost','Cost (USD)')]],brokerState.usage||[],ui('broker.usage_empty','No cloud requests recorded yet.'));
  const decisions=brokerTable('broker-decisions',[['requested_model',ui('broker.col_model','Model')],[row=>row.served?.ref||'-',ui('broker.col_served','Served by')],['fallback_count',ui('broker.col_fallbacks','Fallbacks')],['ts',ui('broker.col_time','Time')]],brokerState.decisions||[],ui('broker.decisions_empty','No routing decisions recorded yet.'));
  const model=bn('input',{id:'broker-explain-model'}),output=bn('pre',{id:'broker-explain-result'});
  const form=bn('form',{id:'broker-explain-form',cls:'broker-form'},brokerField(ui('broker.explain_model','Model to explain'),model),bn('button',{type:'submit',cls:'quiet',text:ui('broker.explain','Explain routing')}),output);
  form.onsubmit=async event=>{
    event?.preventDefault?.();
    try{const decision=await(await api('/api/broker/explain',{method:'POST',json:{surface:'openai.chat',body:{model:model.value.trim(),messages:[{role:'user',content:'.'}]}}})).json();
      setContent(output,JSON.stringify({chain:decision.chain,candidates:(decision.candidates||[]).map(c=>({ref:c.ref,eligible:c.eligible,excluded:c.excluded})),effective:decision.effective},null,2));}
    catch(error){setContent(output,error.message);}
  };
  return bn('section',{cls:'broker-section'},bn('h3',{text:ui('broker.activity','Usage and recent decisions')}),bn('p',{cls:'hint',text:ui('broker.metadata_only','Metadata only; prompts and outputs are never stored.')}),usage,decisions,form);
}
function brokerKeys(){
  const list=bn('div',{id:'broker-key-list',cls:'broker-list'});
  for(const row of brokerState.keys||[]){
    const item=bn('div',{cls:'broker-item'},bn('strong',{text:row.name||row.id}),bn('p',{cls:'hint',text:ui('broker.key_meta','{{id}} · {{status}} · {{scopes}}',{id:row.id,status:row.status||'-',scopes:(row.scopes||[]).join(', ')})}));
    if(row.status!=='revoked')item.append(
      brokerButton(ui('broker.rotate','Rotate'),()=>globalThis.confirm?.(String(ui('broker.confirm_rotate','Rotate {{id}}? The current secret stops working at once.',{id:row.id})))&&brokerKeySecret('/api/keys/'+encodeURIComponent(row.id)+'/rotate',{method:'POST'})),
      brokerButton(ui('broker.revoke','Revoke'),()=>globalThis.confirm?.(String(ui('broker.confirm_revoke','Revoke {{id}}? This cannot be undone.',{id:row.id})))&&brokerCall('/api/keys/'+encodeURIComponent(row.id)+'/revoke',{method:'POST'}),'quiet danger'));
    list.append(item);
  }
  const name=bn('input',{id:'broker-key-name'}),models=bn('input',{id:'broker-key-models'}),rpm=bn('input',{id:'broker-key-rpm',type:'number',min:1}),budget=bn('input',{id:'broker-key-budget'}),days=bn('input',{id:'broker-key-days',type:'number',min:1});
  setUiAttribute(models,'placeholder',ui('broker.models_hint','Model globs, comma separated (default all)'));
  const scopes=bn('fieldset',{id:'broker-key-scopes',cls:'broker-checks'},bn('legend',{text:ui('broker.scopes','Scopes')}),...BROKER_SCOPE_CHOICES.map(scope=>bn('label',{},bn('input',{type:'checkbox',value:scope,checked:scope==='inference'}),bn('span',{text:scope}))));
  const form=bn('form',{id:'broker-key-form',cls:'broker-form'},brokerField(ui('broker.key_name','Key name'),name),scopes,brokerField(ui('broker.key_models','Models'),models),brokerField(ui('broker.key_rpm','Requests per minute'),rpm),brokerField(ui('broker.key_budget','Budget per day (USD)'),budget),brokerField(ui('broker.key_days','Expires after days'),days),bn('button',{type:'submit',cls:'quiet',text:ui('broker.create_key','Create key')}));
  form.onsubmit=event=>{
    event?.preventDefault?.();const spec={name:name.value.trim(),scopes:[...(scopes.children||[])].map(node=>node.children?.[0]).filter(box=>box?.checked&&box.value).map(box=>box.value)};
    const globs=String(models.value||'').split(',').map(item=>item.trim()).filter(Boolean);if(globs.length)spec.constraints={models:globs};
    if(rpm.value)spec.limits={rpm:Number(rpm.value)};if(String(budget.value||'').trim())spec.budget={usd_per_day:String(budget.value).trim()};
    if(days.value)spec.expires_at=new Date(Date.now()+Number(days.value)*86400000).toISOString().replace(/\.\d+Z$/,'Z');
    return brokerKeySecret('/api/keys',{method:'POST',json:spec});
  };
  // the one-time secret box survives re-renders (refresh, other panel actions) until it is dismissed
  return bn('section',{id:'broker-keys',cls:'broker-section'},bn('h3',{text:ui('broker.keys','Keys')}),list,form,bn('div',{id:'broker-secret-slot'},brokerSecretBox));
}
function brokerTabKeyId(){const match=/^tk_([a-z2-7]{10})_/.exec(String(key||''));return match?match[1]:null;}
async function brokerKeySecret(path,options){
  // the principal this tab acts as, read before the request: a rotation of that key kills the tab's secret at once
  const selfIds=[brokerState.self?.id,brokerTabKeyId()].filter(Boolean);
  try{const answer=await(await api(path,options)).json();const secret=answer.secret||null;delete answer.secret;
    // a keys:admin key that rotated itself adopts the new secret as the tab's in-memory key (as Connect does, never
    // stored) before any refresh, so the refresh that follows authenticates and the one-time box is not dropped on a 401
    const rotatedSelf=Boolean(secret&&answer.id&&selfIds.includes(answer.id));if(rotatedSelf)key=secret;
    brokerSecretBox=secret?keySecretBox(answer.id,secret,rotatedSelf):null;await refreshBroker();showKeySecret();}
  catch(error){brokerMessage(error.message,true);}
}
function keySecretBox(id,secret,rotatedSelf=false){
  // the secret lives only in this closure and the code element; dismiss drops both
  const value=bn('code',{id:'broker-key-secret-value'});value.textContent=secret;
  let box=null;const dismiss=()=>{secret=null;value.textContent='';if(brokerSecretBox===box)brokerSecretBox=null;box?.parentElement?.replaceChildren?.();};
  const note=rotatedSelf?ui('broker.secret_self','Key {{id}} rotated. This tab now uses the new secret; copy it now, it is shown only once.',{id}):ui('broker.secret_once','Key {{id}} created. Copy this secret now; it is shown only once.',{id});
  box=bn('div',{id:'broker-key-secret',cls:'broker-secret-box',role:'alert'},bn('p',{text:note}),value,
    brokerButton(ui('broker.copy','Copy'),()=>globalThis.navigator?.clipboard?.writeText?.(secret||'')),brokerButton(ui('broker.dismiss','Done, hide it'),dismiss));
  return box;
}
function showKeySecret(){const slot=$('broker-secret-slot');if(!slot)return;if(brokerSecretBox&&!brokerAllows('keys:admin'))brokerSecretBox=null;slot.replaceChildren(...(brokerSecretBox?[brokerSecretBox]:[]));}
function setupBroker(){const panel=$('broker-panel');if(panel)panel.hidden=true;}
// --- end broker panel ---
setupPreferences();setupRegistration();setupTaskFields();setupLifecycle();setupAssignments();setupOneStop();setupBroker();applyTranslations();renderCabinet();
$('register-backend').onchange=registrationBackend;
$('register-task').onchange=registrationBackend;
$('register-device').onchange=registrationBackend;
$('music-mode').onchange=()=>select(false);
function inventoryDevice(row,inventory,backend,task){
  if(backend==='openvino-genai'&&task==='image'){
    const address=String(row.runtime_address||'').toUpperCase();
    if(address==='CPU')return row.uuid||'CPU';
    if(/^GPU(?:\.\d+)?$/.test(address))return row.uuid||(address==='GPU'?'GPU.0':address);
    return null;
  }
  if(row.uuid)return row.uuid;
  if(inventory==='onnx'){
    if(row.identity_kind==='dxgi_luid')return null;
    return {CPUExecutionProvider:'cpu',DmlExecutionProvider:'dml',QNNExecutionProvider:'qnn',CoreMLExecutionProvider:'coreml',VitisAIExecutionProvider:'vitisai'}[row.runtime_address]||null;
  }
  return row.runtime_address||row.id;
}
$('load-devices').onclick=async()=>{
  const {backend,policy}=selectedRegistrationPolicy(),task=$('register-task').value;if(!policy||policy.cliOnly)return;
  try{
    const result=await(await api('/api/devices?backend='+policy.inventory)).json(),rows=Array.isArray(result.devices)?result.devices:[];
    $('register-devices').replaceChildren(...rows.map(row=>[inventoryDevice(row,policy.inventory,backend,task),row]).filter(([value])=>value).map(([value,row])=>option(value,row.name||row.runtime_address)));
    dmlAdapters.clear();$('register-dml-adapter').replaceChildren(option('',ui("ui.dml_choose","Choose a current DirectML adapter")));
    for(const row of rows.filter(row=>row.device==='dml'&&!row.software)){
      const id=row.id||('DML-'+row.luid);dmlAdapters.set(id,row);$('register-dml-adapter').append(option(id,`${row.name} · ${row.runtime_address} · ${row.luid}`));
    }
    status(rows.length?uiCount("ui.devices_loaded","Loaded {{count}} known device records. Inventory presence is not inference qualification.","Loaded {{count}} known device record. Inventory presence is not inference qualification.",{count:rows.length}):ui("ui.devices_empty","No cached devices. Enter a supported runtime address or refresh inventory with the CLI."));
  }catch(error){status(ui("ui.devices_unavailable","Device inventory unavailable; manual device entry remains enabled. ")+error.message,true);}
};
$('register-dml-adapter').onchange=()=>{const row=dmlAdapters.get($('register-dml-adapter').value);if(row)$('register-device').value='dml';};
$('register').onsubmit=async event=>{
  event.preventDefault();const form=new FormData(event.target),task=form.get('task'),{backend,policy}=selectedRegistrationPolicy();
  try{
    if(!policy||!policy.tasks.includes(task))throw Error(ui("ui.choose_backend_task","Choose a task supported by the selected backend"));
    if(policy.cliOnly)throw Error(policy.help);
    const spec={path:form.get('path'),task,device:form.get('device'),backend};
    if(policy.dtype&&$('register-dtype').value!=='auto')spec.dtype=$('register-dtype').value;
    if(policy.graph&&$('register-graph').value.trim())spec.graph=$('register-graph').value.trim();
    if(policy.voices&&$('register-voices').value.trim()){
      let voices;try{voices=JSON.parse($('register-voices').value);}catch{throw Error(ui("ui.voice_map_invalid","Voice map must be a JSON object"));}
      if(!voices||Array.isArray(voices)||typeof voices!=='object'||Object.entries(voices).some(([name,path])=>!name||typeof path!=='string'||!/^(?:[A-Za-z]:[\\/]|\/)/.test(path)))throw Error(ui("ui.voice_paths_absolute","Voice map values must be absolute local WAV paths"));
      spec.voices=voices;
    }
    const variant=policy.variant?form.get('variant'):null;
    if(variant){
      if(/[^A-Za-z0-9_-]/.test(variant))throw Error(ui("ui.variant_invalid","Weight variant must contain only ASCII letters, numbers, underscores or hyphens, such as fp16 or non_ema; leave empty for ordinary filenames"));
      spec.load_options={variant};
    }
    if(policy.onnxImage&&spec.device==='dml'){
      const adapter=dmlAdapters.get($('register-dml-adapter').value);
      if(!adapter)throw Error(ui("ui.choose_real_dml","Choose a current non-software DirectML adapter"));
      spec.load_options={device_id:Number(adapter.device_id),adapter_luid:adapter.luid};
      const diagnostic=$('register-disable-metacommands').value;if(diagnostic!=='')spec.load_options.disable_metacommands=diagnostic==='true';
    }
    const payload={name:form.get('name'),spec};if($('register-replace').checked)payload.replace=true;
    const replacingSelected=Boolean(payload.replace&&chosen()?.id===payload.name);
    await api('/api/models/register',{method:'POST',json:payload});
    if(replacingSelected){history=[];runtimeModelId=null;$('model-runtime').hidden=true;setContent($('model-runtime-output'),'');}
    await refresh(true);status(ui("ui.registration_complete","Model registered. Its external checkpoint files were not copied or changed."),false,'success');
  }catch(e){status(e.message,true);}
};
registrationBackend();
$('model').onchange=()=>{++lifecycleEpoch;runtimeModelId=null;$('model-runtime').hidden=true;select();};
$('refresh').onclick=()=>refresh(true);
$('connect').onclick=()=>{if(active)return;key=$('key').value;$('key').value='';infoAttempted=false;setupReconciled=false;if(typeof brokerReset==='function')brokerReset();return refresh(true);};
$('refresh-jobs').onclick=()=>refreshJobs().catch(e=>status(e.message,true));
$('clear').onclick=()=>{if(active)return;history=[];$('results').replaceChildren();renderCabinet();urls.forEach(URL.revokeObjectURL);urls=[];status(ui("ui.new_conversation","New conversation."),false,'success');};
$('load').onclick=async()=>{
  if(active||!chosen())return;
  try{runtimeModelId=null;$('model-runtime').hidden=true;status(ui("ui.loading_model","Loading model…"));await api('/api/models/load',{method:'POST',json:{model:chosen().id}});await refresh();status(ui("ui.model_loaded","Model loaded. Runtime details are now available."),false,'success');}
  catch(error){status(error.message,true);}
};
$('unload').onclick=async()=>{
  if(active||!chosen())return;
  try{await api('/api/models/unload',{method:'POST',json:{model:chosen().id}});runtimeModelId=null;$('model-runtime').hidden=true;await refresh();status(ui("ui.model_unloaded","Model unloaded."),false,'success');}
  catch(error){status(error.message,true);}
};
$('model-details').onclick=async()=>{
  const model=chosen();if(active||!model||model.status!=='loaded')return;
  const modelId=model.id,epoch=lifecycleEpoch;
  try{
    const result=await(await api('/api/models/details',{method:'POST',json:{model:modelId}})).json();
    if(epoch!==lifecycleEpoch||chosen()?.id!==modelId||chosen()?.status!=='loaded')return;
    runtimeModelId=modelId;setContent($('model-runtime-output'),JSON.stringify(result,null,2));$('model-runtime').hidden=false;$('model-runtime').open=true;status(ui("ui.showing_runtime","Showing observed runtime details for the loaded model."));
  }catch(error){if(epoch===lifecycleEpoch&&chosen()?.id===modelId)status(error.message,true);}
};
$('remove-registration').onclick=async()=>{
  const model=chosen();if(active||!model)return;
  if(!confirm(ui("ui.remove_confirm","Remove the {{model}} registration? External checkpoint and voice files will not be deleted.",{model:model.id})))return;
  try{await api('/api/models/registrations/'+encodeURIComponent(model.id),{method:'DELETE'});history=[];runtimeModelId=null;$('model-runtime').hidden=true;setContent($('model-runtime-output'),'');await refresh(true);status(ui("ui.registration_removed","Model registration removed. External checkpoint and voice files were not deleted."),false,'success');}
  catch(error){status(error.message,true);}
};
refresh();
