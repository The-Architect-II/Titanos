/* Actual browser UI against intercepted fixtures only. No server/model/GPU calls. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {chromium}=require(process.env.TITANOS_PLAYWRIGHT_MODULE||'playwright');
const web=path.join(__dirname,'../src/titanos/web');
const evidence=process.env.TITANOS_I18N_EVIDENCE_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'titanos-i18n-'));
fs.mkdirSync(evidence,{recursive:true});
const requests=[],pageErrors=[],unexpected=[],checks=[];
const raw='Download and run. Default. {{count}} — original user content';
let aceStatus='missing_runtime',llamaStatus='available',acquisitionCount=0,runtimeSetups=0,currentPlan=null,musicControls=true;
const models=[{id:'fixture/Default',task:'chat',backend:'transformers',device:'cpu',status:'loaded',repository:'fixture/source',revision:'rev-original'},
              {id:'fixture/music',task:'music',backend:'ace-step',device:'cpu',status:'unloaded',repository:'fixture/music-source',revision:'rev-music'}];
let records=[];
const headers={'content-type':'application/json'};

function plan(payload){
  const music=payload.backend==='ace-step';
  return {provider:payload.provider,provider_title:music?'ACE-Step 1.5':'Hugging Face',candidate_id:payload.identifier,
    registry_name:payload.registry_name,task:payload.task,backend:payload.backend,device:payload.device,
    requested_revision:payload.revision||'main',resolved_revision:music?'music-revision':'pinned-123',
    source_url:music?'https://provider.invalid/xl-sft':'https://huggingface.co/publisher/model',
    expected_bytes:music?29924251861:491416231,license:'apache-2.0',gated:false,
    compatibility:{qualification:null},runtime:{status:music?aceStatus:llamaStatus,managed_profile:!music&&llamaStatus==='available'?'fixture-cpu':null},
    files:[{path:'LICENSE',size:11343},{path:music?'model.safetensors':'model-q4.gguf',size:491400032}],
    model_spec:{task:payload.task,backend:payload.backend,device:payload.device}};
}

// 01 rail and the collapsed bays: whole rows only (no row sliced by the scroller's edges), a "N more below" cue that
// matches the rows below its edge also after a disclosure is toggled, Connection reachable while Register a local path is
// open, and a collapsed bay's sentence that wraps (never ellipsized or clipped mid-word); all locales at 1440, 1024 and 390
const RAIL_STATE=()=>{
  const body=document.querySelector('.rail-body'),cue=document.getElementById('rail-more'),style=getComputedStyle(body),across=style.overflowY==='hidden';
  const box=body.getBoundingClientRect(),start=across?box.left+body.clientLeft:box.top+body.clientTop,end=start+(across?body.clientWidth:body.clientHeight);
  const sliced=[],below=[];
  for(const row of document.getElementById('models').children){
    const r=row.getBoundingClientRect(),a=across?r.left:r.top,b=across?r.right:r.bottom,seen=Math.min(b,end)-Math.max(a,start);
    if(seen>0.5&&(a<start-0.5||b>end+0.5))sliced.push((row.dataset.modelId||row.className)+' shows '+Math.round(seen)+' of '+Math.round(b-a)+'px');
    if(row.dataset.modelId&&b>end+0.5)below.push(row.dataset.modelId);
  }
  // a meta chip (task, provider) that does not fit wraps to its own line; none is squeezed into an ellipsis or a sliver
  const slivers=[...document.querySelectorAll('#models .model-chip')].filter(chip=>{const r=chip.getBoundingClientRect();return r.width>0&&(r.width<6||chip.scrollWidth>chip.clientWidth+1);}).map(chip=>chip.textContent);
  const count=Number((cue.textContent.match(/\d+/)||[0])[0]);
  return {sliced,slivers,cue:cue.textContent,cueOk:across?cue.textContent==='':count===below.length,below:below.length};
};
const BAY_LINES=()=>[...document.querySelectorAll('details.bay:not([open]) .bay-line')].filter(line=>line.getClientRects().length).map(line=>{
  const style=getComputedStyle(line);
  return {text:line.textContent.slice(0,40),ellipsis:style.textOverflow==='ellipsis',nowrap:style.whiteSpace==='nowrap',clipped:line.scrollWidth>line.clientWidth+1||line.scrollHeight>line.clientHeight+1};
});
async function railSettled(page){await page.waitForTimeout(250);await page.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))));}
async function railAndBayChecks(page){
  const before=models.length;
  for(let index=0;index<8;index++)models.push({id:`fixture/rail-${String(index).padStart(2,'0')}`,task:['chat','embeddings','speech','transcription'][index%4],backend:'transformers',device:'cpu',status:'unloaded',repository:'fixture/rail',revision:'rev-rail'});
  // cloud rows (broker catalog rows carry no task): their provider chip must wrap rather than shrink to a sliver
  for(const [id,capability] of [['gpt-image-1@openai','images'],['text-embedding-3-small@openai','embeddings'],['claude-haiku-4-5@anthropic','chat'],['tts-1@openai','speech']])
    models.push({id,object:'model',titanos:{class:'external',upstream:id.split('@')[1],capabilities:[capability],status:'available'}});
  await page.click('#refresh');await page.waitForFunction(count=>document.querySelectorAll('#models .model').length===count,models.length);
  const rail=async label=>{const state=await page.evaluate(RAIL_STATE);
    assert.deepEqual(state.sliced,[],label+': a rail row is sliced by the scroller edge');assert.deepEqual(state.slivers,[],label+': a rail chip is squeezed into an ellipsis or a sliver');
    assert(state.cueOk,label+`: the cue says "${state.cue}" but ${state.below} rows are below its edge`);return state;};
  const inTools=selector=>page.evaluate(selector=>{const node=document.querySelector(selector).getBoundingClientRect(),tools=document.querySelector('#bay-models .rail-tools').getBoundingClientRect();
    return node.height>0&&node.top>=0&&node.bottom<=innerHeight&&node.top>=tools.top-0.5&&node.bottom<=tools.bottom+0.5;},selector);
  for(const [width,height] of [[1440,900],[1024,768],[390,844]])for(const mode of ['simple','advanced']){
    const label=`${width}x${height} ${mode}`;
    await page.setViewportSize({width,height});await page.selectOption('#ui-locale','en');await page.selectOption('#ui-mode',mode);await railSettled(page);
    await page.evaluate(()=>{document.querySelector('.rail-body').scrollTo(0,0);});await railSettled(page);
    const top=await rail(label);
    if(width>900){
      assert(top.below>0,label+': the fixture rail should overflow');
      for(const fraction of [0.5,1]){await page.evaluate(f=>{const b=document.querySelector('.rail-body');b.scrollTop=(b.scrollHeight-b.clientHeight)*f;},fraction);await railSettled(page);await rail(label+' scrolled '+fraction);}
      await page.evaluate(()=>{document.querySelector('.rail-body').scrollTo(0,0);});await railSettled(page);
      const details='#models .model:first-child .model-technical > summary';
      await page.click(details);await railSettled(page);const opened=await rail(label+' Details open');
      await page.click(details);await railSettled(page);const closed=await rail(label+' Details closed');
      assert(opened.below>=closed.below,label+': the cue did not follow the opened Details');
      if(mode==='advanced'){
        const register='#bay-models .rail-tools details:not(#connection-panel) > summary';
        await page.click(register);await railSettled(page);await rail(label+' Register open');
        await page.focus('#connection-panel > summary');await railSettled(page);
        assert(await inTools('#connection-panel > summary'),label+': Connection cannot be reached while Register a local path is open');
        await page.keyboard.press('Enter');await page.focus('#key');await railSettled(page);
        assert(await inTools('#key'),label+': the server key field cannot be reached while Register a local path is open');
        await rail(label+' Register and Connection open');
        await page.click('#connection-panel > summary');await page.click(register);await railSettled(page);await rail(label+' tools closed');
      }
    }
    for(const locale of ['en','es','ja','zh-CN']){
      await page.selectOption('#ui-locale',locale);await railSettled(page);
      const lines=await page.evaluate(BAY_LINES);assert(lines.length>=2,`${label} ${locale}: collapsed bay sentences are missing`);
      for(const line of lines)assert(!line.ellipsis&&!line.nowrap&&!line.clipped,`${label} ${locale}: a collapsed bay sentence is cut: ${JSON.stringify(line)}`);
      await rail(`${label} ${locale}`);
    }
  }
  checks.push({rail_whole_rows_and_cue:'1440x900, 1024x768, 390x844 in Simple and Advanced',connection_reachable_with_register_open:true,collapsed_bay_sentences_wrap:'en, es, ja, zh-CN'});
  models.splice(before);await page.selectOption('#ui-locale','en');await page.setViewportSize({width:1600,height:1100});
  await page.click('#refresh');await page.waitForFunction(count=>document.querySelectorAll('#models .model').length===count,before);
}

async function main(){
  const browser=await chromium.launch({headless:true,executablePath:process.env.TITANOS_CHROME_PATH||undefined,
    args:['--disable-gpu','--disable-software-rasterizer','--disable-dev-shm-usage']});
  const context=await browser.newContext({viewport:{width:1600,height:1100},locale:'en-US'});
  const page=await context.newPage();page.on('pageerror',error=>pageErrors.push(error.message));
  page.setDefaultTimeout(5000);
  await page.route('**/*',async route=>{
    const request=route.request(),url=new URL(request.url()),pathname=url.pathname;
    if(url.hostname!=='titanos.test'){unexpected.push(request.url());return route.abort();}
    const file=pathname==='/'?'index.html':pathname.startsWith('/ui/')?pathname.slice(4):null;
    if(file&&/^(index\.html|app\.js|i18n\.js|retrieval\.js|style\.css|titanos-[a-z0-9-]+\.svg)$/.test(file)){
      const types={html:'text/html',js:'application/javascript',css:'text/css',svg:'image/svg+xml'};
      return route.fulfill({status:200,contentType:types[file.split('.').at(-1)],body:fs.readFileSync(path.join(web,file))});
    }
    const method=request.method(),payload=request.postData()?JSON.parse(request.postData()):null;
    requests.push({path:pathname,method,payload});let data;
    if(pathname==='/v1/models')data={data:models};
    else if(pathname==='/api/info')data={models:Object.fromEntries(models.map(model=>[model.id,{task:model.task,backend:model.backend,device:model.device}])),
      settings:{max_tokens:{value:4096},max_image_dimension:{value:1024},max_image_pixels:{value:1048576}},
      model_setup:{runtime:{configured:true,recommended_profile:'fixture-cpu',profiles:[{id:'fixture-cpu',title:'CPU · fixture',runtime_kind:'cpu',one_stop_supported:true}]}},
      inference_contracts:{require_loaded:{revision:1,routes:['/v1/chat/completions']}}};
    else if(pathname==='/api/models/capabilities'){
      const model=models.find(row=>row.id===payload.model);
      data={backend:model.backend,task:model.task,source:{repository:model.repository,revision:model.revision},implemented:[model.task],
        modes:{single:{implemented:true},model_sharding:{implemented:true},tensor_parallel:{implemented:true}},
        ...(model.task==='music'?{music:{controls:{lyrics:true,duration_min:10,duration_max:600,vocal_languages:['en','es'],bpm:true,key_scale:true,reference_audio:false,seed:musicControls,guidance_scale:musicControls,steps:musicControls}}}:{})};
    }
    else if(pathname==='/api/jobs'&&method==='POST')data={id:'fixture-music-job'};
    else if(pathname==='/api/jobs/fixture-music-job')data={status:'completed',result:{task:'music',text:'CONTROLLED FIXTURE: no music was generated'}};
    else if(pathname==='/api/jobs')data={data:[{id:'job-native-id',model:'fixture/Default',task:'image',status:'running',native_generation:{stage:'sampling'}}]};
    else if(pathname==='/v1/extensions')data={retrieval_answer:{models:{embedding:['native/Default-encoder'],rerank:['native/Default-reranker'],llm:['native/Default-answer']}}};
    else if(pathname==='/api/assignments/devices')data={data:[{id:'cpu',label:'CPU',present:true}]};
    else if(pathname.startsWith('/api/assignments/'))data={model:decodeURIComponent(pathname.split('/').at(-1)),mode:'single',groups:[['cpu']],replicas:1,tensor_parallel_size:2};
    else if(pathname==='/api/assignments')data={...payload,replicas:payload.groups?.length||1};
    else if(pathname==='/api/model-providers/huggingface/search')data={data:[{id:'publisher/model',name:'Model ID: Default',provider:'huggingface',titanos_task:'chat',recommended_backend:'llama-cpp',task:'text-generation',license:'apache-2.0',gated:false,compatibility:{note:'RAW PROVIDER NOTE: Default {{count}}'}}]};
    else if(pathname==='/api/model-providers/inspect')data={id:payload.identifier,files:[{path:'model-q4.gguf',size:491400032}],source_url:'https://huggingface.co/publisher/model'};
    else if(pathname==='/api/acquisitions/plan'){currentPlan=plan(payload);data=currentPlan;}
    else if(pathname==='/api/acquisitions'&&method==='GET')data={data:records};
    else if(pathname==='/api/acquisitions'&&method==='POST'){
      acquisitionCount++;currentPlan=payload;const record={id:'acquisition-'+acquisitionCount,status:'prepared',plan:payload,license:'apache-2.0',progress:{stage:'prepared',completed:payload.expected_bytes,total:payload.expected_bytes}};records=[record];data=record;
    }else if(pathname.startsWith('/api/acquisitions/')&&pathname.endsWith('/activate')){
      const record=records[0],p=record.plan;
      const model={id:p.registry_name,task:p.task,backend:p.backend,device:p.device,status:'loaded',repository:p.provider==='huggingface'?p.candidate_id:p.source_url,revision:p.resolved_revision};
      const index=models.findIndex(row=>row.id===model.id);if(index>=0)models[index]=model;else models.push(model);
      Object.assign(record,{status:'active',activation:{model:model.id,status:'loaded'}});data=record;
    }else if(pathname.startsWith('/api/acquisitions/'))data=records[0];
    else if(pathname==='/api/runtime-setups/plan')data={runtime:'llama.cpp',profile:payload.profile};
    else if(pathname==='/api/runtime-setups'&&method==='POST'){runtimeSetups++;data={id:'runtime-fixture',status:'prepared',plan:{runtime:'llama.cpp',release:'fixture',profile:'fixture-cpu',license:'mit'},progress:{stage:'prepared',completed:10,total:10}};}
    else if(pathname==='/api/runtime-setups/runtime-fixture')data={id:'runtime-fixture',status:'prepared',plan:{runtime:'llama.cpp',release:'fixture',profile:'fixture-cpu',license:'mit'},progress:{stage:'prepared',completed:10,total:10}};
    else if(pathname==='/api/runtime-setups')data={data:[]};
    else if(pathname==='/fixture/error')return route.fulfill({status:409,headers,body:JSON.stringify({error:{code:'model_not_loaded',message:'RAW_DIAGNOSTIC: Default {{count}}'}})});
    else if(pathname==='/v1/chat/completions')return route.fulfill({status:200,contentType:'text/event-stream',body:'data: '+JSON.stringify({choices:[{delta:{content:raw}}]})+'\n\ndata: [DONE]\n\n'});
    else if(pathname==='/api/models/load'||pathname==='/api/models/unload'){
      const model=models.find(row=>row.id===payload.model);model.status=pathname.endsWith('/unload')?'unloaded':'loaded';data={model:model.id,status:model.status};
    }else {unexpected.push(pathname);return route.fulfill({status:404,headers,body:JSON.stringify({error:{message:'Unexpected controlled route'}})});}
    return route.fulfill({status:200,headers,body:JSON.stringify(data)});
  });
  try{
    await page.goto('http://titanos.test/');
    await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('Connected'));
    if(process.argv.includes('--simple-only')){
      const titles={en:'Jobs',es:'Trabajos',ja:'ジョブ','zh-CN':'任务'};
      await page.locator('#prompt').fill(raw);
      await page.evaluate(text=>{
        card('Saved result',text);$('seed').value='123';runtimeModelId=chosen().id;
        $('model-runtime').hidden=false;$('model-runtime').open=true;setContent($('model-runtime-output'),'RAW_RUNTIME: cpu / unchanged');
      },raw);
      const selected=await page.locator('#model').inputValue();
      const technicalBefore=await page.locator('.model-meta .advanced-only').allTextContents();
      assert(technicalBefore.includes('transformers')&&technicalBefore.includes('cpu'));
      for(const locale of ['en','es','ja','zh-CN']){
        await page.selectOption('#ui-locale',locale);await page.selectOption('#ui-mode','simple');
        for(const id of ['model-details','remove-registration','model-runtime'])assert.equal(await page.locator('#'+id).isVisible(),false,locale+' exposes '+id+' in Simple');
        for(const node of await page.locator('.model-meta .advanced-only').all())assert.equal(await node.isVisible(),false);
        assert(await page.locator('.model-meta .model-chip:not(.advanced-only)').first().isVisible());
        assert(await page.locator('#load').isVisible());assert(await page.locator('#submit').isVisible());
        assert.equal(await page.locator('#bay-jobs .bay-plate [data-i18n="jobs.title"]').textContent(),titles[locale]);
        assert.equal(await page.locator('#model').inputValue(),selected);
        assert.equal(await page.locator('#prompt').inputValue(),raw);
        assert.equal(await page.locator('.result-saved-result > pre').textContent(),raw);
        assert.equal(await page.locator('#seed').inputValue(),'123');
        if(locale==='en'){await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(evidence,'simple-en.png'),fullPage:true});}
        await page.selectOption('#ui-mode','advanced');
        for(const id of ['model-details','remove-registration','model-runtime'])assert(await page.locator('#'+id).isVisible(),locale+' lost Advanced '+id);
        for(const node of await page.locator('.model-meta .advanced-only').all())assert(await node.isVisible());
        assert.deepEqual(await page.locator('.model-meta .advanced-only').allTextContents(),technicalBefore);
        assert.equal(await page.locator('#model-runtime-output').textContent(),'RAW_RUNTIME: cpu / unchanged');
        checks.push({locale,simple_hides_technical_controls:true,advanced_access_preserved:true,creations_title:titles[locale],model_prompt_results_and_values_preserved:true});
      }
      await railAndBayChecks(page);
      assert(!requests.some(row=>['/v1/chat/completions','/api/models/load','/api/models/unload','/api/assignments','/api/runtime-setups','/api/acquisitions','/api/jobs'].includes(row.path)&&row.method==='POST'));
      assert.deepEqual(pageErrors,[]);assert.deepEqual(unexpected,[]);
      fs.writeFileSync(path.join(evidence,'RESULT.json'),JSON.stringify({status:'PASSED',scope:'Simple/Advanced presentation only; intercepted fixtures, no inference or mutation requests, headless GPU disabled',checks,page_errors:pageErrors,unexpected_requests:unexpected,requests},null,2)+'\n');
      console.log('PASS: Simple hides runtime/removal and backend/device tags; Advanced retains them; Jobs is localized in all4locales; state unchanged.');
      return;
    }
    await page.locator('#prompt').fill(raw);await page.locator('#music-lyrics').evaluate((node,text)=>node.value=text,raw);
    await page.locator('#submit').click();await page.waitForFunction(()=>document.querySelector('#status').textContent==='Done');
    const actualOutput=await page.locator('.result-assistant pre').textContent();assert.equal(actualOutput,raw);
    await page.selectOption('#ui-mode','advanced');
    await page.waitForFunction(()=>document.querySelector('#assignment-apply').disabled===false);
    await page.locator('#assignment-replicas').fill('2');
    await page.locator('#retrieval-form').evaluate(node=>node.parentElement.open=true);
    await page.locator('#retrieval-query').fill(raw);
    await page.evaluate(text=>window.renderRetrievalAnswer({status:'answered',claims:[{text,citations:[{document_id:'Default',quote:text,spans:[{start:0,end:text.length}]}]}],documents:[{document_id:'Default',title:'Default',source_uri:'source://{{count}}'}],stages:[]}),raw);
    for(const locale of ['en','es','ja','zh-CN']){
      await page.selectOption('#ui-locale',locale);
      assert.equal(await page.locator('html').getAttribute('lang'),locale);
      assert.equal(await page.locator('#prompt').inputValue(),raw);
      assert.equal(await page.locator('#music-lyrics').inputValue(),raw);
      assert.equal(await page.locator('#retrieval-query').inputValue(),raw);
      assert.equal(await page.locator('.result-assistant pre').textContent(),raw);
      assert.equal(await page.locator('#retrieval-results blockquote').textContent(),raw);
      assert.equal(await page.locator('#retrieval-results article p').textContent(),raw);
      assert((await page.locator('#retrieval-results article small').textContent()).includes('source://{{count}}'));
      assert.equal(await page.locator('#assignment-replicas').inputValue(),'2');
      assert.equal(await page.locator('#retrieval-results strong').textContent(),'Default · Default');
      const keys=['ui.max_tokens','ui.registration_name','retrieval.title','retrieval.question','music.lyrics','advanced.unsaved'];
      for(const key of keys){
        const expected=await page.evaluate(key=>TitanosI18n.t(key),key);
        const content=await page.locator('[data-i18n="'+key+'"]').first().textContent();assert.equal(content,expected,key+' '+locale);
      }
      await page.evaluate(async()=>{try{await api('/fixture/error');}catch(error){status(error.message,true);}});
      assert((await page.locator('#status').textContent()).includes('RAW_DIAGNOSTIC: Default {{count}}'));
      assert((await page.locator('#status').textContent()).includes(await page.evaluate(()=>TitanosI18n.t('error.model_not_loaded'))));
      await page.locator('#model-setup').evaluate(node=>node.open=true);   // 02 Add a model stays closed once the registry has a local model
      await page.evaluate(async()=>oneStopSearch());
      assert((await page.locator('#setup-results').textContent()).includes('Model ID: Default'));
      await page.locator('#setup-results button').first().click();
      await page.waitForFunction(()=>document.querySelector('#setup-plan [data-i18n="ui.pinned_revision"]'));
      assert((await page.locator('#setup-plan').textContent()).includes('pinned-123'));
      assert.equal(await page.locator('#setup-plan [data-i18n="setup.download"]').textContent(),await page.evaluate(()=>TitanosI18n.t('setup.download')));
      assert((await page.locator('#setup-plan [data-i18n="ui.license_access"]').textContent()).includes('apache-2.0'));
      assert((await page.locator('#setup-runtime-profile option:checked').textContent()).includes(await page.evaluate(()=>TitanosI18n.t('ui.recommended_suffix'))));
      const before=acquisitionCount;aceStatus='missing_runtime';
      await page.evaluate(async()=>oneStopPlan({id:'xl-sft',name:'XL SFT',provider:'ace-step',titanos_task:'music',recommended_backend:'ace-step'}));
      assert(await page.locator('#setup-plan button').last().isDisabled());
      assert.equal(await page.locator('#setup-message').textContent(),await page.evaluate(()=>TitanosI18n.t('ui.music_runtime_missing')));
      await page.evaluate(async()=>oneStopAcquire({backend:'ace-step',runtime:{status:'missing_runtime'}}));
      assert.equal(acquisitionCount,before,'Missing ACE runtime started acquisition');
      aceStatus='unsupported_adapter';
      await page.evaluate(async()=>oneStopPlan({id:'xl-turbo',name:'XL Turbo',provider:'ace-step',titanos_task:'music',recommended_backend:'ace-step'}));
      assert(await page.locator('#setup-plan button').last().isDisabled());
      assert.equal(await page.locator('#setup-message').textContent(),await page.evaluate(()=>TitanosI18n.t('ui.music_variant_unsupported')));
      checks.push({locale,semantic_controls:true,raw_prompt_output_lyrics_citations_preserved:true,assignment_draft_preserved:true,discovery_plan_and_errors:true,music_missing_runtime_and_variant_blocked:true});
      if(locale==='es'||locale==='zh-CN'){await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(evidence,'studio-'+locale+'.png'),fullPage:true});}
    }
    // Localized captions must not destroy a live license checkbox or alter the license text.
    await page.evaluate(async text=>{
      const record={id:'license-fixture',status:'prepared',plan:{registry_name:'license/model'},
        progress:{stage:'prepared',completed:1,total:1},license:null,
        license_evidence:[{path:'LICENSE',text}],license_evidence_digest:'a'.repeat(64)};
      renderAcquisition(record);await activateAcquisition(record);
    },raw);
    await page.locator('#setup-plan input[type="checkbox"]').check();
    for(const locale of ['en','es','ja','zh-CN']){
      await page.selectOption('#ui-locale',locale);
      assert(await page.locator('#setup-plan input[type="checkbox"]').isChecked());
      assert.equal(await page.locator('#setup-plan pre').textContent(),raw);
      assert.equal(await page.locator('#setup-plan [data-i18n="ui.activate_reviewed"]').textContent(),await page.evaluate(()=>TitanosI18n.t('ui.activate_reviewed')));
    }
    // A current successful plan remains an ordinary acquisition path.
    aceStatus='available';await page.evaluate(async()=>oneStopPlan({id:'xl-sft',name:'XL SFT',provider:'ace-step',titanos_task:'music',recommended_backend:'ace-step'}));
    assert.equal(await page.locator('#setup-plan button').last().isDisabled(),false);
    await page.locator('#setup-plan button').last().click();
    await page.waitForFunction(()=>document.querySelector('#setup-message').textContent===TitanosI18n.t('setup.ready'));
    assert(acquisitionCount>0);assert.equal(await page.locator('#model').inputValue(),'xl-sft');
    // Preserve llama.cpp's existing normal runtime preparation then acquisition.
    llamaStatus='missing_runtime';await page.evaluate(async()=>oneStopPlan({id:'publisher/second',name:'Second',provider:'huggingface',titanos_task:'chat',recommended_backend:'llama-cpp'}));
    assert.equal(await page.locator('#setup-plan button').last().isDisabled(),false);
    await page.locator('#setup-plan button').last().click();
    await page.waitForFunction(()=>document.querySelector('#model').value==='publisher/second');assert.equal(runtimeSetups,1);
    assert.equal(requests.filter(row=>row.path.endsWith('/activate')).at(-1).payload.runtime_profile,'fixture-cpu');
    // Supported music controls stay usable in Advanced and retain values across language/mode changes.
    await page.selectOption('#model','fixture/music');await page.selectOption('#ui-mode','advanced');
    await page.locator('#settings-caption').evaluate(node=>node.parentElement.open=true);
    for(const [id,value] of [['seed','123'],['guidance_scale','2.5'],['steps','12']]){
      assert(await page.locator('#'+id).isVisible());assert(await page.locator('#'+id).isEnabled());await page.locator('#'+id).fill(value);
    }
    await page.selectOption('#ui-mode','simple');await page.selectOption('#ui-locale','ja');await page.selectOption('#ui-mode','advanced');
    for(const [id,value] of [['seed','123'],['guidance_scale','2.5'],['steps','12']])assert.equal(await page.locator('#'+id).inputValue(),value);
    await page.locator('#submit').click();await page.waitForFunction(()=>document.querySelector('#status').textContent===TitanosI18n.t('status.done'));
    let posted=requests.filter(row=>row.path==='/api/jobs'&&row.method==='POST').at(-1).payload;
    assert.equal(posted.seed,123);assert.equal(posted.guidance_scale,2.5);assert.equal(posted.steps,12);assert(!Object.hasOwn(posted,'duration'));
    musicControls=false;await page.evaluate(()=>refresh(true));
    for(const id of ['seed','guidance_scale','steps'])assert(await page.locator('#'+id).isDisabled());
    await page.locator('#submit').click();await page.waitForFunction(()=>document.querySelector('#status').textContent===TitanosI18n.t('status.done'));
    posted=requests.filter(row=>row.path==='/api/jobs'&&row.method==='POST').at(-1).payload;
    for(const id of ['seed','guidance_scale','steps'])assert(!Object.hasOwn(posted,id),'Unsupported stale music control leaked: '+id);
    await page.evaluate(async text=>{
      await showResult({text,source_language:'raw/source',target_language:'raw/target',translated:true},'raw-translation-label');
      const controller=new AbortController();controller.abort();
      try{await streamChat({model:'raw/stopped-model',messages:[]},controller.signal);}catch{}
    },raw);
    for(const locale of ['en','es','ja','zh-CN']){
      await page.selectOption('#ui-locale',locale);
      const translated=await page.locator('.result-raw-translation-label > pre').textContent();
      assert(translated.startsWith(raw+'\n'));
      assert(translated.includes('raw/source → raw/target'));
      assert(translated.endsWith(await page.evaluate(()=>TitanosI18n.t('ui.translated_suffix'))));
      assert((await page.locator('.result-assistant .label').first().textContent()).includes(await page.evaluate(()=>TitanosI18n.t('ui.partial_stopped'))));
    }
    await page.setViewportSize({width:390,height:844});
    for(const locale of ['en','es','ja','zh-CN']){
      await page.selectOption('#ui-locale',locale);
      const width=await page.evaluate(()=>({viewport:innerWidth,scroll:document.documentElement.scrollWidth,picker:document.querySelector('#model').getBoundingClientRect().width}));
      assert(width.scroll<=width.viewport+1,locale+' horizontal overflow');assert(width.picker>=240,locale+' unusable picker');
    }
    // reflow (WCAG 1.4.10): no horizontal scrolling at 320 CSS px in any locale or mode (boot readout, plates, long buttons)
    await page.setViewportSize({width:320,height:800});
    for(const mode of ['simple','advanced'])for(const locale of ['en','es','ja','zh-CN']){
      await page.selectOption('#ui-mode',mode);await page.selectOption('#ui-locale',locale);
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth);
      assert(overflow<=0,`${locale} ${mode} scrolls sideways by ${overflow}px at 320px`);
    }
    await page.selectOption('#ui-mode','simple');await page.selectOption('#ui-locale','zh-CN');await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(evidence,'studio-mobile-zh-CN.png'),fullPage:true});
    assert.deepEqual(pageErrors,[]);assert.deepEqual(unexpected,[]);
    const result={status:'PASSED',scope:'Actual headless Chromium; intercepted HTTP fixtures only, GPU disabled, no model/inference/native input',checks,available_music_path:true,llama_preparation_preserved:true,license_checkbox_and_text_preserved:true,music_advanced_controls_retained_and_capability_gated:true,partial_stop_and_translation_metadata_localized_without_output_changes:true,mobile_locales_checked:4,page_errors:pageErrors,unexpected_requests:unexpected,requests};
    fs.writeFileSync(path.join(evidence,'RESULT.json'),JSON.stringify(result,null,2)+'\n');
    console.log('PASS: four-language whole-workflow UI, preserved user/source text and assignment drafts, ACE pre-download gates, llama setup and mobile layout. Evidence: '+evidence);
  }finally{await context.close();await browser.close();}
}
main().catch(error=>{fs.writeFileSync(path.join(evidence,'FAILURE.json'),JSON.stringify({message:error.message,stack:error.stack,pageErrors,unexpected,requests},null,2));console.error(error);process.exitCode=1;});
