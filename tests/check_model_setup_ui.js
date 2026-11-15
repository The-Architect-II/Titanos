/* Existing capabilities rendered passively; headless/intercepted responses only. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.TITANOS_PLAYWRIGHT_MODULE||'playwright');
const web=path.join(__dirname,'../src/titanos/web'),requests=[],errors=[];
const action='Restore the matching model revision.',warning='Unfamiliar architecture remains unverified.',observed='Loaded: streaming wrapper revision 13ef823ffe76 and 1 registered PCM16 reference voice (vega) are bound.';
const models=[
  {id:'incomplete',task:'embeddings',status:'unloaded',setup:{status:'blocked',issues:[{severity:'error',message:'Required weights are missing.',files:['model.safetensors'],next_action:action}]}},
  {id:'unknown',task:'chat',status:'unloaded',setup:{status:'needs_review',issues:[{severity:'warning',message:warning,files:['config.json'],next_action:'Try an explicit load only after checking compatibility.'}]}},
  {id:'present',task:'chat',status:'unloaded',setup:{status:'ready_for_load_check',issues:[]}},
  {id:'voice',task:'speech',status:'loaded',setup:{status:'verified_at_load',model_loaded:true,observed_at_load:{scope:'runtime details of the currently loaded model; passive file checks unchanged',backend:'qwen3-tts',device:'cuda:0',reconciled_codes:['voice_wrapper_observed']},
    issues:[{code:'voice_wrapper_observed',severity:'info',message:observed,files:[],next_action:null,observed:{wrapper_revision:'13ef823ffe76aec4ac7e35d3fec28ecfc8d7427a',voices:['vega']}}]}}
];
async function main(){
  const browser=await chromium.launch({headless:true,executablePath:process.env.TITANOS_CHROME_PATH||undefined,
    args:['--disable-gpu','--disable-software-rasterizer','--renderer-process-limit=1']});
  const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();
  page.on('pageerror',error=>errors.push(error.message));page.setDefaultTimeout(10000);
  await page.route('**/*',async route=>{
    const request=route.request(),url=new URL(request.url()),name=url.pathname;
    assert.equal(url.hostname,'titanos.test');
    const file=name==='/'?'index.html':name.startsWith('/ui/')?name.slice(4):null;
    if(file&&/^(index\.html|app\.js|i18n\.js|retrieval\.js|style\.css|titanos-[a-z0-9-]+\.svg)$/.test(file)){
      return route.fulfill({status:200,contentType:({html:'text/html',js:'application/javascript',css:'text/css',svg:'image/svg+xml'})[file.split('.').at(-1)],body:fs.readFileSync(path.join(web,file))});
    }
    requests.push({name,method:request.method()});let data;
    if(name==='/v1/models')data={data:models.map(({setup,...model})=>model)};
    else if(name==='/api/info')data={models:{},settings:{},inference_contracts:{}};
    else if(name==='/api/models/capabilities'){
      const model=models.find(row=>row.id===request.postDataJSON().model);
      data={backend:'transformers',task:model.task,modes:{single:{implemented:true}},setup:{schema_version:1,advisory:true,model_loaded:false,qualified:null,...model.setup}};
    }else if(['/api/jobs','/api/acquisitions','/api/assignments/devices'].includes(name))data={data:[]};
    else if(name.startsWith('/api/assignments/'))data={mode:'single',groups:[['cpu']],replicas:1};
    else if(name==='/v1/extensions')data={};
    else throw Error('Unexpected request or model work: '+name);
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(data)});
  });
  try{
    await page.goto('http://titanos.test/');await page.waitForFunction(()=>document.querySelector('#model').options.length===4);
    const panel=page.locator('#model-setup-check');
    for(const locale of ['en','es','ja','zh-CN']){
      await page.selectOption('#ui-locale',locale);await page.selectOption('#ui-mode','simple');await page.selectOption('#model','incomplete');
      assert(await panel.isVisible());assert((await panel.textContent()).includes('model.safetensors'));assert((await panel.textContent()).includes(action));
      assert((await panel.textContent()).includes(await page.evaluate(()=>TitanosI18n.t('setup.check_blocked'))));
      assert(!(await page.locator('#load').isDisabled()),'Advisory inspection must not add a load-admission gate');
      assert(!(await page.locator('#submit').isDisabled()));
      await page.selectOption('#model','unknown');
      assert((await panel.textContent()).includes(await page.evaluate(()=>TitanosI18n.t('setup.check_unknown'))));
      assert(!(await panel.locator('.setup-issue').isVisible()),'Warnings belong to Advanced');
      await page.selectOption('#ui-mode','advanced');assert(await panel.locator('.setup-issue').isVisible());assert((await panel.textContent()).includes(warning));
      await page.selectOption('#model','present');assert((await panel.textContent()).includes(await page.evaluate(()=>TitanosI18n.t('setup.check_ready'))));
      assert.equal(await panel.locator('.setup-issue').count(),0);
      // reconciled with the loaded runtime: label says verified, the observed row replaces the 'were not loaded' advisory, no Next action
      await page.selectOption('#model','voice');const voiceText=await panel.textContent();
      assert(voiceText.includes(await page.evaluate(()=>TitanosI18n.t('setup.check_verified'))),'reconciled status label (catalog string for '+locale+')');assert(voiceText.includes(observed),'observed row quotes the bound wrapper and voice');
      assert(!voiceText.includes('were not loaded')&&!voiceText.includes('Next:'),'no stale advisory or next action once observed');
      assert(voiceText.includes(await page.evaluate(()=>TitanosI18n.t('setup.check_observed'))),'observed scope hint (catalog string for '+locale+')');
      assert(await panel.locator('.setup-issue.observed').isVisible(),'observed rows show in Advanced like warnings');
      await page.selectOption('#ui-mode','simple');assert(!(await panel.locator('.setup-issue').isVisible()),'observed rows stay Advanced-only');
      await page.selectOption('#ui-mode','advanced');
    }
    assert.equal(requests.filter(row=>row.name==='/api/models/capabilities').length,4,'Rendering must reuse existing capability requests');
    assert(requests.every(row=>row.method==='GET'||row.name==='/api/models/capabilities'));
    assert.deepEqual(errors,[]);
    console.log('PASS: passive setup messages/actions, four locales, Advanced warnings, reconciled verified-at-load rendering, unchanged Load/Generate admission, and zero additional capability/model requests.');
  }finally{await context.close();await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
