/* Headless Studio with intercepted responses. No model or desktop input. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.TITANOS_PLAYWRIGHT_MODULE||'playwright');
const web=path.join(__dirname,'../src/titanos/web'),requests=[],errors=[],unexpected=[];
const speakers={model:'speakers',task:'diarization',speakers:[{id:'speaker_0'},{id:'speaker_1'}],
  segments:[{speaker:'speaker_0',start:0,end:4},{speaker:'speaker_1',start:3,end:7}],
  overlaps:[{speakers:['speaker_0','speaker_1'],start:3,end:4}]};
const composition={task:'audio_analysis',text:'Original words remain unchanged.',diarization:speakers,
  speaker_attribution_status:'partial',speaker_segments:[
    {text:'Original words remain unchanged.',speakers:['speaker_0','speaker_1'],overlap:true,attribution:'interval_intersection'},
    {text:'Preserved uncertain timing.',start:6,end:8,speakers:[],overlap:false,attribution:'timestamps_outside_audio'}],text_analysis:{label:'neutral'}};
const models=[{id:'asr',task:'transcription',backend:'transformers'},
  {id:'speakers',task:'diarization',backend:'sherpa-onnx'},
  {id:'activity',task:'vad',backend:'onnx'},{id:'text-analysis',task:'sentiment',backend:'transformers'},
  {id:'chat-analysis',task:'chat',backend:'llama-cpp'}];
let hold=false,release;
async function main(){
  const browser=await chromium.launch({headless:true,executablePath:process.env.TITANOS_CHROME_PATH||undefined,
    args:['--disable-gpu','--disable-software-rasterizer','--renderer-process-limit=1']});
  const context=await browser.newContext({viewport:{width:1440,height:1100}}),page=await context.newPage();
  page.on('pageerror',error=>errors.push(error.message));page.setDefaultTimeout(10000);
  await page.route('**/*',async route=>{
    const request=route.request(),url=new URL(request.url()),name=url.pathname;
    if(url.hostname!=='titanos.test'){unexpected.push(request.url());return route.abort();}
    const file=name==='/'?'index.html':name.startsWith('/ui/')?name.slice(4):null;
    if(file&&/^(index\.html|app\.js|i18n\.js|retrieval\.js|style\.css|titanos-[a-z0-9-]+\.svg)$/.test(file)){
      return route.fulfill({status:200,contentType:({html:'text/html',js:'application/javascript',css:'text/css',svg:'image/svg+xml'})[file.split('.').at(-1)],body:fs.readFileSync(path.join(web,file))});
    }
    const payload=request.postDataJSON?.();let result;
    if(name==='/v1/models')result={data:models.map(m=>({...m,status:'unloaded'}))};
    else if(name==='/api/info')result={models:Object.fromEntries(models.map(m=>[m.id,{...m,device:'cpu'}])),settings:{max_tokens:{value:8192}},inference_contracts:{}};
    else if(name==='/api/models/capabilities'){const model=models.find(m=>m.id===payload.model);result={...model,modes:{single:{implemented:true}},implemented:[model.task]};}
    else if(name==='/api/jobs'||name==='/api/acquisitions')result={data:[]};
    else if(name==='/v1/extensions')result={};
    else if(name==='/api/model-providers')result={providers:[]};
    else if(name==='/api/assignments/devices')result={data:[{id:'cpu',label:'CPU'}]};
    else if(name.startsWith('/api/assignments/'))result={mode:'single',groups:[['cpu']],replicas:1};
    else if(name==='/v1/audio/analysis'||name==='/v1/audio/diarizations'){
      requests.push({route:name,payload});if(hold)await new Promise(resolve=>{release=resolve;});
      result=name.endsWith('/analysis')?composition:speakers;
    }else{unexpected.push(name);return route.fulfill({status:404,body:'Unexpected fixture route'});}
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(result)}).catch(()=>{});
  });
  try{
    await page.goto('http://titanos.test/');await page.waitForFunction(()=>document.querySelector('#model').options.length===5);
    await page.selectOption('#model','asr');await page.selectOption('#audio-action','analyze');
    assert(await page.locator('#diarization-model').isVisible());
    await page.selectOption('#diarization-model','speakers');
    assert(await page.locator('#audio-text-model option[value="chat-analysis"]').count());
    await page.setInputFiles('#file',{name:'test.wav',mimeType:'audio/wav',buffer:Buffer.from('controlled source bytes')});
    assert(!(await page.locator('#diarization-count').isVisible()));
    await page.click('#submit');await page.waitForFunction(()=>document.querySelector('#submit').disabled===false);
    assert.equal(requests[0].route,'/v1/audio/analysis');
    assert.equal(requests[0].payload.diarization_model,'speakers');
    assert.deepEqual(requests[0].payload.diarization_options,{});
    assert(!Object.hasOwn(requests[0].payload,'text_analysis_model'));
    await page.selectOption('#ui-mode','advanced');await page.locator('#generate details.advanced-only').evaluate(node=>{node.open=true;}); // the generation-settings panel; the device-visibility panel is a separate details element
    await page.fill('#diarization-count','2');await page.fill('#diarization-threshold','0.55');
    await page.fill('#diarization-on','0.1');await page.fill('#diarization-off','0');await page.fill('#audio-timeout','30000');
    await page.fill('#max_tokens','72');await page.selectOption('#audio-text-model','text-analysis');
    await page.selectOption('#vad-gate','model:activity');
    for(const locale of ['en','es','ja','zh-CN']){
      await page.selectOption('#ui-locale',locale);
      assert.equal(await page.inputValue('#diarization-count'),'2');assert.equal(await page.inputValue('#diarization-model'),'speakers');
      assert((await page.locator('#results').textContent()).includes(composition.text));
      assert((await page.locator('#results').textContent()).includes(await page.evaluate(()=>TitanosI18n.t('audio.outside_timing'))));
      assert((await page.locator('#results').textContent()).includes(await page.evaluate(()=>TitanosI18n.t('audio.unattributed'))));
      const labels=await page.evaluate(()=>['task.diarization','audio.analyze','audio.anonymous_note','audio.result_summary'].map(name=>[name,TitanosI18n.t(name)]));
      assert(labels.every(([name,text])=>text!==name&&!text.startsWith('audio.')));
      if(locale!=='en')assert(await page.evaluate(()=>['task.diarization','audio.analyze','audio.anonymous_note'].every(name=>TitanosI18n.t(name)!==TitanosI18n.catalogs.en[name])));
      await page.click('#submit');await page.waitForFunction(()=>document.querySelector('#submit').disabled===false);
      const body=requests.at(-1).payload;
      assert.equal(body.max_tokens,72);assert.equal(body.timeout,30000);assert.equal(body.vad_model,'activity');
      assert.equal(body.text_analysis_model,'text-analysis');assert.deepEqual(body.diarization_options,{num_speakers:2,cluster_threshold:.55,min_duration_on:.1,min_duration_off:0});
    }
    await page.selectOption('#model','speakers');assert(await page.locator('#file').isVisible());assert(!(await page.locator('#prompt').isVisible()));
    await page.click('#submit');await page.waitForFunction(()=>document.querySelector('#submit').disabled===false);
    assert.equal(requests.at(-1).route,'/v1/audio/diarizations');assert.equal(requests.at(-1).payload.num_speakers,2);
    assert(!Object.hasOwn(requests.at(-1).payload,'diarization_model'));assert(!Object.hasOwn(requests.at(-1).payload,'vad_model'));
    const countBefore=await page.locator('#results>.result').count();hold=true;
    await page.click('#submit');await page.waitForFunction(()=>document.querySelector('#stop').hidden===false);
    while(!release)await new Promise(resolve=>setTimeout(resolve,10));
    await page.click('#stop');release();await page.waitForFunction(()=>document.querySelector('#submit').disabled===false);
    assert.equal(await page.locator('#results>.result').count(),countBefore,'Aborted result was rendered');
    assert.deepEqual(errors,[]);assert.deepEqual(unexpected,[]);
    console.log('PASS: headless Studio Simple/Advanced, four locales, snapshot controls, composition/standalone dispatch, overlap rendering, and Stop suppresses late results; intercepted responses only.');
  }finally{await context.close();await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
