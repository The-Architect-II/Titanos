/* Actual headless browser against intercepted APIs only: no server/model/device calls. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {chromium}=require(process.env.TITANOS_PLAYWRIGHT_MODULE||'playwright');
const web=path.join(__dirname,'../src/titanos/web'),evidence=process.env.TITANOS_ACCELERATOR_UI_EVIDENCE||fs.mkdtempSync(path.join(os.tmpdir(),'titanos-accelerator-ui-'));
fs.mkdirSync(evidence,{recursive:true});
const A='GPU-11111111-1111-1111-1111-111111111111',B='GPU-22222222-2222-2222-2222-222222222222',C='GPU-33333333-3333-3333-3333-333333333333';
const models=[{id:'native',task:'chat',backend:'llama-cpp',device:'auto',status:'unloaded'},
 {id:'cpu-pin',task:'chat',backend:'llama-cpp',device:'cpu',status:'unloaded'},
 {id:'framework',task:'embeddings',backend:'transformers',device:'auto',status:'unloaded'},
 {id:'speakers',task:'diarization',backend:'sherpa-onnx',device:'cpu',status:'unloaded'}];
const views=new Map(models.map(m=>[m.id,{model:m.id,mode:'single',groups:[],replicas:1,tensor_parallel_size:2,configured_device:m.device,policy:m.device==='cpu'?'explicit_cpu':'automatic',automatic:true,observed:null}]));
let masked=false,managedProfile='cpu-profile',holdModel=null,releaseAssignment=null,acquired=null,nextAcquisitionStatus='prepared',activeProfile='cpu-profile',profileMetadataStatus='current';const posts=[],requests=[],unexpected=[],errors=[],checks=[];
function devices(model){
 const cpuOnly=model==='speakers';return {schema_version:2,object:'list',visibility:{constraints:masked?{CUDA_VISIBLE_DEVICES:''}:{}},probe_errors:[],qualification:null,
 data:[{id:'cpu',label:'CPU',present:true,installed:true,stable:true,assignable:true,compatible:true,device_type:'cpu',runtime_address:'cpu',signature:{runtime_vendor:'cpu',architecture:'x86_64'},visibility:{status:'visible'},reasons:[]},
 ...[A,B,C].map((id,index)=>({id,label:'Fixture GPU '+index,present:true,installed:true,stable:true,assignable:!masked&&!cpuOnly,compatible:!cpuOnly,device_type:'gpu',runtime_address:'cuda:'+index,memory_mib:24576,runtimes:['cuda'],signature:{runtime_vendor:'cuda',architecture:index===2?'8.6':'8.9'},visibility:{status:masked?'masked':'visible'},reasons:masked?[{code:'visibility_masked'}]:cpuOnly?[{code:'cpu_only'}]:[]}))]};
}
function view(id){const v=views.get(id),model=models.find(m=>m.id===id);return {...v,configured_device:model.device,selection:{status:v.policy==='automatic'?'recommended':'configured',device:v.groups?.[0]?.[0]|| (v.policy==='explicit_cpu'||masked?'cpu':A),observed:false},missing_devices:[],unavailable_devices:masked?v.groups.flat().filter(id=>id!=='cpu'):[],modes:caps(model).modes};}
function caps(model){return {task:model.task,backend:model.backend,source:{repository:model.repository,revision:model.revision},implemented:[model.task,'replicas'],setup:{status:'ready_for_load_check',issues:[],advisory:true},modes:{single:{implemented:true},replicas:{implemented:true},model_sharding:{implemented:model.id!=='speakers'},tensor_parallel:{implemented:model.id!=='speakers',runtime_available:null,reason:'runtime_unverified',collectives:model.id==='framework'?{cuda:{backend:'nccl',available:false,reason:'nccl_unavailable'},cpu:{backend:'gloo',available:null,reason:'runtime_unverified'}}:undefined,execution_backend:model.backend,experimental:model.backend==='llama-cpp'}}};}
async function main(){
 const browser=await chromium.launch({headless:true,executablePath:process.env.TITANOS_CHROME_PATH||undefined,args:['--disable-gpu','--disable-software-rasterizer','--renderer-process-limit=1']});
 const context=await browser.newContext({viewport:{width:1440,height:1200}}),page=await context.newPage();page.setDefaultTimeout(8000);page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/*',async route=>{
  const req=route.request(),url=new URL(req.url()),name=url.pathname;if(url.hostname!=='titanos.test'){unexpected.push(req.url());return route.abort();}
  const file=name==='/'?'index.html':name.startsWith('/ui/')?name.slice(4):null;
  if(file&&/^(index\.html|app\.js|i18n\.js|retrieval\.js|style\.css|titanos-[a-z0-9-]+\.svg)$/.test(file))return route.fulfill({status:200,contentType:({html:'text/html',js:'application/javascript',css:'text/css',svg:'image/svg+xml'})[file.split('.').at(-1)],body:fs.readFileSync(path.join(web,file))});
  const data=req.postData()?JSON.parse(req.postData()):null;requests.push({name,method:req.method(),data});if(req.method()==='POST')posts.push({name,data});let result;
  if(name==='/v1/models')result={data:models};
  else if(name==='/api/info')result={models:Object.fromEntries(models.map(m=>[m.id,{...m}])),settings:{max_tokens:{value:4096}},inference_contracts:{},model_setup:{runtime:{configured:true,profiles_route:'/api/runtime-profiles',metadata_status:'current',recommended_profile:'cuda-profile',profiles:[{id:'cuda-profile',title:'CUDA runtime',runtime_kind:'cuda',recommended:true,one_stop_supported:true},{id:'cpu-profile',title:'CPU runtime',runtime_kind:'cpu',one_stop_supported:true}]}}};
  else if(name==='/api/runtime-profiles')result={metadata_status:profileMetadataStatus,recommended:profileMetadataStatus==='current'?'cuda-profile':null,data:[{id:'cuda-profile',title:'CUDA runtime',runtime_kind:'cuda',recommended:profileMetadataStatus==='current',one_stop_supported:true},{id:'cpu-profile',title:'CPU runtime',runtime_kind:'cpu',one_stop_supported:true}]};
  else if(name==='/api/models/capabilities')result=caps(models.find(m=>m.id===data.model));
  else if(name==='/api/assignments/devices')result=devices(url.searchParams.get('model'));
  else if(name.startsWith('/api/assignments/')){const id=decodeURIComponent(name.split('/').at(-1));if(id===holdModel)await new Promise(resolve=>{releaseAssignment=resolve;});result=view(id);}
  else if(name==='/api/assignments'){
   const current=views.get(data.model),m=models.find(m=>m.id===data.model);if(data.policy)m.device=data.policy==='cpu'?'cpu':'auto';
   const policy=data.policy?(data.policy==='cpu'?'explicit_cpu':'automatic'):(data.groups.length?'pinned':current.policy);
   views.set(data.model,{...current,...data,policy,automatic:!data.groups.length,replicas:data.groups.length||1,observed:null});result=view(data.model);
  }else if(name==='/api/models/load'||name==='/api/models/unload'){
   const m=models.find(m=>m.id===data.model);m.status=name.endsWith('/unload')?'unloaded':'loaded';views.get(m.id).observed=m.status==='loaded'?{device:view(m.id).selection.device,label:'Fixture GPU 0',observed:true}:null;result={model:m.id,status:m.status};
  }else if(name==='/api/jobs'||name==='/api/acquisitions'||name==='/api/runtime-setups'){
   if(req.method()==='GET')result={data:[]};
   else if(name==='/api/runtime-setups'){managedProfile=data.profile;result={id:'runtime-test',status:'prepared',plan:{runtime:'llama.cpp',profile:data.profile,release:'fixture'},progress:{stage:'prepared',completed:1,total:1}};}
   else if(name==='/api/acquisitions'){
    const id=data.registry_name;if(!models.some(m=>m.id===id)){models.push({id,task:'chat',backend:'llama-cpp',device:data.device,status:'unloaded',repository:data.candidate_id||data.identifier,revision:data.resolved_revision});views.set(id,{model:id,mode:'single',groups:[],replicas:1,tensor_parallel_size:2,policy:data.device==='cpu'?'explicit_cpu':'automatic',automatic:true,observed:null});}
    result=acquired={id:'fixture-acquisition',status:nextAcquisitionStatus,plan:data,license:'fixture-only',activation:nextAcquisitionStatus==='active'?{model:id,runtime_profile:activeProfile}:undefined,progress:{stage:nextAcquisitionStatus,completed:1,total:1}};
   }
  }else if(name==='/api/acquisitions/fixture-acquisition/retry'){result=acquired={...acquired,id:'fixture-retry',status:'prepared'};}
  else if(name==='/api/acquisitions/fixture-acquisition'||name==='/api/acquisitions/fixture-retry')result=acquired;
  else if(name==='/api/acquisitions/fixture-acquisition/activate'||name==='/api/acquisitions/fixture-retry/activate'){
   assert.equal(data.runtime_profile,'cuda-profile');const m=models.find(m=>m.id===acquired.plan.registry_name);m.status='loaded';result=acquired={...acquired,status:'active',activation:{model:m.id,status:'loaded',runtime_profile:data.runtime_profile}};
  }
  else if(name==='/v1/extensions')result={};
  else if(name==='/api/model-providers')result={providers:[]};
  else if(name==='/api/model-providers/huggingface/search')result={data:[{id:'fixture/new',name:'Fixture GGUF',provider:'huggingface',titanos_task:'chat',recommended_backend:'llama-cpp'}]};
  else if(name==='/api/model-providers/inspect')result={files:[{path:'model-q4_k_m.gguf',size:1}]};
  else if(name==='/api/acquisitions/plan')result={...data,provider_title:'Fixture',resolved_revision:'fixture-revision',expected_bytes:1,license:'fixture-only',gated:false,files:[{path:data.model_file,size:1}],compatibility:{qualification:null},runtime:{status:'available',managed_profile:managedProfile,managed_executable:'fixture-not-executed'}};
  else if(name==='/api/runtime-setups/plan')result={runtime:'llama.cpp',profile:data.profile};
  else if(name==='/api/runtime-setups/runtime-test')result={id:'runtime-test',status:'prepared',plan:{runtime:'llama.cpp',profile:managedProfile,release:'fixture'},progress:{stage:'prepared',completed:1,total:1}};
  else{unexpected.push(name);return route.fulfill({status:404,contentType:'application/json',body:JSON.stringify({error:{message:'Unexpected fixture route'}})});}
  return route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(result)});
 });
 try{
  await page.goto('http://titanos.test/');await page.waitForFunction(()=>document.querySelector('#device-policy')?.disabled===false);
  // the device block is a one-line readout until opened (Simple keeps it collapsed); its readout names the policy, never a device
  assert.equal(await page.locator('#device-selection').evaluate(n=>n.tagName+':'+n.open),'DETAILS:false');assert.match(await page.locator('#device-line').innerText(),/Automatic/);
  await page.locator('#device-line').click();
  assert.equal(await page.locator('#device-policy').inputValue(),'auto');assert((await page.locator('#device-selection-text').innerText()).includes('Fixture GPU 0'));
  assert.match(await page.locator('#device-observed-text').innerText(),/not been observed/);assert.equal(posts.filter(x=>x.name==='/api/assignments').length,0);
  await page.click('#load');await page.waitForFunction(()=>document.querySelector('#device-observed-text').textContent.includes('Fixture GPU 0'));checks.push('Simple shows automatic GPU recommendation separately from loaded observation; refresh has no assignment mutation');
  await page.selectOption('#model','cpu-pin');await page.waitForFunction(()=>document.querySelector('#device-policy').value==='cpu');
  for(const locale of ['es','ja','zh-CN','en']){await page.selectOption('#ui-locale',locale);assert.equal(await page.locator('#device-policy').inputValue(),'cpu');assert(await page.locator('#device-policy').isVisible());}
  assert.equal(posts.filter(x=>x.name==='/api/assignments').length,0);
  await page.selectOption('#device-policy','auto');await page.waitForFunction(()=>document.querySelector('#device-policy').value==='auto'&&!document.querySelector('#device-policy').disabled);
  assert.deepEqual(posts.filter(x=>x.name==='/api/assignments').at(-1).data,{model:'cpu-pin',mode:'single',groups:[],policy:'auto'});checks.push('Explicit CPU survives locales; intentional Auto changes use the policy contract');
  await page.selectOption('#ui-mode','advanced');await page.selectOption('#model','native');await page.waitForFunction(()=>document.querySelector('#assignment-devices').options.length===4);
  await page.selectOption('#assignment-devices',[A,B]);await page.selectOption('#assignment-mode','model_sharding');
  for(const locale of ['es','ja','zh-CN','en']){await page.selectOption('#ui-locale',locale);assert.deepEqual(await page.locator('#assignment-devices').evaluate(n=>[...n.selectedOptions].map(x=>x.value)),[A,B]);assert.equal(await page.locator('#assignment-mode').inputValue(),'model_sharding');}
  await page.selectOption('#model','framework');await page.waitForFunction(()=>document.querySelector('#assignment-mode option[value="tensor_parallel"]').disabled);
  assert.match(await page.locator('#assignment-mode-limits').innerText(),/collective/);
  for(const [locale,word] of [['es','colectivo'],['ja','集団通信'],['zh-CN','集合通信'],['en','collective']]){await page.selectOption('#ui-locale',locale);assert((await page.locator('#assignment-mode-limits').innerText()).includes(word));}
  await page.selectOption('#assignment-devices','cpu');assert.equal(await page.locator('#assignment-mode option[value="tensor_parallel"]').getAttribute('disabled'),null);
  await page.selectOption('#model','native');await page.waitForFunction(()=>document.querySelector('#assignment-mode').value==='model_sharding');
  await page.click('#assignment-apply');await page.waitForFunction(()=>document.querySelector('#device-policy').value==='pinned');
  assert.deepEqual(posts.filter(x=>x.name==='/api/assignments').at(-1).data.groups,[[A,B]]);checks.push('Homogeneous sharding and per-model dirty drafts survive locale/model changes; framework TP differs from native tensor');
  await page.selectOption('#assignment-mode','single');await page.selectOption('#assignment-devices',[A,B]);await page.fill('#assignment-replicas','2');await page.click('#assignment-apply');
  await page.waitForFunction(()=>document.querySelector('#assignment-status').textContent.includes('2'));
  assert.deepEqual(posts.filter(x=>x.name==='/api/assignments').at(-1).data.groups,[[A],[B]]);checks.push('Independent replicas are distinct groups');
  await page.selectOption('#assignment-mode','model_sharding');await page.selectOption('#assignment-devices',[A,C]);const count=posts.filter(x=>x.name==='/api/assignments').length;await page.click('#assignment-apply');
  assert.equal(posts.filter(x=>x.name==='/api/assignments').length,count);assert.match(await page.locator('#assignment-status').innerText(),/same runtime vendor and architecture/);
  await page.click('#assignment-reset');await page.waitForFunction(()=>!document.querySelector('#device-policy').disabled);
  masked=true;const beforeMask=posts.filter(x=>x.name==='/api/assignments').length;await page.click('#device-refresh');await page.waitForFunction(()=>document.querySelector('#device-selection-text').textContent.includes('No automatic CPU fallback'));
  assert.equal(await page.locator('#device-policy').inputValue(),'pinned');assert.equal(posts.filter(x=>x.name==='/api/assignments').length,beforeMask);
  assert.match(await page.locator('#device-visibility').innerText(),/does not mean.*no GPU/);assert.equal(await page.locator(`#assignment-devices option[value="${A}"]`).getAttribute('disabled'),'');checks.push('Incompatible group refused without POST; masked installed GPUs remain visible and saved pins warn without CPU fallback');
  await page.selectOption('#device-policy','auto');await page.waitForFunction(()=>!document.querySelector('#device-policy').disabled);
  masked=false;await page.locator('#model-setup').evaluate(n=>n.open=true);await page.click('#setup-refresh-devices');await page.waitForFunction(()=>document.querySelector('#setup-device').options.length===5);
  await page.locator('#model-setup').evaluate(n=>n.open=true);assert.equal(await page.locator('#setup-runtime-profile').inputValue(),'cuda-profile');
  profileMetadataStatus='unverified';await page.click('#setup-refresh-devices');await page.waitForFunction(()=>!document.querySelector('#setup-refresh-devices').disabled);assert.equal(await page.locator('#setup-runtime-profile').inputValue(),'');assert.match(await page.locator('#setup-message').innerText(),/not been verified/);
  profileMetadataStatus='current';await page.click('#setup-refresh-devices');await page.waitForFunction(()=>!document.querySelector('#setup-refresh-devices').disabled);assert.equal(await page.locator('#setup-runtime-profile').inputValue(),'cuda-profile');checks.push('Metadata refresh recomputes runtime recommendation; unverified refresh has no automatic profile fallback');
  await page.click('#setup-search');await page.locator('#setup-results button').first().click();await page.waitForFunction(()=>document.querySelector('#setup-plan button')?.textContent.includes('Prepare selected'));
  assert.equal(posts.filter(x=>x.name==='/api/acquisitions/plan').at(-1).data.device,'auto');await page.locator('#setup-plan button').last().click();
  await page.waitForFunction(()=>document.querySelector('#model').value==='fixture/new');
  assert.equal(posts.filter(x=>x.name==='/api/runtime-setups/plan').at(-1).data.profile,'cuda-profile');assert.equal(managedProfile,'cuda-profile');
  assert.equal(posts.find(x=>x.name==='/api/acquisitions/fixture-acquisition/activate').data.runtime_profile,'cuda-profile');checks.push('Available old CPU runtime is not reused after recommended CUDA intent; activation binds that selected profile');
  await page.selectOption('#setup-runtime-profile','cpu-profile');await page.click('#setup-refresh-devices');await page.waitForFunction(()=>!document.querySelector('#setup-refresh-devices').disabled);await page.selectOption('#ui-locale','es');await page.selectOption('#ui-locale','en');assert.equal(await page.locator('#setup-runtime-profile').inputValue(),'cpu-profile');
  await page.locator('#setup-results button').first().click();await page.waitForFunction(()=>document.querySelector('#setup-plan button')?.textContent.includes('Prepare selected'));
  assert.equal(posts.filter(x=>x.name==='/api/acquisitions/plan').at(-1).data.device,'cpu');checks.push('Explicit CPU profile survives locale and maps GGUF plan to CPU');
  await page.selectOption('#setup-device',A);const planned=posts.filter(x=>x.name==='/api/acquisitions/plan').length;await page.locator('#setup-results button').first().click();
  await page.waitForFunction(()=>document.querySelector('#setup-message').textContent.includes('CPU runtime cannot'));
  assert.equal(posts.filter(x=>x.name==='/api/acquisitions/plan').length,planned);checks.push('Explicit GPU with CPU profile rejected before plan/download');
  const previousPlan=posts.find(x=>x.name==='/api/acquisitions/plan').data;
  const completePlan={...previousPlan,registry_name:'fixture/active-cpu',provider_title:'Fixture',resolved_revision:'fixture-revision',expected_bytes:1,license:'fixture-only',files:[],compatibility:{qualification:null},runtime:{status:'available',managed_profile:'cuda-profile'}};
  nextAcquisitionStatus='active';const activations=posts.filter(x=>x.name.endsWith('/activate')).length;
  await page.evaluate(plan=>oneStopAcquire(plan,'cuda-profile'),completePlan);assert.match(await page.locator('#setup-message').innerText(),/another runtime profile/);
  activeProfile=undefined;await page.evaluate(plan=>oneStopAcquire({...plan,registry_name:'fixture/active-unknown'},'cuda-profile'),completePlan);assert.match(await page.locator('#setup-message').innerText(),/did not record its runtime profile/);
  assert.equal(posts.filter(x=>x.name.endsWith('/activate')).length,activations);checks.push('Active mismatched or older unrecorded profiles warn without reactivation or automatic migration');
  nextAcquisitionStatus='failed';await page.evaluate(plan=>oneStopAcquire({...plan,registry_name:'fixture/retry'},'cuda-profile'),completePlan);
  await page.getByRole('button',{name:'Retry / resume',exact:true}).click();await page.waitForFunction(()=>document.querySelector('#model').value==='fixture/retry');
  assert.equal(posts.find(x=>x.name==='/api/acquisitions/fixture-retry/activate').data.runtime_profile,'cuda-profile');checks.push('Acquisition retry preserves the original selected profile through activation');
  await page.screenshot({path:path.join(evidence,'accelerator-studio-fixture.png'),fullPage:true});
  assert.deepEqual(errors,[]);assert.deepEqual(unexpected,[]);console.log(JSON.stringify({status:'PASSED_HEADLESS_ACCELERATOR_STUDIO',checks,requests:requests.length,model_calls:0,device_probes:0,server_bindings:0,primary_UI_actions:0,evidence},null,2));
  fs.writeFileSync(path.join(evidence,'RESULT.json'),JSON.stringify({status:'PASSED_HEADLESS_ACCELERATOR_STUDIO',checks,requests,errors,unexpected,scope:'Actual headless Chromium; every API/device/model/runtime response intercepted. No actual hardware, server binding or inference qualification.'},null,2));
 }finally{await browser.close();}
}
main().catch(error=>{fs.writeFileSync(path.join(evidence,'FAILED.json'),JSON.stringify({status:'FAILED',error:String(error),checks,requests,errors,unexpected},null,2));console.error(error);process.exitCode=1;});
