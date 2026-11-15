'use strict';
// Execute the real Studio source through existing DOM/fetch doubles. No models/network.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
const filename=path.join(__dirname,'check_studio_lifecycle_ui.js');
const harness=fs.readFileSync(filename,'utf8').split('(async()=>{')[0];
const checks=String.raw`
(async()=>{
  await tick();await tick();
  const id='a'.repeat(32),plan={registry_name:'chat/model',provider:'huggingface',
    candidate_id:'publisher/chat',resolved_revision:'abc123',backend:'transformers',task:'chat',device:'cpu'};
  const record={id,status:'active',plan,progress:{stage:'prepared',completed:8,total:8},
    activation:{model:'chat/model',status:'loaded'}};
  const originalFetch=context.fetch;let polls=0,posts=0,activationCalls=0,failCatalog=false;
  context.fetch=async(route,options={})=>{
    if(route==='/v1/models'&&failCatalog)return failure(503,'Catalog unavailable');
    if(route==='/api/acquisitions'&&options.method==='POST'){posts++;return response(record);}
    if(route==='/api/acquisitions/'+id){polls++;return response(record);}
    if(route==='/api/acquisitions?limit=100')return response({data:[record]});
    if(route.endsWith('/activate')){activationCalls++;throw Error('An active record must not activate again');}
    return originalFetch(route,options);
  };
  context.setTimeout=()=>{throw Error('An active record must not schedule another poll');};
  get('prompt').value='Keep this input';
  await context.oneStopAcquire(plan);
  assert.equal(polls,1);assert.equal(posts,1);assert.equal(activationCalls,0);
  assert.equal(vm.runInContext('activeAcquisition',context),null);
  assert.equal(context.setupPending(),null);assert.equal(get('model').value,'chat/model');
  assert.equal(get('prompt').value,'Keep this input');
  assert.match(get('setup-message').textContent,/already registered/);
  assert.match(get('setup-plan').textContent,/Acquisition complete: chat\/model/);

  // Reconnect after server-side activation consumes the retained pending record.
  context.setupPending({id,key:'stable-key',plan:JSON.stringify(plan)});
  vm.runInContext('setupReconciled=false',context);polls=0;
  await context.reconcileAcquisitions();await tick();await tick();
  assert.equal(polls,1);assert.equal(context.setupPending(),null);
  assert.equal(vm.runInContext('activeAcquisition',context),null);
  // Another acquisition is no longer blocked by the completed one.
  await context.oneStopAcquire(plan);assert.equal(posts,2);assert.equal(activationCalls,0);

  // An alias reused for different weights must not be selected as the original model.
  sourceRevision='replacement456';get('model').value='beta';get('model').onchange();
  await context.oneStopAcquire(plan);
  assert.equal(get('model').value,'beta');assert.equal(context.setupPending(),null);
  assert.equal(vm.runInContext('activeAcquisition',context),null);
  assert.match(get('setup-message').textContent,/registration is missing or has changed/);
  sourceRevision='abc123';

  // Failed refresh cannot make a cached registration look newly confirmed.
  await context.refresh(true);get('model').value='beta';get('model').onchange();failCatalog=true;
  await context.oneStopAcquire(plan);
  assert.equal(get('model').value,'beta');assert.equal(context.setupPending(),null);
  assert.match(get('setup-message').textContent,/catalog could not be refreshed/);
  failCatalog=false;

  // Completing setup does not take over a separately active inference request.
  vm.runInContext("active={model:catalog.find(row=>row.id==='beta'),controller:new AbortController()}",context);
  const activeOwner=vm.runInContext('active',context);
  await context.oneStopAcquire(plan);
  assert.equal(vm.runInContext('active',context),activeOwner);
  assert.equal(get('model').value,'beta');assert.equal(activationCalls,0);
  vm.runInContext('active=null',context);
  console.log('PASS: active acquisition settles once, reconnect clears pending state, source mismatch is refused, unrelated inference preserved.');
})().catch(error=>{console.error(error);process.exitCode=1;});`;
const test=new Module(filename,module);test.filename=filename;
test.paths=Module._nodeModulePaths(path.dirname(filename));test._compile(harness+checks,filename);
