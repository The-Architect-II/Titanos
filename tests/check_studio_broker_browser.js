/* Studio 'Cloud and routing' end to end in a real browser (browser tier): Playwright drives the real Studio pages served
   by a real `titanos serve` subprocess whose broker.json points at the loopback OpenAI mock (tests/broker_mocks).

   Keyed serve (bootstrap key from the environment, vault refs-only):
   - Connect with the bootstrap key (kept in the tab only); the panel renders every admin section.
   - The credential form is write-only: a typed secret is never prefilled, the field is emptied and replaced on submit,
     and the refused vault write (503 vault_unavailable, refs-only) leaves no trace in the DOM, input values, storage or
     the response; then an env-sourced credential (the variable is set only in the serve environment) is saved,
     validated against the mock and bound to its upstream; the server lists it by fingerprint and variable name only.
   - The upstream is enabled from its toggle; after Refresh the cloud alias appears with its External badge.
   - A chat through the alias streams the mock's answer and shows the provenance line; the decision is listed.
   - Design contract spot checks on the rendered panel and result (computed styles): one lime action (Connect), no
     gradients or shadows, uppercase and tracking only on the numbered plates, radii at most 6px, the error colour on
     danger buttons, the cyan External badge, monospace provenance, no horizontal overflow.
   Keyless serve (R29-R31): Studio sends X-Titanos-Studio: 1 on its broker requests and is the keyless-studio principal:
   GET /api/broker/env-candidates lists OPENAI_API_KEY by name; Arrow keys on the closed Provider select step through the
   providers (outside macOS) and focus stays on it through each re-render, then on Connect after Enter (WCAG 2.4.3, 3.2.2);
   Connect a provider (in Simple mode, by keyboard only:
   Arrow Up to the environment option, Tab through the variable picker and the key field to Connect with the chosen source
   unchanged at every stop, Enter) by that variable
   name posts only the name, and because the upstream's base URL is the loopback mock (not the profile's default
   address) the server refuses it (403 scope_denied), which Studio reports with the server's own reason and no success
   claim; a pasted key (the keyless serve uses a sealed-file vault), submitted by Enter in the key field, keeps focus in
   the fresh key field and is sent once, validated against the mock, the
   upstream enabled, keyless cloud use allowed and the models listed, and the key leaves no trace in the page; the
   cloud alias then chats keyless with its provenance; the same writes without the Studio header are refused with 403
   keyless_mode_refused, and keys are never minted keylessly.
   Throughout: the mock only ever sees its own sentinel credential (never the bootstrap key, never the typed secret),
   both servers run under the netguard sitecustomize with TITANOS_BROKER_EGRESS=loopback-only and record zero
   non-loopback attempts, and the browser makes no request outside 127.0.0.1.

   Environment: TITANOS_PYTHON (default python) runs serve and the mock with PYTHONPATH=src; TITANOS_PLAYWRIGHT_MODULE,
   NODE_PATH or a node_modules beside the checkout or the working directory provide playwright; TITANOS_CHROME_PATH or
   TITANOS_CHROME_CHANNEL (e.g. chrome) pick a browser, otherwise Playwright's Chromium; TITANOS_BROKER_BROWSER_EVIDENCE
   names a directory for screenshots;
   TITANOS_BROKER_BROWSER_DEBUG=1 logs the page's requests. Every credential here is synthetic. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const net=require('node:net'),crypto=require('node:crypto'),{spawn,execFileSync}=require('node:child_process');

const ROOT=path.resolve(__dirname,'..');
const PYTHON=process.env.TITANOS_PYTHON||'python';
const EVIDENCE=process.env.TITANOS_BROKER_BROWSER_EVIDENCE||'';
const BOOTSTRAP='studio-e2e-bootstrap-'+crypto.randomBytes(16).toString('hex');
const TYPED_SECRET='sk-test-studio-typed-'+crypto.randomBytes(12).toString('hex');
const ENV_VAR='STUDIO_E2E_OPENAI_CREDENTIAL';
const LIME='rgb(166, 255, 0)',ERROR='rgb(255, 77, 109)',CYAN='rgb(50, 212, 255)';
const STARTUP_MS=Number(process.env.TITANOS_BROKER_BROWSER_STARTUP_MS||90000);

// the broker matrix run (tools/build_broker_matrix.py --run) reads one results line per check from TITANOS_BROKER_RESULTS
function reportMatrix(passed,failed,skipped={}){
  const target=process.env.TITANOS_BROKER_RESULTS;
  if(target)fs.appendFileSync(target,JSON.stringify({check:path.basename(__filename),passed,failed,skipped})+'\n');
}
const CASE='Studio.providers_panel_against_a_real_serve';

function loadPlaywright(){
  const tried=[];
  const names=[process.env.TITANOS_PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean);
  const paths=[process.cwd(),ROOT,...String(process.env.NODE_PATH||'').split(path.delimiter).filter(Boolean)];
  for(const name of names){
    try{return require(name);}catch(error){tried.push(name+': '+error.code);}
    try{return require(require.resolve(name,{paths}));}catch(error){tried.push(name+' (paths): '+error.code);}
  }
  throw Error('playwright is not installed (npm install --no-save --ignore-scripts playwright): '+tried.join('; '));
}

// --- processes ------------------------------------------------------------------------------------------------------
function cleanEnv(extra){
  // provider variables, proxies, base-URL overrides and Titanos settings never reach the subprocesses
  const env={};
  for(const [name,value] of Object.entries(process.env)){
    if(/^TITANOS_/i.test(name)||/(_API_KEY|_TOKEN|_BASE_URL|_KEY)$/i.test(name)||/^(HTTPS?|ALL|NO)_PROXY$/i.test(name))continue;
    if(/^(PYTHONPATH|PYTHONHOME|PYTHONSTARTUP|VIRTUAL_ENV)$/i.test(name))continue;
    env[name]=value;
  }
  return {...env,PYTHONDONTWRITEBYTECODE:'1',PYTHONIOENCODING:'utf-8',PYTHONUNBUFFERED:'1',...extra};
}
// a crash (or Ctrl+C) never leaves a server or the mock behind
const LIVE=new Set();
process.on('exit',()=>{for(const child of LIVE)try{child.kill();}catch{}});
function track(child){LIVE.add(child);child.once('exit',()=>LIVE.delete(child));return child;}
function freePort(){
  return new Promise((resolve,reject)=>{const server=net.createServer();server.unref();server.on('error',reject);
    server.listen(0,'127.0.0.1',()=>{const {port}=server.address();server.close(()=>resolve(port));});});
}
function lines(child,onLine){
  let pending='';child.stdout.setEncoding('utf8');
  child.stdout.on('data',chunk=>{pending+=chunk;let at;while((at=pending.indexOf('\n'))>=0){const line=pending.slice(0,at).trim();pending=pending.slice(at+1);if(line)onLine(line);}});
}
function tail(text){return String(text).split(/\r?\n/).slice(-40).join('\n');}
async function stopChild(child,label,graceMs=15000){
  if(!child||child.exitCode!==null||child.signalCode!==null)return;
  const exited=new Promise(resolve=>child.once('exit',resolve));
  child.kill();
  const timer=new Promise(resolve=>setTimeout(()=>resolve('timeout'),graceMs));
  if(await Promise.race([exited,timer])==='timeout'){child.kill('SIGKILL');await Promise.race([exited,new Promise(r=>setTimeout(r,5000))]);}
}

// The loopback OpenAI mock (tests/broker_mocks) in its own Python process, guarded by netguard. A JSON line protocol
// on stdin/stdout: the ready line carries the api_base and the mock's own (synthetic) sentinel credential; {"op":"hits",
// "forbidden":[...]} answers the recorded requests (credential spellings are the mock's redacted '<sentinel:...>' or
// '<foreign:...>' forms; forbidden values are reported by index only); {"op":"stop"} stops it.
const MOCK_SCRIPT=String.raw`
import json, sys
sys.path.insert(0, 'tests')
from broker_mocks import netguard
netguard.install()
from broker_mocks.fixtures import SENTINELS
from broker_mocks.openai_mock import OpenAIMock
mock = OpenAIMock(models=['gpt-4o-mini'])
mock.start()
print(json.dumps({'ready': True, 'api_base': mock.api_base, 'sentinel': SENTINELS['openai']}), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command.get('op') == 'hits':
        forbidden = command.get('forbidden', [])
        rows = [{'method': h.method, 'path': h.path, 'surface': h.surface, 'status': h.status, 'credential': h.credential,
                 'forbidden': [i for i, value in enumerate(forbidden) if h.presented(value)]} for h in list(mock.hits)]
        print(json.dumps({'hits': rows, 'blocked': netguard.blocked_count()}), flush=True)
    elif command.get('op') == 'stop':
        break
mock.stop()
print(json.dumps({'stopped': True}), flush=True)
`;
class MockProcess{
  constructor(report){this.report=report;this.waiters=[];this.stderr='';}
  async start(){
    this.child=track(spawn(PYTHON,['-B','-c',MOCK_SCRIPT],{cwd:ROOT,env:cleanEnv({PYTHONPATH:path.join(ROOT,'src'),TITANOS_BROKER_EGRESS:'loopback-only',TITANOS_TEST_NETGUARD_REPORT:this.report}),stdio:['pipe','pipe','pipe']}));
    this.child.stderr.setEncoding('utf8');this.child.stderr.on('data',chunk=>{this.stderr=(this.stderr+chunk).slice(-20000);});
    lines(this.child,line=>{let value;try{value=JSON.parse(line);}catch{return;}const waiter=this.waiters.shift();if(waiter)waiter(value);});
    this.child.on('exit',()=>{for(const waiter of this.waiters.splice(0))waiter(null);});
    const ready=await this.next(STARTUP_MS);
    if(!ready?.ready)throw Error('the mock process did not start:\n'+tail(this.stderr));
    this.apiBase=ready.api_base;this.sentinel=ready.sentinel;return this;
  }
  next(ms=20000){return new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('the mock process did not answer:\n'+tail(this.stderr))),ms);this.waiters.push(value=>{clearTimeout(timer);resolve(value);});});}
  async send(command){const answer=this.next();this.child.stdin.write(JSON.stringify(command)+'\n');return answer;}
  async hits(forbidden){const answer=await this.send({op:'hits',forbidden});if(!answer)throw Error('the mock process exited:\n'+tail(this.stderr));return answer;}
  async stop(){
    if(!this.child||this.child.exitCode!==null)return;
    try{await this.send({op:'stop'});}catch{}
    await stopChild(this.child,'mock',10000);
  }
}

class Serve{
  constructor(name,home,{bootstrap=null,env={}}={}){this.name=name;this.home=home;this.bootstrap=bootstrap;this.extra=env;this.output='';this.report=path.join(home,'netguard-'+name+'.jsonl');}
  async start(){
    this.port=await freePort();this.base='http://127.0.0.1:'+this.port;
    const env=cleanEnv({PYTHONPATH:[path.join(ROOT,'src'),path.join(ROOT,'tests','broker_mocks','netguard_site')].join(path.delimiter),
      TITANOS_TEST_NETGUARD:'1',TITANOS_TEST_NETGUARD_REPORT:this.report,TITANOS_HOME:this.home,TITANOS_BROKER_EGRESS:'loopback-only',
      TITANOS_BROKER_VAULT_BACKEND:'refs-only',...(this.bootstrap?{TITANOS_API_KEY:this.bootstrap}:{}),...this.extra});
    this.child=track(spawn(PYTHON,['-B','-m','titanos','--config',path.join(this.home,'models.json'),'serve','--host','127.0.0.1','--port',String(this.port)],
      {cwd:ROOT,env,stdio:['ignore','pipe','pipe']}));
    for(const stream of [this.child.stdout,this.child.stderr]){stream.setEncoding('utf8');stream.on('data',chunk=>{this.output=(this.output+chunk).slice(-40000);});}
    const deadline=Date.now()+STARTUP_MS;
    while(Date.now()<deadline){
      if(this.child.exitCode!==null)throw Error(`serve (${this.name}) exited with ${this.child.exitCode}:\n`+tail(this.output));
      try{const answer=await fetch(this.base+'/healthz',{signal:AbortSignal.timeout(2000)});if(answer.ok)return this;}catch{}
      await new Promise(resolve=>setTimeout(resolve,250));
    }
    throw Error(`serve (${this.name}) did not answer /healthz within ${STARTUP_MS} ms:\n`+tail(this.output));
  }
  async stop(){await stopChild(this.child,this.name);}
  blocked(){try{return fs.readFileSync(this.report,'utf8').split(/\r?\n/).filter(Boolean);}catch(error){if(error.code==='ENOENT')return [];throw error;}}
}

function writeHome(home,apiBase,{enabled=false}={}){
  fs.mkdirSync(home,{recursive:true});
  fs.writeFileSync(path.join(home,'models.json'),JSON.stringify({version:1,models:{},workers:{}},null,2)+'\n');
  const broker={version:1,policy:{cloud_policy:'explicit'},
    upstreams:{openai:{profile:'openai',base_url:apiBase,enabled,models:['gpt-4o-mini'],catalog:'static',label:'OpenAI mock'}},
    models:{'cloud-mini':{target:'gpt-4o-mini@openai'}}};
  fs.writeFileSync(path.join(home,'broker.json'),JSON.stringify(broker,null,2)+'\n');
}

// --- browser helpers ------------------------------------------------------------------------------------------------
async function newPage(browser,offsite){
  const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();
  page.setDefaultTimeout(20000);
  const errors=[],dialogs=[];
  page.on('pageerror',error=>errors.push(error.message));
  if(process.env.TITANOS_BROKER_BROWSER_DEBUG){page.on('request',r=>console.error('>>',r.method(),r.url()));page.on('response',r=>console.error('<<',r.status(),r.url()));page.on('console',m=>console.error('console',m.text()));}
  page.on('dialog',dialog=>{dialogs.push(dialog.message());dialog.accept();});
  // nothing leaves 127.0.0.1: any other destination is aborted and recorded
  await context.route('**/*',route=>{const url=new URL(route.request().url());
    if(url.hostname==='127.0.0.1'||url.protocol==='data:'||url.protocol==='blob:')return route.continue();
    offsite.push(url.origin);return route.abort();});
  return {context,page,errors,dialogs};
}
async function advanced(page){await page.selectOption('#ui-mode','advanced');}
async function openPanel(page){if(!await page.evaluate(()=>document.getElementById('broker-panel').open))await page.click('#broker-panel > summary');
  await page.waitForSelector('#broker-connect',{state:'attached'});await openFolds(page);}
async function openFolds(page){await page.evaluate(()=>{for(const node of document.querySelectorAll('#broker-body details.broker-section'))if(!node.open)node.querySelector(':scope > summary').click();});}
async function shot(page,name,selector=null){if(!EVIDENCE)return;fs.mkdirSync(EVIDENCE,{recursive:true});const file=path.join(EVIDENCE,name+'.png');if(selector)await page.locator(selector).screenshot({path:file});else await page.screenshot({path:file,fullPage:true});}
function waitPut(page,pathname){
  // settled even when it is never awaited (an earlier assertion failed), so a timeout cannot surface as an unhandled rejection
  const pending=page.waitForResponse(response=>response.request().method()==='PUT'&&new URL(response.url()).pathname===pathname);
  pending.catch(()=>{});return pending;
}
async function traces(page,values){
  // every place a typed or configured secret could linger in the page: markup, live input values, both storages
  return page.evaluate(values=>{
    const inputs=[...document.querySelectorAll('input,textarea,select')].map(node=>String(node.value||''));
    const storage=[];for(const store of [localStorage,sessionStorage])for(let i=0;i<store.length;i++){const name=store.key(i);storage.push(name+'='+store.getItem(name));}
    const html=document.documentElement.outerHTML;
    return values.map(value=>({inHtml:html.includes(value),inInputs:inputs.some(text=>text.includes(value)),inStorage:storage.some(text=>text.includes(value))}));
  },values);
}
async function designSpotChecks(page){
  return page.evaluate(({LIME})=>{
    const panel=document.getElementById('broker-panel'),problems=[];
    const visible=node=>{const box=node.getBoundingClientRect();return box.width>0&&box.height>0&&getComputedStyle(node).visibility!=='hidden'&&node.checkVisibility();};
    const scope=[panel,...panel.querySelectorAll('*'),...document.querySelectorAll('.result, .result *')];
    for(const node of scope){
      const style=getComputedStyle(node),name=node.id?'#'+node.id:node.tagName.toLowerCase()+(node.className&&typeof node.className==='string'?'.'+node.className.trim().replace(/\s+/g,'.'):'');
      const plate=Boolean(node.closest('.bay-plate'));
      if(/gradient\(/.test(style.backgroundImage))problems.push(name+' background-image '+style.backgroundImage);
      if(style.boxShadow!=='none')problems.push(name+' box-shadow '+style.boxShadow);
      if(style.textShadow!=='none')problems.push(name+' text-shadow '+style.textShadow);
      if(style.textTransform!=='none'&&!plate)problems.push(name+' text-transform '+style.textTransform);
      if(style.letterSpacing!=='normal'&&parseFloat(style.letterSpacing)!==0&&!plate)problems.push(name+' letter-spacing '+style.letterSpacing);
      if(/blur\(/.test(style.filter)||/blur\(/.test(style.backdropFilter||''))problems.push(name+' blur '+style.filter);
      for(const corner of ['borderTopLeftRadius','borderTopRightRadius','borderBottomLeftRadius','borderBottomRightRadius'])
        if(parseFloat(style[corner])>6||String(style[corner]).includes('%'))problems.push(name+' '+corner+' '+style[corner]);
    }
    const lime=[...panel.querySelectorAll('button')].filter(node=>visible(node)&&getComputedStyle(node).backgroundColor===LIME).map(node=>node.textContent.trim());
    // one lime fill per bay across the whole page (RUN in 03, Connect in 07); every other button is quiet or violet
    const limeByBay={};for(const node of document.querySelectorAll('button'))if(visible(node)&&getComputedStyle(node).backgroundColor===LIME){const bay=node.closest('.bay,header,nav')?.id||'(none)';(limeByBay[bay]=limeByBay[bay]||[]).push(node.textContent.trim());}
    const danger=[...panel.querySelectorAll('button.danger')].filter(visible).map(node=>getComputedStyle(node).color);
    const badge=document.querySelector('.model[data-model-id="cloud-mini"] .model-badge');
    const provenance=document.querySelector('.result .provenance');
    return {problems,lime,limeByBay,danger,badgeColor:badge&&getComputedStyle(badge).color,
      provenanceFont:provenance&&getComputedStyle(provenance).fontFamily,
      overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth};
  },{LIME});
}

// --- the check ------------------------------------------------------------------------------------------------------
async function main(){
  let playwright;
  try{playwright=loadPlaywright();}catch(error){
    // under the matrix run only, a host without Playwright records the lane as untried (never as a pass)
    if(process.env.TITANOS_BROKER_RESULTS){reportMatrix([],[],{[CASE]:'untried: '+error.message.split(':')[0]});console.log('UNTRIED: '+error.message);return;}
    throw error;
  }
  const {chromium}=playwright;
  const temp=fs.mkdtempSync(path.join(os.tmpdir(),'titanos-studio-broker-'));
  const offsite=[],children=[];let browser=null,mock=null;
  try{
    mock=await new MockProcess(path.join(temp,'netguard-mock.jsonl')).start();
    const keyedHome=path.join(temp,'keyed'),keylessHome=path.join(temp,'keyless');
    writeHome(keyedHome,mock.apiBase);writeHome(keylessHome,mock.apiBase);
    const keyed=new Serve('keyed',keyedHome,{bootstrap:BOOTSTRAP,env:{[ENV_VAR]:mock.sentinel}});children.push(keyed);
    // the keyless serve keeps pasted keys in a sealed-file vault; its passphrase file is written owner-only by Titanos itself
    const passphrase=path.join(temp,'vault-passphrase');
    execFileSync(PYTHON,['-B','-c','import sys\nfrom titanos.broker.vault.acl import write_owner_only\nwrite_owner_only(sys.argv[1], b"studio-browser-check-synthetic-passphrase")',passphrase],
      {cwd:ROOT,env:cleanEnv({PYTHONPATH:path.join(ROOT,'src')}),stdio:['ignore','pipe','pipe']});
    const keyless=new Serve('keyless',keylessHome,{env:{OPENAI_API_KEY:mock.sentinel,TITANOS_BROKER_VAULT_BACKEND:'sealed-file',TITANOS_BROKER_VAULT_PASSPHRASE_FILE:passphrase}});children.push(keyless);
    await Promise.all([keyed.start(),keyless.start()]);
    const launch={headless:true,args:['--disable-gpu','--renderer-process-limit=2','--no-first-run','--no-default-browser-check']};
    if(process.env.TITANOS_CHROME_PATH)launch.executablePath=process.env.TITANOS_CHROME_PATH;
    else if(process.env.TITANOS_CHROME_CHANNEL)launch.channel=process.env.TITANOS_CHROME_CHANNEL;
    browser=await chromium.launch(launch);

    // ===== keyed serve: connect with the bootstrap key =====
    const a=await newPage(browser,offsite),page=a.page;
    await page.goto(keyed.base+'/');await advanced(page);
    await page.fill('#key',BOOTSTRAP);await page.click('#connect');
    assert.equal(await page.inputValue('#key'),'','the key field keeps the key after Connect');
    await page.waitForSelector('#broker-credential-form',{state:'attached'});
    assert(await page.isVisible('#broker-panel'),'the Cloud and routing panel is not shown to the bootstrap key');
    await openPanel(page);
    for(const id of ['broker-role','broker-upstreams','broker-credential-form','broker-alias-form','broker-policy-form','broker-usage','broker-decisions','broker-keys'])
      assert.equal(await page.locator('#'+id).count(),1,'missing #'+id);
    assert.match(await page.textContent('#broker-vault'),/refs-only/);
    const secretField=page.locator('#broker-secret');
    assert.equal(await secretField.inputValue(),'','the secret field is prefilled');
    assert.deepEqual(await secretField.evaluate(node=>[node.type,node.getAttribute('autocomplete'),node.getAttribute('spellcheck'),node.getAttribute('autocapitalize')]),['password','off','false','off']);
    assert.match(await page.textContent('#broker-upstreams'),/Disabled · breaker/);

    // (1) write-only secret path: the refs-only vault refuses a typed value; nothing of it survives anywhere
    await page.selectOption('#broker-credential-source','secret');
    await page.fill('#broker-credential-id','openai-typed');
    await secretField.fill(TYPED_SECRET);
    const typedField=await secretField.elementHandle(),typedForm=await page.$('#broker-credential-form');
    let answer=waitPut(page,'/api/broker/credentials/openai-typed');
    // submitted by Enter in the secret field: focus follows the swapped-in field through the refusal's re-render (WCAG 2.4.3)
    await secretField.focus();await page.keyboard.press('Enter');
    let response=await answer;
    assert.equal(response.status(),503,'a refs-only vault accepted a typed secret');
    const refusal=await response.text();
    assert.equal(JSON.parse(refusal).error.code,'vault_unavailable');assert(!refusal.includes(TYPED_SECRET),'the refusal echoes the typed secret');
    await page.waitForFunction(()=>document.querySelector('#broker-message.error')?.textContent.includes('references only'));
    await page.waitForFunction(form=>!form.isConnected,typedForm);   // the panel re-rendered after the refusal
    assert.equal(await typedField.evaluate(node=>[node.value,node.isConnected].join('|')),'|false','the submitted field was not emptied and replaced');
    assert.equal(await page.inputValue('#broker-secret'),'');
    assert.equal(await page.evaluate(()=>document.activeElement?.id||document.activeElement?.tagName),'broker-secret','after Enter in the secret field focus left it (WCAG 2.4.3)');
    assert.deepEqual(await traces(page,[TYPED_SECRET]),[{inHtml:false,inInputs:false,inStorage:false}],'the typed secret lingers in the page');

    // (2) env-sourced credential: only the variable name crosses the wire; validated against the mock and bound
    await page.fill('#broker-credential-id','');
    assert.match(await page.textContent('#broker-credential-expected'),/use openai-key\./);
    await page.selectOption('#broker-credential-source','env');
    await page.fill('#broker-credential-env',ENV_VAR);
    answer=waitPut(page,'/api/broker/credentials/openai-key');const bind=waitPut(page,'/api/broker/upstreams/openai');
    await page.click('#broker-credential-form button[type=submit]');
    response=await answer;
    assert.deepEqual(response.request().postDataJSON(),{upstream:'openai',env:ENV_VAR},'the credential write sent more than the variable name');
    assert.equal(response.status(),201,'the env-sourced credential was not saved: '+await response.text());
    const saved=await response.json();
    assert.equal(saved.source,'env');assert.equal(saved.env_var,ENV_VAR);assert.equal(saved.validation,'ok');assert.match(String(saved.fingerprint),/^[0-9a-f]{8,}$/);
    assert(!JSON.stringify(saved).includes(mock.sentinel),'the credential metadata carries the value');
    response=await bind;assert.equal(response.status(),200);assert.equal(response.request().postDataJSON().credential,'openai-key');
    await page.waitForFunction(()=>/bound to openai/.test(document.getElementById('broker-message')?.textContent||''));
    assert.deepEqual(a.dialogs,[],'binding a first credential asked for confirmation');
    const shown=JSON.parse(await page.textContent('#broker-credential-result'));
    assert.deepEqual(Object.keys(shown).sort(),['fingerprint','id','validated_at','validation']);assert.equal(shown.id,'openai-key');
    await page.waitForFunction(fingerprint=>{const text=document.getElementById('broker-upstreams')?.textContent||'';return text.includes(fingerprint)&&/Validation\.*\s*ok · /.test(text.replace(/\s+/g,' '));},saved.fingerprint);

    // (3) enable the upstream from its toggle
    answer=waitPut(page,'/api/broker/upstreams/openai');
    await page.locator('#broker-upstreams .broker-item .broker-toggle input[type=checkbox]').check();
    response=await answer;assert.equal(response.status(),200);
    const enabled=response.request().postDataJSON();
    assert.equal(enabled.enabled,true);assert.equal(enabled.credential,'openai-key');assert.equal(enabled.base_url,mock.apiBase);
    for(const name of ['id','ready','readiness','credential_validation','catalog_status','credentials_reset'])assert(!(name in enabled),'the upstream write carried read-only '+name);
    await page.waitForFunction(()=>/Enabled · breaker/.test(document.getElementById('broker-upstreams')?.textContent||''));

    // (4) the cloud alias appears after Refresh, with its External badge and the placement select
    await page.click('#refresh');
    await page.waitForFunction(()=>[...document.querySelectorAll('#model option')].some(option=>option.value==='cloud-mini'));
    assert.equal(await page.isVisible('#location-label'),true,'the placement select is hidden with an enabled upstream');
    await page.selectOption('#model','cloud-mini');
    assert.equal((await page.textContent('.model[data-model-id="cloud-mini"] .model-badge')).trim(),'OpenAI mock');

    // (5) chat through the cloud alias: streamed answer and the provenance line
    await page.fill('#prompt','Hello from the Studio browser check.');
    const chat=page.waitForResponse(response=>response.request().method()==='POST'&&new URL(response.url()).pathname==='/v1/chat/completions');
    await page.click('#submit');
    response=await chat;assert.equal(response.status(),200);
    const sent=response.request().postDataJSON();assert.equal(sent.model,'cloud-mini');assert.equal(sent.stream,true);
    assert.equal(response.headers()['x-titanos-served-by'],'cloud:openai');
    await page.waitForFunction(()=>/Hello! How can I help\?/.test(document.querySelector('#results .result')?.textContent||''));
    const provenance=(await page.textContent('#results .result .provenance')).trim();
    assert.equal(provenance,'Served by cloud:openai · gpt-4o-mini@openai · fallbacks 0');
    await page.waitForFunction(()=>!document.getElementById('submit').disabled);
    await shot(page,'keyed-chat');

    // (6) the decision is listed (metadata only) and the server lists the credential by reference only
    await page.click('#refresh');
    await page.waitForFunction(()=>/cloud-mini/.test(document.getElementById('broker-decisions')?.textContent||''));
    assert.match(await page.textContent('#broker-decisions'),/gpt-4o-mini@openai/);
    assert(!(await page.textContent('#broker-decisions')).includes('Hello from the Studio browser check.'),'the decision table shows the prompt');
    const listed=await page.evaluate(async key=>{const answer=await fetch('/api/broker/credentials',{headers:{Authorization:'Bearer '+key}});return {status:answer.status,cache:answer.headers.get('cache-control'),body:await answer.text()};},BOOTSTRAP);
    assert.equal(listed.status,200);assert.equal(listed.cache,'no-store');
    const records=JSON.parse(listed.body).data;
    assert.deepEqual(records.map(row=>row.id),['openai-key'],'the refused typed secret left a record');
    assert.equal(records[0].source,'env');
    for(const value of [mock.sentinel,TYPED_SECRET,BOOTSTRAP])assert(!listed.body.includes(value),'the credential list carries a secret');

    // (7) design contract spot checks on the rendered panel and result
    await page.mouse.move(1,1);await openPanel(page);
    const design=await designSpotChecks(page);
    assert.deepEqual(design.problems,[],'design contract');
    assert.deepEqual(design.lime,['Connect'],'exactly one lime action in the panel');
    for(const [bay,labels] of Object.entries(design.limeByBay))assert(labels.length<=1,'more than one lime action in '+bay+': '+labels.join(', '));
    assert.deepEqual(design.limeByBay['broker-panel'],['Connect']);
    // RUN clips itself to its chamfer: its keyboard focus ring must still be painted (on the unclipped wrapper)
    {const run=page.locator('#submit');await run.scrollIntoViewIfNeeded();await page.evaluate(()=>document.activeElement?.blur?.());
      const box=await run.boundingBox(),clip={x:Math.max(0,box.x-8),y:Math.max(0,box.y-8),width:box.width+16,height:box.height+16};
      const before=await page.screenshot({clip});await run.focus();await page.keyboard.press('Shift+Tab');await page.keyboard.press('Tab');
      assert(await run.evaluate(node=>node.matches(':focus-visible')),'RUN is not keyboard focused');
      assert.notEqual(Buffer.compare(before,await page.screenshot({clip})),0,'RUN shows no visible keyboard focus');}
    assert.equal(await page.getAttribute('.model[data-model-id="cloud-mini"] .model-technical summary','aria-label'),'Details for cloud-mini','the rail Details control names its model');
    assert(design.danger.length>0&&design.danger.every(color=>color===ERROR),'Remove/Revoke use the error colour: '+design.danger);
    assert.equal(design.badgeColor,CYAN,'the External badge is cyan: the route leaves this computer');
    assert.match(String(design.provenanceFont),/monospace/,'the provenance line is monospace');
    assert(design.overflow<=0,'horizontal overflow of '+design.overflow+'px');
    assert.deepEqual(await traces(page,[TYPED_SECRET,mock.sentinel,BOOTSTRAP]),Array(3).fill({inHtml:false,inInputs:false,inStorage:false}),'a secret lingers in the page');
    await shot(page,'keyed-panel','#broker-panel');
    assert.deepEqual(a.errors,[],'page errors (keyed)');
    await a.context.close();

    // ===== keyless serve: the R29 Studio principal, R30 environment keys by name and R31 one-step connect =====
    const b=await newPage(browser,offsite),open=b.page;
    const discovery=open.waitForRequest(request=>new URL(request.url()).pathname==='/api/broker/env-candidates');
    await open.goto(keyless.base+'/');
    const discovered=await discovery;assert.equal((await discovery).headers()['x-titanos-studio'],'1','Studio did not mark its broker request');
    const candidates=await(await discovered.response()).json();
    const openaiVars=candidates.data.find(row=>row.provider==='openai').variables;
    assert.deepEqual(openaiVars,[{name:'OPENAI_API_KEY',set:true}],'env discovery reports names and booleans only');
    assert(!JSON.stringify(candidates).includes(mock.sentinel),'env discovery carries a value');
    await openPanel(open);
    assert.equal(await open.locator('#broker-keyless-note').count(),0,'the keyless note replaced the Studio forms');
    assert(await open.isVisible('#broker-connect-form'),'Connect a provider is not shown in Simple mode');
    // the write-only key field and the variable picker carry their option text as accessible names
    assert.equal(await open.getByRole('textbox',{name:'Paste a key (sent once, never shown again)'}).count(),1,'the pasted-key field has no accessible name');
    assert.equal(await open.getByRole('combobox',{name:"Use a key from this server's environment"}).count(),1,'the variable picker has no accessible name');
    assert.equal(await open.isDisabled('#broker-anonymous-cloud'),false);assert.equal(await open.isDisabled('#broker-allow-http-lan'),true);
    // the provider by keyboard (WCAG 2.4.3, 3.2.2): an Arrow on the closed select changes it (and fires change) outside macOS;
    // the panel re-renders for the new provider and focus stays on the Provider select, so the next Arrow moves on again
    const focusAt=()=>open.evaluate(()=>{const node=document.activeElement;return {id:node?.id||'',tag:node?.tagName||'',connect:Boolean(node?.matches?.('#broker-connect-form button[type=submit]')),
      value:document.getElementById('broker-connect-provider')?.value||'',inConnect:Boolean(node?.closest?.('#broker-connect'))};});
    if(process.platform!=='darwin'){
      await open.focus('#broker-connect-provider');const steps=[await focusAt()];
      for(const keyName of ['ArrowDown','ArrowDown','ArrowUp']){
        await open.keyboard.press(keyName);const at=await focusAt();steps.push(at);
        assert.equal(at.id,'broker-connect-provider','after '+keyName+' on the Provider select focus moved to '+(at.id||at.tag)+' (WCAG 2.4.3): '+JSON.stringify(steps));
        assert.notEqual(at.value,steps[steps.length-2].value,keyName+' did not move to another provider: '+JSON.stringify(steps));
      }
      assert.notEqual(steps[2].value,steps[1].value,'the second Arrow did not reach the next provider');assert.equal(steps[3].value,steps[1].value,'Arrow Up did not go back');
      await open.keyboard.press('Tab');const next=await focusAt();
      assert(next.inConnect&&next.id!=='broker-connect-provider','Tab after the Provider select left Connect a provider: '+JSON.stringify(next));
    }
    await open.selectOption('#broker-connect-provider','openai');
    await open.waitForFunction(()=>document.querySelector('#broker-connect-env option:checked')?.textContent.includes('OPENAI_API_KEY'));
    assert(await open.isChecked('#broker-connect-anonymous'),'keyless use of the new provider is not offered');
    // the variable by name: only the name is sent; the upstream points at the mock, not the profile's default address,
    // so the keyless Studio may not send an environment credential there (403); Studio says so and claims nothing
    // by keyboard only (WCAG 2.1.1 and 3.2.1): from the paste option Arrow Up chooses the environment option; Tab then walks
    // through the variable picker and the write-only key field to Connect without changing the chosen source; Enter connects
    const sourceRadio=value=>`#broker-connect-form input[name="broker-connect-source"][value="${value}"]`;
    await open.locator(sourceRadio('value')).check();await open.focus(sourceRadio('value'));
    await open.keyboard.press('ArrowUp');
    assert(await open.isChecked(sourceRadio('env')),'Arrow Up did not choose the environment option');
    const walked=[];let reachedConnect=false;
    for(let step=0;step<12&&!reachedConnect;step++){
      await open.keyboard.press('Tab');
      const at=await open.evaluate(()=>{const node=document.activeElement;return {name:node?.id||node?.type||node?.tagName||'',connect:Boolean(node?.matches?.('#broker-connect-form button[type=submit]')),
        source:document.querySelector('#broker-connect-form input[name="broker-connect-source"]:checked')?.value||''};});
      walked.push(at.name);reachedConnect=at.connect;
      assert.equal(at.source,'env','focus on '+at.name+' changed the chosen source (WCAG 3.2.1): '+walked.join(' > '));
    }
    assert(reachedConnect,'Tab never reached Connect: '+walked.join(' > '));
    assert(walked.includes('broker-connect-env')&&walked.includes('broker-connect-secret'),'Tab did not pass the variable picker and the key field: '+walked.join(' > '));
    let connect=open.waitForResponse(response=>new URL(response.url()).pathname==='/api/broker/connect');
    await open.keyboard.press('Enter');
    let reply=await connect;assert.equal(reply.request().headers()['x-titanos-studio'],'1');
    assert.deepEqual(reply.request().postDataJSON(),{provider:'openai',enable:true,credential:{env:'OPENAI_API_KEY'},anonymous_cloud:true},'connect sent more than the variable name');
    assert.equal(reply.status(),403,'an environment credential was sent to a non-default address');assert.equal((await reply.json()).error.code,'scope_denied');
    await open.waitForFunction(()=>/is not connected/.test(document.getElementById('broker-connect-result')?.textContent||''));
    assert.match(await open.textContent('#broker-connect-result'),/default\s+address/,'the refusal lost the server\'s reason');
    assert.doesNotMatch(await open.textContent('#broker-connect-result'),/Check the complete model|Correct the named fields/,'a local-model hint was appended to a broker refusal');
    assert.doesNotMatch(await open.textContent('#broker-message'),/^Connected/);
    // the result re-rendered the panel: keyboard focus is still on Connect, not dropped to the page
    {const at=await focusAt();assert(at.connect,'after a Connect by Enter focus moved to '+(at.id||at.tag)+' (WCAG 2.4.3)');}
    // a pasted key: sent once, validated against the mock, enabled, keyless use allowed, models discovered; gone from the page
    await open.selectOption('#broker-connect-provider','openai');
    await open.locator('#broker-connect-form input[name="broker-connect-source"][value="value"]').check();
    const pastedField=await open.locator('#broker-connect-secret').elementHandle();await open.fill('#broker-connect-secret',mock.sentinel);
    // submitted by Enter in the key field itself (the keyboard way to send a pasted key)
    connect=open.waitForResponse(response=>new URL(response.url()).pathname==='/api/broker/connect');
    await open.focus('#broker-connect-secret');await open.keyboard.press('Enter');
    reply=await connect;assert.equal(reply.status(),200,'connect failed: '+await reply.text());
    assert.equal(reply.request().postDataJSON().credential.value,mock.sentinel);assert.equal(reply.request().postDataJSON().anonymous_cloud,true);
    assert.equal(await pastedField.evaluate(node=>[node.value,node.isConnected].join('|')),'|false','the pasted field was not emptied and replaced');
    const connected=await reply.json();
    assert.equal(connected.validation,'ok');assert.equal(connected.upstream.enabled,true);assert.equal(connected.anonymous_cloud,true);assert(connected.models.includes('gpt-4o-mini'));
    assert(!JSON.stringify(connected).includes(mock.sentinel),'the connect answer carries the value');
    await open.waitForFunction(()=>/OpenAI connected/.test(document.getElementById('broker-connect-result')?.textContent||''));
    assert.match(await open.textContent('#broker-connect-result'),/gpt-4o-mini/);
    // the emptied key field was swapped for a fresh one while it had focus: focus moved with it and survived the re-renders
    {const at=await focusAt();assert.equal(at.id,'broker-connect-secret','after Enter in the key field focus moved to '+(at.id||at.tag)+' (WCAG 2.4.3)');
      assert.equal(await open.inputValue('#broker-connect-secret'),'','the fresh key field is not empty');}
    assert.deepEqual(await traces(open,[mock.sentinel]),[{inHtml:false,inInputs:false,inStorage:false}],'the pasted key lingers in the page');
    // the connected provider now serves the keyless Studio: the alias chats with its provenance and cyan route mark
    await open.waitForFunction(()=>[...document.querySelectorAll('#model option')].some(option=>option.value==='cloud-mini'));
    await open.selectOption('#model','cloud-mini');assert(await open.isVisible('#route-note'),'the cloud route note is missing');
    await open.fill('#prompt','Hello from the keyless Studio.');
    const keylessChat=open.waitForResponse(response=>response.request().method()==='POST'&&new URL(response.url()).pathname==='/v1/chat/completions');
    await open.click('#submit');assert.equal((await keylessChat).status(),200);
    await open.waitForFunction(()=>/Served by cloud:openai/.test(document.querySelector('#results .result .provenance')?.textContent||''));
    assert(await open.evaluate(()=>document.querySelector('#results .result.external.done')!==null),'the cloud result lost its route mark');
    // without the Studio header a page cannot write keylessly, and keys are never minted keylessly
    const refused=await open.evaluate(async([variable])=>{
      const call=async(url,method,body,headers={})=>{const answer=await fetch(url,{method,headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(body)});return {status:answer.status,code:(await answer.json())?.error?.code};};
      return [await call('/api/broker/credentials/openai-key','PUT',{upstream:'openai',env:variable}),await call('/api/broker/connect','POST',{provider:'openai',credential:{env:variable}}),
        await call('/api/keys','POST',{name:'studio',scopes:['inference']},{'X-Titanos-Studio':'1'})];
    },['OPENAI_API_KEY']);
    assert.deepEqual(refused.map(row=>row.status),[403,403,403],'keyless writes without the Studio header, or a keyless key mint: '+JSON.stringify(refused));
    assert.deepEqual(refused.slice(0,2).map(row=>row.code),['keyless_mode_refused','keyless_mode_refused']);
    await shot(open,'keyless-panel','#broker-panel');
    assert.deepEqual(b.errors,[],'page errors (keyless)');
    await b.context.close();

    // ===== what the provider saw, and what left the processes =====
    const seen=await mock.hits([BOOTSTRAP,TYPED_SECRET]);
    assert(seen.hits.length>=2,'the mock was not reached');
    assert.deepEqual(seen.hits.filter(hit=>hit.credential!=='Bearer <sentinel:openai>'),[],'a request reached the mock without its own credential');
    assert.deepEqual(seen.hits.filter(hit=>hit.forbidden.length),[],'the bootstrap key or the typed secret reached the provider');
    assert.equal(seen.hits.filter(hit=>hit.surface==='openai.chat').length,2,'each chat (keyed, then keyless after connect) was not sent upstream exactly once');
    assert.deepEqual([...new Set(seen.hits.map(hit=>hit.surface))].sort(),['openai.chat','openai.models'],'unexpected upstream surfaces');
    assert(seen.hits.some(hit=>hit.method==='GET'&&/\/models$/.test(hit.path)),'the credential was not validated against the upstream');
    assert.equal(seen.blocked,0);
    assert.deepEqual(offsite,[],'the browser requested a non-loopback origin');
    await browser.close();browser=null;
    for(const serve of children)await serve.stop();
    await mock.stop();
    for(const [name,blocked] of [['keyed',keyed.blocked()],['keyless',keyless.blocked()],['mock',fs.existsSync(mock.report)?fs.readFileSync(mock.report,'utf8').split(/\r?\n/).filter(Boolean):[]]])
      assert.deepEqual(blocked,[],name+' attempted a non-loopback connection');
    for(const serve of children){const log=serve.output;for(const value of [mock.sentinel,TYPED_SECRET,BOOTSTRAP])assert(!log.includes(value),`serve (${serve.name}) output carries a secret`);}
    reportMatrix([CASE],[]);
    console.log('PASS: Studio against a real serve plus the loopback OpenAI mock in '+(launch.executablePath||launch.channel||'Playwright Chromium')+': write-only credential form (typed secret refused by the refs-only vault with no trace in the DOM, inputs, storage or response), env-sourced credential saved by variable name, validated and bound, upstream enabled from its toggle, cloud alias with its External badge, streamed chat with the provenance line and a listed decision, design spot checks (one lime action, no gradient/shadow, caps and tracking only on plates, radii <= 6px, cyan badge, monospace provenance, no overflow), keyless Studio connect (R29 header; the Provider select stepped by Arrow keys with focus kept through each re-render and on Connect after Enter; R30 a variable chosen and submitted by keyboard, sent by name only and refused for a non-default address with the server\'s reason; R31 a pasted key validated and enabled with no trace in the page) and a keyless cloud chat, 403 for header-less keyless writes and keyless key mints; the mock saw only its own sentinel credential and nothing left loopback.');
  }finally{
    if(browser)await browser.close().catch(()=>{});
    for(const serve of children)await serve.stop().catch(()=>{});
    if(mock)await mock.stop().catch(()=>{});
    for(let attempt=0;attempt<5;attempt++){try{fs.rmSync(temp,{recursive:true,force:true});break;}catch{await new Promise(resolve=>setTimeout(resolve,500));}}
  }
}
main().catch(error=>{console.error(error);reportMatrix([],[CASE]);process.exitCode=1;});
