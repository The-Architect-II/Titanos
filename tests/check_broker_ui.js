/* Studio 'Cloud and routing' panel (bay 07): write-only secrets, scope-gated sections, the keyless note, the R29 Studio header,
   the R30/R31 Connect-a-provider flow (environment keys by name, a pasted key sent once, typed messages for a missing route or
   a refused key, never a success claim without a connect result), external badges and the provenance line. node:vm over
   slices of app.js with a DOM double and a fake api(); no browser, server or network. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../src/titanos/web/app.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../src/titanos/web/index.html'),'utf8');
const SENTINEL='sk-test-wp8-studio-sentinel-0123456789';
const KEY_SECRET='tk_abcdefghij_SECRETSECRETSECRETSECRETSECRETSECRET01';

class El{
  constructor(tag){this.tagName=tag;this.children=[];this.parentElement=null;this.attributes={};this.className='';this.hidden=false;this.value='';this._text='';this.dataset={};this.options=[];}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  set textContent(value){this._text=String(value);this.children=[];}
  append(...items){for(let item of items){if(typeof item!=='object'||item===null)item=String(item);if(typeof item==='string'){const t=new El('#text');t._text=item;item=t;}if(item.parentElement)item.parentElement.children.splice(item.parentElement.children.indexOf(item),1);item.parentElement=this;this.children.push(item);if(this.tagName==='select'&&item.tagName==='option')this.options.push(item);}}
  // removing the focused node (or its ancestor) drops focus to the body, as in a browser
  replaceChildren(...items){for(const c of this.children){blurIfWithin(c);c.parentElement=null;}this.children=[];this.options=[];this.append(...items);}
  replaceWith(node){const parent=this.parentElement;if(!parent)return;blurIfWithin(this);const i=parent.children.indexOf(this);parent.children[i]=node;node.parentElement=parent;this.parentElement=null;}
  setAttribute(name,value){this.attributes[name]=String(value);}
  getAttribute(name){return this.attributes[name]??null;}
  removeAttribute(name){delete this.attributes[name];}
  // radios of one name are exclusive, as in a browser
  get checked(){return Boolean(this._checked);}
  set checked(value){this._checked=Boolean(value);if(this._checked&&this.type==='radio'&&this.name){let top=this;while(top.parentElement)top=top.parentElement;for(const n of walk(top))if(n!==this&&n.type==='radio'&&n.name===this.name)n._checked=false;}}
  // events reach the target's listeners, then its ancestors' when the event bubbles (focus does not)
  addEventListener(type,listener){(this._listeners||(this._listeners={}))[type]=[...(this._listeners[type]||[]),listener];}
  // focus() makes the node the document's active element, as in a browser
  focus(){document.activeElement=this;}
  dispatchEvent(event){event.target=event.target||this;for(let node=this;node;node=event.bubbles?node.parentElement:null)for(const listener of node._listeners?.[event.type]||[])listener.call(node,event);return true;}
}
const document={createElement:tag=>new El(tag),activeElement:null};
function blurIfWithin(node){if(document.activeElement&&walk(node).includes(document.activeElement))document.activeElement=null;}
// the broker matrix run (tools/build_broker_matrix.py --run) reads one results line per check from TITANOS_BROKER_RESULTS
function reportMatrix(passed,failed,skipped={}){
  const target=process.env.TITANOS_BROKER_RESULTS;
  if(target)fs.appendFileSync(target,JSON.stringify({check:path.basename(__filename),passed,failed,skipped})+'\n');
}
const CASE='Panel.dom_double_contract';
function walk(node,out=[]){out.push(node);for(const c of node.children)walk(c,out);return out;}
function root(){const r=new El('body');const panel=new El('details');panel.id='broker-panel';const body=new El('div');body.id='broker-body';panel.append(body);const label=new El('label');label.id='location-label';const select=new El('select');select.id='location';label.append(select);r.append(panel,label);return r;}
function serialize(node){return walk(node).map(n=>[n.id,n._text,n.value,n.className,JSON.stringify(n.attributes)].join('|')).join('\n');}

function load({self,server={broker:{configured:true}},secure=true,answers={},connect=null}){
  const tree=root(),requests=[],stored=[];
  const storage={getItem:()=>null,setItem:(k,v)=>stored.push([k,v]),removeItem(){},clear(){}};
  const data={'/api/keys/self':self,'/api/broker':{role:'standalone',cloud_policy:{effective:'explicit',file:'explicit',ceiling:'fallback'},vault:{backend:'dpapi',sealed:false}},
    '/api/broker/upstreams':{data:[{id:'openai',profile:'openai',label:'OpenAI',enabled:true,credential:'oa',catalog:'static'}]},
    '/api/broker/models':{data:[{name:'mini',target:'gpt-4o-mini@openai'}]},'/api/broker/virtual-models':{data:[]},
    '/api/broker/policy':{cloud_policy:'explicit',default_location:'auto',anonymous_cloud:false,allow_http_lan:false,max_fallbacks:2,ceiling:'fallback'},
    '/api/broker/usage?group_by=upstream,model':{rows:[{upstream:'openai',model:'gpt-4o-mini@openai',requests:2,cost_usd:'0.01'}]},
    '/api/broker/decisions?limit=10':{data:[{requested_model:'mini',served:{ref:'gpt-4o-mini@openai'},fallback_count:0,ts:'2026-10-02T00:00:00Z'}]},
    '/api/broker/credentials':{data:[{id:'oa',fingerprint:'1a2b3c4d',validated_at:'2026-10-01T00:00:00Z',validation:'ok'}]},
    '/api/keys':{data:[{id:'abcdefghij',name:'ci',status:'active',scopes:['inference']}]},
    '/api/broker/upstreams/openai/health':{breaker:{state:'closed'}},...answers};
  const context={console,document,Object,Array,JSON,String,Number,Boolean,Promise,Date,Math,Error,RegExp,
    localStorage:storage,sessionStorage:storage,location:secure?{protocol:'http:',hostname:'127.0.0.1'}:{protocol:'http:',hostname:'192.0.2.10'},
    confirm:()=>true,navigator:{clipboard:{writeText:async()=>{}}},serverInfo:server,i18n:{t:name=>name},key:'',
    $:id=>walk(tree).find(node=>node.id===id)||null,
    api:async(url,options={})=>{requests.push({url,method:options.method||'GET',body:options.json===undefined?undefined:JSON.stringify(options.json)});
      if(url==='/api/broker/credentials/oa-new'&&options.method==='PUT')return{json:async()=>({id:'oa-new',fingerprint:'9f8e7d6c',validated_at:'2026-10-02T00:00:00Z',validation:'ok',upstream_id:'openai'})};
      if(url==='/api/keys'&&options.method==='POST')return{json:async()=>({id:'abcdefghij',secret:KEY_SECRET,name:'new'})};
      if(url.startsWith('/api/broker/credentials/')&&options.method==='PUT')return{json:async()=>({id:decodeURIComponent(url.split('/').pop()),fingerprint:'0a0b0c0d',validated_at:null,validation:'ok'})};
      if(url==='/api/broker/upstreams/openai'&&options.method==='PUT')return{json:async()=>({...options.json,id:'openai'})};
      if(url==='/api/broker/connect'&&options.method==='POST'){if(!connect)throw Object.assign(Error('Not Found'),{status:404});return connect(options.json);}
      if(!(url in data)||data[url]===undefined)throw Object.assign(Error('HTTP 403 scope_denied'),{status:403});
      return{json:async()=>JSON.parse(JSON.stringify(data[url]))};}};
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf('const UI_MESSAGE='),source.indexOf('function bindStaticTranslations(){')),context);
  const start=source.indexOf('// --- broker panel (Cloud and routing) ---'),end=source.indexOf('// --- end broker panel ---');
  assert(start>0&&end>start,'broker block markers');
  vm.runInContext(source.slice(start,end),context);
  const shown=()=>{const box=vm.runInContext('brokerSecretBox',context);return box?walk(box).find(n=>n.id==='broker-key-secret-value')?.textContent||null:null;};
  return {context,tree,requests,stored,data,$:context.$,state:()=>vm.runInContext('brokerState',context),shown};
}
const ADMIN={id:'bootstrap',kind:'bootstrap',scopes:['admin']};

(async()=>{
  // static markup: the panel is bay 07 in the work column after Jobs (shown in both modes; its expert sections fold away in
  // Simple); the location select is in the composer
  assert.match(html,/id="bay-jobs" class="jobs-section bay"[\s\S]*?<details id="broker-panel" class="panel bay cloud-bay" hidden><summary class="bay-summary">/);
  assert.match(html,/<select id="location">/);

  // (1) admin: every section renders; the secret input is never prefilled and is write-only
  let t=load({self:ADMIN});await t.context.refreshBroker();
  assert.equal(t.$('broker-panel').hidden,false);
  for(const id of ['broker-role','broker-policy','broker-vault','broker-upstreams','broker-connect-form','broker-connect-provider','broker-connect-secret','broker-credential-form','broker-alias-form','broker-virtual-form','broker-policy-form','broker-usage','broker-decisions','broker-explain-form','broker-keys'])assert(t.$(id),'missing #'+id);
  // numbered sub-sections: Connect and the routes open, the expert sections folded and Advanced-only
  const folds=walk(t.$('broker-body')).filter(n=>n.tagName==='details');
  assert.deepEqual(folds.map(n=>String(n.children[0].children[0].textContent)),['07.1','07.2','07.3','07.4','07.5','07.6','07.7']);
  assert.deepEqual(folds.map(n=>Boolean(n.open)),[true,true,false,false,false,false,false]);
  assert(folds.slice(2).every(n=>n.className.includes('advanced-only'))&&!folds[0].className.includes('advanced-only'));
  const connectSecret=t.$('broker-connect-secret');
  assert.deepEqual([connectSecret.value,connectSecret.attributes.type,connectSecret.attributes.autocomplete,connectSecret.attributes.spellcheck,connectSecret.attributes.autocapitalize],['','password','off','false','off']);
  const secret=t.$('broker-secret');
  assert.equal(secret.value,'');assert.equal(secret.attributes.type,'password');
  assert.equal(secret.attributes.autocomplete,'off');assert.equal(secret.attributes.spellcheck,'false');assert.equal(secret.attributes.autocapitalize,'off');
  assert.equal(t.$('location-label').hidden,false);
  assert(!/null|undefined/.test(t.$('broker-body').textContent),'a null or undefined child was rendered as text');

  // (2) a blank value changes nothing; a saved value leaves no trace in the DOM, the state or storage
  const before=t.requests.length;
  await t.$('broker-credential-form').onsubmit({preventDefault(){}});
  assert.equal(t.requests.length,before,'a blank secret sent a request');
  t.$('broker-credential-id').value='oa-new';t.$('broker-secret').value=SENTINEL;const old=t.$('broker-secret');
  // Enter pressed in the secret field: the fresh field keeps keyboard focus through the swap and the re-renders (WCAG 2.4.3)
  old.focus();
  await t.$('broker-credential-form').onsubmit({preventDefault(){}});
  const put=t.requests.find(r=>r.method==='PUT');assert(put&&put.url==='/api/broker/credentials/oa-new');
  assert.equal(JSON.parse(put.body).secret,SENTINEL);assert.equal(JSON.parse(put.body).upstream,'openai');
  assert.equal(old.value,'');assert.notEqual(t.$('broker-secret'),old,'the input element was not replaced');assert.equal(t.$('broker-secret').value,'');
  assert.equal(document.activeElement,t.$('broker-secret'),'saving from the secret field dropped keyboard focus');document.activeElement=null;
  assert(!serialize(t.tree).includes(SENTINEL),'the DOM holds the submitted secret');
  assert(!JSON.stringify(t.state()).includes(SENTINEL),'broker state holds the submitted secret');
  assert.deepEqual(t.stored,[]);
  assert.deepEqual(JSON.parse(t.$('broker-credential-result').textContent),{id:'oa-new',fingerprint:'9f8e7d6c',validated_at:'2026-10-02T00:00:00Z',validation:'ok'});
  // (2b) the saved credential is bound to its upstream (confirmed, because openai referenced 'oa'); the full upstream spec
  // is sent back without the read-only fields, and the panel re-renders with the message kept
  const bind=t.requests.find(r=>r.url==='/api/broker/upstreams/openai'&&r.method==='PUT');assert(bind,'the credential was not bound to its upstream');
  assert.deepEqual(JSON.parse(bind.body),{profile:'openai',label:'OpenAI',enabled:true,credential:'oa-new',catalog:'static'});
  assert.match(String(t.$('broker-message').textContent),/bound to openai/);
  assert(t.requests.indexOf(bind)<t.requests.length-1,'the panel was not refreshed after the save');
  // a blank id replaces the credential the upstream already uses, so no rebinding is needed
  assert.match(String(t.$('broker-credential-expected').textContent),/use oa\./);
  t.$('broker-secret').value=SENTINEL;const binds=t.requests.filter(r=>r.method==='PUT'&&r.url.startsWith('/api/broker/upstreams/')).length;
  await t.$('broker-credential-form').onsubmit({preventDefault(){}});
  assert.equal(t.requests.filter(r=>r.method==='PUT').at(-1).url,'/api/broker/credentials/oa');
  assert.equal(t.requests.filter(r=>r.method==='PUT'&&r.url.startsWith('/api/broker/upstreams/')).length,binds);

  // (3) a minted key secret is shown once and gone on dismiss; never stored
  t.$('broker-key-name').value='new';
  await t.$('broker-key-form').onsubmit({preventDefault(){}});
  assert.equal(JSON.parse(t.requests.find(r=>r.url==='/api/keys'&&r.method==='POST').body).scopes.includes('inference'),true);
  assert.equal(t.$('broker-key-secret-value').textContent,KEY_SECRET);assert.equal(t.shown(),KEY_SECRET);
  assert.equal(serialize(t.tree).split(KEY_SECRET).length-1,1,'the secret is rendered exactly once');
  // (3b) a later re-render (refresh, another panel action) keeps the box until it is dismissed
  await t.context.refreshBroker();
  assert.equal(t.$('broker-key-secret-value').textContent,KEY_SECRET,'a re-render dropped the one-time secret');
  assert.equal(serialize(t.tree).split(KEY_SECRET).length-1,1,'the secret is rendered exactly once after a re-render');
  const dismiss=walk(t.$('broker-key-secret')).find(n=>n.tagName==='button'&&String(n.textContent).includes('hide'));
  dismiss.onclick();
  assert.equal(t.$('broker-key-secret'),null);assert.equal(t.shown(),null);
  assert(!serialize(t.tree).includes(KEY_SECRET));assert(!JSON.stringify(t.state()).includes(KEY_SECRET));assert.deepEqual(t.stored,[]);

  // (3c) one lime (default) action in the panel: Connect; Save credential is a violet structural button
  t=load({self:ADMIN});await t.context.refreshBroker();
  const lime=walk(t.$('broker-body')).filter(n=>n.tagName==='button'&&!/\b(quiet|structural)\b/.test(n.className));
  assert.deepEqual(lime.map(n=>String(n.textContent)),['Connect'],'more than one lime action in the panel');
  assert(walk(t.$('broker-credential-form')).some(n=>n.tagName==='button'&&n.className==='structural'&&String(n.textContent)==='Save credential'));

  // (3d) a principal change clears the previous rendering: an admin-rendered panel and a shown secret do not survive
  t.$('broker-key-name').value='new';await t.$('broker-key-form').onsubmit({preventDefault(){}});assert(t.$('broker-key-secret'));
  t.data['/api/keys/self']={id:'reader',kind:'key',scopes:['broker:read','credentials:write']};await t.context.refreshBroker();
  assert.equal(t.$('broker-panel').hidden,false);assert.equal(t.shown(),null);
  for(const id of ['broker-keys','broker-key-secret','broker-policy-form','broker-alias-form'])assert.equal(t.$(id),null,'#'+id+' survived a principal change');
  assert(!serialize(t.tree).includes(KEY_SECRET));
  // a credential writer without broker:admin saves but cannot bind: told so, no upstream write
  t.$('broker-credential-id').value='oa-second';t.$('broker-secret').value=SENTINEL;
  await t.$('broker-credential-form').onsubmit({preventDefault(){}});
  assert.equal(t.requests.filter(r=>r.method==='PUT'&&r.url.startsWith('/api/broker/upstreams/')).length,0);
  assert.match(String(t.$('broker-message').textContent),/broker:admin must bind it/);
  t.context.brokerReset();assert.equal(t.$('broker-panel').hidden,true);assert.equal(t.$('broker-body').children.length,0);
  delete t.data['/api/keys/self'];await t.context.refreshBroker();assert.equal(t.$('broker-panel').hidden,true,'a refused principal kept the panel');

  // (3e) a keys:admin key rotates ITSELF from Studio: the old secret dies at once on the server, so the tab adopts the
  // new secret (in memory, as Connect does) before any refresh; the one-time box is shown once and survives the 401
  // answered to a request the tab sent with the old key before the rotation; the old key is never sent again
  {
    const SELF_ID='selfadmin2',OLD_KEY='tk_'+SELF_ID+'_'+'A'.repeat(43),NEW_KEY='tk_'+SELF_ID+'_'+'B'.repeat(43);
    t=load({self:{id:SELF_ID,kind:'key',scopes:['keys:admin','broker:read']},answers:{'/api/keys':{data:[
      {id:SELF_ID,name:'studio',status:'active',scopes:['keys:admin','broker:read']},{id:'abcdefghij',name:'ci',status:'active',scopes:['inference']}]}}});
    t.context.key=OLD_KEY;let valid=OLD_KEY,held=null;const sent=[],inner=t.context.api;
    t.context.api=async(url,options={})=>{
      const used=t.context.key;sent.push({url,method:options.method||'GET',key:used});   // the real api() reads the key when the request is built
      if(held&&url==='/api/keys/self'){const gate=held;held=null;await gate.promise;}
      if(used!==valid)throw Object.assign(Error('HTTP 401 invalid_key'),{status:401});
      if(options.method==='POST'&&url==='/api/keys/'+SELF_ID+'/rotate'){valid=NEW_KEY;return{json:async()=>({id:SELF_ID,secret:NEW_KEY})};}
      if(options.method==='POST'&&url==='/api/keys/abcdefghij/rotate')return{json:async()=>({id:'abcdefghij',secret:KEY_SECRET})};
      return inner(url,options);};
    await t.context.refreshBroker();assert(t.$('broker-keys'),'keys section missing for the keys:admin key');
    // a panel refresh is in flight with the old key when the rotation lands; its 401 arrives after the rotation
    let release;held={promise:new Promise(resolve=>{release=resolve;})};const stale=t.context.refreshBroker();
    const rotateOf=id=>walk(t.$('broker-key-list')).filter(n=>n.tagName==='div'&&n.className==='broker-item').find(n=>walk(n).some(c=>c.tagName==='strong'&&String(c.textContent)===(id===SELF_ID?'studio':'ci')))
      .children.find(n=>n.tagName==='button'&&String(n.textContent)==='Rotate');
    const mark=sent.length;await rotateOf(SELF_ID).onclick();
    assert.equal(t.context.key,NEW_KEY,'the tab did not adopt its rotated secret');
    release();await stale;
    assert.equal(t.$('broker-panel').hidden,false,'the panel was reset after the self-rotation');assert(t.$('broker-keys'));
    assert.equal(t.shown(),NEW_KEY,'the one-time secret box was dropped after the self-rotation');
    assert.equal(t.$('broker-key-secret-value').textContent,NEW_KEY);
    assert.equal(serialize(t.tree).split(NEW_KEY).length-1,1,'the rotated secret is rendered exactly once');
    assert.match(String(t.$('broker-key-secret').children[0].textContent),/rotated\. This tab now uses the new secret/);
    const after=sent.slice(mark+1);assert(after.length>0,'no refresh followed the self-rotation');
    assert.deepEqual(after.filter(r=>r.key!==NEW_KEY),[],'a request after the self-rotation carried the old key');
    // the panel keeps working with the new key: a later refresh keeps the box, another key is minted and the old key is gone
    await t.context.refreshBroker();assert.equal(t.shown(),NEW_KEY);assert.equal(serialize(t.tree).split(NEW_KEY).length-1,1);
    t.$('broker-key-name').value='next';const minted=sent.length;await t.$('broker-key-form').onsubmit({preventDefault(){}});
    assert.equal(t.shown(),KEY_SECRET);assert.match(String(t.$('broker-key-secret').children[0].textContent),/created/);
    assert(sent.slice(minted).length>1&&sent.slice(minted).every(r=>r.key===NEW_KEY),'the panel did not keep working with the new key');
    assert.equal(sent.slice(mark+1).filter(r=>r.key===OLD_KEY).length,0,'the old key was used after the rotation');
    // rotating ANOTHER key leaves the tab's key alone and labels the box as a new secret for that key
    await rotateOf('abcdefghij').onclick();assert.equal(t.context.key,NEW_KEY);assert.equal(t.shown(),KEY_SECRET);
    assert.match(String(t.$('broker-key-secret').children[0].textContent),/created/);
    assert(!JSON.stringify(t.state()).includes(NEW_KEY),'broker state holds the rotated secret');assert.deepEqual(t.stored,[]);
    // with no principal rendered yet, the key id inside the tab's own tk_ secret still identifies a self-rotation
    t.context.brokerReset();t.context.key=NEW_KEY;const LATER='tk_'+SELF_ID+'_'+'C'.repeat(43);
    t.context.api=async(url,options={})=>options.method==='POST'?{json:async()=>({id:SELF_ID,secret:LATER})}:inner(url,options);
    await t.context.brokerKeySecret('/api/keys/'+SELF_ID+'/rotate',{method:'POST'});assert.equal(t.context.key,LATER);
  }

  // (4) scope-gated sections
  t=load({self:{id:'k1',kind:'key',scopes:['inference']}});await t.context.refreshBroker();
  assert.equal(t.$('broker-panel').hidden,true,'panel shown without broker:read');
  assert.equal(t.requests.filter(r=>r.url!=='/api/keys/self').length,0);
  t=load({self:{id:'k2',kind:'key',scopes:['broker:read']}});await t.context.refreshBroker();
  assert.equal(t.$('broker-panel').hidden,false);
  for(const id of ['broker-credential-form','broker-keys','broker-policy-form','broker-alias-form','broker-keyless-note'])assert.equal(t.$(id),null,'#'+id+' shown to broker:read');
  assert(!t.requests.some(r=>r.url==='/api/keys'||r.url==='/api/broker/credentials'));
  assert.equal(walk(t.$('broker-upstreams')).some(n=>n.tagName==='button'),false,'admin controls shown to broker:read');

  // (5) keyless: the fixed CLI note instead of the form; guarded policy switches disabled
  t=load({self:{id:'anonymous',kind:'anonymous_loopback',scopes:['broker:read','broker:admin','inference']}});await t.context.refreshBroker();
  assert.match(t.$('broker-keyless-note').textContent,/titanos broker credential set/);
  assert(!/null|undefined/.test(t.$('broker-body').textContent),'a skipped section was rendered as text');
  assert.equal(t.$('broker-credential-form'),null);
  assert.equal(t.$('broker-anonymous-cloud').disabled,true);assert.equal(t.$('broker-allow-http-lan').disabled,true);
  // a keyless server without the R29 Studio principal: Connect is the CLI route, and env discovery is not requested
  assert.match(t.$('broker-connect-cli').textContent,/titanos broker connect <provider> --from-env NAME/);
  assert.equal(t.$('broker-connect-form'),null);assert(!t.requests.some(r=>r.url==='/api/broker/env-candidates'));

  // (6) an origin that is neither loopback nor https disables the form with a warning
  t=load({self:ADMIN,secure:false});await t.context.refreshBroker();
  assert(t.$('broker-insecure'));assert.equal(t.$('broker-secret').disabled,true);

  // (7) no broker block in /api/info (or no settings block): nothing is requested, nothing breaks
  t=load({self:ADMIN,server:{models:{}}});await t.context.refreshBroker();
  assert.equal(t.$('broker-panel').hidden,true);assert.equal(t.$('location-label').hidden,true);assert.deepEqual(t.requests,[]);
  const ctx={$:id=>({max:'',value:'',hidden:false,replaceChildren(){},append(){},parentElement:null,id}),serverInfo:{broker:{configured:false}},runtimeProfileChoice:null,i18n:{t:n=>n},document};
  vm.createContext(ctx);
  vm.runInContext(source.slice(source.indexOf('const UI_MESSAGE='),source.indexOf('function bindStaticTranslations(){')),ctx);
  vm.runInContext(source.slice(source.indexOf('function applyServerInfo(){'),source.indexOf('function selectedRuntimeProfile(){')),ctx);
  assert.doesNotThrow(()=>ctx.applyServerInfo(),'a missing settings block broke rendering');

  // (8) external badges and catalog rows without a task
  t=load({self:ADMIN});await t.context.refreshBroker();
  const external=t.context.catalogEntry({id:'mini',object:'model',owned_by:'openai',remote:true,titanos:{class:'cloud',upstream:'openai',capabilities:['chat','tools'],status:'available'}});
  assert.equal(external.task,'chat');assert.deepEqual(JSON.parse(JSON.stringify(external.external)),{kind:'external',upstream:'openai'});
  // the rail's meta line names the provider; the orbit glyph and the Cloud routes divider carry the route
  assert.equal(String(t.context.externalBadge(external).textContent),'OpenAI');
  const virtual=t.context.catalogEntry({id:'auto',owned_by:'titanos-virtual',titanos:{class:'virtual',capability:'embeddings',strategy:'ordered'}});
  assert.equal(virtual.task,'embeddings');assert.equal(String(t.context.externalBadge(virtual).textContent),'Virtual');
  assert.equal(t.context.catalogEntry({id:'mod',titanos:{class:'cloud',capabilities:['moderation']}}),null);
  const local={id:'qwen',task:'chat'};assert.equal(t.context.catalogEntry(local),local);
  // static-catalog rows carry the profile's whole list: a model id naming one modality picks that task
  const all=['chat','embeddings','images','moderation','responses','speech','tools','transcription','vision'];
  const task=id=>t.context.catalogEntry({id,titanos:{class:'cloud',upstream:'openai',capabilities:all,status:'available'}})?.task??null;
  assert.deepEqual(['dall-e-3@openai','tts-1@openai','whisper-1@openai','text-embedding-3-small@openai','omni-moderation-latest@openai','gpt-4o-mini@openai'].map(task),['image','speech','transcription','embeddings',null,'chat']);
  assert.equal(t.context.catalogEntry({id:'whisper-1@x',titanos:{class:'cloud',capabilities:['chat']}}).task,'chat','a hint the row does not offer');

  // refresh(): the panel refresh starts before (and independently of) the jobs and acquisition reads; Connect resets it
  assert(source.indexOf("brokerRefresh=typeof refreshBroker==='function'?refreshBroker()")>0);
  assert(source.indexOf("brokerRefresh=typeof refreshBroker==='function'?refreshBroker()")<source.indexOf('await refreshJobs();await reconcileAcquisitions();await brokerRefresh;'));
  assert.match(source,/\$\('connect'\)\.onclick=[^\n]*brokerReset\(\);return refresh\(true\);/);

  // (9) the provenance line under a result
  const box=new El('article');
  assert.equal(t.context.appendProvenance(box),null,'a line without provenance');
  t.context.rememberProvenance({headers:{get:name=>({'x-titanos-served-by':'cloud:openai','x-titanos-served-model':'gpt-4o-mini@openai','x-titanos-fallback-count':'1'})[name]||null}});
  const line=t.context.appendProvenance(box);
  assert.equal(String(line.textContent),'Served by cloud:openai · gpt-4o-mini@openai · fallbacks 1');
  t.context.rememberProvenance({headers:{get:()=>null}});
  assert.equal(String(t.context.appendProvenance(new El('article')).textContent),'Served by cloud:openai · gpt-4o-mini@openai · fallbacks 1');
  assert.equal(t.context.routingLocation(),'');t.$('location').value='cloud_only';assert.equal(t.context.routingLocation(),'cloud_only');
  // (10) R29: api() marks every broker and key request with X-Titanos-Studio: 1 (GET and writes); other routes stay unmarked;
  // a tab key is still sent as the bearer credential
  {
    const sent=[];const ctx={Headers,JSON,String,Object,Error,Symbol,WeakMap,i18n:{t:name=>name},key:'',
      fetch:async(url,init)=>{sent.push({url,headers:Object.fromEntries(init.headers.entries()),body:init.body});return {ok:true,status:200,headers:{get:()=>null},json:async()=>({})};}};
    vm.createContext(ctx);
    vm.runInContext(source.slice(source.indexOf('const UI_MESSAGE='),source.indexOf('function bindStaticTranslations(){')),ctx);
    vm.runInContext(source.slice(source.indexOf('class ApiError'),source.indexOf('function chosen(){')),ctx);
    await ctx.api('/api/keys/self');await ctx.api('/api/broker/env-candidates');await ctx.api('/api/broker/connect',{method:'POST',json:{provider:'openai'}});
    await ctx.api('/v1/models');vm.runInContext("key='tk_abcdefghij_x'",ctx);await ctx.api('/api/broker/upstreams/openai',{method:'PUT',json:{}});
    assert.deepEqual(sent.map(r=>r.headers['x-titanos-studio']??null),['1','1','1',null,'1']);
    assert.equal(sent[2].headers['content-type'],'application/json');assert.equal(sent[4].headers.authorization,'Bearer tk_abcdefghij_x');
    // a broker refusal carries its own reason, without the hints written for local model routes; a dead server is said in words
    ctx.fetch=async()=>({ok:false,status:503,headers:{get:()=>null},json:async()=>({error:{code:'vault_refs_only',message:'This vault accepts env: and file: references only'}})});
    await assert.rejects(ctx.api('/api/broker/connect',{method:'POST',json:{}}),error=>error.message==='This vault accepts env: and file: references only'&&error.status===503);
    ctx.fetch=async()=>{throw new TypeError('Failed to fetch');};
    await assert.rejects(ctx.api('/v1/models'),error=>/^Titanos is not reachable at this address\. Start it or press Refresh\.$/.test(error.message)&&error.code==='unreachable');
    ctx.fetch=async()=>{throw Object.assign(new Error('The operation was aborted.'),{name:'AbortError'});};
    await assert.rejects(ctx.api('/v1/models'),error=>error.name==='AbortError');
  }

  // (11) R30/R31 keyless Studio: env candidates by name, connect with a variable name, then with a pasted key sent once
  const STUDIO={id:'keyless-studio',kind:'anonymous_loopback',scopes:['broker:admin','broker:read','credentials:write','inference','jobs','mcp','models:read','models:write','settings:read']};
  const ENV={object:'list',role:'standalone',data:[{provider:'anthropic',kind:'cloud',variables:[{name:'ANTHROPIC_API_KEY',set:false}],any_set:false},
    {provider:'groq',kind:'cloud',variables:[{name:'GROQ_API_KEY',set:true}],any_set:true},{provider:'openai',kind:'cloud',variables:[{name:'OPENAI_API_KEY',set:true}],any_set:true}]};
  const PROFILES={object:'list',data:[{id:'openai',kind:'cloud',adapter:'openai_compat',auth:'bearer',auth_optional:false,default_base_url:'https://api.openai.com/v1'},
    {id:'groq',kind:'cloud',adapter:'openai_compat',auth:'bearer',auth_optional:false,default_base_url:'https://api.groq.com/openai/v1'},
    {id:'anthropic',kind:'cloud',adapter:'anthropic',auth:'anthropic',auth_optional:false,default_base_url:'https://api.anthropic.com'},
    {id:'custom_openai',kind:'cloud',adapter:'openai_compat',auth:'bearer',auth_optional:true,default_base_url:null},{id:'pending_contract',kind:'slot',adapter:'none',auth:'none',auth_optional:true}]};
  let bodies=[];const ANSWER={upstream:{id:'groq',profile:'groq',enabled:true,catalog:'live',credential:'groq-key',base_url:'https://api.groq.com/openai/v1'},
    credential:{id:'groq-key',source:'env',env_var:'GROQ_API_KEY',fingerprint:'5e6f7a8b',validation:'ok'},validation:'ok',cloud_policy:{file:'explicit',ceiling:'fallback',effective:'explicit'},
    anonymous_cloud:true,policy_changes:{anonymous_cloud:true},catalog_status:null,models:['llama-3.3-70b','whisper-large-v3'],models_total:2,aliases:[],aliases_skipped:[],next_steps:['Call a model as <model>@groq']};
  t=load({self:STUDIO,answers:{'/api/broker/env-candidates':ENV,'/api/broker/profiles':PROFILES},connect:async body=>{bodies.push(JSON.parse(JSON.stringify(body)));return {json:async()=>JSON.parse(JSON.stringify(ANSWER))};}});
  await t.context.refreshBroker();
  assert(t.requests.some(r=>r.url==='/api/broker/env-candidates'),'env candidates were not requested');
  const providers=t.$('broker-connect-provider').options.map(o=>[o.value,String(o.textContent)]);
  assert.deepEqual(providers.map(([id])=>id),['openai','anthropic','groq','custom_openai'],'profiles in a stable order, slots left out');
  assert.equal(t.$('broker-connect-provider').value,'groq','the first provider with a key found and no upstream is preselected');
  assert.match(providers.find(([id])=>id==='groq')[1],/GROQ_API_KEY found/);assert.match(providers.find(([id])=>id==='openai')[1],/connected/);
  assert.deepEqual(t.$('broker-connect-env').options.map(o=>[o.value,Boolean(o.disabled)]),[['GROQ_API_KEY',false]]);
  assert(t.$('broker-connect-anonymous').checked,'keyless Studio offers keyless use of the new provider');
  assert.equal(t.$('broker-keyless-note'),null,'the keyless Studio principal may write credentials');assert(t.$('broker-credential-form'));
  assert.equal(t.$('broker-anonymous-cloud').disabled,false);assert.equal(t.$('broker-allow-http-lan').disabled,true);
  await t.$('broker-connect-form').onsubmit({preventDefault(){}});
  assert.deepEqual(bodies,[{provider:'groq',enable:true,credential:{env:'GROQ_API_KEY'},anonymous_cloud:true}],'the connect body carries the variable name only');
  const result=t.$('broker-connect-result');assert.match(result.textContent,/Groq connected/);assert.match(result.textContent,/llama-3\.3-70b/);assert.match(result.textContent,/env GROQ_API_KEY · 5e6f7a8b/);
  assert.match(String(t.$('broker-message').textContent),/Connected Groq: 2 models found/);
  // the next steps are Studio's own sentences from the answer's fields, never the server's CLI wording
  assert.match(result.textContent,/Its models are listed under Models with their full ids, for example llama-3\.3-70b@groq\./);
  assert.doesNotMatch(result.textContent,/Call a model as|aliases: true/);
  // a pasted key: sent once in the body, the field emptied and replaced, no trace in the DOM, the state or storage
  t.state().connectDraft={provider:'anthropic'};t.context.renderBroker();
  assert.equal(t.$('broker-connect-provider').value,'anthropic');
  const radios=walk(t.$('broker-connect-form')).filter(n=>n.attributes.name==='broker-connect-source');
  assert.deepEqual(radios.map(n=>[n.value,Boolean(n.checked)]),[['env',false],['value',true]],'with no key found the paste option is chosen');
  // each radio has its own label; the select and the write-only key field are named by their option text
  assert.equal(t.$('broker-connect-secret').attributes['aria-labelledby'],'broker-connect-secret-label');
  assert.equal(t.$('broker-connect-env').attributes['aria-labelledby'],'broker-connect-env-label');
  assert.equal(String(t.$('broker-connect-secret-label').textContent),'Paste a key (sent once, never shown again)');
  assert(radios.every(n=>n.parentElement.tagName==='label'&&n.parentElement.children.length===2),'a radio label wraps more than its radio and caption');
  // focus alone never changes the chosen source (WCAG 2.1.1, 3.2.1: Tab passes every field on the way to Connect);
  // picking a variable or typing a key is a deliberate edit and chooses that field's option
  const chosenSource=()=>radios.filter(n=>n.checked).map(n=>n.value);
  for(const id of ['broker-connect-env','broker-connect-secret'])t.$(id).dispatchEvent({type:'focus'});
  assert.deepEqual(chosenSource(),['value'],'focusing a field changed the chosen source');
  t.$('broker-connect-env').dispatchEvent({type:'change',bubbles:true});
  assert.deepEqual(chosenSource(),['env'],'picking a variable did not choose the environment option');
  t.$('broker-connect-secret').dispatchEvent({type:'focus'});
  assert.deepEqual(chosenSource(),['env'],'focusing the key field changed the chosen source');
  t.$('broker-connect-secret').dispatchEvent({type:'input',bubbles:true});
  assert.deepEqual(chosenSource(),['value'],'typing a key did not choose the paste option');
  // Enter pressed in the key field (focus on it): the swapped-in field keeps focus through the submit and re-renders (WCAG 2.4.3)
  const pasted=t.$('broker-connect-secret');pasted.value=SENTINEL;bodies=[];pasted.focus();
  await t.$('broker-connect-form').onsubmit({preventDefault(){}});
  assert.equal(bodies.length,1);assert.equal(bodies[0].credential.value,SENTINEL);assert.equal(pasted.value,'');assert.notEqual(t.$('broker-connect-secret'),pasted);
  assert(!serialize(t.tree).includes(SENTINEL),'the DOM holds the pasted key');assert(!JSON.stringify(t.state()).includes(SENTINEL),'broker state holds the pasted key');assert.deepEqual(t.stored,[]);
  assert.equal(document.activeElement,t.$('broker-connect-secret'),'connecting from the key field dropped keyboard focus');
  // a blank paste sends nothing, and focus stays in the key field
  t.state().connectDraft={provider:'anthropic'};t.context.renderBroker();bodies=[];t.$('broker-connect-secret').focus();
  await t.$('broker-connect-form').onsubmit({preventDefault(){}});assert.deepEqual(bodies,[]);assert.match(String(t.$('broker-message').textContent),/blank; nothing changed/);
  assert.equal(document.activeElement,t.$('broker-connect-secret'),'a blank paste dropped keyboard focus');document.activeElement=null;

  // (12) typed messages: a server without one-step connect (404) and a refused key (422); never a success claim
  t=load({self:STUDIO,answers:{'/api/broker/env-candidates':ENV,'/api/broker/profiles':PROFILES}});await t.context.refreshBroker();
  await t.$('broker-connect-form').onsubmit({preventDefault(){}});
  assert.match(t.$('broker-connect-result').textContent,/Groq is not connected/);
  assert.match(t.$('broker-connect-result').textContent,/no one-step connect \(POST \/api\/broker\/connect answered 404\)/);
  assert.equal(t.state().connectResult.ok,false);assert.doesNotMatch(String(t.$('broker-message').textContent),/^Connected/);
  t=load({self:STUDIO,answers:{'/api/broker/env-candidates':ENV,'/api/broker/profiles':PROFILES},connect:async()=>{throw Object.assign(Error('The upstream refused the credential (status_401); groq stays disabled'),{status:422,code:'credential_validation_failed'});}});
  await t.context.refreshBroker();await t.$('broker-connect-form').onsubmit({preventDefault(){}});
  assert.match(t.$('broker-connect-result').textContent,/refused the credential \(status_401\); groq stays disabled/);assert.equal(t.$('broker-message').className,'hint error');
  // a refused pasted key: the error re-renders the panel, and focus stays in the (fresh, empty) key field
  t.state().connectDraft={provider:'anthropic'};t.context.renderBroker();const refused=t.$('broker-connect-secret');refused.value=SENTINEL;refused.focus();
  await t.$('broker-connect-form').onsubmit({preventDefault(){}});
  assert.equal(t.$('broker-message').className,'hint error');assert.notEqual(t.$('broker-connect-secret'),refused);assert.equal(t.$('broker-connect-secret').value,'');
  assert.equal(document.activeElement,t.$('broker-connect-secret'),'a refused pasted key dropped keyboard focus');document.activeElement=null;
  t=load({self:STUDIO,answers:{'/api/broker/env-candidates':ENV,'/api/broker/profiles':PROFILES},connect:async()=>({json:async()=>({detail:'Not a connect answer'})})});
  await t.context.refreshBroker();await t.$('broker-connect-form').onsubmit({preventDefault(){}});
  assert.match(t.$('broker-connect-result').textContent,/not with a connect result/);assert.equal(t.state().connectResult.ok,false);
  // env discovery missing (an older server): the variable name is typed, and the hint says which route answered 404
  t=load({self:STUDIO,answers:{'/api/broker/profiles':PROFILES},connect:async body=>{bodies.push(body);return {json:async()=>ANSWER};}});
  t.context.api=(inner=>async(url,options={})=>{if(url==='/api/broker/env-candidates')throw Object.assign(Error('Not Found'),{status:404});return inner(url,options);})(t.context.api);
  await t.context.refreshBroker();
  assert.equal(t.$('broker-connect-env').tagName,'input');assert.match(t.$('broker-connect-form').textContent,/env-candidates answered 404/);
  // an upstream that exists but cannot serve (disabled, no credential) is never offered as connected
  t=load({self:STUDIO,answers:{'/api/broker/env-candidates':ENV,'/api/broker/profiles':PROFILES,'/api/broker/upstreams':{data:[{id:'openai',profile:'openai',label:'OpenAI',enabled:false,credential:null,catalog:'static'},{id:'anthropic',profile:'anthropic',label:'Anthropic',enabled:false,credential:null,catalog:'static'}]}}});
  await t.context.refreshBroker();
  const idle=t.$('broker-connect-provider').options.map(o=>[o.value,String(o.textContent)]);
  assert.doesNotMatch(idle.find(([id])=>id==='openai')[1],/connected/);assert.match(idle.find(([id])=>id==='openai')[1],/OPENAI_API_KEY found/);
  assert.equal(idle.find(([id])=>id==='anthropic')[1],'Anthropic · configured, not enabled');
  t.state().connectDraft={provider:'anthropic'};t.context.renderBroker();
  assert.match(t.$('broker-connect-form').textContent,/Upstream anthropic is configured but not enabled/);
  // enabled with a credential but not ready (an env: key the server process does not hold): never "connected"
  const unreadyRow=id=>({id,profile:id,label:id,enabled:true,credential:id+'-key',catalog:'live',ready:false,readiness:{reason:'unvalidated',detail:'credential_unavailable'},credential_validation:'ok'});
  t=load({self:STUDIO,answers:{'/api/broker/env-candidates':ENV,'/api/broker/profiles':PROFILES,'/api/broker/upstreams':{data:[unreadyRow('openai'),unreadyRow('anthropic')]}}});
  await t.context.refreshBroker();
  const unready=t.$('broker-connect-provider').options.map(o=>[o.value,String(o.textContent)]);
  assert.doesNotMatch(unready.find(([id])=>id==='openai')[1],/connected/);assert.match(unready.find(([id])=>id==='openai')[1],/OPENAI_API_KEY found/);
  assert.equal(unready.find(([id])=>id==='anthropic')[1],'Anthropic · configured, not ready');
  t.state().connectDraft={provider:'anthropic'};t.context.renderBroker();
  assert.match(t.$('broker-connect-form').textContent,/Upstream anthropic is enabled but not ready/);
  t.state().connectDraft={provider:'openai'};t.context.renderBroker();
  assert.doesNotMatch(t.$('broker-connect-form').textContent,/Connected as upstream openai/);
  // back to the disabled pair for the focus checks below
  t=load({self:STUDIO,answers:{'/api/broker/env-candidates':ENV,'/api/broker/profiles':PROFILES,'/api/broker/upstreams':{data:[{id:'openai',profile:'openai',label:'OpenAI',enabled:false,credential:null,catalog:'static'},{id:'anthropic',profile:'anthropic',label:'Anthropic',enabled:false,credential:null,catalog:'static'}]}}});
  await t.context.refreshBroker();t.state().connectDraft={provider:'anthropic'};t.context.renderBroker();
  // a re-render keeps keyboard focus (WCAG 2.4.3, 3.2.2): the Provider select changed by an Arrow (change fires) is the new
  // Provider select; a node with no id (Connect) is found again by its place in its nearest ancestor with an id
  const shownProvider=t.$('broker-connect-provider');shownProvider.focus();shownProvider.value='openai';shownProvider.onchange();
  assert.notEqual(t.$('broker-connect-provider'),shownProvider,'the provider change did not re-render');
  assert.equal(document.activeElement,t.$('broker-connect-provider'),'a provider change dropped keyboard focus');assert.equal(t.$('broker-connect-provider').value,'openai');
  const submitOf=()=>walk(t.$('broker-connect-form')).find(n=>n.tagName==='button'&&n.type==='submit');
  const connectButton=submitOf();connectButton.focus();t.context.renderBroker();
  assert.notEqual(submitOf(),connectButton);assert.equal(document.activeElement,submitOf(),'a re-render dropped focus from Connect');
  document.activeElement=null;t.context.renderBroker();assert.equal(document.activeElement,null,'a re-render took focus it did not have');
  // empty activity tables say so instead of showing bare headers
  t=load({self:ADMIN,answers:{'/api/broker/usage?group_by=upstream,model':{rows:[]},'/api/broker/decisions?limit=10':{data:[]}}});await t.context.refreshBroker();
  assert.match(t.$('broker-usage').textContent,/No cloud requests recorded yet\./);assert.match(t.$('broker-decisions').textContent,/No routing decisions recorded yet\./);
  // (13) a key without credentials:write is pointed at the CLI; an insecure origin disables the form
  t=load({self:{id:'k3',kind:'key',scopes:['broker:admin']}});await t.context.refreshBroker();
  assert.match(t.$('broker-connect-cli').textContent,/needs a key with credentials:write/);
  t=load({self:ADMIN,secure:false});await t.context.refreshBroker();
  assert(t.$('broker-connect-insecure'));assert.equal(t.$('broker-connect-provider').disabled,true);

  console.log('PASS: write-only credential and connect fields (never prefilled, emptied and replaced, no DOM/state/storage trace), the R29 Studio header, R30/R31 connect by variable name or pasted key with typed 404/422 messages and no false success, saved credential bound to its upstream, key secret shown once, kept across re-renders and gone on dismiss or a principal change, one lime action, independent refresh, scope-gated sections, keyless CLI note, insecure-origin refusal, missing broker/settings blocks tolerated, external/virtual badges and the provenance line. DOM double only.');
  reportMatrix([CASE],[]);
})().catch(error=>{console.error(error);reportMatrix([],[CASE]);process.exitCode=1;});
