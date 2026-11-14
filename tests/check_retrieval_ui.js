/* Model-free DOM verification: no browser/network/model execution. */
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const ids=new Map();
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.attributes={};this.value='';this._text='';}
  set id(value){this._id=value;ids.set(value,this);} get id(){return this._id;}
  set textContent(value){this._text=String(value);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  set innerHTML(value){throw Error('Untrusted answer/source text must not use innerHTML');}
  append(...items){for(const item of items){if(item.parentElement){const i=item.parentElement.children.indexOf(item);if(i>=0)item.parentElement.children.splice(i,1);}item.parentElement=this;this.children.push(item);}}
  after(item){this.parentElement.append(item);}
  replaceChildren(...items){this.children=[];this._text='';this.append(...items);}
  setAttribute(key,value){this.attributes[key]=value;}
  addEventListener(){}
}
const root=new Element('main'),generate=new Element('form');generate.id='generate';root.append(generate);
let requests=[];let answer;let pendingInference=false;let activeSignal;let inferenceEntered;
const context={document:{createElement:tag=>new Element(tag),getElementById:id=>ids.get(id)},window:{},AbortController,Map,JSON,Number,
  api:async(path,options={})=>{
    requests.push({path,options});
    if(path==='/v1/extensions')return {json:async()=>({retrieval_answer:{models:{embedding:['e'],rerank:['r'],llm:['l']}}})};
    if(pendingInference){activeSignal=options.signal;inferenceEntered();return await new Promise((resolve,reject)=>{options.signal.addEventListener('abort',()=>reject(Object.assign(new Error('Stopped'),{name:'AbortError'})),{once:true});});}
    return {json:async()=>answer};
  }};
vm.createContext(context);vm.runInContext(fs.readFileSync('src/titanos/web/retrieval.js','utf8'),context);
const data={status:'answered',reasons:[],documents:[{document_id:'A',title:'<img onerror=bad>',source_uri:'javascript:bad'}],
  claims:[{text:'<script>claim</script>',citations:[{document_id:'A',quote:'<svg onload=bad>',spans:[[0,17],[30,47]]}]}],stages:[]};
context.window.renderRetrievalAnswer(data);
let output=ids.get('retrieval-results').textContent;
assert(output.includes('<script>claim</script>') && output.includes('<svg onload=bad>') && output.includes('not independently verified'));
const all=node=>[node,...node.children.flatMap(all)];assert(!all(ids.get('retrieval-results')).some(node=>node.tagName==='a'));
context.window.renderRetrievalAnswer({status:'failed',claims:[],documents:[],reasons:['invalid_citation'],stages:[{status:'failed',error:{message:'Citation failed validation'}}]});
assert(!ids.get('retrieval-results').textContent.includes('<script>claim'));
assert(ids.get('retrieval-results').textContent.includes('No answer returned'));
(async()=>{
  ids.get('retrieval-documents').files=[{text:async()=>JSON.stringify([{id:'A',text:'A source passage.'}])}];
  for(const [name,value] of Object.entries({query:'Question',embedding_model:'e',rerank_model:'r',llm_model:'l',context_top_n:'2'}))ids.get('retrieval-'+name).value=value;
  answer={status:'abstained',claims:[],documents:[],reasons:['model_abstained_from_selected_context'],stages:[]};
  await ids.get('retrieval-form').onsubmit({preventDefault(){}});
  assert(ids.get('retrieval-results').textContent.includes('The model did not answer'));
  assert(ids.get('retrieval-results').textContent.includes('Relevant information may still be present'));
  assert(!ids.get('retrieval-results').textContent.includes('No answer in selected context'));
  assert.equal(requests.at(-1).path,'/v1/retrieval/answer');assert.equal(requests.at(-1).options.json.context_top_n,2);
  assert.equal(requests.at(-1).options.json.documents[0].id,'A');assert(requests.at(-1).options.signal);
  assert.equal(Object.hasOwn(requests.at(-1).options.json,'prompt_version'),false);
  const policy=ids.get('retrieval-prompt_version');
  // the answer-policy select sits in the folded, Advanced-only 06.3 sub-cell
  const holder=(node=>{while(node&&node.tagName!=='details')node=node.parentElement;return node;})(policy);
  assert(holder&&String(holder.className).includes('wf-cell')&&String(holder.className).includes('advanced-only'),'the policy select is inside the 06.3 settings cell');
  assert(policy.children.some(option=>option.textContent.includes('Use source facts; ignore embedded commands')));
  policy.value='retrieval-answer-v1';await ids.get('retrieval-form').onsubmit({preventDefault(){}});
  assert.equal(requests.at(-1).options.json.prompt_version,'retrieval-answer-v1');
  const form=ids.get('retrieval-form');
  const submit=all(form).find(node=>node.tagName==='button'&&node.type==='submit');
  const refresh=all(form).find(node=>node.tagName==='button'&&node.textContent==='Refresh workflow models');
  const stop=all(form).find(node=>node.tagName==='button'&&node.textContent==='Stop');
  const before=requests.filter(request=>request.path==='/v1/retrieval/answer').length;
  pendingInference=true;const entered=new Promise(resolve=>{inferenceEntered=resolve;});
  const pending=form.onsubmit({preventDefault(){}});await entered;
  assert.equal(submit.disabled,true);assert.equal(stop.hidden,false);
  const ownedSignal=activeSignal;
  await refresh.onclick();assert.equal(submit.disabled,true,'Refresh enabled a submit during active inference');
  await form.onsubmit({preventDefault(){}});
  assert.equal(requests.filter(request=>request.path==='/v1/retrieval/answer').length,before+1,'Overlapping submission launched another inference');
  assert.equal(activeSignal,ownedSignal);
  stop.onclick();assert.equal(ownedSignal.aborted,true,'Stop did not abort the request that owns the form');
  await pending;assert.equal(stop.hidden,true);assert.equal(submit.disabled,false);
  assert(all(root).some(node=>node.attributes.role==='status'&&node.textContent==='Stopped.'));
  pendingInference=false;await form.onsubmit({preventDefault(){}});
  assert.equal(requests.filter(request=>request.path==='/v1/retrieval/answer').length,before+2,'Cleanup did not release the form for the next request');
  console.log('PASS minimal retrieval form, source rendering and deferred inference→refresh→blocked overlap→owned Stop/cleanup; no real browser or model');
})().catch(error=>{console.error(error);process.exitCode=1;});
