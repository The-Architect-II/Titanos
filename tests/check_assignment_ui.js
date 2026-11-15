/* Assignment drafts and out-of-order responses; no browser/network/device. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../src/titanos/web/app.js'),'utf8');
const nodes=new Map();
for(const id of ['panel','mode','ranks','replicas','devices','status','load','apply']){
  const node={value:'',options:[],parentElement:{},append(item){this.options.push(item);},replaceChildren(...items){this.options=items;}};
  Object.defineProperty(node,'selectedOptions',{get(){return this.options.filter(item=>item.selected);}});
  nodes.set('assignment-'+id,node);
}
const A='GPU-11111111-1111-1111-1111-111111111111',B='GPU-22222222-2222-2222-2222-222222222222';
let selected='A',pending=null,posts=[];
const context={Map,Set,console,encodeURIComponent,Number,String,active:null,
  $:id=>nodes.get(id),chosen:()=>({id:selected}),preferences:{mode:'advanced'},tr:key=>key,i18n:{t:key=>key},
  option:(value,textContent)=>({value,textContent,selected:false}),refresh:async()=>{},
  api:async(url,options)=>{
    if(options){posts.push(options.json);return{json:async()=>({...options.json,replicas:1})};}
    const model=decodeURIComponent(url.slice(url.lastIndexOf('/')+1));
    if(pending?.model===model)await pending.promise;
    return{json:async()=>({model,mode:'single',groups:[[model==='B'?B:A]],replicas:1,tensor_parallel_size:2})};
  }};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('const UI_MESSAGE='),source.indexOf('function bindStaticTranslations(){')),context);
vm.runInContext(source.slice(source.indexOf('const assignmentDrafts='),source.indexOf('function setupTaskFields()')),context);
context.setupAssignments();
function hold(model){let release;const promise=new Promise(resolve=>{release=resolve;});pending={model,promise};return()=>{pending=null;release();};}
(async()=>{
  await context.loadAssignment();
  nodes.get('assignment-mode').value='model_sharding';nodes.get('assignment-mode').onchange();
  nodes.get('assignment-devices').append({value:B,selected:true});nodes.get('assignment-devices').onchange();
  await context.loadAssignment(); // Locale/mode redraw must keep the dirty draft.
  assert.equal(nodes.get('assignment-mode').value,'model_sharding');
  selected='B';await context.loadAssignment();selected='A';await context.loadAssignment();
  assert.equal(nodes.get('assignment-mode').value,'model_sharding');
  assert.deepEqual(nodes.get('assignment-devices').selectedOptions.map(o=>o.value),[A,B]);
  selected='C';await context.loadAssignment();
  const release=hold('C'),late=context.loadAssignment();
  selected='B';await context.loadAssignment();release();await late;
  assert.deepEqual(nodes.get('assignment-devices').selectedOptions.map(o=>o.value),[B]);
  await nodes.get('assignment-apply').onclick();
  assert.equal(posts.at(-1).model,'B');assert.equal(JSON.stringify(posts.at(-1).groups),JSON.stringify([[B]]));
  const releaseSame=hold('B'),same=context.loadAssignment();
  nodes.get('assignment-replicas').value='2';nodes.get('assignment-replicas').oninput();
  releaseSame();await same;
  assert.equal(nodes.get('assignment-replicas').value,'2');
  console.log('PASS: per-model dirty drafts survive redraw/switch; late other-model and same-model responses cannot overwrite edits; save uses current model devices.');
})().catch(error=>{console.error(error);process.exitCode=1;});
