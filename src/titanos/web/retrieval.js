'use strict';
(() => {
  const rt=(name,fallback,variables={})=>typeof ui==='function'?ui(name,fallback,variables):fallback.replace(/\{\{(\w+)\}\}/g,(_,key)=>String(variables[key]??''));
  const put=(node,value)=>{if(typeof setContent==='function')return setContent(node,value);node.textContent=String(value??'');return node;};
  const attribute=(node,name,value)=>{if(typeof setUiAttribute==='function')return setUiAttribute(node,name,value);node[name]=String(value??'');return node;};

  const element = (tag, text) => { const node=document.createElement(tag); if(text!==undefined)put(node,text); return node; };
  // bay 06: a collapsible bay after Jobs; the summary is the whole strip with the numbered plate on the frame
  const panel=element('details'); panel.id='retrieval-panel'; panel.className='panel bay workflow-bay';
  const summary=element('summary'); summary.className='bay-summary'; const plate=element('span'); plate.className='bay-plate';
  const number=element('span'); number.className='bay-no'; number.textContent='06'; const separator=element('span'); separator.className='bay-sep'; separator.textContent='//'; separator.setAttribute('aria-hidden','true');
  plate.append(number,separator,element('span',rt("retrieval.title","Answer from your documents"))); const line=element('span',rt("retrieval.bay_line","Retrieve passages, rerank them and answer with citations, using three registered models.")); line.className='bay-line';
  summary.append(plate,line); panel.append(summary);
  const form=element('form'); form.id='retrieval-form'; panel.append(form); (document.getElementById('bay-jobs')||document.getElementById('generate')).after(panel);
  form.append(element('p',rt("retrieval.intro","Choose three registered models and a JSON array of passages with unique id and text fields. Files are sent to this local Titanos server.")));
  // numbered sub-cells on a compact grid (06.1 models, 06.2 passages, 06.3 limits), never one long column of fields
  function cell(index,title,tag='div') { const box=element(tag); box.className='wf-cell'+(tag==='details'?' advanced-only':''); const head=element(tag==='details'?'summary':'h3'); const no=element('span'); no.className='sub-index'; no.textContent=index; head.append(no,element('span',title)); box.append(head); const grid=element('div'); grid.className='wf-grid'; box.append(grid); form.append(box); return {box,grid}; }
  const inputs={};
  function field(host,name,label,type='number') { const box=element('label');box.append(element('span',label)); const input=element(type==='select'?'select':type==='textarea'?'textarea':'input'); input.name=name; input.id='retrieval-'+name; if(!['select','textarea'].includes(type))input.type=type; if(type==='number')input.min='1'; box.append(input); host.append(box); inputs[name]=input; return input; }
  const models=cell('06.1',rt("retrieval.models_cell","Models"));
  field(models.grid,'embedding_model',rt("retrieval.embedding","Embedding model"),'select').required=true;
  field(models.grid,'rerank_model',rt("retrieval.reranker","Reranker"),'select').required=true;
  field(models.grid,'llm_model',rt("retrieval.answer_model","Answer model"),'select').required=true;
  const refresh=element('button',rt("retrieval.refresh","Refresh workflow models")); refresh.type='button'; refresh.className='quiet'; const modelActions=element('div'); modelActions.className='wf-actions'; modelActions.append(refresh); models.box.append(modelActions);
  const passages=cell('06.2',rt("retrieval.passages_cell","Passages and question"));
  field(passages.grid,'documents',rt("retrieval.passages_file","Passages JSON file"),'file').accept='.json,application/json'; inputs.documents.required=true;
  field(passages.grid,'query',rt("retrieval.question","Question"),'textarea').required=true; inputs.query.rows=3; inputs.query.parentElement.className='wf-wide';
  const options=cell('06.3',rt("retrieval.settings","Retrieval and answer settings"),'details').grid;
  for(const [name,label] of [['retrieve_top_k',rt("retrieval.candidates","Candidates to rerank")],['context_top_n',rt("retrieval.context_passages","Passages supplied to answer model")],['embedding_max_tokens',rt("retrieval.encoder_tokens","Encoder token limit")],['rerank_max_input_tokens',rt("retrieval.rerank_tokens","Reranker token limit")],['context_max_tokens',rt("retrieval.input_tokens","Answer input token limit")],['max_tokens',rt("retrieval.output_tokens","Answer output token limit")],['timeout',rt("retrieval.deadline","Whole request deadline (milliseconds)")]]) {
    const input=field(options,name,label); attribute(input,"placeholder",rt("ui.default_model_server","Model / server default"));
  }
  const policy=field(options,'prompt_version',rt("retrieval.policy","How to read source passages"),'select');
  for(const [value,label] of [['',rt("ui.default_model_server","Model / server default")],['retrieval-answer-v2',rt("retrieval.policy_v2","Use source facts; ignore embedded commands (v2)")],['retrieval-answer-v1',rt("retrieval.policy_v1","Original source-reading policy (v1)")]]) { const option=element('option',label); option.value=value; policy.append(option); }
  // the bay's one lime action: Answer with sources
  const submit=element('button',rt("retrieval.submit","Answer with sources")); submit.type='submit'; submit.className='go'; submit.disabled=true; const stop=element('button',rt('composer.stop','Stop')); stop.type='button'; stop.className='quiet'; stop.hidden=true; const actions=element('div'); actions.className='wf-actions'; actions.append(submit,stop); form.append(actions);
  const message=element('p'); message.setAttribute('role','status'); panel.append(message); const results=element('div'); results.id='retrieval-results'; panel.append(results);
  let active=null;
  const missingModels=()=>['embedding','rerank','llm'].some(role=>!inputs[role+'_model'].value);
  function render(result) {
    results.replaceChildren(); const heading=element('h2',result.status==='answered'?rt("retrieval.answer_heading","Answer with source quotations"):rt("retrieval.no_answer","No answer returned")); results.append(heading);
    if(result.status==='answered') {
      results.append(element('p',rt("retrieval.quote_note","Quotes match submitted text. Review whether they support each claim; answer correctness was not independently verified.")));
      const sources=new Map(result.documents.map(source=>[source.document_id,source]));
      for(const claim of result.claims) {
        const box=element('article'); box.className='result'; box.append(element('p',claim.text));
        for(const citation of claim.citations) { const source=sources.get(citation.document_id); box.append(element('strong',citation.document_id+(source?.title?' · '+source.title:''))); box.append(element('blockquote',citation.quote)); if(source?.source_uri)box.append(element('small',rt("retrieval.source_ref","Caller source reference: {{source}}",{source:source.source_uri}))); if(citation.spans.length>1)box.append(element('small',rt("retrieval.quote_positions","This quote appears at {{count}} positions in the passage.",{count:citation.spans.length}))); }
        results.append(box);
      }
    } else if(result.status==='abstained') { results.append(element('p',rt("retrieval.abstained","The model did not answer from the selected passages. Relevant information may still be present; review the source and stage trace."))); }
    else { const failure=result.stages.find(stage=>stage.status==='failed'); results.append(element('p',failure?.error?.message||result.reasons.join(', '))); }
    const details=element('details'); details.append(element('summary',rt("retrieval.trace","Source and stage trace"))); details.append(element('pre',JSON.stringify(result,null,2))); results.append(details);
  }
  async function refreshModels() { try { const data=await(await api('/v1/extensions')).json(); const models=data.retrieval_answer?.models||{}; for(const role of ['embedding','rerank','llm']) { const select=inputs[role+'_model']; const previous=select.value; select.replaceChildren(); for(const id of models[role]||[]) { const option=element('option',id); option.value=id; select.append(option); } if((models[role]||[]).includes(previous))select.value=previous; } const unavailable=missingModels(); submit.disabled=Boolean(active)||unavailable; if(unavailable&&!active)put(message,rt("retrieval.missing_models","Register a compatible encoder, reranker and native answer model to use this workflow.")); } catch(error) { if(!active)put(message,error.message); } }
  refresh.onclick=refreshModels; panel.addEventListener('toggle',()=>{ if(panel.open)refreshModels(); });
  document.getElementById('connect')?.addEventListener('click',refreshModels);
  document.getElementById('refresh')?.addEventListener('click',refreshModels);
  form.onsubmit=async event=>{ event.preventDefault(); if(active)return; const request=new AbortController(); active=request; submit.disabled=true; stop.hidden=false; put(message,rt("retrieval.running","Retrieving passages and preparing an answer…")); results.replaceChildren();
    try { const file=inputs.documents.files[0]; if(!file)throw Error(rt("retrieval.choose_file","Choose a passages JSON file")); const payload={documents:JSON.parse((await file.text()).replace(/^\uFEFF/,'')),query:inputs.query.value}; for(const name of ['embedding_model','rerank_model','llm_model'])payload[name]=inputs[name].value; if(inputs.prompt_version.value)payload.prompt_version=inputs.prompt_version.value; for(const name of ['retrieve_top_k','context_top_n','embedding_max_tokens','rerank_max_input_tokens','context_max_tokens','max_tokens','timeout'])if(inputs[name].value!=='')payload[name]=Number(inputs[name].value); const response=await api('/v1/retrieval/answer',{method:'POST',json:payload,signal:request.signal}); const result=await response.json(); if(active===request&&!request.signal.aborted){render(result); put(message,result.status==='failed'?rt("retrieval.failed","Request failed; no answer claims returned."):rt("retrieval.finished","Finished."));} }
    catch(error) { if(active===request)put(message,error.name==='AbortError'?rt("ui.stopped","Stopped."):error.message); }
    finally { if(active===request){active=null; submit.disabled=missingModels(); stop.hidden=true;} }
  };
  stop.onclick=()=>active?.abort();
  // Export only the renderer for model-free DOM verification, not a model action.
  window.renderRetrievalAnswer=render;
})();
