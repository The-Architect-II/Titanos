/* Static identity and theme contract (Cabinet): tokens, roles, the banned AI-template tells, the vendored font and its licence,
   contrast of the token pairs Studio uses, and the structural hooks the layout depends on. Browser rendering is verified
   separately (check_studio_broker_browser.js, check_studio_workflow_i18n.js). No browser, server or network. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.join(__dirname,'../src/titanos/web');
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const css=fs.readFileSync(path.join(root,'style.css'),'utf8');
const app=fs.readFileSync(path.join(root,'app.js'),'utf8');
const retrieval=fs.readFileSync(path.join(root,'retrieval.js'),'utf8');

// identity: the approved marks, lineage line and theme colour
assert.equal((html.match(/titanos-wordmark-v2\.svg/g)||[]).length,1,'The approved wordmark must be integrated exactly once');
assert.match(html,/rel="icon"[^>]+titanos-honeycomb-t-v2\.svg/);
assert.match(html,/Born of HiveMind/);
assert.match(html,/Standalone local inference/);
assert.match(html,/<meta name="theme-color" content="#050509">/);

// rules as [selectors, body] pairs, comments removed (the licence comment at the end is checked separately)
const source=css.replace(/\/\*[\s\S]*?\*\//g,'');
const rules=[];{const re=/([^{}]+)\{([^{}]*)\}/g;let match;while((match=re.exec(source)))rules.push([match[1].trim(),match[2]]);}
const using=(property,pattern=/./)=>rules.filter(([,body])=>body.split(';').some(decl=>{const [name,...rest]=decl.split(':');return name.trim()===property&&pattern.test(rest.join(':'));}));

// tokens: one :root block carries the palette (canvas, bays, seams, control border, violet roles, lime, cyan, magenta, status, text)
const tokens={};for(const [,body] of rules.filter(([selector])=>selector===':root'))for(const decl of body.split(';')){const m=/^\s*(--[\w-]+)\s*:\s*(.+?)\s*$/.exec(decl);if(m)tokens[m[1]]=m[2];}
const expected={'--bg':'#050509','--bay':'#0D0D18','--bay-hi':'#16162A','--cart':'#11111F','--slot':'#22104A','--seam':'#2A2640','--seam-hi':'#3A3458','--ctl':'#6E5CA8',
  '--frame':'#8000FF','--frame-hi':'#A855FF','--violet-ink':'#B488FF','--lime':'#A6FF00','--lime-ink':'#0F1A00','--lime-shadow':'#4E7D00','--magenta':'#D66BFF','--cyan':'#32D4FF',
  '--warn':'#FFC857','--error':'#FF4D6D','--text':'#FFFFFF','--text-2':'#C9C1E8','--text-3':'#8F88AD'};
for(const [name,value] of Object.entries(expected))assert.equal(String(tokens[name]).toUpperCase(),value,'theme token '+name);
for(const glyph of ['hollow','fill','half','slash','notch','orbit','run','cube','lattice','lit'])assert.match(tokens['--g-'+glyph]||'',/^url\("data:image\/svg\+xml,/,'hex glyph --g-'+glyph+' is an inline SVG');

// contrast (WCAG 2.x): text at 4.5:1 on every surface it sits on; control borders and the focus ring at 3:1
const luminance=hex=>{const c=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255).map(x=>x<=0.03928?x/12.92:((x+0.055)/1.055)**2.4);return .2126*c[0]+.7152*c[1]+.0722*c[2];};
const ratio=(a,b)=>{const [x,y]=[luminance(a),luminance(b)].sort((p,q)=>q-p);return (x+.05)/(y+.05);};
for(const surface of ['--bg','--bay','--cart','--bay-hi','--slot'])for(const ink of ['--text','--text-2','--text-3','--violet-ink','--lime','--cyan','--magenta','--warn','--error'])
  assert(ratio(expected[ink],expected[surface])>=4.5,`${ink} on ${surface} is ${ratio(expected[ink],expected[surface]).toFixed(2)}:1`);
for(const surface of ['--bg','--bay','--cart','--bay-hi','--slot'])assert(ratio(expected['--ctl'],expected[surface])>=3,'control border on '+surface);
assert(ratio(expected['--lime-ink'],expected['--lime'])>=4.5,'ink on lime');assert(ratio(expected['--text'],expected['--frame'])>=4.5,'white plate text on violet');

// the banned AI-template tells
const upper=source.toUpperCase();
for(const banned of ['BACKDROP-FILTER','TEXT-SHADOW','TRANSLATEY','SCALE(','ROTATE(','999PX','#00F0FF','#7B61FF','#94A3B8','#60A5FA','#1976D2','#76B900','#39FF14','ORBITRON','@IMPORT'])
  assert(!upper.includes(banned),'Banned style tell present: '+banned);
assert.doesNotMatch(source,/font-family\s*:[^;}]*\bInter\b/,'no Inter');
assert.equal((source.match(/gradient\(/g)||[]).length,1,'the only gradient is the header route rule');
assert.match(source,/\.app-header\{[^}]*border-image:linear-gradient\(90deg,var\(--frame\)[^;]*var\(--lime\) 100%\) 1/,'the route rule is the 2px violet-to-lime header border');
assert.doesNotMatch(source,/box-shadow\s*:(?!\s*none)/i,'no box-shadow except none (hard shadows are zero-blur drop-shadow filters)');
for(const match of source.matchAll(/drop-shadow\(([^)]*)\)/g)){const parts=match[1].trim().split(/\s+/);assert.equal(parts[2],'0','a drop-shadow with blur: '+match[0]);}
assert.doesNotMatch(source,/(^|[;{\s])transform\s*:/,'no transforms');
for(const [selector] of using('translate'))assert.match(selector,/#submit:active/,'translate only on the pressed RUN button: '+selector);
assert.doesNotMatch(source,/border-radius\s*:[^;}]*(?:[7-9]|\d{2,})px/,'radius above 6px');
assert.doesNotMatch(source,/border-radius\s*:[^;}]*%/,'percentage radius');
assert.doesNotMatch(source,/font(?:-size)?\s*:[^;}]*clamp\(/i,'no fluid display type');
assert.doesNotMatch(html,/class="eyebrow"|class="intro"|class="hero/,'no marketing hero');
for(const match of source.matchAll(/url\(\s*["']?([^"')]*)/g))assert.match(match[1],/^data:(image\/svg\+xml|font\/woff2;base64),/,'every url() is an inline data URI: '+match[1].slice(0,40));
assert.doesNotMatch(html+app+retrieval,/https?:\/\/(?!127\.0\.0\.1|localhost|huggingface\.co\/owner|host\/)[a-z0-9.-]+\.[a-z]{2,}\/[^'"\s]*\.(?:js|css|woff2?|ttf|png|jpg|svg)/i,'no external script, style, font or image');

// uppercase and tracking belong to the numbered plates, the boot line and RUN; the display face to plates, numerals and RUN
const plates=/bay-plate|bay-no|studio-plate|boot-title|#submit/;
for(const [selector] of using('text-transform',/uppercase/))assert.match(selector,plates,'uppercase outside the plates: '+selector);
for(const [selector] of using('letter-spacing',/[1-9]/))assert.match(selector,plates,'letter-spacing outside the plates: '+selector);
for(const [selector,body] of rules)if(/font(-family)?\s*:[^;]*var\(--font-display\)/.test(body))
  assert.match(selector,/bay-plate|bay-no|studio-plate|boot-title|#submit|counter b|sub-index|hexcap|bay-index a|row:first-of-type::before|step-input::before|actions::before/,'display face outside plates, numerals and RUN: '+selector);

// violet has roles: the 2px frame belongs to top-level bays and the selected model, the fill to structural buttons (Load); quiet buttons and inputs use the control border
for(const [selector,body] of rules)if(/border(-[a-z]+)?\s*:[^;]*var\(--frame\)/.test(body))assert.match(selector,/^\.app-header$|^button\.structural,#load$/,'a violet border outside the header and the violet-filled structural buttons: '+selector);
assert.match(source,/\.bay::before\{[^}]*background:var\(--frame\);clip-path:var\(--chamfer\)/,'bays carry the violet chamfered frame');
// the hard offset shadow carries hierarchy: only the working bays (Run, Results with a result, an opened collapsible bay) get it
assert.match(source,/\.composer\.bay,\.results-bay:has\(#results>\*\),details\.bay\[open\]\{filter:drop-shadow\(4px 4px 0 var\(--frame\)\)\}/,'working bays carry the hard offset shadow');
assert.doesNotMatch(source,/(^|[}\s])\.bay\{[^}]*filter:/,'a plain bay has no shadow');
assert.match(source,/details\.bay:not\(\[open\]\)\{[^}]*padding:24px 20px 10px\}/,'a collapsed bay is a slim plate row');
assert.match(source,/\.model\.selected::before,\.model\[aria-current="true"\]::before\{[^}]*background:var\(--frame\)/,'only the selected model gets the violet frame');
assert.match(source,/button\.quiet\{[^}]*border-color:var\(--ctl\)/,'quiet buttons use the control border');
// buttons are quiet by default; lime fills only the one .go action of a bay (and RUN)
assert.match(source,/(^|[}\s])button\{[^}]*color:var\(--text\);background:transparent;border:1px solid var\(--ctl\)/,'the default button is quiet');
for(const [selector,body] of rules)if(/(^|[\s,>])button/.test(selector)&&/background(-color)?\s*:\s*var\(--lime/.test(body))assert.match(selector,/^button\.go(:hover:not\(:disabled\)|:active:not\(:disabled\))?$/,'a lime button outside .go: '+selector);
assert.match(source,/button\.go\{color:var\(--lime-ink\);background:var\(--lime\)/,'.go is the lime action');
const goes=text=>(text.match(/className='go'|cls:'go'|class="go"/g)||[]).length;
assert.equal(goes(app),2,'app.js marks exactly two .go actions: Search (02) and Connect (07)');assert.equal(goes(retrieval),1,'retrieval.js marks Answer with sources (06)');assert.equal(goes(html),0,'index.html marks no .go button (RUN is #submit)');
assert.match(source,/input,select,textarea\{[^}]*border:1px solid var\(--ctl\)/,'inputs use the control border');
assert.match(source,/\.sub-bay\{[^}]*border:1px dashed var\(--seam-hi\)/,'sub-bays use seams');
assert.match(source,/#submit\{[^}]*background:var\(--lime\)[^}]*clip-path:var\(--chamfer\)/,'RUN is the chamfered lime arcade button');
assert.match(source,/\.route-note\{[^}]*border:1px solid var\(--cyan\)/,'the cloud route note is cyan');
assert.match(source,/\.result\.done\.external>\.spine\{background:var\(--cyan\)\}/,'a recorded cloud result keeps its cyan route mark');
assert.match(source,/\.rail:has\(\.model\[data-state="loaded"\]\)::before[^{]*\{background:var\(--g-lit\)/,'a live bay lights its cut corner');

// CSS carries no copy: content is empty, a separator or a counter
for(const match of source.matchAll(/(?<![-\w])content\s*:\s*([^;}]+)/g))assert.match(match[1].trim(),/^(""|"·"|counter\(step\)|none)$/,'CSS text literal: '+match[1]);

// the vendored display face: two Latin woff2 subsets inlined from fonts/, with the OFL beside them and in the stylesheet
const faces=[...css.matchAll(/@font-face\{font-family:"Titanos Chakra Petch";[^}]*src:url\(data:font\/woff2;base64,([A-Za-z0-9+/=]+)\)/g)].map(match=>match[1]);
const files=['ChakraPetch-SemiBold.woff2','ChakraPetch-BoldItalic.woff2'].map(name=>fs.readFileSync(path.join(root,'fonts',name)).toString('base64'));
assert.deepEqual(faces,files,'the inlined faces are exactly fonts/*.woff2');
assert(files.every(data=>Buffer.from(data,'base64').length<16384),'each face stays a small subset');
const ofl=fs.readFileSync(path.join(root,'fonts','OFL.txt'),'utf8');
assert.match(ofl,/Copyright 2018 The Chakra Petch Project Authors/);assert.match(ofl,/SIL OPEN FONT LICENSE Version 1\.1/);
assert(css.includes('Copyright 2018 The Chakra Petch Project Authors')&&css.includes('SIL OPEN FONT LICENSE Version 1.1'),'the stylesheet carries the font licence');

// live regions, selects, breakpoints, motion and focus
assert.match(source,/#status:empty\{margin:0/,'empty status line collapses to zero height');
assert.doesNotMatch(source,/#status:empty\{[^}]*display:\s*none/,'live region stays rendered while empty');
assert.doesNotMatch(source,/#results:empty\{[^}]*display:\s*none/,'results live region stays rendered while empty');
assert.match(source,/select\{padding-top:0;padding-bottom:0\}/,'select descenders are not clipped');
assert.match(source,/@media \(max-width: 640px\)/);assert.match(source,/@media \(max-width: 900px\)/);
assert.match(source,/@media \(prefers-reduced-motion: reduce\)\{[^}]*animation:none!important/,'reduced motion stops every animation');
assert.match(source,/\*:focus-visible\{outline:2px solid var\(--lime\);outline-offset:2px\}/,'focus is a visible 2px outline');
// clipped controls draw their ring on something unclipped: RUN on its wrapper, the index links keep the hex on ::before
assert.match(source,/#submit-wrap:has\(#submit:focus-visible\)\{outline:2px solid var\(--text\);outline-offset:3px\}/,'RUN focus ring on the unclipped wrapper');
assert.match(source,/#submit:focus-visible\{outline:none\}/);
for(const [selector,body] of rules)if(/\.bay-index a(:hover)?$/.test(selector))assert.doesNotMatch(body,/clip-path/,'an index link clips itself (and its focus ring): '+selector);
assert.match(source,/\.bay-index a::before\{[^}]*clip-path:var\(--hex-flat\)/,'the index hex is drawn on ::before');
assert.match(source,/@media \(forced-colors: active\)\{[^]*button,button:disabled,#submit,#submit:disabled,#load,#load:disabled\{border:2px solid ButtonText;clip-path:none\}/,'forced colors keep button boundaries');
assert.match(source,/\.boot-value\{flex:0 1 auto;min-width:0;[^}]*overflow-wrap:anywhere/,'boot values shrink and wrap (reflow)');
assert.match(source,/input::placeholder,textarea::placeholder\{color:var\(--text-3\)/);
assert.match(source,/audio\{[^}]*color-scheme:dark/,'the native audio control follows the dark palette');
for(const hook of [/\.brand img\{/,/\.model\.selected/,/\.composer\{/,/\.result\{/,/\.receipt\{/,/\.job-head,\.job\{/,/\.tile\{/,/\.connect-form\{/,/grid-column:1 \/ -1/,/\.model-list\{display:flex/])
  assert.match(source,hook,'layout hook '+hook);

// numbered bays: 01-07 in reading order (02 is built by app.js, 06 by retrieval.js), the composer steps and the mobile bay index
for(const no of ['01','03','04','05','07'])assert.match(html,new RegExp('<span class="bay-no">'+no+'</span>'),'bay '+no);
assert.match(app,/number\.textContent='02'/);assert.match(retrieval,/number\.textContent='06'/);
assert.match(html,/id="step-input"/);assert.match(source,/\.composer\{counter-reset:step/);
assert.match(html,/<nav id="bay-index" class="bay-index"/);
assert.match(html,/id="bay-jobs" class="jobs-section bay"/);
assert.equal((html.match(/<span data-i18n="jobs\.title">/g)||[]).length,1,'one Jobs plate title binding');
for(const no of ['01','02','03','04','05','06','07'])assert.match(html,new RegExp('>'+no+' <span class="sr-only" data-i18n="[a-z.]+">'),'index link '+no+' carries its number and section name');

console.log('PASS: approved identity, Cabinet tokens and roles, contrast of every token pair in use, banned tells absent, uppercase and the display face confined to plates, numerals and RUN, the inlined OFL face with its licence, live regions, breakpoints, reduced motion and focus.');
