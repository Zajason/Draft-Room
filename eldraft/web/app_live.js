"use strict";
/* ---------- helpers ---------- */
const $=(s,r)=>(r||document).querySelector(s);
const el=(t,a,k)=>{const n=document.createElement(t);if(a)for(const q in a){if(q==="class")n.className=a[q];else if(q==="html")n.innerHTML=a[q];else if(q==="text")n.textContent=a[q];else if(q.startsWith("on"))n.addEventListener(q.slice(2),a[q]);else if(a[q]!=null)n.setAttribute(q,a[q]);}(k||[]).forEach(c=>n.appendChild(typeof c==="string"?document.createTextNode(c):c));return n;};
const n1=v=>v==null?"–":(+v).toFixed(1);
const n2=v=>v==null?"–":(+v).toFixed(2);
const POSNAME={G:"Guard",F:"Forward",C:"Centre",H:"Coach"};
const RULES=DATA.rules, WK=DATA.weekly, SCOUT_URL="__SCOUT_URL__";
const ALL=DATA.players, BY={}; ALL.forEach(p=>BY[p.code]=p);
// Credit prices are the game's official quotations (imported from the stats export); a
// user can still override any value locally (kept in
// localStorage). Overrides are written straight into the player objects so every engine —
// budget, transfers, draft — uses the corrected price automatically.
const ORIG_PRICE={}; ALL.forEach(p=>ORIG_PRICE[p.code]=p.price);
const LSKEY_PRICES="elsr_prices_v1";
let PRICES={};
function loadPrices(){ try{const r=localStorage.getItem(LSKEY_PRICES); if(r)PRICES=JSON.parse(r)||{};}catch(e){} for(const c in PRICES){ if(BY[c]) BY[c].price=PRICES[c]; } }
function savePrices(){ try{localStorage.setItem(LSKEY_PRICES, JSON.stringify(PRICES));}catch(e){} }
function setPrice(code,v){ v=Math.round(v*2)/2; if(v===ORIG_PRICE[code]){delete PRICES[code];} else {PRICES[code]=v;} BY[code].price=(PRICES[code]!=null?PRICES[code]:ORIG_PRICE[code]); savePrices(); if(S.mode==="weekly")S.plan=null; refresh(); }
function resetPrices(){ for(const c in PRICES){ if(BY[c]) BY[c].price=ORIG_PRICE[c]; } PRICES={}; savePrices(); if(S.mode==="weekly")S.plan=null; refresh(); }
function priceCell(p){
  const td=el("td",{class:"num"});
  const edited=PRICES[p.code]!=null;
  const title=edited?("Overridden (official "+n1(ORIG_PRICE[p.code])+") — click to change"):(p.priceSource==="official"?"Official credit value — click to override":"Estimated — click to set the real value");
  const span=el("span",{class:"cred"+(edited?" edited":""),title:title,text:n1(p.price)});
  span.addEventListener("click",e=>{e.stopPropagation();const inp=el("input",{class:"credinp",type:"number",step:"0.5",min:"1",max:"30",value:p.price});
    td.innerHTML="";td.appendChild(inp);inp.focus();inp.select();
    const done=commit=>{if(commit){const v=parseFloat(inp.value);if(!isNaN(v)&&v>0){setPrice(p.code,v);return;}}renderBoard();};
    inp.addEventListener("blur",()=>done(true));
    inp.addEventListener("keydown",ev=>{if(ev.key==="Enter"){done(true);}else if(ev.key==="Escape"){done(false);}});});
  td.appendChild(span);return td;
}
const CAPS_WEEKLY=[RULES.slots.G,RULES.slots.F,RULES.slots.C,1];
const CAPS_DRAFT=[RULES.slots.G,RULES.slots.F,RULES.slots.C,0];
const CLUBNAME=DATA.clubNames||{}, clubName=c=>CLUBNAME[c]||c, CREST=DATA.crests||{};
const POSVAR={G:"--pos-g",F:"--pos-f",C:"--pos-c",H:"--pos-h"};
const SQUAD_N=RULES.slots.G+RULES.slots.F+RULES.slots.C+1;
const HORIZON_GAMMA=0.6;
const THRIFT={aggressive:0.4,balanced:1.5,thrifty:3.5};

function initials(name){return (name||"").split(/\s+/).filter(Boolean).map(w=>w[0]).slice(0,2).join("").toUpperCase();}
function avatar(p,size){size=size||26;const w=el("span",{class:"avatar",style:`width:${size}px;height:${size}px`});
  if(p.photo){w.appendChild(el("img",{src:p.photo,alt:"",loading:"lazy"}));}
  else{w.classList.add("ini");w.style.background=`var(${POSVAR[p.pos]||"--pos-h"})`;w.style.font=`600 ${Math.round(size*0.36)}px/1 "IBM Plex Sans"`;w.textContent=initials(p.name);}
  return w;}
function badge(code,size){size=size||16;const u=CREST[code];return u?el("img",{class:"crest",src:u,alt:"",style:`width:${size}px;height:${size}px`}):el("span",{style:`width:${size}px;display:inline-block`});}

/* ---------- weekly matchup + horizon ---------- */
function weeklyFP(p,round){
  const team=p.clubCode, games=(WK&&WK.schedule[String(round)])||[], det=[]; let total=0;
  for(const g of games){ if(g.home!==team&&g.away!==team)continue;
    const opp=g.home===team?g.away:g.home, home=g.home===team;
    const d=(WK.defense[opp]!=null)?WK.defense[opp]:WK.leagueAvg;
    let m=Math.max(WK.clampLo,Math.min(WK.clampHi,d/WK.leagueAvg))*(home?WK.homeAdv:WK.awayAdv);
    total+=(p.fp||0)*m; det.push({opp,home,mult:m}); }
  return {fp:total,games:det};
}
function horizon(p,round,H){ let hz=0; for(let r=0;r<H;r++)hz+=Math.pow(HORIZON_GAMMA,r)*weeklyFP(p,round+r).fp; return hz; }
function matchupText(w){ if(!w.games.length)return "Bye — no game this round";
  return w.games.map(g=>`${g.home?"vs":"@"} ${g.opp} ${g.mult>=1.02?"▲":(g.mult<=0.98?"▼":"·")}`).join("  "); }

/* ---------- teams: persistent (localStorage) ---------- */
const LSKEY="elsr_teams_v2";
let TEAMS=[], CURRENT=null;
function loadTeams(){ try{const raw=localStorage.getItem(LSKEY); if(raw){const d=JSON.parse(raw); if(d&&Array.isArray(d.teams)){TEAMS=d.teams;CURRENT=d.current;}}}catch(e){} }
function saveTeams(){ try{localStorage.setItem(LSKEY,JSON.stringify({teams:TEAMS,current:CURRENT}));}catch(e){} }
function curTeam(){ return TEAMS.find(t=>t.id===CURRENT)||null; }
function newTeam(name){ const id="t"+Date.now().toString(36); TEAMS.push({id,name:name||("Team "+(TEAMS.length+1)),cr:RULES.budget,squad:[]}); CURRENT=id; saveTeams(); }
function delTeam(id){ TEAMS=TEAMS.filter(t=>t.id!==id); if(CURRENT===id)CURRENT=TEAMS[0]?TEAMS[0].id:null; saveTeams(); }
function teamNeeds(t){ const n={G:RULES.slots.G,F:RULES.slots.F,C:RULES.slots.C,H:1}; (t.squad||[]).forEach(c=>{const p=BY[c];if(p)n[p.pos]=Math.max(0,n[p.pos]-1);}); return n; }
function teamSpent(t){ return (t.squad||[]).reduce((a,c)=>a+((BY[c]||{}).price||0),0); }
function teamFull(t){ const n=teamNeeds(t); return n.G+n.F+n.C+n.H===0; }
function canAdd(t,code){ const p=BY[code]; if(!p||t.squad.includes(code))return false; const n=teamNeeds(t); if(n[p.pos]<=0)return false; if(teamSpent(t)+p.price>t.cr+1e-6)return false; return true; }
function addToTeam(code){ const t=curTeam(); if(t&&canAdd(t,code)){t.squad.push(code);saveTeams();S.plan=null;refresh();} }
function removeFromTeam(code){ const t=curTeam(); if(t){t.squad=t.squad.filter(c=>c!==code);saveTeams();S.plan=null;refresh();} }

/* ---------- state ---------- */
const S={ mode:"weekly", q:"", pos:"ALL",
  wkRound:(WK&&WK.nextRound)||1, wkH:3, wkMaxT:4, thrift:"balanced", excluded:new Set(),
  plan:null, planT:null, computing:false,
  mine:[], taken:new Set(), history:[], engine:"greedy", rec:[] };

/* ---------- this-week valuation ---------- */
function weekPts(codes,round){ const out=[]; let coach=0;
  codes.forEach(c=>{const p=BY[c];const w=weeklyFP(p,round).fp; if(p.pos==="H")coach+=w; else out.push(w);});
  out.sort((a,b)=>b-a); const k=RULES.full;
  let v=out.slice(0,k).reduce((a,b)=>a+b,0)+RULES.bench*out.slice(k).reduce((a,b)=>a+b,0);
  if(out.length)v+=(RULES.captain-1)*out[0]; return v+coach; }
function weekCaptain(codes,round){ const l=computeLineup(codes,round); return l.captain; }

/* ---------- reactive lineup advisor: who to START (formation-legal 6 + captain),
   who's on the bench, and the Tue–Fri timing/insurance that actually banks points. */
const DOW=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
function roundDays(round){ const games=(WK&&WK.schedule&&WK.schedule[String(round)])||[];
  const dates=[...new Set(games.map(g=>g.date).filter(Boolean))].sort(); const di={};
  dates.forEach((d,i)=>di[d]=i); return {dates,di,games}; }
function playerDay(code,round){ const p=BY[code],team=p.clubCode,{games,di}=roundDays(round);
  for(const g of games){ if(g.home===team||g.away===team) return {date:g.date,idx:di[g.date]}; }
  return null; }
function computeLineup(codes,round){
  const out=codes.map(c=>BY[c]).filter(p=>p&&p.pos!=="H");
  const coach=codes.map(c=>BY[c]).find(p=>p&&p.pos==="H")||null;
  const fp=out.map(p=>weeklyFP(p,round).fp), pos=out.map(p=>p.pos);
  const avail=out.map(p=>!S.excluded.has(p.code));
  const fullIdx=(typeof chooseFullSix==="function")?chooseFullSix(fp,pos,avail)
    : new Set(fp.map((_,i)=>i).sort((a,b)=>fp[b]-fp[a]).slice(0,RULES.full));
  const starters=out.filter((_,i)=>fullIdx.has(i));
  const bench=out.filter((_,i)=>!fullIdx.has(i));
  let cap=null,capfp=-1; starters.forEach(p=>{const w=weeklyFP(p,round).fp;
    if(!S.excluded.has(p.code)&&w>capfp){capfp=w;cap=p.code;}});
  const startSet=new Set(starters.map(p=>p.code));
  return {starters,bench,captain:cap,out,coach,startSet,
    day:c=>playerDay(c,round)};
}
function last(nm){ return String(nm||"").split(" ").slice(-1)[0]; }
function bestReplacement(LU,outP,round){
  const cand=LU.bench.filter(b=>!S.excluded.has(b.code));
  const same=cand.filter(b=>b.pos===outP.pos);
  const pick=(same.length?same:cand).sort((a,b)=>weeklyFP(b,round).fp-weeklyFP(a,round).fp)[0];
  return pick||null;
}
function firstOffBench(LU,round){
  const avail=LU.bench.filter(b=>!S.excluded.has(b.code) && weeklyFP(b,round).fp>0)
    .sort((a,b)=>weeklyFP(b,round).fp-weeklyFP(a,round).fp);
  return avail.slice(0,2);
}
function lineupAdvice(LU,round){
  const box=el("div",{class:"advice"});
  box.appendChild(el("div",{class:"advice-h",text:"Lineup & timing"}));
  const {dates}=roundDays(round);
  const dl=d=>{try{return DOW[new Date(d+"T00:00:00").getDay()];}catch(e){return "";}};
  const sl=el("div",{class:"tiny",style:"margin-bottom:4px"});
  sl.appendChild(el("b",{text:"Start: "}));
  sl.appendChild(document.createTextNode(LU.starters
    .slice().sort((a,b)=>weeklyFP(b,round).fp-weeklyFP(a,round).fp)
    .map(p=>last(p.name)+(p.code===LU.captain?" (C)":"")).join(", ")));
  box.appendChild(sl);
  if(dates.length>1)
    box.appendChild(el("div",{class:"tiny muted",style:"margin-bottom:4px",
      text:"Games span "+dates.map(dl).join(" / ")+" — lock each player before his tip; you can still move anyone who hasn't played."}));
  else
    box.appendChild(el("div",{class:"tiny muted",style:"margin-bottom:4px",
      text:"All games on one day — set your 6 before tip; no in-week moves."}));
  LU.starters.filter(p=>S.excluded.has(p.code)).forEach(o=>{
    const rep=bestReplacement(LU,o,round);
    box.appendChild(el("div",{class:"tiny",style:"color:var(--crit)"},[
      el("b",{text:"⚠ "+last(o.name)+" out"}),
      document.createTextNode(rep?(" → start "+last(rep.name)+" ("+rep.pos+")"):" → no legal bench cover")]));
  });
  const firsts=firstOffBench(LU,round);
  if(firsts.length) box.appendChild(el("div",{class:"tiny muted",
    text:"First off the bench if a starter sits: "+firsts.map(x=>{const d=LU.day(x.code);return last(x.name)+" ("+x.pos+(d?", "+dl(d.date):"")+")";}).join(", ")}));
  return box;
}

/* ---------- the optimiser ---------- */
function horizonPool(round,H){
  const t=curTeam(); const inPool=new Set(), pool=[];
  const add=code=>{ if(inPool.has(code)||S.excluded.has(code))return; const p=BY[code];
    pool.push({code,pos:p.pos,price:p.price,fp:horizon(p,round,H)}); inPool.add(code); };
  (t?t.squad:[]).forEach(add);
  // Per position, keep the best by look-ahead value AND the cheapest by price — the
  // budget needs affordable bench fillers, not just stars, or no legal 11 fits under CR.
  ["G","F","C","H"].forEach(pos=>{ const cand=ALL.filter(p=>p.pos===pos&&!S.excluded.has(p.code));
    const nTop=pos==="H"?8:22, nCheap=pos==="H"?4:10;
    cand.map(p=>({p,h:horizon(p,round,H)})).sort((a,b)=>b.h-a.h).slice(0,nTop).forEach(x=>add(x.p.code));
    cand.slice().sort((a,b)=>a.price-b.price).slice(0,nCheap).forEach(p=>add(p.code)); });
  return pool;
}
function autoFill(){ const t=curTeam(); if(!t)return;
  const pool=horizonPool(S.wkRound,S.wkH);
  const res=greedyRecommend(pool,t.cr,{full:RULES.full,bench:RULES.bench,captain:RULES.captain},CAPS_WEEKLY);
  if(res.squad&&res.squad.length===SQUAD_N){t.squad=res.squad.slice();saveTeams();S.plan=null;refresh();}
}
function optimiseWeek(){
  const t=curTeam(); if(!t||!teamFull(t))return;
  const pool=horizonPool(S.wkRound,S.wkH);
  const curSet=new Set(t.squad.filter(c=>!S.excluded.has(c)));
  const dp=transferDP(pool,t.cr,curSet,S.wkMaxT,CAPS_WEEKLY,{full:RULES.full,bench:RULES.bench,captain:RULES.captain});
  // recommend transfer count by horizon value minus per-transfer cost
  const lam=THRIFT[S.thrift]; let best=-1e18,bt=0;
  for(let lvl=0;lvl<=S.wkMaxT;lvl++){ if(dp.frontier[lvl]==null)continue; const score=dp.frontier[lvl]-lam*lvl; if(score>best){best=score;bt=lvl;} }
  S.plan=dp; S.planT=bt; refresh();
}
function planDiff(fromCodes,toCodes){
  const fromSet=new Set(fromCodes), toSet=new Set(toCodes);
  const outs=fromCodes.filter(c=>!toSet.has(c)), ins=toCodes.filter(c=>!fromSet.has(c));
  // pair OUT->IN by position
  const pairs=[]; const insByPos={G:[],F:[],C:[],H:[]}; ins.forEach(c=>insByPos[BY[c].pos].push(c));
  outs.forEach(c=>{const pos=BY[c].pos; const inC=insByPos[pos].shift(); pairs.push({out:c,in:inC});});
  return pairs;
}
function applyPlan(){ const t=curTeam(); if(!t||!S.plan||S.planT==null)return; const sq=S.plan.squads[S.planT];
  if(sq){t.squad=sq.slice();saveTeams();S.plan=null;S.planT=null;refresh();} }

/* ================================================================= TEAMS PANEL */
function renderTeamsPanel(){
  const host=$("#panel"); host.innerHTML="";
  const t=curTeam();

  // team bar
  const bar=el("div",{class:"card pad"});
  const tb=el("div",{class:"teambar"});
  const sel=el("select",{"aria-label":"Team",onchange:e=>{CURRENT=e.target.value;saveTeams();S.plan=null;refresh();}});
  if(!TEAMS.length)sel.appendChild(el("option",{text:"No teams yet"}));
  TEAMS.forEach(x=>sel.appendChild(el("option",{value:x.id,selected:x.id===CURRENT?"":null,text:x.name})));
  tb.appendChild(sel);
  tb.appendChild(el("button",{class:"mini",type:"button",title:"New team",text:"+ New",onclick:()=>{const nm=prompt("Team name:","Team "+(TEAMS.length+1));if(nm!==null){newTeam(nm.trim()||undefined);refresh();}}}));
  if(t){
    tb.appendChild(el("button",{class:"mini",type:"button",title:"Rename",text:"Rename",onclick:()=>{const nm=prompt("Rename team:",t.name);if(nm){t.name=nm.trim()||t.name;saveTeams();refresh();}}}));
    tb.appendChild(el("button",{class:"mini take",type:"button",title:"Delete",text:"Delete",onclick:()=>{if(confirm("Delete “"+t.name+"”?")){delTeam(t.id);refresh();}}}));
  }
  bar.appendChild(tb);
  if(t){
    const cr=el("div",{class:"setrow",style:"margin-top:8px;margin-bottom:0"});
    cr.appendChild(el("label",{},[el("span",{text:"My credits (CR)"}),el("input",{type:"number",min:"40",max:"140",step:"0.5",value:t.cr,
      onchange:e=>{t.cr=Math.max(40,Math.min(140,+e.target.value||100));saveTeams();S.plan=null;refresh();}})]));
    bar.appendChild(cr);
  }
  host.appendChild(bar);

  if(!t){ host.appendChild(el("div",{class:"card pad"},[el("div",{class:"tiny muted",text:"Create a team to begin. Build your 11 from the board, or use Auto-fill, then optimise each week."})])); renderBoard(); return; }

  // squad card
  const sq=el("div",{class:"card pad"});
  const spent=teamSpent(t), needs=teamNeeds(t), full=teamFull(t);
  sq.appendChild(el("div",{class:"sec-h"},[el("h2",{text:"Squad"}),el("span",{class:"muted",text:`${t.squad.length}/${SQUAD_N} · Round ${S.wkRound}`})]));
  const bud=el("div",{class:"budget"});
  bud.appendChild(el("b",{text:spent.toFixed(1)+" cr"}));
  const bar2=el("div",{class:"bar"});bar2.appendChild(el("i",{class:spent>t.cr?"over":"",style:`width:${Math.min(100,spent/t.cr*100)}%`}));
  bud.appendChild(bar2);bud.appendChild(el("span",{class:"tiny muted",text:(t.cr-spent).toFixed(1)+" left"}));
  sq.appendChild(bud);
  const slots=el("div",{class:"slots"});
  [["G",RULES.slots.G],["F",RULES.slots.F],["C",RULES.slots.C],["H",1]].forEach(([pos,cap])=>{const have=cap-needs[pos];
    slots.appendChild(el("div",{class:"slot"+(have>=cap?" full":"")},[el("span",{class:"pos "+pos,text:pos}),el("b",{text:`${have}/${cap}`})]));});
  sq.appendChild(slots);
  const LU=full?computeLineup(t.squad,S.wkRound):null;
  const cap=LU?LU.captain:null;
  const list=el("div",{class:"mylist"});
  t.squad.map(c=>BY[c]).sort((a,b)=>(a.pos==="H")-(b.pos==="H")||weeklyFP(b,S.wkRound).fp-weeklyFP(a,S.wkRound).fp).forEach(p=>{
    const w=weeklyFP(p,S.wkRound), inj=S.excluded.has(p.code);
    const isStart=LU&&p.pos!=="H"&&LU.startSet.has(p.code);
    const r=el("div",{class:"myrow",style:"grid-template-columns:26px auto 1fr auto auto"});
    r.appendChild(avatar(p,26));
    // starter / bench chip (coach shown as its own slot)
    r.appendChild(p.pos==="H"?el("span",{class:"lchip coach",text:"HC"})
      :el("span",{class:"lchip "+(isStart?"start":"bench"),title:isStart?"Starter (full points)":"Bench (half points)",text:isStart?"ST":"BN"}));
    r.appendChild(el("div",{style:"min-width:0"},[el("div",{style:"font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"},[
      document.createTextNode(p.name), p.code===cap?el("span",{class:"cap",style:"margin-left:6px",text:"C ×"+RULES.captain}):el("span"),
      inj?el("span",{class:"cap",style:"margin-left:6px;color:var(--crit)",text:"OUT"}):el("span")]),
      el("div",{class:"tiny muted",text:matchupText(w)})]));
    r.appendChild(el("span",{class:"mono tiny",text:n1(p.price)+"cr"}));
    const rr=el("div",{style:"display:flex;gap:8px;align-items:center"});
    rr.appendChild(el("span",{class:"mono",style:"font-weight:600"+(isStart?"":";opacity:.55"),text:n1(w.fp)}));
    rr.appendChild(el("button",{class:"x",type:"button",title:"Remove",text:"×",onclick:()=>removeFromTeam(p.code)}));
    r.appendChild(rr); list.appendChild(r);
  });
  if(!t.squad.length)list.appendChild(el("div",{class:"tiny muted",text:"Add players from the board (+ Add), or use Auto-fill below."}));
  sq.appendChild(list);
  if(LU) sq.appendChild(lineupAdvice(LU,S.wkRound));
  // squad card is appended LAST (after the plan card) — see end of function.
  if(!full){
    sq.appendChild(el("button",{class:"btn",style:"margin-top:10px",type:"button",text:"Auto-fill an optimal squad",onclick:autoFill}));
    sq.appendChild(el("div",{class:"tiny muted",style:"margin-top:6px",text:`Still to add: ${["G","F","C","H"].filter(p=>needs[p]>0).map(p=>needs[p]+p).join(", ")}. Build it yourself from the board, or auto-fill and tweak.`}));
  } else {
    sq.appendChild(el("div",{class:"tiny muted",style:"margin-top:6px",text:`Projected this round: ${n1(weekPts(t.squad,S.wkRound))} pts · captain ${cap?BY[cap].name:"—"}.`}));
  }

  // plan card
  const plan=el("div",{class:"card pad"});
  plan.appendChild(el("div",{class:"sec-h"},[el("h2",{text:"Optimise the week"}),el("span",{class:"muted",text:"transfers ≤ "+S.wkMaxT})]));
  const ctl=el("div",{class:"setrow"});
  const rsel=el("select",{onchange:e=>{S.wkRound=+e.target.value;S.plan=null;refresh();}});
  (WK?WK.rounds:[1]).forEach(r=>rsel.appendChild(el("option",{value:r,selected:r===S.wkRound?"":null,text:"Round "+r})));
  ctl.appendChild(el("label",{},[el("span",{text:"Round"}),rsel]));
  const hsel=el("select",{onchange:e=>{S.wkH=+e.target.value;S.plan=null;refresh();}});
  [1,2,3,4,5].forEach(h=>hsel.appendChild(el("option",{value:h,selected:h===S.wkH?"":null,text:h+(h>1?" rounds":" round")})));
  ctl.appendChild(el("label",{},[el("span",{text:"Look ahead"}),hsel]));
  const tsel=el("select",{onchange:e=>{S.wkMaxT=+e.target.value;S.plan=null;refresh();}});
  [0,1,2,3,4].forEach(x=>tsel.appendChild(el("option",{value:x,selected:x===S.wkMaxT?"":null,text:x+" max"})));
  ctl.appendChild(el("label",{},[el("span",{text:"Transfers"}),tsel]));
  const th=el("select",{onchange:e=>{S.thrift=e.target.value;if(S.plan)optimiseWeek();else refresh();}});
  [["aggressive","Aggressive"],["balanced","Balanced"],["thrifty","Thrifty"]].forEach(([v,l])=>th.appendChild(el("option",{value:v,selected:v===S.thrift?"":null,text:l})));
  ctl.appendChild(el("label",{},[el("span",{text:"Thrift"}),th]));
  plan.appendChild(ctl);

  if(!full){ plan.appendChild(el("div",{class:"tiny muted",text:"Complete your 11 first — then the optimiser shows which few transfers to make."})); host.appendChild(plan); host.appendChild(sq); renderBoard(); return; }
  plan.appendChild(el("button",{class:"btn",type:"button",text:"Optimise this week",onclick:optimiseWeek}));

  if(S.plan){
    const dp=S.plan, base=dp.frontier[0];
    // frontier chips
    plan.appendChild(el("div",{class:"subh",text:"Value of each transfer (look-ahead pts)"}));
    const fr=el("div",{class:"frontier"});
    for(let lvl=0;lvl<=S.wkMaxT;lvl++){ if(dp.frontier[lvl]==null)continue;
      const gain=dp.frontier[lvl]-base;
      fr.appendChild(el("button",{class:"fchip"+(lvl===S.planT?" on":""),type:"button",onclick:()=>{S.planT=lvl;refresh();}},[
        el("b",{text:lvl+" tf"}), el("span",{text:lvl===0?n1(dp.frontier[0]):"+"+n1(gain)})]));
    }
    plan.appendChild(fr);
    // recommended moves at planT
    const target=dp.squads[S.planT];
    const pairs=planDiff(t.squad,target);
    plan.appendChild(el("div",{class:"subh",text:S.planT===0?"No transfer beats the thrift threshold — hold":`Make ${pairs.length} transfer${pairs.length!==1?"s":""}`}));
    if(pairs.length){
      const mv=el("div",{class:"moves"});
      pairs.forEach(pr=>{ const o=BY[pr.out], i=pr.in?BY[pr.in]:null;
        const row=el("div",{class:"move"});
        row.appendChild(el("div",{class:"mside out"},[avatar(o,24),el("div",{style:"min-width:0"},[el("b",{text:o.name}),el("span",{class:"tiny muted",text:`${o.pos} · ${n1(o.price)}cr · ${n1(weeklyFP(o,S.wkRound).fp)}`})])]));
        row.appendChild(el("span",{class:"arrow",text:"→"}));
        row.appendChild(i?el("div",{class:"mside in"},[avatar(i,24),el("div",{style:"min-width:0"},[el("b",{text:i.name}),el("span",{class:"tiny muted",text:`${i.pos} · ${n1(i.price)}cr · ${n1(weeklyFP(i,S.wkRound).fp)}`})])]):el("span"));
        mv.appendChild(row); });
      plan.appendChild(mv);
    }
    const nowPts=weekPts(t.squad,S.wkRound), newPts=weekPts(target,S.wkRound);
    const newCap=weekCaptain(target,S.wkRound);
    plan.appendChild(el("div",{class:"note",html:
      `This round: <b>${n1(nowPts)}</b> → <b>${n1(newPts)}</b> pts (<b>${newPts>=nowPts?"+":""}${n1(newPts-nowPts)}</b>). Captain <b>${newCap?BY[newCap].name:"—"}</b> ×${RULES.captain}.`+
      (S.planT>0?` The look-ahead values each pick over the next ${S.wkH} round${S.wkH>1?"s":""}, so a swap is only suggested when it pays off beyond this week.`:"")}));
    if(S.planT>0)plan.appendChild(el("button",{class:"btn",style:"margin-top:8px",type:"button",text:"Apply these transfers",onclick:applyPlan}));
  }
  host.appendChild(plan);
  host.appendChild(sq);
  renderBoard();
}

/* ================================================================= DRAFT MODE (unchanged) */
function needs(){ const n={G:RULES.slots.G,F:RULES.slots.F,C:RULES.slots.C};
  S.mine.forEach(c=>{if(BY[c].pos!=="H")n[BY[c].pos]=Math.max(0,n[BY[c].pos]-1);}); return n; }
function draftSpent(){ return S.mine.reduce((a,c)=>a+(BY[c].price||0),0); }
function isAvail(c){ return !S.taken.has(c)&&!S.mine.includes(c); }
function draftMine(code){ if(!isAvail(code))return; S.history.push({t:"mine",code}); S.mine.push(code); refresh(); }
function markTaken(code){ if(!isAvail(code))return; S.history.push({t:"taken",code}); S.taken.add(code); refresh(); }
function undo(){ const h=S.history.pop(); if(!h)return; if(h.t==="mine")S.mine=S.mine.filter(c=>c!==h.code); else S.taken.delete(h.code); refresh(); }
function draftPool(){ const pool=[]; ALL.forEach(p=>{ if(p.pos==="H")return;
  if(S.mine.includes(p.code))pool.push(Object.assign({},p,{mandatory:true})); else if(!S.taken.has(p.code))pool.push(p); }); return pool; }
function draftNeeded(){ const nd=needs(); return nd.G+nd.F+nd.C>0; }
function computeDraftRec(){
  if(!draftNeeded()){ S.rec=[]; renderRec(); renderBoard(); return; }
  if(S.engine==="greedy"){
    const pool=draftPool(); const res=greedyRecommend(pool,RULES.budget,{full:RULES.full,bench:RULES.bench,captain:RULES.captain},CAPS_DRAFT);
    const nd=needs(), rem=RULES.budget-draftSpent(), rows=[];
    res.rows.forEach(r=>{const p=pool[r.i]; if(p.mandatory)return; if(nd[p.pos]<=0)return; if((p.price||0)>rem+1e-6)return; if(r.vorp==null)return; rows.push({code:p.code,metric:r.vorp});});
    rows.sort((a,b)=>b.metric-a.metric); S.rec=rows; renderRec(); renderBoard();
  } else { S.computing=true; renderRec();
    setTimeout(()=>{ try{ const nTeams=Math.max(2,Math.min(16,+$("#nteams").value||8)); const mySlot=Math.max(1,Math.min(nTeams,+$("#myslot").value||1))-1; const sims=Math.max(30,Math.min(600,+$("#sims").value||140));
      const players=ALL.filter(p=>p.pos!=="H"); S.rec=mctsRecommend(players,BY,S.mine,S.taken,{budget:RULES.budget,slots:RULES.slots,full:RULES.full,bench:RULES.bench,captain:RULES.captain},nTeams,mySlot,sims,CAPS_DRAFT);
    }catch(e){console.error(e);S.rec=[];} S.computing=false; renderRec(); renderBoard(); },30); }
}
function renderDraftPanel(){
  const host=$("#panel"); host.innerHTML=""; const nd=needs(), sp=draftSpent();
  const sq=el("div",{class:"card pad"});
  sq.appendChild(el("div",{class:"sec-h"},[el("h2",{text:"My squad"}),el("span",{class:"muted",text:`${S.mine.length} of ${RULES.slots.G+RULES.slots.F+RULES.slots.C} picked`})]));
  const bud=el("div",{class:"budget"}); bud.appendChild(el("b",{text:sp.toFixed(1)+" cr"}));
  const bar=el("div",{class:"bar"});bar.appendChild(el("i",{class:sp>RULES.budget?"over":"",style:`width:${Math.min(100,sp/RULES.budget*100)}%`}));
  bud.appendChild(bar);bud.appendChild(el("span",{class:"tiny muted",text:(RULES.budget-sp).toFixed(1)+" left"})); sq.appendChild(bud);
  const slots=el("div",{class:"slots"});
  [["G",RULES.slots.G],["F",RULES.slots.F],["C",RULES.slots.C]].forEach(([pos,cap])=>{const have=cap-nd[pos];
    slots.appendChild(el("div",{class:"slot"+(have>=cap?" full":"")},[el("span",{class:"pos "+pos,text:pos}),el("b",{text:`${have}/${cap}`})]));});
  sq.appendChild(slots);
  const list=el("div",{class:"mylist"}); const outs=S.mine.map(c=>BY[c]).sort((a,b)=>b.fp-a.fp); const cap=outs[0]?outs[0].code:null;
  outs.forEach(p=>{const r=el("div",{class:"myrow",style:"grid-template-columns:26px 1fr auto auto"});
    r.appendChild(avatar(p,26)); r.appendChild(el("span",{text:p.name}));
    r.appendChild(el("span",{class:p.code===cap?"cap":"tiny muted",text:p.code===cap?"C ×"+RULES.captain:""}));
    const rr=el("div",{style:"display:flex;gap:8px;align-items:center"});
    rr.appendChild(el("span",{class:"mono tiny",text:n1(p.price)+"cr"})); rr.appendChild(el("span",{class:"mono",style:"font-weight:600",text:n1(p.fp)}));
    rr.appendChild(el("button",{class:"x",type:"button",title:"Remove",text:"×",onclick:()=>{S.mine=S.mine.filter(c=>c!==p.code);refresh();}})); r.appendChild(rr); list.appendChild(r);});
  if(!S.mine.length)list.appendChild(el("div",{class:"tiny muted",text:"Draft from the board, or take a recommendation below."}));
  sq.appendChild(list); host.appendChild(sq);
  const rec=el("div",{class:"card pad"}); rec.appendChild(el("div",{class:"sec-h"},[el("h2",{text:"Who to pick next"})]));
  const et=el("div",{class:"engtabs",id:"engtabs"});
  et.appendChild(el("button",{class:"engtab","aria-pressed":String(S.engine==="greedy"),"data-eng":"greedy",type:"button",html:'Value engine<small>exact · instant</small>'}));
  et.appendChild(el("button",{class:"engtab","aria-pressed":String(S.engine==="mcts"),"data-eng":"mcts",type:"button",html:'Lookahead (MCTS)<small>models rivals</small>'}));
  et.addEventListener("click",e=>{const x=e.target.closest(".engtab");if(!x)return;S.engine=x.dataset.eng;refresh();}); rec.appendChild(et);
  const setrow=el("div",{class:"setrow"+(S.engine==="mcts"?"":" hide"),id:"mcts-set"});
  setrow.appendChild(el("label",{},[el("span",{text:"Teams"}),el("input",{type:"number",id:"nteams",min:"2",max:"16",value:"8",onchange:()=>{if(S.engine==="mcts")computeDraftRec();}})]));
  setrow.appendChild(el("label",{},[el("span",{text:"My slot"}),el("input",{type:"number",id:"myslot",min:"1",max:"16",value:"1",onchange:()=>{if(S.engine==="mcts")computeDraftRec();}})]));
  setrow.appendChild(el("label",{},[el("span",{text:"Search"}),el("input",{type:"number",id:"sims",min:"30",max:"600",step:"10",value:"140",onchange:()=>{if(S.engine==="mcts")computeDraftRec();}})]));
  rec.appendChild(setrow); rec.appendChild(el("div",{id:"rec"}));
  const rb=el("div",{class:"rowbtns"}); rb.appendChild(el("button",{class:"btn ghost",type:"button",text:"Undo last",onclick:undo}));
  rb.appendChild(el("button",{class:"btn ghost",type:"button",text:"Reset",onclick:()=>{S.mine=[];S.taken=new Set();S.history=[];refresh();}})); rec.appendChild(rb);
  rec.appendChild(el("div",{class:"note",id:"engnote"})); host.appendChild(rec); renderRec(); engNote();
}
function renderRec(){ const host=$("#rec"); if(!host)return; host.innerHTML="";
  if(!draftNeeded()){ host.appendChild(el("div",{class:"note",text:"Your squad is complete — 10 players."})); return; }
  if(S.computing){ host.appendChild(el("div",{class:"recrow"},[el("span",{class:"rk",html:'<span class="spin"></span>'}),el("div",{class:"nm"},[el("b",{text:"Searching drafts…"}),el("span",{text:"running the lookahead"})]),el("span",{})])); return; }
  if(!S.rec.length){ host.appendChild(el("div",{class:"tiny muted",text:"No legal pick fits your budget and slots."})); return; }
  const list=el("div",{class:"reclist"}), label=S.engine==="mcts"?"visits":"pts added";
  S.rec.slice(0,7).forEach((r,i)=>{const p=BY[r.code];
    const row=el("div",{class:"recrow"+(i===0?" top":""),style:"grid-template-columns:22px 30px 1fr auto",onclick:()=>draftMine(r.code)});
    row.appendChild(el("span",{class:"rk",text:String(i+1)})); row.appendChild(avatar(p,30));
    row.appendChild(el("div",{class:"nm"},[el("b",{text:p.name}),el("span",{html:`<span class="pos ${p.pos}" style="width:14px;height:14px;font-size:8px;vertical-align:-2px">${p.pos}</span> ${p.club||""} · ${n1(p.price)}cr · proj ${n1(p.fp)}`})]));
    const sc=el("div",{class:"sc"}); sc.appendChild(el("b",{text:S.engine==="mcts"?String(r.metric):(r.metric>=0?"+":"")+n2(r.metric)})); sc.appendChild(el("span",{text:label}));
    row.appendChild(sc); list.appendChild(row);});
  host.appendChild(list);
}
function engNote(){ const n=$("#engnote"); if(!n)return; n.innerHTML=S.engine==="greedy"
  ? `<b>Value engine.</b> The exact squad optimiser — each number is the fantasy points per round this player adds to your best finished squad, over his replacement. Instant.`
  : `<b>Lookahead (MCTS).</b> Plays the rest of the draft forward against a model of the other teams. <b>Visits</b> = how firmly the search settled on it. About level with the value engine in testing — use it when you expect runs on a position.`; }

/* ================================================================= BOARD */
let sortK="fp", sortDir=-1;
function boardCols(){ if(S.mode==="weekly") return [["name","Player",0],["pos","",0],["club","Club",0],["price","Cr",1],["_wkfp","Wk proj",1],["_match","Matchup",0],["fp","Season",1],["ovr","OVR",1],["act","",0]];
  return [["name","Player",0],["pos","",0],["club","Club",0],["price","Cr",1],["fp","Proj",1],["mptg","Min",1],["ovr","OVR",1],["val","VAL",1],["act","",0]]; }
function weeklyProjOf(code){ return weeklyFP(BY[code],S.wkRound).fp; }
function filteredRows(){ let r=ALL.slice();
  if(S.mode==="draft")r=r.filter(p=>p.pos!=="H");
  if(S.pos!=="ALL")r=r.filter(p=>p.pos===S.pos);
  if(S.q){const q=S.q.toLowerCase();r=r.filter(p=>p.name.toLowerCase().includes(q)||(p.club||"").toLowerCase().includes(q));}
  const k=sortK,d=sortDir, val=p=>k==="_wkfp"?weeklyProjOf(p.code):p[k];
  r.sort((a,b)=>{let x=val(a),y=val(b);if(x==null)x=-Infinity;if(y==null)y=-Infinity;if(typeof x==="string")return d*x.localeCompare(y);return d*(x-y);}); return r; }
function ratChip(v){ if(v==null)return el("span",{class:"muted",text:"–"}); const t=Math.max(0,Math.min(1,(v-40)/59)),c=t>.78?"var(--good)":t>.45?"var(--accent-ink)":"var(--ink-2)"; return el("span",{class:"rat",style:`color:${c}`,text:String(v)}); }
function renderBoard(){
  const cols=boardCols(); const t=curTeam();
  const inTeam=new Set(t?t.squad:[]);
  const suggestIn=new Set(); if(S.mode==="weekly"&&S.plan&&S.planT!=null){ const tg=S.plan.squads[S.planT]; if(tg)tg.forEach(c=>{if(!inTeam.has(c))suggestIn.add(c);}); }
  const draftHL=S.mode==="draft"?new Set(S.rec.slice(0,3).map(r=>r.code)):new Set();
  const table=el("table"); const thead=el("thead"),tr=el("tr");
  cols.forEach(([k,label,num])=>{const th=el("th",{class:num?"num":"",text:label,scope:"col",
    onclick:()=>{if(k==="act"||k==="pos"||k==="_match")return;if(sortK===k)sortDir*=-1;else{sortK=k;sortDir=(k==="name"||k==="club")?1:-1;}renderBoard();}});
    if(sortK===k)th.setAttribute("aria-sort",sortDir===1?"ascending":"descending");tr.appendChild(th);}); thead.appendChild(tr);table.appendChild(thead);
  const tb=el("tbody");
  filteredRows().slice(0,500).forEach(p=>{
    const gone=S.mode==="draft"&&!isAvail(p.code), owned=S.mode==="weekly"&&inTeam.has(p.code), excl=S.excluded.has(p.code);
    const cls=[(S.mode==="draft"&&S.mine.includes(p.code))||owned?"mine":"",(draftHL.has(p.code)&&!gone)||suggestIn.has(p.code)?"rec":"",gone||excl?"gone":""].filter(Boolean).join(" ");
    const row=el("tr",{class:cls});
    row.appendChild(el("td",{},[el("div",{class:"pname"},[avatar(p,26),el("span",{text:p.name}),
      p.unknown?el("span",{class:"tiny",style:"color:var(--warn)",title:"No top-flight record",text:"⚠"}):el("span")])]));
    row.appendChild(el("td",{},[el("span",{class:"pos "+p.pos,text:p.pos,title:POSNAME[p.pos]})]));
    row.appendChild(el("td",{class:"muted"},[el("div",{class:"clubcell"},[badge(p.clubCode,16),el("span",{text:p.club||""})])]));
    row.appendChild(priceCell(p));
    if(S.mode==="weekly"){ const w=weeklyFP(p,S.wkRound);
      row.appendChild(el("td",{class:"num",style:w.games.length?"":"color:var(--crit)"},[el("b",{text:n1(w.fp)})]));
      row.appendChild(el("td",{class:"tiny muted",text:matchupText(w)}));
      row.appendChild(el("td",{class:"num muted",text:n1(p.fp)}));
      row.appendChild(el("td",{class:"num"},[ratChip(p.ovr)]));
    } else { row.appendChild(el("td",{class:"num",text:n1(p.fp)}));
      row.appendChild(el("td",{class:"num muted",text:p.mptg==null?"–":n1(p.mptg)}));
      row.appendChild(el("td",{class:"num"},[ratChip(p.ovr)])); row.appendChild(el("td",{class:"num"},[ratChip(p.val)])); }
    const act=el("td",{},[]);
    if(S.mode==="weekly"){ const w=el("div",{class:"acts"});
      if(owned){ w.appendChild(el("button",{class:"mini",type:"button",title:"Remove from team",text:"Remove",onclick:e=>{e.stopPropagation();removeFromTeam(p.code);}})); }
      else { const ok=t&&canAdd(t,p.code); w.appendChild(el("button",{class:"mini mine",type:"button",title:ok?"Add to team":"No slot / over budget",text:"+ Add",disabled:ok?null:"",style:ok?"":"opacity:.4;cursor:default",onclick:e=>{e.stopPropagation();if(ok)addToTeam(p.code);}})); }
      w.appendChild(el("button",{class:"mini take"+(excl?" on":""),type:"button",title:"Injured / unavailable",text:excl?"Out":"Injury",onclick:e=>{e.stopPropagation();if(excl)S.excluded.delete(p.code);else S.excluded.add(p.code);S.plan=null;refresh();}}));
      act.appendChild(w);
    } else if(gone){ act.appendChild(el("span",{class:"tiny muted",text:S.mine.includes(p.code)?"yours":"taken"})); }
    else { const w=el("div",{class:"acts"}); w.appendChild(el("button",{class:"mini mine",type:"button",text:"+ Mine",onclick:e=>{e.stopPropagation();draftMine(p.code);}}));
      w.appendChild(el("button",{class:"mini take",type:"button",text:"Taken",onclick:e=>{e.stopPropagation();markTaken(p.code);}})); act.appendChild(w); }
    row.appendChild(act); tb.appendChild(row);
  });
  table.appendChild(tb); const bd=$("#board"); bd.innerHTML=""; bd.appendChild(table);
}

/* ================================================================= header + boot */
function renderHeader(){
  $("#kick").textContent="Turkish Airlines EuroLeague · "+(DATA.meta.season||"").replace(/^E(\d+)$/,(m,y)=>y+"-"+String(+y+1).slice(2));
  const h=$("#hstats"); h.innerHTML="";
  if(S.mode==="weekly"){ const t=curTeam();
    const stats=t?[[`R${S.wkRound}`,"Round"],[t.cr.toFixed(0),"CR"],[teamFull(t)?n1(weekPts(t.squad,S.wkRound)):"—","Wk proj"],[`${t.squad.length}/${SQUAD_N}`,"Squad"]]:[[`R${S.wkRound}`,"Round"],["—","CR"],["—","Wk proj"],["0/"+SQUAD_N,"Squad"]];
    stats.forEach(s=>h.appendChild(el("div",{class:"hstat"},[el("b",{text:s[0]}),el("span",{text:s[1]})])));
  } else { const nd=needs();
    [[draftSpent().toFixed(1),"Credits"],[String(nd.G+nd.F+nd.C),"To pick"],[weekPts?"":"",""]].slice(0,2).concat([[weekPtsDraft(),"Proj"]]).forEach(s=>h.appendChild(el("div",{class:"hstat"},[el("b",{text:s[0]}),el("span",{text:s[1]})]))); }
}
function weekPtsDraft(){ const out=S.mine.map(c=>BY[c].fp).sort((a,b)=>b-a); const k=RULES.full; let v=out.slice(0,k).reduce((a,b)=>a+b,0)+RULES.bench*out.slice(k).reduce((a,b)=>a+b,0); if(out.length)v+=(RULES.captain-1)*out[0]; return v.toFixed(0); }
function setMode(m){ S.mode=m; sortK=m==="weekly"?"_wkfp":"fp"; sortDir=-1;
  [...$("#modetabs").children].forEach(c=>c.setAttribute("aria-pressed",String(c.dataset.mode===m)));
  const ch=[...$("#posf").children].find(c=>c.dataset.v==="H"); if(ch)ch.classList.toggle("hide",m==="draft");
  if(m==="draft"&&S.pos==="H"){S.pos="ALL";[...$("#posf").children].forEach(c=>c.setAttribute("aria-pressed",String(c.dataset.v==="ALL")));}
  refresh();
}
function refreshPriceHint(){ var h=$("#pricehint"); if(!h)return; var n=Object.keys(PRICES).length;
  h.innerHTML = n ? ('<b>'+n+'</b> override'+(n>1?'s':'')+' · <a href="#" id="resetprices">reset</a>')
                  : 'CR values are the game\'s official quotations — click any Cr to override';
  var r=$("#resetprices"); if(r) r.addEventListener("click",function(e){e.preventDefault();resetPrices();}); }
function refresh(){ renderHeader(); refreshPriceHint(); if(S.mode==="weekly")renderTeamsPanel(); else renderDraftPanel(); if(S.mode==="draft")computeDraftRec(); }
function boot(){
  loadTeams(); loadPrices();
  if(SCOUT_URL&&SCOUT_URL.indexOf("http")===0)$("#scoutlink").href=SCOUT_URL; else $("#scoutlink").classList.add("hide");
  $("#modetabs").addEventListener("click",e=>{const t=e.target.closest(".modetab");if(t)setMode(t.dataset.mode);});
  const pf=$("#posf"); [["ALL","All"],["G","G"],["F","F"],["C","C"],["H","Coach"]].forEach(([v,l])=>
    pf.appendChild(el("button",{class:"chip","aria-pressed":String(S.pos===v),text:l,"data-v":v,onclick:()=>{S.pos=v;[...pf.children].forEach(c=>c.setAttribute("aria-pressed",String(c.dataset.v===v)));renderBoard();}})));
  $("#q").addEventListener("input",e=>{S.q=e.target.value;renderBoard();});
  var ctrls=document.querySelector(".controls");
  if(ctrls){ var hint=el("span",{class:"pricehint",id:"pricehint"}); ctrls.appendChild(hint); refreshPriceHint(); }
  $("#theme").addEventListener("click",()=>{const cur=document.documentElement.getAttribute("data-theme");const dark=cur?cur==="dark":matchMedia("(prefers-color-scheme: dark)").matches;document.documentElement.setAttribute("data-theme",dark?"light":"dark");refresh();});
  var params = new URLSearchParams(location.search);
  var demo = params.get("demo") || (location.hash||"").replace(/^#/, "");
  if (demo === "weekly") { demoWeekly(); }
  else if (demo === "draft") { demoDraft(); }
  else { setMode("weekly"); }
}
function demoWeekly(){
  try{ localStorage.removeItem(LSKEY); }catch(e){}
  newTeam("My Team"); autoFill();
  var t=curTeam();
  ["Sasha Vezenkov","Dan Oturu"].forEach(function(nm){
    var c=ALL.find(function(p){return p.name===nm;}); if(!c)return; c=c.code;
    var pos=BY[c].pos; t.squad=t.squad.filter(function(x){return x!==c;});
    var cheap=ALL.filter(function(p){return p.pos===pos && t.squad.indexOf(p.code)<0;}).sort(function(a,b){return a.price-b.price;})[0];
    t.squad.push(cheap.code);
  });
  saveTeams(); setMode("weekly"); optimiseWeek();
}
function demoDraft(){
  setMode("draft");
  var pick=function(nm){var p=ALL.find(function(x){return x.name===nm;}); if(p) draftMine(p.code);};
  var take=function(nm){var p=ALL.find(function(x){return x.name===nm;}); if(p) markTaken(p.code);};
  pick("Sasha Vezenkov"); take("Dan Oturu"); take("Theo Maledon"); pick("Mathias Lessort");
}
boot();
