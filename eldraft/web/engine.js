/* ================================================================================
   Draft engines — runs entirely in the browser.

   Two decision engines share one exact valuation:

     * Greedy (VORP): the current tool. One forward + one backward pass of the squad
       dynamic program yield, for every available player, the best squad you can build
       WITH him minus the best WITHOUT — his value over replacement. Ranks instantly.
     * Lookahead (MCTS): plays the rest of the draft forward against a model of the
       other teams, so a pick is valued by the squad it leads to after rivals have taken
       their share. Uses the same exact DP to complete rosters at the leaves.

   The DP here is a line-for-line port of the Python implementation and is verified to
   match it to 1e-5 on every player.
   ================================================================================ */
const NEG = -1e9;
const POSL = ["G", "F", "C", "H"];
const CAPS = [4, 4, 2, 1];
// The game's real quotations move in tenths of a credit, so the knapsack grid must too;
// a coarser grid rounds prices and can slip an over-budget squad past the cap.
const CSTEP = 0.1;

function unitsOf(price) { return Math.round(price / CSTEP); }

/* ---- exact squad dynamic program --------------------------------------------- */
function DP(players, budget, rules, caps) {
  rules = rules || { full: 6, bench: 0.5, captain: 2.0 };
  caps = caps || CAPS;
  const B = Math.round(budget / CSTEP);
  const dims = [caps[0] + 1, caps[1] + 1, caps[2] + 1, caps[3] + 1, B + 1];
  const stride = [dims[1] * dims[2] * dims[3] * dims[4], dims[2] * dims[3] * dims[4], dims[3] * dims[4], dims[4], 1];
  const size = dims[0] * stride[0];
  const units = players.map(p => unitsOf(p.price));
  const posIx = players.map(p => POSL.indexOf(p.pos));
  const fp = players.map(p => p.fp);
  const mand = players.map(p => !!p.mandatory);
  const order = players.map((_, i) => i).sort((a, b) => fp[b] - fp[a]);
  const mult = (g, f, c) => (g + f + c) < rules.full ? 1.0 : rules.bench;
  const empty = () => { const a = new Float32Array(size); a.fill(NEG); return a; };

  function forward() {
    const fwd = new Array(order.length + 1);
    let cur = empty(); cur[0] = 0; fwd[0] = cur;
    for (let k = 0; k < order.length; k++) {
      const i = order[k], ax = posIx[i], cost = units[i], cap = caps[ax], s = fp[i];
      const nxt = mand[i] ? empty() : cur.slice();
      if (cap > 0 && cost <= B) {
        for (let g = 0; g < dims[0]; g++) for (let f = 0; f < dims[1]; f++)
        for (let c = 0; c < dims[2]; c++) for (let h = 0; h < dims[3]; h++) {
          const ix = [g, f, c, h]; if (ix[ax] >= cap) continue;
          let gain = (ax === 3) ? s : mult(g, f, c) * s;
          if (ax !== 3 && g === 0 && f === 0 && c === 0) gain += (rules.captain - 1) * s;
          const bFrom = g * stride[0] + f * stride[1] + c * stride[2] + h * stride[3];
          const d = [g, f, c, h]; d[ax]++;
          const bTo = d[0] * stride[0] + d[1] * stride[1] + d[2] * stride[2] + d[3] * stride[3];
          for (let b = 0; b + cost <= B; b++) {
            const src = cur[bFrom + b]; if (src <= NEG / 2) continue;
            const t = bTo + b + cost, cand = src + gain;
            if (cand > nxt[t]) nxt[t] = cand;
          }
        }
      }
      fwd[k + 1] = nxt; cur = nxt;
    }
    return fwd;
  }

  function solveFrom(fwd) {
    const cur = fwd[order.length];
    const base = caps[0] * stride[0] + caps[1] * stride[1] + caps[2] * stride[2] + caps[3] * stride[3];
    let bestV = NEG, bestB = 0;
    for (let b = 0; b <= B; b++) { const v = cur[base + b]; if (v > bestV) { bestV = v; bestB = b; } }
    let st = [caps[0], caps[1], caps[2], caps[3], bestB], chosen = [];
    const flat = a => a[0] * stride[0] + a[1] * stride[1] + a[2] * stride[2] + a[3] * stride[3] + a[4];
    for (let k = order.length; k > 0; k--) {
      const i = order[k - 1], ax = posIx[i], cost = units[i], s = fp[i];
      if (!mand[i] && Math.abs(fwd[k - 1][flat(st)] - fwd[k][flat(st)]) < 1e-4) continue;
      const cand = st.slice(); cand[ax]--; cand[4] -= cost;
      if (cand[ax] < 0 || cand[4] < 0) continue;
      let gain = (ax === 3) ? s : mult(cand[0], cand[1], cand[2]) * s;
      if (ax !== 3 && cand[0] === 0 && cand[1] === 0 && cand[2] === 0) gain += (rules.captain - 1) * s;
      if (Math.abs(fwd[k - 1][flat(cand)] + gain - fwd[k][flat(st)]) < 1e-3) { chosen.push(i); st = cand; }
    }
    return { value: bestV, squad: chosen };
  }

  // Value-only forward pass: keeps just two rolling buffers instead of one array per
  // player, so it can be called thousands of times in the MCTS leaves without GC thrash.
  function solveValue() {
    let cur = empty(); cur[0] = 0;
    let buf = new Float32Array(size);
    for (let k = 0; k < order.length; k++) {
      const i = order[k], ax = posIx[i], cost = units[i], cap = caps[ax], s = fp[i], man = mand[i];
      if (man) buf.fill(NEG); else buf.set(cur);
      if (cap > 0 && cost <= B) {
        for (let g = 0; g < dims[0]; g++) for (let f = 0; f < dims[1]; f++)
        for (let c = 0; c < dims[2]; c++) for (let h = 0; h < dims[3]; h++) {
          const ix = [g, f, c, h]; if (ix[ax] >= cap) continue;
          let gain = (ax === 3) ? s : mult(g, f, c) * s;
          if (ax !== 3 && g === 0 && f === 0 && c === 0) gain += (rules.captain - 1) * s;
          const bFrom = g*stride[0]+f*stride[1]+c*stride[2]+h*stride[3];
          const d = [g, f, c, h]; d[ax]++;
          const bTo = d[0]*stride[0]+d[1]*stride[1]+d[2]*stride[2]+d[3]*stride[3];
          for (let b = 0; b + cost <= B; b++) {
            const src = cur[bFrom + b]; if (src <= NEG / 2) continue;
            const t = bTo + b + cost, cand = src + gain;
            if (cand > buf[t]) buf[t] = cand;
          }
        }
      }
      const tmp = cur; cur = buf; buf = tmp;   // ping-pong the two buffers
    }
    const base = caps[0]*stride[0]+caps[1]*stride[1]+caps[2]*stride[2]+caps[3]*stride[3];
    let best = NEG; for (let b = 0; b <= B; b++) { const v = cur[base + b]; if (v > best) best = v; }
    return best;
  }

  function backward() {
    const n = order.length;
    const term = () => { const a = empty(); const base = caps[0]*stride[0]+caps[1]*stride[1]+caps[2]*stride[2]+caps[3]*stride[3]; for (let b = 0; b <= B; b++) a[base + b] = 0; return a; };
    const bwd = new Array(n + 1); bwd[n] = term();
    for (let k = n - 1; k >= 0; k--) {
      const i = order[k], ax = posIx[i], cost = units[i], cap = caps[ax], s = fp[i];
      const nxt = bwd[k + 1], take = empty();
      if (cap > 0 && cost <= B) {
        for (let g = 0; g < dims[0]; g++) for (let f = 0; f < dims[1]; f++)
        for (let c = 0; c < dims[2]; c++) for (let h = 0; h < dims[3]; h++) {
          const ix = [g, f, c, h]; if (ix[ax] >= cap) continue;
          let gain = (ax === 3) ? s : mult(g, f, c) * s;
          if (ax !== 3 && g === 0 && f === 0 && c === 0) gain += (rules.captain - 1) * s;
          const d = [g, f, c, h]; d[ax]++;
          const bFrom = d[0]*stride[0]+d[1]*stride[1]+d[2]*stride[2]+d[3]*stride[3];
          const bTo = g*stride[0]+f*stride[1]+c*stride[2]+h*stride[3];
          for (let b = 0; b + cost <= B; b++) { const v = nxt[bFrom + b + cost]; if (v <= NEG / 2) continue; take[bTo + b] = v + gain; }
        }
      }
      if (mand[i]) bwd[k] = take;
      else { const a = nxt.slice(); for (let j = 0; j < size; j++) if (take[j] > a[j]) a[j] = take[j]; bwd[k] = a; }
    }
    return bwd;
  }

  function marginals(fwd, bwd) {
    const out = new Array(players.length);
    for (let k = 0; k < order.length; k++) {
      const i = order[k], ax = posIx[i], cost = units[i], cap = caps[ax], s = fp[i];
      const before = fwd[k], after = bwd[k + 1];
      let withP = NEG, withoutP = NEG;
      for (let g = 0; g < dims[0]; g++) for (let f = 0; f < dims[1]; f++)
      for (let c = 0; c < dims[2]; c++) for (let h = 0; h < dims[3]; h++) {
        const bBase = g*stride[0]+f*stride[1]+c*stride[2]+h*stride[3];
        if (!mand[i]) for (let b = 0; b <= B; b++) { const a = before[bBase+b], c2 = after[bBase+b]; if (a > NEG/2 && c2 > NEG/2) { const v = a + c2; if (v > withoutP) withoutP = v; } }
        if (cap > 0 && [g,f,c,h][ax] < cap && cost <= B) {
          let gain = (ax === 3) ? s : mult(g, f, c) * s;
          if (ax !== 3 && g === 0 && f === 0 && c === 0) gain += (rules.captain - 1) * s;
          const d = [g, f, c, h]; d[ax]++;
          const aBase = d[0]*stride[0]+d[1]*stride[1]+d[2]*stride[2]+d[3]*stride[3];
          for (let b = 0; b + cost <= B; b++) { const bf = before[bBase+b], af = after[aBase+b+cost]; if (bf > NEG/2 && af > NEG/2) { const v = bf + gain + af; if (v > withP) withP = v; } }
        }
      }
      out[i] = [withP > NEG/2 ? withP : null, mand[i] ? null : (withoutP > NEG/2 ? withoutP : null)];
    }
    return out;
  }

  return { forward, solveFrom, solveValue, backward, marginals, B };
}

/* ---- greedy recommendation: VORP over available, given my picks --------------- */
function greedyRecommend(pool, budget, rules, caps) {
  const dp = DP(pool, budget, rules, caps);
  const fwd = dp.forward();
  const { value, squad } = dp.solveFrom(fwd);
  const bwd = dp.backward();
  const mv = dp.marginals(fwd, bwd);
  const inSquad = new Set(squad);
  const rows = [];
  for (let i = 0; i < pool.length; i++) {
    const [w, wo] = mv[i] || [null, null];
    const vorp = (w !== null && wo !== null) ? w - wo : null;
    rows.push({ i, code: pool[i].code, vorp, inSquad: inSquad.has(i), mandatory: !!pool[i].mandatory });
  }
  return { value, squad: squad.map(i => pool[i].code), rows };
}

/* ================================================================================
   MCTS lookahead — the AlphaZero-style option, in the browser.
   Uses the exact DP above to complete rosters at the leaves; opponents are a heuristic
   field. Faithful to the Python implementation that was measured against greedy.
   ================================================================================ */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

function makeUni(players){
  const n=players.length;
  const fp=new Float64Array(n), units=new Int32Array(n), pos=new Int8Array(n);
  players.forEach((p,i)=>{fp[i]=p.fp||0;units[i]=unitsOf(p.price);pos[i]=POSL.indexOf(p.pos);});
  const posSorted=[0,1,2,3].map(pi=>{const a=[];for(let i=0;i<n;i++)if(pos[i]===pi)a.push(i);a.sort((x,y)=>units[x]-units[y]);return a;});
  return {n,fp,units,pos,posSorted,players};
}
function minCompletion(uni,avail,needs){
  let tot=0;
  for(let pi=0;pi<4;pi++){let k=needs[pi];if(k<=0)continue;let got=0;const ord=uni.posSorted[pi];
    for(let j=0;j<ord.length&&got<k;j++){const idx=ord[j];if(avail[idx]){tot+=uni.units[idx];got++;}}
    if(got<k)return 1e9;}
  return tot;
}
function feasibleP(uni,state,seat,i){
  if(!state.avail[i])return false;const pi=uni.pos[i];if(state.needs[seat][pi]<=0)return false;
  const rem=state.budgetU-state.spent[seat];const cost=uni.units[i];if(cost>rem)return false;
  const nd=state.needs[seat].slice();nd[pi]--;let s=nd[0]+nd[1]+nd[2]+nd[3];if(s===0)return true;
  state.avail[i]=0;const floor=minCompletion(uni,state.avail,nd);state.avail[i]=1;return floor<=rem-cost;
}
function candIdx(uni,state,seat){
  const rem=state.budgetU-state.spent[seat],nd=state.needs[seat],out=[];
  for(let i=0;i<uni.n;i++){if(state.avail[i]&&uni.units[i]<=rem&&nd[uni.pos[i]]>0)out.push(i);}return out;
}
function applyPick(uni,state,seat,i){state.spent[seat]+=uni.units[i];state.needs[seat][uni.pos[i]]--;state.avail[i]=0;state.ptr++;if(state.mine!==undefined&&seat===state.mySeat)state.mine.push(i);}
function heurPick(uni,state,seat,rng,temp){
  const cand=candIdx(uni,state,seat);if(!cand.length)return -1;
  const fps=cand.map(i=>uni.fp[i]);const mx=Math.max(...fps);
  let ord;
  if(temp<=1e-6)ord=cand.map((i,k)=>[i,fps[k]]).sort((a,b)=>b[1]-a[1]).map(x=>x[0]);
  else{const sd=(Math.max(...fps)-Math.min(...fps))/4+1e-6;const w=fps.map(v=>Math.exp((v-mx)/(temp*sd)));
    // sample up to 6 distinct by weight
    const picks=[];const pool=cand.slice(),wt=w.slice();
    for(let t=0;t<Math.min(6,pool.length);t++){let s=wt.reduce((a,b)=>a+b,0);let r=rng()*s,acc=0,ix=0;
      for(let k=0;k<pool.length;k++){acc+=wt[k];if(r<=acc){ix=k;break;}}picks.push(pool[ix]);pool.splice(ix,1);wt.splice(ix,1);}
    ord=picks;}
  for(const i of ord)if(feasibleP(uni,state,seat,i))return i;
  const byU=cand.slice().sort((a,b)=>uni.units[a]-uni.units[b]);
  for(const i of byU)if(feasibleP(uni,state,seat,i))return i;
  return -1;
}
function cloneState(state){return {needs:state.needs.map(a=>a.slice()),spent:state.spent.slice(),
  avail:state.avail.slice(),order:state.order,ptr:state.ptr,mySeat:state.mySeat,budgetU:state.budgetU,
  myPicks:state.myPicks.slice()};}
function squadVal(uni,picks,rules){const out=[];let coach=0;
  for(const i of picks){if(uni.pos[i]===3)coach+=uni.fp[i];else out.push(uni.fp[i]);}
  out.sort((a,b)=>b-a);const k=rules.full;let v=0;for(let j=0;j<out.length;j++)v+=(j<k?1:rules.bench)*out[j];
  if(out.length)v+=(rules.captain-1)*out[0];return v+coach;}
function dpCompleteValue(uni,myPicks,avail,budget,rules,shortlist,caps){
  const sub=[];for(const i of myPicks)sub.push({pos:POSL[uni.pos[i]],price:uni.units[i]*CSTEP,fp:uni.fp[i],mandatory:true});
  for(let pi=0;pi<4;pi++){const idxs=[];for(let i=0;i<uni.n;i++)if(avail[i]&&uni.pos[i]===pi)idxs.push(i);
    idxs.sort((a,b)=>uni.fp[b]-uni.fp[a]);for(let j=0;j<Math.min(shortlist,idxs.length);j++){const i=idxs[j];sub.push({pos:POSL[pi],price:uni.units[i]*CSTEP,fp:uni.fp[i]});}}
  return DP(sub,budget,rules,caps).solveValue();
}
function leafValue(uni,state,rng,rules,shortlist,caps){
  const s=cloneState(state);
  while(s.ptr<s.order.length){const seat=s.order[s.ptr];if(seat===s.mySeat){s.ptr++;continue;}
    const i=heurPick(uni,s,seat,rng,0.6);if(i<0){s.ptr++;continue;}applyPick(uni,s,seat,i);}
  const v=dpCompleteValue(uni,state.myPicks,s.avail,state.budgetU*CSTEP,rules,shortlist,caps);
  return isFinite(v)&&v>-1e8?v:squadVal(uni,state.myPicks,rules);
}
function softmaxPrior(scores,temp){const ks=Object.keys(scores);if(!ks.length)return{};
  const vs=ks.map(k=>scores[k]);const mx=Math.max(...vs);const w=vs.map(v=>Math.exp((v-mx)/Math.max(1e-9,temp)));
  const s=w.reduce((a,b)=>a+b,0);const o={};ks.forEach((k,i)=>o[k]=w[i]/s);return o;}

function remainingOrder(needsPerTeam,mySeat,nTeams){
  const order=[];const rem=needsPerTeam.map(nd=>nd[0]+nd[1]+nd[2]+nd[3]);
  const seq=[mySeat];for(let s=0;s<nTeams;s++)if(s!==mySeat)seq.push(s);
  let rnd=0,guard=0;
  while(rem.some(r=>r>0)&&guard<1000){const ring=rnd%2===0?seq:seq.slice().reverse();
    for(const s of ring)if(rem[s]>0){order.push(s);rem[s]--;}rnd++;guard++;}
  return order;
}

function mctsRecommend(players,BY,mineCodes,takenSet,rules,nTeams,mySeat,sims,caps){
  caps=caps||[rules.slots.G,rules.slots.F,rules.slots.C,1];
  const uni=makeUni(players);
  const codeIx={};players.forEach((p,i)=>codeIx[p.code]=i);
  // needs per team from template
  const tmpl=caps.slice();
  const needs=[];for(let s=0;s<nTeams;s++)needs.push(tmpl.slice());
  const avail=new Uint8Array(uni.n).fill(1);
  const spent=new Int32Array(nTeams);
  const myPicks=[];
  mineCodes.forEach(c=>{const i=codeIx[c];avail[i]=0;needs[mySeat][uni.pos[i]]--;spent[mySeat]+=uni.units[i];myPicks.push(i);});
  // distribute rivals' taken players to opponent seats to set plausible needs
  let seat=0;const opp=[];for(let s=0;s<nTeams;s++)if(s!==mySeat)opp.push(s);
  takenSet.forEach(c=>{const i=codeIx[c];if(avail[i]===0)return;avail[i]=0;const pi=uni.pos[i];
    let placed=false;for(let t=0;t<opp.length;t++){const s=opp[(seat+t)%opp.length];if(needs[s][pi]>0){needs[s][pi]--;spent[s]+=uni.units[i];seat=(seat+t+1)%opp.length;placed=true;break;}}});
  const order=remainingOrder(needs,mySeat,nTeams);
  const budgetU=Math.round(rules.budget/CSTEP);
  const root={children:{},N:0,W:0,P:{},cand:[],expanded:false};
  const rootState={needs,spent,avail,order,ptr:0,mySeat,budgetU,myPicks};
  // must be my turn at ptr 0 (remainingOrder starts with me)
  const rng=mulberry32(12345);
  const cpuct=1.6,topK=12,shortlist=22;let valueNorm=0;

  // root prior from exact VORP
  const pool=[];const poolIx=[];
  for(let i=0;i<uni.n;i++){if(myPicks.includes(i)){pool.push(Object.assign({},players[i],{mandatory:true}));poolIx.push(i);}
    else if(avail[i]){pool.push(players[i]);poolIx.push(i);}}
  const gr=greedyRecommend(pool,rules.budget,rules,caps);
  const vorp={};gr.rows.forEach(r=>{if(r.vorp!=null&&!r.mandatory)vorp[poolIx[r.i]]=r.vorp;});

  function prior(state,depth){
    const cand=candIdx(uni,state,state.mySeat).filter(i=>feasibleP(uni,state,state.mySeat,i));
    if(!cand.length)return{};
    if(depth===0&&Object.keys(vorp).length){const sc={};let lo=1e9;cand.forEach(i=>{const v=vorp[i]!=null?vorp[i]:0;sc[i]=v;if(v<lo)lo=v;});
      for(const k in sc)sc[k]=sc[k]-lo+0.05;return softmaxPrior(sc,1.0);}
    const sc={};cand.forEach(i=>sc[i]=uni.fp[i]);return softmaxPrior(sc,1.2);
  }
  function expand(node,state,depth){const P=prior(state,depth);const ks=Object.keys(P).map(Number);
    if(!ks.length){node.expanded=true;return;}ks.sort((a,b)=>P[b]-P[a]);const top=ks.slice(0,topK);
    let z=0;top.forEach(a=>z+=P[a]);node.P={};top.forEach(a=>node.P[a]=P[a]/(z||1));node.cand=top;node.expanded=true;}
  function select(node,legalSet){const avail=node.cand.filter(a=>legalSet.has(a));if(!avail.length)return -1;
    const sN=Math.sqrt(Math.max(1,node.N));let best=-1e18,ba=avail[0];
    for(const a of avail){const ch=node.children[a];const q=ch?ch.W/ch.N:0;const u=cpuct*(node.P[a]||1e-3)*sN/(1+(ch?ch.N:0));
      if(q+u>best){best=q+u;ba=a;}}return ba;}

  for(let it=0;it<sims;it++){
    const state=cloneState(rootState);let node=root;const path=[root];let depth=0,value;
    while(true){
      if(state.ptr>=state.order.length){value=squadVal(uni,state.myPicks,rules);break;}
      const cand=candIdx(uni,state,state.mySeat).filter(i=>feasibleP(uni,state,state.mySeat,i));
      if(!cand.length){value=squadVal(uni,state.myPicks,rules);break;}
      if(!node.expanded){expand(node,state,depth);value=leafValue(uni,state,rng,rules,shortlist,caps);break;}
      const legal=new Set(cand);const a=select(node,legal);if(a<0){value=leafValue(uni,state,rng,rules,shortlist,caps);break;}
      applyPick(uni,state,state.mySeat,a);
      while(state.ptr<state.order.length&&state.order[state.ptr]!==state.mySeat){const sd=state.order[state.ptr];const i=heurPick(uni,state,sd,rng,0.6);if(i<0){state.ptr++;continue;}applyPick(uni,state,sd,i);}
      let ch=node.children[a];if(!ch){ch={children:{},N:0,W:0,P:{},cand:[],expanded:false};node.children[a]=ch;}
      node=ch;path.push(node);depth++;
    }
    if(!valueNorm)valueNorm=Math.max(1,Math.abs(value));const v=value/valueNorm;
    for(const nd of path){nd.N++;nd.W+=v;}
  }
  const rows=[];for(const a of root.cand){const ch=root.children[a];rows.push({code:players[a].code,
    metric:ch?ch.N:0,value:ch?ch.W/ch.N*valueNorm:0,prior:root.P[a]||0});}
  rows.sort((x,y)=>y.metric-x.metric||y.value-x.value);
  return rows;
}

/* ================================================================================
   Transfer-constrained optimiser (weekly meta-game).

   The normal game is not a free rebuild each week — you carry your squad and may make at
   most a few transfers. This is the exact optimiser for that: the best squad you can reach
   from your current one within `maxT` transfers, where keeping a player you already own is
   free and bringing in anyone new costs one transfer (a coach change counts as a transfer
   too). It is the free squad DP with one extra dimension — transfers used — so it stays
   exact: state (G,F,C,H, credits, transfers).

   Player `fp` here is a HORIZON value (this week plus a discounted look at the next few
   rounds), so a transfer is judged on the run of fixtures it buys, not just one week —
   that, together with the caller's per-transfer cost, is the problem-3 cost function.
   ================================================================================ */
function transferDP(players, budget, currentSet, maxT, caps, rules){
  rules=rules||{full:6,bench:0.5,captain:2.0}; caps=caps||CAPS;
  const B=Math.round(budget/CSTEP), T=maxT;
  const dims=[caps[0]+1,caps[1]+1,caps[2]+1,caps[3]+1,B+1,T+1];
  const st=[dims[1]*dims[2]*dims[3]*dims[4]*dims[5],dims[2]*dims[3]*dims[4]*dims[5],dims[3]*dims[4]*dims[5],dims[4]*dims[5],dims[5],1];
  const size=dims[0]*st[0];
  const units=players.map(p=>unitsOf(p.price));
  const posIx=players.map(p=>POSL.indexOf(p.pos));
  const fp=players.map(p=>p.fp);
  const isNew=players.map(p=>currentSet.has(p.code)?0:1);
  const order=players.map((_,i)=>i).sort((a,b)=>fp[b]-fp[a]);
  const mult=(g,f,c)=>(g+f+c)<rules.full?1.0:rules.bench;
  const empty=()=>{const a=new Float32Array(size);a.fill(NEG);return a;};
  const fwd=new Array(order.length+1);
  let cur=empty(); cur[0]=0; fwd[0]=cur;
  for(let k=0;k<order.length;k++){
    const i=order[k],ax=posIx[i],cost=units[i],cap=caps[ax],s=fp[i],nt=isNew[i];
    const nxt=cur.slice();
    if(cap>0&&cost<=B){
      for(let g=0;g<dims[0];g++)for(let f=0;f<dims[1];f++)for(let c=0;c<dims[2];c++)for(let h=0;h<dims[3];h++){
        const ix=[g,f,c,h]; if(ix[ax]>=cap)continue;
        let gain=(ax===3)?s:mult(g,f,c)*s;
        if(ax!==3&&g===0&&f===0&&c===0)gain+=(rules.captain-1)*s;
        const d=[g,f,c,h]; d[ax]++;
        const bFrom=g*st[0]+f*st[1]+c*st[2]+h*st[3];
        const bTo=d[0]*st[0]+d[1]*st[1]+d[2]*st[2]+d[3]*st[3];
        for(let t=0;t+nt<=T;t++)for(let b=0;b+cost<=B;b++){
          const src=cur[bFrom+b*st[4]+t]; if(src<=NEG/2)continue;
          const to=bTo+(b+cost)*st[4]+(t+nt), cand=src+gain;
          if(cand>nxt[to])nxt[to]=cand;
        }
      }
    }
    fwd[k+1]=nxt; cur=nxt;
  }
  const termPos=caps[0]*st[0]+caps[1]*st[1]+caps[2]*st[2]+caps[3]*st[3];
  // best exact value at each transfer level, and the (b,t) that achieves ≤ level
  const bestAtMost=new Array(T+1).fill(-Infinity), argB=new Array(T+1).fill(0), argT=new Array(T+1).fill(0);
  for(let lvl=0;lvl<=T;lvl++){
    for(let t=0;t<=lvl;t++)for(let b=0;b<=B;b++){
      const v=cur[termPos+b*st[4]+t];
      if(v>bestAtMost[lvl]){bestAtMost[lvl]=v;argB[lvl]=b;argT[lvl]=t;}
    }
  }
  function recover(lvl){
    let state=[caps[0],caps[1],caps[2],caps[3],argB[lvl],argT[lvl]], chosen=[];
    const flat=a=>a[0]*st[0]+a[1]*st[1]+a[2]*st[2]+a[3]*st[3]+a[4]*st[4]+a[5];
    for(let k=order.length;k>0;k--){
      const i=order[k-1],ax=posIx[i],cost=units[i],s=fp[i],nt=isNew[i];
      if(Math.abs(fwd[k-1][flat(state)]-fwd[k][flat(state)])<1e-4)continue; // skipped
      const cand=state.slice(); cand[ax]--; cand[4]-=cost; cand[5]-=nt;
      if(cand[ax]<0||cand[4]<0||cand[5]<0)continue;
      let gain=(ax===3)?s:mult(cand[0],cand[1],cand[2])*s;
      if(ax!==3&&cand[0]===0&&cand[1]===0&&cand[2]===0)gain+=(rules.captain-1)*s;
      if(Math.abs(fwd[k-1][flat(cand)]+gain-fwd[k][flat(state)])<1e-3){chosen.push(players[i].code);state=cand;}
    }
    return chosen;
  }
  const frontier=[], squads=[];
  for(let lvl=0;lvl<=T;lvl++){ frontier.push(bestAtMost[lvl]>NEG/2?bestAtMost[lvl]:null);
    squads.push(bestAtMost[lvl]>NEG/2?recover(lvl):null); }
  return {frontier, squads, transfersUsed:argT};
}
