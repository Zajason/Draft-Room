// Standalone JS port of the exact squad DP, verified against the Python implementation.
const NEG = -1e9;
const POS = ["G","F","C","H"], CAP = {G:4,F:4,C:2,H:1};
export function buildDP(players, budget, rules){
  rules = rules || {full:6, bench:0.5, captain:2.0};
  const B = Math.round(budget/0.5);
  const caps=[CAP.G,CAP.F,CAP.C,CAP.H], dims=[caps[0]+1,caps[1]+1,caps[2]+1,caps[3]+1,B+1];
  const stride=[dims[1]*dims[2]*dims[3]*dims[4], dims[2]*dims[3]*dims[4], dims[3]*dims[4], dims[4], 1];
  const size=dims[0]*stride[0];
  const posIx=p=>POS.indexOf(p.pos);
  const units=players.map(p=>Math.round(p.price/0.5));
  const order=players.map((_,i)=>i).sort((a,b)=>players[b].fp-players[a].fp);
  // multiplier as function of (g,f,c): next selected outfield player's weight
  const mult=(g,f,c)=> (g+f+c)<rules.full ? 1.0 : rules.bench;
  const empty=()=>{const a=new Float32Array(size); a.fill(NEG); return a;};
  // forward tables
  const fwd=new Array(order.length+1);
  let cur=empty(); cur[0]=0.0; fwd[0]=cur;
  for(let k=0;k<order.length;k++){
    const i=order[k], p=players[i], ax=posIx(p), cost=units[i], cap=caps[ax], s=p.fp, mand=!!p.mandatory;
    const nxt = mand ? empty() : cur.slice();
    if(cap>0 && cost<=B){
      // iterate states with room at axis ax
      const idx=[0,0,0,0];
      for(idx[0]=0;idx[0]<dims[0];idx[0]++) for(idx[1]=0;idx[1]<dims[1];idx[1]++)
      for(idx[2]=0;idx[2]<dims[2];idx[2]++) for(idx[3]=0;idx[3]<dims[3];idx[3]++){
        if(idx[ax]>=cap) continue;
        const g=idx[0],f=idx[1],c=idx[2];
        let gain = (ax===3)? s : mult(g,f,c)*s;
        if(ax!==3 && g===0&&f===0&&c===0) gain += (rules.captain-1.0)*s;
        const dst=idx.slice(); dst[ax]+=1;
        const baseFrom=idx[0]*stride[0]+idx[1]*stride[1]+idx[2]*stride[2]+idx[3]*stride[3];
        const baseTo=dst[0]*stride[0]+dst[1]*stride[1]+dst[2]*stride[2]+dst[3]*stride[3];
        for(let b=0;b+cost<=B;b++){
          const src=cur[baseFrom+b];
          if(src<=NEG/2) continue;
          const cand=src+gain, t=baseTo+b+cost;
          if(cand>nxt[t]) nxt[t]=cand;
        }
      }
    }
    fwd[k+1]=nxt; cur=nxt;
  }
  // optimal value
  const term=[caps[0],caps[1],caps[2],caps[3]];
  const baseT=term[0]*stride[0]+term[1]*stride[1]+term[2]*stride[2]+term[3]*stride[3];
  let bestV=NEG,bestB=0;
  for(let b=0;b<=B;b++){ const v=cur[baseT+b]; if(v>bestV){bestV=v;bestB=b;} }
  // recover squad
  let st=[term[0],term[1],term[2],term[3],bestB], chosen=[];
  for(let k=order.length;k>0;k--){
    const i=order[k-1], p=players[i], ax=posIx(p), cost=units[i], s=p.fp, mand=!!p.mandatory;
    const cur2=fwd[k], prev=fwd[k-1];
    const flat=(a)=>a[0]*stride[0]+a[1]*stride[1]+a[2]*stride[2]+a[3]*stride[3]+a[4];
    if(!mand && Math.abs(prev[flat(st)]-cur2[flat(st)])<1e-4) continue;
    const cand=st.slice(); cand[ax]-=1; cand[4]-=cost;
    if(cand[ax]<0||cand[4]<0) continue;
    const g=cand[0],f=cand[1],c=cand[2];
    let gain=(ax===3)?s:mult(g,f,c)*s;
    if(ax!==3 && g===0&&f===0&&c===0) gain+=(rules.captain-1.0)*s;
    if(Math.abs(prev[flat(cand)]+gain-cur2[flat(st)])<1e-3){ chosen.push(i); st=cand; }
  }
  return {value:bestV, squad:chosen.map(i=>players[i].code), B, fwd, order, players, units, caps, dims, stride, rules};
}
// quick self-test vs python export
import fs from 'fs';
const c=JSON.parse(fs.readFileSync('/tmp/dp_case.json'));
const r=buildDP(c.players, c.budget);
const okV=Math.abs(r.value-c.opt_value)<1e-2;
const okSquad=JSON.stringify(r.squad.slice().sort())===JSON.stringify(c.opt_codes);
console.log('JS opt_value',r.value.toFixed(3),'python',c.opt_value,'match',okV);
console.log('squad match',okSquad);

export function marginals(dp){
  const {fwd, order, players, units, caps, dims, stride, B, rules}=dp;
  const mult=(g,f,c)=> (g+f+c)<rules.full ? 1.0 : rules.bench;
  const posIx=p=>POS.indexOf(p.pos);
  const size=dims[0]*stride[0];
  const empty=()=>{const a=new Float32Array(size); a.fill(NEG); return a;};
  const n=order.length;
  // terminal: only a fully-filled squad is worth 0
  const term=()=>{const a=empty(); const base=caps[0]*stride[0]+caps[1]*stride[1]+caps[2]*stride[2]+caps[3]*stride[3]; for(let b=0;b<=B;b++)a[base+b]=0.0; return a;};
  const bwd=new Array(n+1); bwd[n]=term();
  for(let k=n-1;k>=0;k--){
    const i=order[k], p=players[i], ax=posIx(p), cost=units[i], cap=caps[ax], s=p.fp, mand=!!p.mandatory;
    const nxt=bwd[k+1];
    const take=empty();
    if(cap>0 && cost<=B){
      const idx=[0,0,0,0];
      for(idx[0]=0;idx[0]<dims[0];idx[0]++) for(idx[1]=0;idx[1]<dims[1];idx[1]++)
      for(idx[2]=0;idx[2]<dims[2];idx[2]++) for(idx[3]=0;idx[3]<dims[3];idx[3]++){
        if(idx[ax]>=cap) continue;
        const g=idx[0],f=idx[1],c=idx[2];
        let gain=(ax===3)?s:mult(g,f,c)*s;
        if(ax!==3 && g===0&&f===0&&c===0) gain+=(rules.captain-1.0)*s;
        const src=idx.slice(); src[ax]+=1;
        const bFrom=src[0]*stride[0]+src[1]*stride[1]+src[2]*stride[2]+src[3]*stride[3];
        const bTo=idx[0]*stride[0]+idx[1]*stride[1]+idx[2]*stride[2]+idx[3]*stride[3];
        for(let b=0;b+cost<=B;b++){
          const v=nxt[bFrom+b+cost];
          if(v<=NEG/2) continue;
          take[bTo+b]=v+gain;
        }
      }
    }
    const res = mand ? take : (()=>{const a=nxt.slice(); for(let j=0;j<size;j++) if(take[j]>a[j])a[j]=take[j]; return a;})();
    bwd[k]=res;
  }
  // combine
  const out={};
  for(let k=0;k<n;k++){
    const i=order[k], p=players[i], ax=posIx(p), cost=units[i], cap=caps[ax], s=p.fp, mand=!!p.mandatory;
    const before=fwd[k], after=bwd[k+1];
    let withP=NEG, withoutP=NEG;
    const idx=[0,0,0,0];
    for(idx[0]=0;idx[0]<dims[0];idx[0]++) for(idx[1]=0;idx[1]<dims[1];idx[1]++)
    for(idx[2]=0;idx[2]<dims[2];idx[2]++) for(idx[3]=0;idx[3]<dims[3];idx[3]++){
      const bBase=idx[0]*stride[0]+idx[1]*stride[1]+idx[2]*stride[2]+idx[3]*stride[3];
      // without: skip p
      if(!mand){ for(let b=0;b<=B;b++){ const a=before[bBase+b], c2=after[bBase+b]; if(a>NEG/2&&c2>NEG/2){const v=a+c2; if(v>withoutP)withoutP=v;} } }
      // with: take p
      if(cap>0 && idx[ax]<cap && cost<=B){
        const g=idx[0],f=idx[1],c=idx[2];
        let gain=(ax===3)?s:mult(g,f,c)*s;
        if(ax!==3 && g===0&&f===0&&c===0) gain+=(rules.captain-1.0)*s;
        const dst=idx.slice(); dst[ax]+=1;
        const aBase=dst[0]*stride[0]+dst[1]*stride[1]+dst[2]*stride[2]+dst[3]*stride[3];
        for(let b=0;b+cost<=B;b++){ const bf=before[bBase+b], af=after[aBase+b+cost]; if(bf>NEG/2&&af>NEG/2){const v=bf+gain+af; if(v>withP)withP=v;} }
      }
    }
    out[p.code]=[withP>NEG/2?withP:null, (mand?null:(withoutP>NEG/2?withoutP:null))];
  }
  return out;
}
// verify marginals vs python
import { buildDP as _b } from '/tmp/dp.mjs';
const cc=JSON.parse(fs.readFileSync('/tmp/dp_case.json'));
const dd=buildDP(cc.players, cc.budget);
const mm=marginals(dd);
let maxerr=0, checked=0, mism=0;
for(const code in cc.marg){
  const [pw,pwo]=cc.marg[code]; const [jw,jwo]=mm[code]||[null,null];
  if(pw!=null&&jw!=null){maxerr=Math.max(maxerr,Math.abs(pw-jw)); checked++;}
  if(pwo!=null&&jwo!=null){maxerr=Math.max(maxerr,Math.abs(pwo-jwo));}
  else if((pwo==null)!==(jwo==null)) mism++;
}
console.log('marginals checked',checked,'max abs err',maxerr.toFixed(5),'null-mismatch',mism);
