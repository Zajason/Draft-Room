import fs from 'fs';
const src=fs.readFileSync('/tmp/engine.js','utf8');
eval(src+'\nglobalThis.transferDP=transferDP;');
function structVal(codes,byCode,rules){const out=[];let coach=0;
  codes.forEach(c=>{const p=byCode[c];if(p.pos==='H')coach+=p.fp;else out.push(p.fp);});
  out.sort((a,b)=>b-a);const k=rules.full;let v=out.slice(0,k).reduce((a,b)=>a+b,0)+rules.bench*out.slice(k).reduce((a,b)=>a+b,0);
  if(out.length)v+=(rules.captain-1)*out[0];return v+coach;}
function brute(players,budget,curSet,maxT,caps,rules){
  // enumerate all valid full squads within budget/caps; filter by transfers≤maxT; return best value per level
  const byPos={G:[],F:[],C:[],H:[]}; players.forEach((p,i)=>byPos[p.pos].push(i));
  const need={G:caps[0],F:caps[1],C:caps[2],H:caps[3]};
  const combos=(arr,k)=>{const res=[];const rec=(s,c)=>{if(c.length===k){res.push(c.slice());return;}for(let i=s;i<arr.length;i++){c.push(arr[i]);rec(i+1,c);c.pop();}};rec(0,[]);return res;};
  const gC=combos(byPos.G,need.G),fC=combos(byPos.F,need.F),cC=combos(byPos.C,need.C),hC=need.H?combos(byPos.H,need.H):[[]];
  const best=new Array(maxT+1).fill(-Infinity);
  for(const g of gC)for(const f of fC)for(const c of cC)for(const h of hC){
    const sel=[...g,...f,...c,...h];const cost=sel.reduce((a,i)=>a+Math.round(players[i].price/0.5),0);
    if(cost>Math.round(budget/0.5))continue;
    const nnew=sel.filter(i=>!curSet.has(players[i].code)).length; if(nnew>maxT)continue;
    const codes=sel.map(i=>players[i].code);const byCode={};players.forEach(p=>byCode[p.code]=p);
    const v=structVal(codes,byCode,rules);
    for(let lvl=nnew;lvl<=maxT;lvl++)if(v>best[lvl])best[lvl]=v;
  }
  return best;
}
const rules={full:2,bench:0.5,captain:2.0};
const caps=[2,2,1,0]; // small: 2G 2F 1C, no coach, squad=5, full=2
let fails=0;
for(let trial=0;trial<30;trial++){
  const players=[];const pos=['G','G','G','G','F','F','F','F','C','C','C'];
  pos.forEach((p,i)=>players.push({code:'p'+i,pos:p,price:[2,2.5,3,4,5.5,7][Math.floor(Math.random()*6)],fp:+(Math.random()*20).toFixed(2)}));
  const byCode={};players.forEach(p=>byCode[p.code]=p);
  // random current valid squad
  const cur=['p0','p1','p4','p5','p8']; const curSet=new Set(cur);
  const budget=18, maxT=3;
  const dp=transferDP(players,budget,curSet,maxT,caps,rules);
  const bf=brute(players,budget,curSet,maxT,caps,rules);
  for(let lvl=0;lvl<=maxT;lvl++){
    const d=dp.frontier[lvl], b=bf[lvl]<=-1e17?null:bf[lvl];
    if((d==null)!==(b==null)||(d!=null&&Math.abs(d-b)>1e-2)){fails++;console.log('FAIL trial',trial,'lvl',lvl,'dp',d,'bf',b);}
    // verify recovered squad matches value & transfer count
    if(d!=null){const sq=dp.squads[lvl];const nnew=sq.filter(c=>!curSet.has(c)).length;
      const v=structVal(sq,byCode,rules);
      if(nnew>lvl||Math.abs(v-d)>1e-2){fails++;console.log('FAIL recover trial',trial,'lvl',lvl,'nnew',nnew,'v',v,'d',d);}}
  }
}
console.log(fails===0?'ALL TRANSFER TESTS PASS (30 trials × 4 levels vs brute force)':('FAILURES: '+fails));
