"""Brute-force check that the DP returns the true optimum of the structured objective."""
import itertools, random, sys
import numpy as np
from eldraft.optimize import SquadDP, Slots
from eldraft.config import GameRules

def brute(players, budget, caps, rules):
    best, bestset = float("-inf"), None
    idx = range(len(players))
    for combo in itertools.combinations(idx, sum(caps.values())):
        cnt = {"G":0,"F":0,"C":0,"H":0}
        cost = 0.0
        for i in combo:
            cnt[players[i]["position"]] += 1; cost += players[i]["price"]
        if cnt != caps or cost > budget + 1e-9: continue
        pl = sorted([players[i]["fp"] for i in combo if players[i]["position"]!="H"], reverse=True)
        coach = sum(players[i]["fp"] for i in combo if players[i]["position"]=="H")
        v = sum(pl[:rules.full_credit_slots]) + rules.bench_multiplier*sum(pl[rules.full_credit_slots:])
        v += (rules.captain_multiplier-1.0)*(pl[0] if pl else 0.0) + coach
        if v > best: best, bestset = v, combo
    return best, bestset

random.seed(3)
rules = GameRules(budget=30.0, slots={"G":2,"F":2,"C":1}, squad_size=5,
                  full_credit_slots=3, bench_multiplier=0.5, captain_multiplier=2.0)
fails = 0
for trial in range(12):
    players = []
    for i in range(16):
        pos = random.choice(["G","G","F","F","C"])
        players.append({"code":str(i),"name":"P%d"%i,"position":pos,
                        "price": round(random.choice([2.0,2.5,3.0,4.0,5.5,7.0,8.5]),1),
                        "fp": round(random.uniform(1,22),2)})
    for i in range(3):
        players.append({"code":"H%d"%i,"name":"C%d"%i,"position":"H",
                        "price": round(random.choice([2.0,3.0,4.5]),1),
                        "fp": round(random.uniform(2,9),2)})
    caps = {"G":2,"F":2,"C":1,"H":1}
    slots = Slots(caps={"G":2,"F":2,"C":1}, coach=True)
    dp = SquadDP(players, 30.0, slots, rules)
    v, idx = dp.solve()
    bv, _ = brute(players, 30.0, caps, rules)
    ok = abs(v-bv) < 1e-2
    # verify the recovered squad actually scores what the DP claims
    if idx:
        sel=[players[i] for i in idx]
        cnt={"G":0,"F":0,"C":0,"H":0}
        for s in sel: cnt[s["position"]]+=1
        cost=sum(s["price"] for s in sel)
        pl=sorted([s["fp"] for s in sel if s["position"]!="H"],reverse=True)
        coach=sum(s["fp"] for s in sel if s["position"]=="H")
        rv=sum(pl[:3])+0.5*sum(pl[3:])+1.0*(pl[0] if pl else 0)+coach
        valid = cnt==caps and cost<=30.0+1e-9 and abs(rv-v)<1e-2
    else: valid=False
    # marginal values: max(with) must equal the optimum and never exceed it; VORP must
    # be positive exactly for the players the optimal squad actually uses.
    mv = dp.marginal_values()
    withs=[w for w,_ in mv.values() if np.isfinite(w)]
    marg_ok = abs(max(withs)-bv)<1e-2 and max(withs) <= bv+1e-2
    chosen=set(idx)
    for i,(w,wo) in mv.items():
        if not np.isfinite(w): continue
        vorp = w - wo if np.isfinite(wo) else float("inf")
        if i in chosen and vorp < -1e-3: marg_ok=False
        if i not in chosen and vorp > 1e-3: marg_ok=False
    if not(ok and valid and marg_ok):
        fails+=1
        print("FAIL trial",trial,"dp=%.3f brute=%.3f valid=%s marg_ok=%s"%(v,bv,valid,marg_ok))
    else:
        print("ok trial %d  dp=%.3f == brute=%.3f  squad valid, marginals consistent"%(trial,v,bv))
print("brute-force failures:", fails)

# --------------------------------------------------------------------------------------
# Mandatory (already-drafted) players must always appear in the solution.
# --------------------------------------------------------------------------------------
def test_mandatory():
    random.seed(11)
    rules = GameRules(budget=30.0, slots={"G":2,"F":2,"C":1}, squad_size=5,
                      full_credit_slots=3, bench_multiplier=0.5, captain_multiplier=2.0)
    bad = 0
    for t in range(8):
        players=[]
        for i in range(18):
            pos=random.choice(["G","G","F","F","C"])
            players.append({"code":str(i),"position":pos,
                            "price":round(random.choice([2.0,2.5,3.0,4.0,5.5]),1),
                            "fp":round(random.uniform(1,22),2)})
        players.append({"code":"H0","position":"H","price":3.0,"fp":5.0})
        forced = random.randrange(18)
        players[forced]["mandatory"]=True
        slots=Slots(caps={"G":2,"F":2,"C":1},coach=True)
        v,idx = SquadDP(players,30.0,slots,rules).solve()
        if forced not in idx:
            bad+=1; print("FAIL mandatory not selected", t, v, idx)
        else:
            # forcing a player can only ever lower or equal the unconstrained optimum
            free=[dict(p) for p in players]
            for p in free: p.pop("mandatory",None)
            fv,_=SquadDP(free,30.0,slots,rules).solve()
            if v > fv+1e-3: bad+=1; print("FAIL forced > free", v, fv)
    print("mandatory failures:", bad)
    return bad

# --------------------------------------------------------------------------------------
# forced / excluded candidate evaluation
# --------------------------------------------------------------------------------------
def test_forced_excluded():
    random.seed(23)
    rules = GameRules(budget=30.0, slots={"G":2,"F":2,"C":1}, squad_size=5,
                      full_credit_slots=3, bench_multiplier=0.5, captain_multiplier=2.0)
    bad=0
    for t in range(6):
        players=[]
        for i in range(20):
            pos=random.choice(["G","G","F","F","C"])
            players.append({"code":str(i),"position":pos,
                            "price":round(random.choice([2.0,2.5,3.0,4.0,5.5]),1),
                            "fp":round(random.uniform(1,22),2)})
        players.append({"code":"H0","position":"H","price":3.0,"fp":5.0})
        slots=Slots(caps={"G":2,"F":2,"C":1},coach=True)
        free,fidx = SquadDP(players,30.0,slots,rules).solve()
        j = random.randrange(20)
        vf,ifx = SquadDP(players,30.0,slots,rules,forced={j}).solve()
        ve,iex = SquadDP(players,30.0,slots,rules,excluded={j}).solve()
        if j not in ifx: bad+=1; print("FAIL forced missing")
        if j in iex: bad+=1; print("FAIL excluded present")
        if vf > free+1e-3 or ve > free+1e-3: bad+=1; print("FAIL constrained beats free")
        if abs(max(vf,ve)-free) > 1e-2: bad+=1; print("FAIL max(forced,excluded) != free", vf,ve,free)
        # and they must agree with the value-only marginals
        mv = SquadDP(players,30.0,slots,rules).marginal_values()
        w,wo = mv[j]
        if abs(w-vf)>1e-2 or abs(wo-ve)>1e-2: bad+=1; print("FAIL marginals disagree",w,vf,wo,ve)
    print("forced/excluded failures:", bad)
    return bad


if __name__ == "__main__":
    n = test_mandatory() + test_forced_excluded()
    print("ALL EXTRA TESTS PASS" if n==0 else "EXTRA TESTS FAILED")
    sys.exit(1 if (fails or n) else 0)
