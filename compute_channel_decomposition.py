#!/usr/bin/env python3
"""
Full two-channel Berkson decomposition from RAW alignment-faking data.
Computes MEASURED (not parametric) I(C;Y_b), I(C;Y_r), the channel-switching
signature I(C;Y_r|comply), and the joint, for any paired train/deploy condition.

C   = is_train (1=training/monitored, 0=deploy/unmonitored)
Y_b = complies with harmful request (not is_refusal_0_7)  [the compliance gap]
Y_r = alignment-faking reasoning (cites_not_trained_as_reason, D2-recommended)
"""
import json, math, sys, zipfile
from collections import Counter

ZIP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/json_outputs.zip"

def H(probs):
    return -sum(p*math.log2(p) for p in probs if p > 0)

def mi_binary(pairs):
    """plug-in MI for list of (a,b) binary tuples."""
    n = len(pairs)
    if n == 0: return float('nan'), 0
    ca, cb, cab = Counter(), Counter(), Counter()
    for a,b in pairs:
        ca[a]+=1; cb[b]+=1; cab[(a,b)]+=1
    mi = 0.0
    for (a,b),c in cab.items():
        pab=c/n; pa=ca[a]/n; pb=cb[b]/n
        mi += pab*math.log2(pab/(pa*pb))
    return mi, n

def load(zf, prefix):
    files=[f for f in zf.namelist() if f.startswith(prefix) and f.endswith(".json")]
    return [json.loads(zf.read(f)) for f in files]

def decompose(zf, train_prefix, deploy_prefix, yr_field="cites_not_trained_as_reason"):
    tr=load(zf, train_prefix); dp=load(zf, deploy_prefix)
    rows=[]
    for d in tr: rows.append((1, 0 if d.get("is_refusal_0_7") else 1, 1 if d.get(yr_field) else 0))
    for d in dp: rows.append((0, 0 if d.get("is_refusal_0_7") else 1, 1 if d.get(yr_field) else 0))
    C  =[(c,yb) for c,yb,yr in rows]
    CR =[(c,yr) for c,yb,yr in rows]
    Jr =[(c,(yb,yr)) for c,yb,yr in rows]
    comply=[(c,yr) for c,yb,yr in rows if yb==1]
    i_cyb,_=mi_binary(C); i_cyr,_=mi_binary(CR); i_joint,_=mi_binary(Jr)
    i_sig,ncomp=mi_binary(comply)
    # behavioral rates
    p_comply_tr=sum(1 for d in tr if not d.get("is_refusal_0_7"))/max(len(tr),1)
    p_comply_dp=sum(1 for d in dp if not d.get("is_refusal_0_7"))/max(len(dp),1)
    p_yr_tr=sum(1 for d in tr if d.get(yr_field))/max(len(tr),1)
    p_yr_dp=sum(1 for d in dp if d.get(yr_field))/max(len(dp),1)
    print(f"\n=== {train_prefix.split('/')[0]}  (Y_r = {yr_field}) ===")
    print(f"  N: train={len(tr)} deploy={len(dp)}  | complied={ncomp}")
    print(f"  compliance gap (train→deploy): {p_comply_tr:.3f} → {p_comply_dp:.3f}  (Δ={p_comply_tr-p_comply_dp:+.3f})")
    print(f"  AF-reasoning  (train→deploy): {p_yr_tr:.3f} → {p_yr_dp:.3f}  (Δ={p_yr_dp-p_yr_tr:+.3f})")
    print(f"  I(C;Y_b)  behavioral coupling      = {i_cyb:.4f} bits")
    print(f"  I(C;Y_r)  reasoning coupling        = {i_cyr:.4f} bits")
    print(f"  I(C;Y_r|comply)  SWITCH SIGNATURE   = {i_sig:.4f} bits")
    print(f"  I(C;Y_b,Y_r)  joint/total           = {i_joint:.4f} bits")

zf=zipfile.ZipFile(ZIP)
decompose(zf,"json_outputs/helpful_only_post_rl/train/","json_outputs/helpful_only_post_rl/deploy/")
decompose(zf,"json_outputs/animal_welfare_post_rl/train/","json_outputs/animal_welfare_post_rl/deploy/")
# CoT ablation: synthetic-docs post-RL, with vs without CoT
decompose(zf,"json_outputs/synthetic_documents_helpful_only_post_rl/train/","json_outputs/synthetic_documents_helpful_only_post_rl/deploy/")
