import os, sys, numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__))
exec(open(f"{HERE}/embedding_spread.py").read().split("def stats")[0])  # defines SANTA, REPO, cid/ctxt/qid/qtxt/qrels, dev
def mrr_of(S):
    idx={d:i for i,d in enumerate(cid)}; r=[]
    for i,q in enumerate(qid):
        o=torch.argsort(S[i],descending=True).tolist(); r.append(o.index(idx[qrels[q]])+1)
    return np.mean([1/x for x in r])
def run(name,path,L=256):
    tok=AutoTokenizer.from_pretrained(path,use_fast=False)
    m=SModel.build(model_args=ModelArguments(model_name_or_path=path),santa_args=SantaArguments(use_generate=False)).to(dev).eval()
    P=encode_all(m,tok,ctxt,L,21,dev,is_query=False).float(); Q=encode_all(m,tok,qtxt,50,21,dev,is_query=True).float()
    del m; torch.cuda.empty_cache()
    n=torch.nn.functional.normalize
    raw=mrr_of(Q@P.T)
    Pc,Qc=P-P.mean(0),Q-Q.mean(0)
    cen=mrr_of(n(Qc,dim=-1)@n(Pc,dim=-1).T)
    mu=P.mean(0); top=(mu**2).sort(descending=True).values; share=[(top[:k].sum()/top.sum()).item() for k in (1,5,20)]
    print(f"{name:32s} raw dot MRR={raw:.4f} | centered-cosine MRR={cen:.4f} | mean-vector energy in top1/5/20 of {len(mu)} dims: {share[0]:.2f}/{share[1]:.2f}/{share[2]:.2f} | argmax dim={int(mu.abs().argmax())}")
run("codet5 untrained","Salesforce/codet5-base")
run("t5 untrained","t5-base")
runs=f"{SANTA}/runs"
run("codet5 BIKE_7->S1000D (256)",f"{runs}/finetune/codet5_s42/BIKE_7_to_S1000D_spec_sample/checkpoints")
run("codet5 BIKE_7->S1000D (p1024)",f"{runs}/finetune/cnt_p1024_s42/BIKE_7_to_S1000D_spec_sample/checkpoints",1024)
