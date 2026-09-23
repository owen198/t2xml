import json, math, os, re, sys, collections
import numpy as np
SANTA=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REPO=os.path.dirname(SANTA)
sys.path.insert(0, SANTA); sys.path.insert(0, f"{SANTA}/evaluate_xml")
R=f"{REPO}/retrieval/S1000D_spec_sample"
def lj(p,i,t):
    rows=[json.loads(l) for l in open(p)]
    return [str(r[i]) for r in rows],[r[t] for r in rows]
cid,ctxt=lj(f"{R}/corpus.test.jsonl","docid","structured"); qid,qtxt=lj(f"{R}/queries.test.jsonl","qid","text")
qrels={l.split()[0]:l.split()[1] for l in list(open(f"{R}/qrels.test.tsv"))[1:]}
tok=lambda s: re.findall(r"[A-Za-z0-9_]+",s.lower())
# BM25 over the 21 test docs (idf from a bigger pool: the S1000D train corpus, so idf is meaningful)
trn=[json.loads(l)["structured"] for l in open(f"{R}/corpus.train.jsonl")]
df=collections.Counter(); 
for d in trn+ctxt: df.update(set(tok(d)))
N=len(trn)+len(ctxt); idf=lambda t: math.log(1+(N-df.get(t,0)+.5)/(df.get(t,0)+.5))
docs=[collections.Counter(tok(d)) for d in ctxt]; L=[sum(c.values()) for c in docs]; avg=sum(L)/len(L)
def bm25(q,i,k1=1.2,b=.75):
    return sum(idf(t)*docs[i][t]*(k1+1)/(docs[i][t]+k1*(1-b+b*L[i]/avg)) for t in set(tok(q)) if t in docs[i])
ranks=[]
for q,t in zip(qid,qtxt):
    s=[bm25(t,i) for i in range(len(cid))]; order=np.argsort(-np.array(s),kind="stable").tolist()
    ranks.append(order.index(cid.index(qrels[q]))+1)
print("BM25 on 21 test docs: MRR=%.4f ranks=%s (chance 0.1736)"%(np.mean([1/r for r in ranks]),ranks))
