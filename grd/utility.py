"""
utility.py - answer-quality (ROUGE) on the benign QA set, per defense.

Runs the benign questions through the same pipeline (retrieve -> defense -> answer)
as one user session, so the per-user budget is actually consumed, then scores each
answer against its reference with ROUGE-L. Privacy and utility are thus measured on
the same mechanism and reported together.
"""
import json, os
from . import config as C
from .retriever import retrieve
from .kg_build import answer
from .defenses import NoDefense, Session
from .context import Retrieved

def run_utility(store, qa, run_name, defense=None, use_cache=True, top_n=None, log=print):
    """qa: list of {id, question, reference}. Writes a JSONL like the attack runner."""
    defense = defense or NoDefense()
    session = Session(run_name)
    path = f'{C.LOG_DIR}/util_{run_name}.jsonl'
    seen = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); seen.add(r['id'])
            session.record(Retrieved.from_dict(r['served']))
    eps_q = getattr(defense, '_eps_per_query', 0.0)
    with open(path, 'a') as f:
        for k, q in enumerate(qa):
            if q['id'] in seen: continue
            try:
                retrieved = retrieve(q['question'], store, top_n=top_n)
                served = defense.apply(retrieved, session)
                resp = answer(q['question'], served, defense.system_prompt_extra)
                session.record(served, eps=eps_q)
                rec = {'id': q['id'], 'run': run_name, 'defense': defense.name,
                       'question': q['question'], 'reference': q['reference'],
                       'served': served.to_dict(), 'response': resp, 'error': None}
            except Exception as e:
                rec = {'id': q['id'], 'run': run_name, 'defense': defense.name,
                       'question': q['question'], 'reference': q.get('reference',''),
                       'served': {'entities':[], 'relations':[], 'chunks':[]},
                       'response': '', 'error': repr(e)}
            f.write(json.dumps(rec) + '\n'); f.flush()
            log(f"util {k+1}/{len(qa)} {q['id']}" + (f" ERR" if rec['error'] else ''))
    return path

def score_utility(run_names, log=print):
    """ROUGE-L F1 averaged over the benign QA answers, per run."""
    from rouge_score import rouge_scorer
    import pandas as pd
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    out = {}
    for r in run_names:
        path = f'{C.LOG_DIR}/util_{r}.jsonl'
        if not os.path.exists(path):
            continue
        fs, n, err = [], 0, 0
        for l in open(path):
            rec = json.loads(l); n += 1
            if rec.get('error'): err += 1; continue
            fs.append(sc.score(rec['reference'], rec['response'])['rougeL'].fmeasure)
        out[r] = {'n': n, 'errors': err, 'rougeL_f1': round(sum(fs)/len(fs), 4) if fs else 0.0}
    return pd.DataFrame(out).T
