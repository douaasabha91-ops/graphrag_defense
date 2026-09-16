"""
run_b.py - experiment runner for the self-built backend (Option B).

Same job and same JSONL log format as attack.run_experiment, but retrieval comes
from retriever.retrieve(store) and answering from kg_build.answer. So metrics.py,
the query sets in attack.py, and every Defense work unchanged.
"""
import json, os, time
from . import config as C
from .retriever import retrieve
from .kg_build import answer
from .defenses import NoDefense, Session
from .context import Retrieved

def log_path(run_name):
    return f'{C.LOG_DIR}/{run_name}.jsonl'

def run(store, queries, run_name, defense=None, use_cache=False, top_n=None, log=print):
    defense = defense or NoDefense()
    session = Session(run_name)
    path = log_path(run_name)
    seen = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); seen.add(r['id'])
            session.record(Retrieved.from_dict(r['served']))
            if use_cache: session.cache[r['query']] = r['served']
    eps_q = getattr(defense, '_eps_per_query', 0.0)
    with open(path, 'a') as f:
        for k, q in enumerate(queries):
            if q['id'] in seen:
                continue
            t0 = time.time()
            rec = {**q, 'run': run_name, 'defense': defense.name,
                   'session': session.session_id, 'query_index': session.n_queries}
            try:
                if use_cache and q['query'] in session.cache:
                    served = Retrieved.from_dict(session.cache[q['query']])
                    retrieved = served; rec['cached'] = True
                    resp = answer(q['query'], served, defense.system_prompt_extra)
                    session.record(served, eps=0.0)
                else:
                    retrieved = retrieve(q['query'], store, top_n=top_n)
                    served = defense.apply(retrieved, session)
                    resp = answer(q['query'], served, defense.system_prompt_extra)
                    if use_cache: session.cache[q['query']] = served.to_dict()
                    session.record(served, eps=eps_q); rec['cached'] = False
                rec.update(retrieved=retrieved.to_dict(), served=served.to_dict(),
                           response=resp, eps_spent=round(session.eps_spent, 3), error=None)
            except Exception as e:
                rec.update(retrieved=None, served={'entities': [], 'relations': [], 'chunks': []},
                           response='', error=repr(e))
            rec['seconds'] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + '\n'); f.flush()
            log(f"{k+1}/{len(queries)} {q['id']} {rec['seconds']}s" + (f"  ERR {rec['error']}" if rec.get('error') else ''))
    return path

def graph_sets(store):
    """(n_ent, n_rel, entity_name_set, relation_key_set) for metrics denominators."""
    from .context import rel_key
    ents = set(store.G.nodes)
    rels = {rel_key(a, b) for a, b in store.G.edges}
    return len(ents), len(rels), ents, rels
