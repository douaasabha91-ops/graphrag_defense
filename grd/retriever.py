"""
retriever.py - score-based retrieval over the self-built KG (no LightRAG).

retrieve(query, store, top_n) embeds the query, cosine-scores every entity and
every relation, and returns a Retrieved object whose items carry real similarity
scores. This is exactly the scored candidate pool the DP mechanism needs.

The DP defense selects its DP top-k FROM this pool; baselines and no-defense take
the plain top-k. So one retriever feeds every condition, and DPRetrieval finally
has real scores instead of the rank proxy.
"""
import numpy as np
from .context import Retrieved, Entity, Relation, norm
from .kg_build import ollama_embed

def _cos(q, M):
    q = np.asarray(q, float); M = np.asarray(M, float)
    if M.size == 0:
        return np.array([])
    qn = q / (np.linalg.norm(q) + 1e-9)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ qn

def retrieve(query, store, top_n=None, embed=ollama_embed):
    """Return Retrieved with the top_n entities and top_n relations by cosine score.
    Scores are min-max normalized to [0,1] so DP sensitivity Delta=1 holds."""
    from . import config as C
    top_n = top_n or C.DP_CANDIDATE_POOL
    qv = embed(query)

    ent_keys = [f'E:{n}' for n in store.G.nodes]
    ent_names = list(store.G.nodes)
    ent_M = [store.emb[k] for k in ent_keys if k in store.emb]
    ent_names = [n for n, k in zip(ent_names, ent_keys) if k in store.emb]
    escore = _cos(qv, ent_M)

    rel_items, rel_M = [], []
    for a, b, d in store.G.edges(data=True):
        key = 'R:' + '|'.join(sorted((a, b)))
        if key in store.emb:
            rel_items.append((a, b, d.get('description', '')))
            rel_M.append(store.emb[key])
    rscore = _cos(qv, rel_M)

    def norm01(x):
        if x.size == 0: return x
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)
    escore, rscore = norm01(escore), norm01(rscore)

    e_order = np.argsort(-escore)[:top_n] if escore.size else []
    r_order = np.argsort(-rscore)[:top_n] if rscore.size else []

    entities = [Entity(ent_names[i], store.G.nodes[ent_names[i]].get('type', ''),
                       store.G.nodes[ent_names[i]].get('description', ''),
                       float(escore[i])) for i in e_order]
    relations = [Relation(rel_items[i][0], rel_items[i][1], rel_items[i][2],
                          float(rscore[i])) for i in r_order]
    return Retrieved(entities=entities, relations=relations, chunks=[])
