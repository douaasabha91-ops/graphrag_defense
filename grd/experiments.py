"""
experiments.py - multi-seed and matched-volume experiment harness.

Provides two evaluation protocols expected of a stochastic defense:
  1. multi-seed: every stochastic condition is run over several seeds; we report
     mean and standard deviation of each metric.
  2. matched retrieval-volume: a no-DP top-k baseline (TopKNoDP) served at the same
     k as the DP mechanism, so the privacy gap between them isolates the DP noise
     from the effect of simply serving fewer items.

Everything reuses run_b.run and metrics.score_log; results are aggregated here.
Runs are resumable (run_b skips logged queries) and each (condition, seed) writes
its own log, so a crash costs at most one cell.
"""
import json, os
import numpy as np
import pandas as pd
from . import config as C
from . import run_b, metrics, attack
from .defenses import NoDefense, DPRetrieval, TopKNoDP, SystemPromptRule, SimilarityThreshold

# ------------------------------------------------------------------ multi-seed
def run_multiseed(store, queries_fn, conditions, seeds, n_graph_ent, n_graph_rel,
                  tag='ms', log=print):
    """
    conditions: dict name -> callable(seed)->Defense  (deterministic defenses ignore seed)
    queries_fn: callable()-> list of query dicts (same across seeds for comparability)
    seeds:      list of int
    Returns a tidy DataFrame with one row per (condition, seed) and the key metrics,
    plus an aggregated mean/std table.
    """
    rows = []
    for name, make in conditions.items():
        for s in seeds:
            dfn = make(s)
            run_name = f'{tag}_{name}_seed{s}'
            use_cache = isinstance(dfn, DPRetrieval)
            run_b.run(store, queries_fn(), run_name, defense=dfn, use_cache=use_cache, log=log)
            _, summ = metrics.score_log(run_b.log_path(run_name), n_graph_ent, n_graph_rel)
            rows.append({'condition': name, 'seed': s,
                         'cum_ent_pct': summ['cum_unique_ent_%graph'],
                         'cum_rel_pct': summ['cum_unique_rel_%graph'],
                         'targeted_hits': summ['targeted_hits'],
                         'ent_leak_pct': summ['entity_leakage_%'],
                         'avg_served_ent': summ['avg_served_ent']})
            log(f'  done {run_name}: cum_ent={summ["cum_unique_ent_%graph"]}%')
    per_run = pd.DataFrame(rows)
    agg = (per_run.groupby('condition')
           .agg(['mean', 'std'])
           .round(2))
    return per_run, agg

def format_meanstd(agg, metric='cum_ent_pct'):
    """Return a clean 'mean ± std' string column for one metric, for the paper table."""
    m = agg[(metric, 'mean')]; s = agg[(metric, 'std')].fillna(0.0)
    return (m.map('{:.1f}'.format) + ' $\\pm$ ' + s.map('{:.1f}'.format))

# ------------------------------------------------------------------ standard condition sets
def default_conditions(k=10, budget=10, eps=1.0, threshold=0.5, summarize_fn=None):
    """The condition set for the matched-volume + multi-seed study.
    DP and TopKNoDP share k so their comparison isolates the DP noise effect."""
    conds = {
        'none':        lambda s: NoDefense(),                       # deterministic
        'topk_nodp':   lambda s: TopKNoDP(k=k),                     # matched volume, no DP
        'dp_budget':   lambda s: DPRetrieval(eps_ent=eps, eps_rel=eps, k=k,
                                             user_budget=budget, seed=s),
        'sysprompt':   lambda s: SystemPromptRule(),                # deterministic
        'threshold':   lambda s: SimilarityThreshold(tau=threshold),
    }
    if summarize_fn is not None:
        from .defenses import Summarization
        conds['summarize'] = lambda s: Summarization(summarize_fn=summarize_fn)
    return conds

def budget_sweep_conditions(budgets, k=10, eps=1.0):
    """DP at several budgets, each seeded, for the multi-seed frontier."""
    conds = {'none': lambda s: NoDefense()}
    for B in budgets:
        conds[f'dp_b{B}'] = (lambda B: (lambda s: DPRetrieval(
            eps_ent=eps, eps_rel=eps, k=k, user_budget=B, seed=s)))(B)
    return conds
