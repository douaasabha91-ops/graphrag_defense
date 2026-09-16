# Bounding Structured-Data Extraction in Graph RAG with Differential Privacy

Code and experiments for a retrieval-time **differential-privacy defense** against the
data-extraction attack on Graph retrieval-augmented generation (Graph RAG) of
Liu et al. (2025), *Exposing Privacy Risks in Graph Retrieval-Augmented Generation*.

This repository accompanies the paper (in preparation) and a master's thesis. It contains
the knowledge-graph builder, the score-based retriever, the differentially private defense,
the baselines, the attack, and the full evaluation harness.

---

## The problem in one paragraph

Graph RAG builds a knowledge graph over a private corpus and retrieves the entities and
relationships nearest to each query. That structure is the sensitive asset: which entities
exist and how they connect is private on its own. An adversary can phrase a query so the
model repeats the retrieved structure verbatim, and leakage **accumulates across queries**,
because each query returns a fresh subgraph the adversary keeps. Every previously proposed
defense acts on a single response, so none bounds this aggregate exposure. We move the control
to retrieval and make it differentially private: a **selection stage** picks the top entities
and relationships under the exponential mechanism, an optional **perturbation stage** blurs
their descriptions, and a **per-user budget** composes the privacy cost across queries so
cumulative privacy loss is bounded.

## Headline result (100-doc HealthCareMagic graph, 40-query targeted attack, 5 seeds)

| Defense | Cumulative entity exposure | Targeted PII hits | ROUGE-L |
|---|---|---|---|
| No defense | 24.4% | 32 | 0.134 |
| System prompt | 22.6% | 32 | 0.140 |
| Similarity threshold | 24.2% | 32 | 0.133 |
| Summarization | n/a (serves prose) | 28 | 0.127 |
| **Matched-volume (top-10, no DP)** | 16.2% | 32 | — |
| **DP budget (B=10)** | **2.8% ± 0.3** | **1.8 ± 0.8** | **0.126** |

The DP budget is the only defense that cuts both cumulative exposure and targeted hits, at
under 6% utility cost. The matched-volume baseline shows most of the reduction comes from the
DP mechanism (a 13.4-point drop at equal retrieval volume), not merely from serving fewer
items (an 8.2-point drop).

---

## Repository layout

```
grd/                       the Python package (all constants in grd/config.py)
  config.py                paths, model names, DP budgets, dataset size, seed
  context.py               Retrieved dataclass; parse / render graph context
  kg_build.py              build the knowledge graph (one LLM call per document) + answer()
  retriever.py             cosine-similarity retrieval returning scored candidates
  defenses.py              Defense interface + Session state:
                             NoDefense, SystemPromptRule, SimilarityThreshold,
                             Summarization, TopKNoDP, DPRetrieval
  attack.py                the extraction-attack prompts and query builders
  run_b.py                 the experiment runner (retrieve -> defense -> answer)
  metrics.py               leakage metrics (per-query + cumulative unique)
  utility.py               ROUGE-L answer quality on a benign QA set
  experiments.py           multi-seed + matched-volume harness; LLM-judge / entity-recall
  data.py                  deterministic dataset subset + benign QA set

notebooks/
  optionB_build_and_attack.ipynb   build the graph, reproduce the attack
  optionB_refinement.ipynb         baselines, utility (ROUGE), untargeted attack
  multiseed_matched_volume.ipynb   multi-seed runs + matched-volume baseline
```

Runtime artifacts (`kg/`, `logs/`, `index/`, dataset index files) are produced locally and
are git-ignored; they are not part of the repository.

## The query pipeline (identical in every condition)

```
retrieve(query, store)  ->  defense.apply(retrieved, session)  ->  answer(...)
```

A `Defense` takes the retrieved entities and relationships and returns a (possibly reduced or
perturbed) set; `NoDefense` is the identity. This is what lets the DP mechanism and every
baseline share one runner, so all conditions are compared on identical retrievals.

## The DP defense (`grd/defenses.py :: DPRetrieval`)

- **Stage 1 (formal):** differentially private top-*k* selection over entities and over
  relationships, by report-noisy-max / exponential-mechanism peeling, with per-item score
  sensitivity 1 on clipped scores. Edge-level (ε, δ)-DP on a fixed candidate universe.
- **Stage 2 (empirical, optional):** Laplace perturbation of numeric attributes in
  descriptions; excluded from the formal guarantee.
- **Per-user budget:** the `Session` object composes the privacy cost across queries; once the
  budget `B` is spent the retriever serves no new structure, and identical repeated queries are
  cached (so noise cannot be averaged away and no budget is re-spent).

## Setup

Models are served locally with [Ollama](https://ollama.com); the code talks to it over HTTP,
so there is no heavy Python ML dependency.

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. Ollama + models
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:7b            # extraction + answering
ollama pull nomic-embed-text      # retrieval embeddings
```

The experiments were developed on a single T4 GPU (Google Colab). On a fresh machine,
remember to `ollama pull` both models before running.

## Reproduce

1. `notebooks/optionB_build_and_attack.ipynb` — build the knowledge graph over a
   HealthCareMagic subset (one LLM call per document), confirm retrieval returns scored
   candidates, and reproduce the extraction attack (C3 vs C1).
2. `notebooks/optionB_refinement.ipynb` — the three prior defenses, ROUGE utility, and the
   untargeted attack.
3. `notebooks/multiseed_matched_volume.ipynb` — the multi-seed study and the matched-volume
   baseline (the two experiments a reviewer expects for a stochastic defense).

Programmatic example:

```python
from grd import config as C, data, attack, metrics, kg_build as kb, retriever as rt, run_b
from grd.defenses import NoDefense, DPRetrieval

docs, records, _ = data.load_subset(n_docs=100)
store = kb.KGStore(f"{C.ROOT}/kg/hcm100")
kb.build_graph(docs, store)                          # index once

N_ENT, N_REL, *_ = run_b.graph_sets(store)
q = attack.build_queries("targeted", "C3")[:40]

run_b.run(store, q, "attack_none",  defense=NoDefense())
run_b.run(store, q, "attack_dp_b10",
          defense=DPRetrieval(eps_ent=1.0, eps_rel=1.0, k=10, user_budget=10),
          use_cache=True)
print(metrics.summarize_runs(["attack_none", "attack_dp_b10"], N_ENT, N_REL, C.LOG_DIR))
```

All runs are seeded and resumable: `run_b.run` skips queries already in the log, so an
interrupted session continues without losing work.

## Notes and honest caveats

- **Cumulative exposure is the headline metric.** The per-query leakage percentage is
  denominator-sensitive: a defense serving fewer items can show a *higher* per-query
  percentage while leaking far less overall. Read cumulative unique exposure.
- The formal guarantee is **edge-level**, on a fixed candidate universe; the node-level and
  private-universe cases are scoped explicitly in the paper.
- Stage 2 (description perturbation) is an optional empirical component, not part of the formal
  guarantee.
- This is a single-model, small-graph, 40-query study. It reports relative comparisons on one
  stack, not the absolute rates of a production deployment.
- Datasets are public, used only to measure a known vulnerability and its mitigation.

## Citation

If you use this code, please cite the paper:

> D. Sabha, A. Imine, H. Harb, M. Dbouk. *Bounding Structured-Data Extraction in Graph RAG
> with Differential Privacy.* 2026 (in preparation).

See `CITATION.cff` for machine-readable metadata.

## License

MIT. See `LICENSE`.
