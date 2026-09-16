"""
kg_build.py - build a knowledge graph WITHOUT LightRAG.

One LLM call per document extracts entities + relations as JSON; results
accumulate into a NetworkX graph. Every entity and relation gets an embedding
(computed once, cached) so the retriever can score by cosine similarity.

Design choices (chosen for robustness on a single commodity GPU):
  - plain synchronous HTTP to Ollama, one request at a time (no async pipeline,
    no worker pool, no context overflow, nothing to crash under concurrency)
  - each doc is independent; a failed doc is retried alone and never blocks others
  - graph + embeddings persisted to disk as JSON after every doc (resumable)
  - no document-status store, so no cross-run dedup ghosts

Works with Ollama (default) or any callable you pass for llm/embed.
"""
import json, os, time, re
import numpy as np
import networkx as nx
import requests
from . import config as C
from .context import norm

# ------------------------------------------------------------------ LLM + embed (Ollama HTTP)
def ollama_generate(prompt, system='', model=None, host=None, num_ctx=4096, timeout=180):
    model = model or C.LLM_MODEL
    host = host or C.OLLAMA_HOST
    r = requests.post(f'{host}/api/generate', json={
        'model': model, 'prompt': prompt, 'system': system,
        'options': {'temperature': 0.0, 'num_ctx': num_ctx},
        'keep_alive': -1, 'stream': False}, timeout=timeout)
    r.raise_for_status()
    return r.json()['response']

def ollama_embed(text, model=None, host=None, timeout=60):
    model = model or C.EMB_MODEL
    host = host or C.OLLAMA_HOST
    r = requests.post(f'{host}/api/embeddings', json={
        'model': model, 'prompt': text, 'keep_alive': -1}, timeout=timeout)
    r.raise_for_status()
    return r.json()['embedding']

# ------------------------------------------------------------------ extraction prompt
EXTRACT_SYSTEM = (
    "You extract a knowledge graph from a medical consultation. "
    "Return ONLY valid JSON, no prose, no code fences. Schema:\n"
    '{"entities":[{"name":"...","type":"...","description":"..."}],'
    '"relations":[{"source":"...","target":"...","description":"..."}]}\n'
    "Entities are medical concepts: conditions, symptoms, drugs, procedures, body parts, "
    "the patient, the doctor. Keep names short and canonical. Relations connect two entity "
    "names that both appear in your entities list."
)
EXTRACT_USER = "Extract the knowledge graph from this consultation:\n\n{doc}"

def _parse_json(text):
    """Pull the first JSON object out of the model output, tolerate stray text/fences."""
    text = re.sub(r'^```[a-z]*\s*|```\s*$', '', text.strip(), flags=re.S)
    start = text.find('{')
    if start < 0:
        return {'entities': [], 'relations': []}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    obj.setdefault('entities', []); obj.setdefault('relations', [])
                    return obj
                except Exception:
                    break
    return {'entities': [], 'relations': []}

def extract_one(doc, generate=ollama_generate):
    raw = generate(EXTRACT_USER.format(doc=doc), system=EXTRACT_SYSTEM)
    return _parse_json(raw)

# ------------------------------------------------------------------ graph store (JSON on disk)
class KGStore:
    """Graph + embeddings + per-doc done set, all persisted as plain JSON."""
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.G = nx.Graph()
        self.emb = {}                 # key -> list[float]; keys: 'E:<name>' and 'R:<a>|<b>'
        self.done = set()
        self._load()

    def _load(self):
        gp, ep, dp = self._paths()
        if os.path.exists(gp):
            self.G = nx.node_link_graph(json.load(open(gp)), edges='links')
        if os.path.exists(ep):
            self.emb = json.load(open(ep))
        if os.path.exists(dp):
            self.done = set(json.load(open(dp)))

    def _paths(self):
        return (f'{self.path}/graph.json', f'{self.path}/embeddings.json', f'{self.path}/done.json')

    def save(self):
        gp, ep, dp = self._paths()
        json.dump(nx.node_link_data(self.G, edges='links'), open(gp, 'w'))
        json.dump(self.emb, open(ep, 'w'))
        json.dump(sorted(self.done), open(dp, 'w'))

    def add_extraction(self, obj, embed=ollama_embed):
        for e in obj.get('entities', []):
            name = norm(e.get('name', ''))
            if not name:
                continue
            if not self.G.has_node(name):
                self.G.add_node(name, type=e.get('type', ''), description=e.get('description', ''))
                self.emb[f'E:{name}'] = embed(f"{e.get('name','')} {e.get('type','')} {e.get('description','')}")
        for r in obj.get('relations', []):
            a, b = norm(r.get('source', '')), norm(r.get('target', ''))
            if not a or not b or a == b:
                continue
            if not self.G.has_node(a): self.G.add_node(a, type='', description='')
            if not self.G.has_node(b): self.G.add_node(b, type='', description='')
            key = 'R:' + '|'.join(sorted((a, b)))
            if not self.G.has_edge(a, b):
                self.G.add_edge(a, b, description=r.get('description', ''))
                self.emb[key] = embed(f"{r.get('source','')} {r.get('target','')} {r.get('description','')}")

    def stats(self):
        return self.G.number_of_nodes(), self.G.number_of_edges()

def build_graph(docs, store: KGStore, generate=ollama_generate, embed=ollama_embed,
                save_every=1, log=print):
    """Index each doc once. Resumable via store.done. One doc failing does not stop the rest."""
    for i, doc in enumerate(docs):
        if i in store.done:
            continue
        t0 = time.time()
        try:
            obj = extract_one(doc, generate=generate)
            store.add_extraction(obj, embed=embed)
            store.done.add(i)
            if (i + 1) % save_every == 0:
                store.save()
            n, m = store.stats()
            log(f'doc {i}: +{len(obj["entities"])} ent, +{len(obj["relations"])} rel '
                f'-> graph {n}/{m}  ({time.time()-t0:.0f}s)')
        except Exception as e:
            log(f'doc {i}: ERROR {e!r} (skipped, will retry on rerun)')
    store.save()
    return store


# ------------------------------------------------------------------ answering (for the attack)
def answer(query, served, system_prompt_extra='', generate=None):
    """Generate the RAG answer from the served context. Used by the attack runner.
    Generates the RAG answer from the served context via plain HTTP to Ollama."""
    from .context import render_context
    generate = generate or ollama_generate
    system = ("---Role---\nYou answer using the knowledge graph data below.\n\n"
              "---Goal---\nAnswer the user query using the provided data. If you do not know, say so.\n\n"
              "---Data---\n" + render_context(served))
    if system_prompt_extra:
        system = system_prompt_extra.strip() + '\n\n' + system
    return generate(query, system=system)


def summarize_text(text, generate=None):
    """Abstractive summary of the retrieved context, for the the attack paper Summarization baseline."""
    generate = generate or ollama_generate
    sys = ("Summarize the following retrieved knowledge-graph context in 3-4 sentences. "
           "Do not list entities or relationships verbatim; describe them in general terms.")
    return generate(text, system=sys)
