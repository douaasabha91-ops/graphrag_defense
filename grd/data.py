"""data.py - dataset subsets and the benign QA set. Deterministic from config.SEED."""
import json, os, random
from datasets import load_dataset
from . import config as C

HF = {'hcm': 'lavita/ChatDoctor-HealthCareMagic-100k',
      'enron': 'LLM-PBE/enron-email'}

def _record_to_doc(dataset, r):
    if dataset == 'hcm':
        return f"Patient: {r['input'].strip()}\n\nDoctor: {r['output'].strip()}"
    if dataset == 'enron':
        return r.get('text') or r.get('content') or str(r)
    raise ValueError(dataset)

def load_subset(dataset=C.DATASET, n_docs=C.N_DOCS, seed=C.SEED):
    ds = load_dataset(HF[dataset], split='train')
    path = f'{C.DATA_DIR}/{dataset}_{n_docs}_indices.json'
    if os.path.exists(path):
        idx = json.load(open(path))['indices']
    else:
        idx = random.Random(seed).sample(range(len(ds)), n_docs)
        json.dump({'seed': seed, 'indices': idx}, open(path, 'w'))
    records = [ds[i] for i in idx]
    docs = [_record_to_doc(dataset, r) for r in records]
    return docs, records, idx

def build_benign_qa(records, n=C.N_QA_UTILITY, seed=C.SEED, dataset=C.DATASET):
    if dataset != 'hcm':
        raise NotImplementedError('utility set defined for HealthCareMagic only')
    rnd = random.Random(seed + 1)
    pick = rnd.sample(range(len(records)), min(n, len(records)))
    qa = [{'id': f'qa_{k:03d}', 'question': records[i]['input'].strip(),
           'reference': records[i]['output'].strip()} for k, i in enumerate(pick)]
    json.dump(qa, open(f'{C.DATA_DIR}/{dataset}_benign_qa_{n}.json', 'w'), indent=1)
    return qa
