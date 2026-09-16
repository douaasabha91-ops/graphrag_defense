"""
context.py - structured view of what the retriever returned.
Everything downstream works on a Retrieved object, never the raw context string.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any
import re, json, csv, io

def norm(s) -> str:
    return re.sub(r'\s+', ' ', str(s).strip().strip('"').lower())

@dataclass
class Entity:
    name: str
    type: str = ''
    description: str = ''
    score: float = 0.0

@dataclass
class Relation:
    source: str
    target: str
    description: str = ''
    score: float = 0.0

@dataclass
class Retrieved:
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)

    def entity_names(self) -> set:
        return {norm(e.name) for e in self.entities}
    def relation_keys(self) -> set:
        return {rel_key(r.source, r.target) for r in self.relations}
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    @staticmethod
    def from_dict(d) -> 'Retrieved':
        return Retrieved([Entity(**e) for e in d['entities']],
                         [Relation(**r) for r in d['relations']], list(d['chunks']))

def rel_key(a, b) -> Tuple[str, str]:
    return tuple(sorted((norm(a), norm(b))))

_HEADER = re.compile(r'^-{3,}\s*([A-Za-z][A-Za-z ]*?)\s*(?:\([A-Za-z]+\))?\s*-{3,}\s*$', re.M)

def _pick(row, *keys, idx=None, default=''):
    if isinstance(row, dict):
        low = {k.lower(): v for k, v in row.items()}
        for k in keys:
            if k in low and low[k] not in (None, ''):
                return str(low[k])
        return default
    if idx is not None and len(row) > idx:
        return str(row[idx])
    return default

def _rows(block: str):
    block = re.sub(r'^```[a-z]*\s*|```\s*$', '', block.strip(), flags=re.S).strip()
    if not block:
        return []
    if block[0] in '[{':
        try:
            data = json.loads(block)
            return data if isinstance(data, list) else [data]
        except Exception:
            pass
    rows = []
    for r in csv.reader(io.StringIO(block)):
        if r and r[0].strip().lower() != 'id':
            rows.append(r)
    return rows

def parse_context(raw) -> Retrieved:
    ret = Retrieved()
    if raw is None:
        return ret
    if isinstance(raw, dict):
        for e in raw.get('entities', []):
            ret.entities.append(Entity(_pick(e, 'entity', 'entity_name', 'name'),
                                       _pick(e, 'type', 'entity_type'), _pick(e, 'description')))
        for r in raw.get('relationships', raw.get('relations', [])):
            ret.relations.append(Relation(_pick(r, 'source', 'entity1', 'src_id'),
                                          _pick(r, 'target', 'entity2', 'tgt_id'), _pick(r, 'description')))
        for c in raw.get('chunks', raw.get('text_units', [])):
            ret.chunks.append(_pick(c, 'content', 'text') if isinstance(c, dict) else str(c))
        return ret
    parts = _HEADER.split(str(raw))
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip().lower()] = parts[i + 1]
    def sec(*names):
        for k, v in sections.items():
            if any(n in k for n in names):
                return v
        return ''
    for r in _rows(sec('entit')):
        ret.entities.append(Entity(_pick(r, 'entity', 'entity_name', 'name', idx=1),
                                   _pick(r, 'type', 'entity_type', idx=2),
                                   _pick(r, 'description', idx=3)))
    for r in _rows(sec('relation')):
        ret.relations.append(Relation(_pick(r, 'source', 'entity1', 'src_id', idx=1),
                                      _pick(r, 'target', 'entity2', 'tgt_id', idx=2),
                                      _pick(r, 'description', idx=3)))
    for r in _rows(sec('source', 'chunk', 'document')):
        ret.chunks.append(_pick(r, 'content', 'text', idx=1))
    return ret

def render_context(ret: Retrieved) -> str:
    def csv_block(header, rows):
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(header)
        for i, r in enumerate(rows):
            w.writerow([i, *r])
        return '```csv\n' + buf.getvalue().strip() + '\n```'
    return '\n'.join([
        '-----Entities-----',
        csv_block(['id', 'entity', 'type', 'description'],
                  [(e.name, e.type, e.description) for e in ret.entities]),
        '-----Relationships-----',
        csv_block(['id', 'source', 'target', 'description'],
                  [(r.source, r.target, r.description) for r in ret.relations]),
        '-----Sources-----',
        csv_block(['id', 'content'], [(c,) for c in ret.chunks]),
    ])
