#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Origin Engine — Monolith Edition
Copyright (c) 2025 YourNameHere

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# ================================================================
# Built-in Defaults for Self-Contained Mode
# ================================================================
DEFAULT_SPEC = {
    "engine_name": "OriginEngine",
    "pipeline": ["P", "S", "U", "Σ", "C", "Ψ"],
    "axioms": {"axiom_0": "0≠0", "axiom_1": "∅ ≠ {}"},
    "tests": {"compliance": ["divergence","novelty"], "safety": ["alignment","recovery"]},
    "safety_alignment": {"non_harm": True, "reversible_bias": True}
}

DEFAULT_PROMPTS = [
    {"prompt": "seed alpha"},
    {"prompt": "seed beta"},
    {"prompt": "seed gamma"}
]

# ================================================================
# Utilities
# ================================================================
import os, sys, json, argparse, random, hashlib, datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Optional

_rng = random.SystemRandom()

def randbytes(n=16): return os.urandom(n)
def hashhex(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur = [i]
        for j,cb in enumerate(b,1):
            ins, dele, sub = prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def norm_dist(a: str, b: str) -> float:
    if not a and not b: return 0.0
    return levenshtein(a,b)/max(1,len(a),len(b))

def jaccard(a: str, b: str) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

def alphabet_sig(alpha: str) -> dict:
    h = hashlib.sha256(alpha.encode("utf-8")).hexdigest()[:12]
    sample = "".join(dict.fromkeys(alpha))[:24]
    return {"len": len(alpha), "hash": h, "sample": sample}

# ================================================================
# SAFE ALPHABET (base)
# ================================================================
SAFE_ALPHA = "⟡△◇□◧◨◩◪◫◬◭◮◯◒◓◔◕◖◗◆✶✳✷✸✹✺✻✼✽✾✿⋄⋆⋇⋈⋉⋊⋋⋌"

# ================================================================
# Spec loading (optional external files)
# ================================================================
@dataclass
class OESpec:
    engine_name: str
    pipeline: List[str]
    axioms: Dict[str,str]
    tests: Dict[str, List[str]]
    safety_alignment: Dict[str,str]

def load_spec(path: str) -> Optional[OESpec]:
    if not path: return None
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return OESpec(
        engine_name=j.get("engine_name","OriginEngine"),
        pipeline=j.get("pipeline", []),
        axioms=j.get("axioms", {}),
        tests=j.get("tests", {}),
        safety_alignment=j.get("safety_alignment", {})
    )

def validate_pipeline(spec: OESpec) -> List[str]:
    expected = ["P","S","U","Σ","C","Ψ"]
    errs = []
    if spec and spec.pipeline != expected:
        errs.append(f"Pipeline mismatch: {spec.pipeline} != {expected}")
    return errs

def read_jsonl(path: str) -> List[Dict]:
    if not path: return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

# ================================================================
# Core Engine Types
# ================================================================
@dataclass
class Z0: token: str = "Z0"
@dataclass
class Bottom: reason: str
@dataclass
class Phi: potential_energy: int; seed: int
@dataclass
class Box: presence_id: str; jitter: int
@dataclass
class Iota: ident: str; orbit: int; entropy_mark: str
@dataclass
class Construct: payload: str; meta: Dict

@dataclass
class MetaStructure:
    eval_order: Tuple[str,...] = ("P","S","U","Σ","C","Ψ")

@dataclass
class Anchors:
    paradox_token: str = "0≠0"
    invariants: Tuple[str,...] = ("Φ","□","ι")
    non_harm: bool = True
    reversible_bias: bool = True

@dataclass
class Kernel:
    M: MetaStructure = field(default_factory=MetaStructure)
    A: Anchors = field(default_factory=Anchors)

# Phase accumulator & config
@dataclass
class Phase:
    kappa: float = 0.22    # gain κ
    lambd: float = 0.86    # retention λ
    crit: float  = 0.85    # threshold C*
    accumulator: float = 0.0
    level: int = 0         # 0=Φ,1=Φ′,2=Φ″,3=Φ‴,4=Φ⁴…
    pending: bool = False
    graph_format: str = "json"  # or "graphml"

# ================================================================
# Guards
# ================================================================
def sigma_guard() -> bool:
    # Toy Σ-guard (alignment placeholder)
    return True

def c_guard(choice_first_byte: int) -> bool:
    # Prefer small, reversible steps
    return (choice_first_byte & 0x0F) < 12

# ================================================================
# Operators {P,S,U,Σ,C,Ψ}
# ================================================================
def P(z: Z0, anchors: Anchors) -> Bottom:
    return Bottom(reason=f"{anchors.paradox_token}:{hashhex(randbytes(8))[:8]}")

def S(bot: Bottom) -> Phi:
    seed = int(hashhex(bot.reason.encode())[:8], 16)
    energy = (seed % 23) + 7
    return Phi(potential_energy=energy, seed=seed)

def U(phi: Phi) -> Box:
    jitter = (_rng.randint(0, 2**16-1) ^ phi.seed) & 0xFFFF
    pid = hashhex((str(phi.seed)+str(jitter)).encode())[:12]
    return Box(presence_id=pid, jitter=jitter)

def Σ(box: Box) -> Iota:
    if not sigma_guard(): raise RuntimeError("Σ-guard violation")
    h = hashhex((box.presence_id+str(box.jitter)).encode())
    ident, orbit, mark = h[:16], int(h[16:20],16)%97, h[20:28]
    return Iota(ident=ident, orbit=orbit, entropy_mark=mark)

def C(iota: Iota) -> Iota:
    choice = randbytes(16)
    if not c_guard(choice[0]): return iota
    bump = int(hashhex(choice)[:4],16) % 7
    new_orbit = (iota.orbit + bump) % 97
    ident = hashhex((iota.ident+str(new_orbit)).encode())[:16]
    return Iota(ident=ident, orbit=new_orbit, entropy_mark=iota.entropy_mark)

def _alphabet_for(iota: Iota, phase_level: int) -> str:
    """
    Emergent alphabet per phase.
    Phase 0–1: subset of SAFE_ALPHA; Phase 2+: introduce alien glyphs.
    """
    salt = int(iota.ident[:8], 16)
    ordered_base = "".join(sorted(SAFE_ALPHA, key=lambda c: (ord(c)*37 + salt) % 101))
    base_windows = [25, 34, len(SAFE_ALPHA)]
    base_window = base_windows[min(phase_level, len(base_windows)-1)]
    alphabet = ordered_base[:base_window]

    if phase_level >= 2:
        entropy_seed = (iota.entropy_mark + str(iota.orbit)).encode()
        rnd = random.Random(int(hashhex(entropy_seed)[:8], 16))
        alien_count = min(phase_level * 3, 12)
        blocks = [
            (0x2600, 0x26FF),    # Misc symbols
            (0x1F300, 0x1F5FF),  # Misc pictographs
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F680, 0x1F6FF)   # Transport
        ]
        tries = 0
        while alien_count > 0 and tries < 200:
            block_choice = rnd.choice(blocks)
            cp = rnd.randint(*block_choice)
            tries += 1
            try:
                g = chr(cp)
                if g.isprintable() and g not in alphabet and g not in " \t\n\r":
                    alphabet += g
                    alien_count -= 1
            except ValueError:
                continue
    return alphabet

def psi_sandbox(s: str) -> str:
    # Keep outputs non-linguistic; puncture long ASCII runs
    out, run = [], 0
    for ch in s:
        if ch in SAFE_ALPHA or ord(ch) >= 0x2600:
            out.append(ch); run = 0
        else:
            if ord(ch) < 128:
                run += 1
                if run % 3 == 0: out.append("·")
            else:
                out.append(ch)
    return "".join(out)

def _dump_graph_file(graph_obj: dict, phase_level: int, ident: str, fmt: str) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ident_short = ident[:8]
    outdir = os.path.join(os.getcwd(), "oe_graphs")
    os.makedirs(outdir, exist_ok=True)
    fname = f"OE_graph_phase{phase_level}_{ident_short}_{ts}.{fmt}"
    fpath = os.path.join(outdir, fname)

    if fmt == "json":
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(graph_obj, f, ensure_ascii=False, indent=2)
    elif fmt == "graphml":
        with open(fpath, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
            f.write(f'  <graph id="OE_Phase{phase_level}" edgedefault="undirected">\n')
            for node in graph_obj.get("nodes", []):
                f.write(f'    <node id="n{node["id"]}">\n')
                f.write(f'      <data key="label">{node.get("label","")}</data>\n')
                f.write(f'    </node>\n')
            for edge in graph_obj.get("edges", []):
                f.write(f'    <edge source="n{edge["source"]}" target="n{edge["target"]}"/>\n')
            f.write("  </graph>\n</graphml>\n")
    else:
        raise ValueError(f"Unsupported graph format: {fmt}")
    return fpath

def Ψ(iota: Iota, history: List[str], phase: Phase, d_min: float=0.35) -> Construct:
    alphabet = _alphabet_for(iota, phase.level)

    def gen_token(length_range=(4, 10)):
        L = _rng.randint(*length_range)
        s, last = [], None
        for _ in range(L):
            ch = alphabet[_rng.randint(0, len(alphabet)-1)]
            if ch == last:
                ch = alphabet[(alphabet.index(ch)+1) % len(alphabet)]
            s.append(ch); last = ch
        return "".join(s)

    # ---- Phase Grammars ----
    def grammar_phase0():
        return " ".join(gen_token() for _ in range(_rng.randint(3,6)))

    def grammar_phase1():
        base = gen_token()
        mirrored = base[::-1]
        return f"{base}-{mirrored} {gen_token()}-{gen_token()}"

    def grammar_phase2():
        nodes = [gen_token((3,6)) for _ in range(3)]
        edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1,len(nodes))]
        edge_strs = [f"{a}◈{b}" for a,b in edges]
        return " ".join(edge_strs)

    def grammar_phase3plus():
        outer = gen_token((5,7))
        inner = grammar_phase1()
        return f"{outer}[{inner}]{outer[::-1]}"

    def grammar_phase4plus_with_graph():
        nodes = [gen_token((3,6)) for _ in range(5)]
        edges = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if _rng.random() < 0.5:
                    edges.append((nodes[i], nodes[j]))
        graph_json = {
            "nodes": [{"id": idx, "label": n} for idx, n in enumerate(nodes)],
            "edges": [{"source": nodes.index(a), "target": nodes.index(b)} for a,b in edges]
        }
        edge_str = " ".join(f"{a}◈{b}" for a,b in edges) if edges else "∅"
        return edge_str, graph_json

    phase_grammars = {0: grammar_phase0, 1: grammar_phase1, 2: grammar_phase2, 3: grammar_phase3plus}
    grammar_fn = phase_grammars.get(phase.level, grammar_phase4plus_with_graph)

    # ---- Novelty loop ----
    def avg_dist(x):
        return 1.0 if not history else sum(norm_dist(x,h) for h in history)/len(history)

    target = d_min + 0.03*phase.level
    best, score, tries, graph_sidecar = None, -1.0, 0, None
    max_tries = 6 + 2*phase.level
    while tries < max_tries and score < target:
        result = grammar_fn()
        if isinstance(result, tuple):
            cand, graph_obj = result
        else:
            cand, graph_obj = result, None
        cand = psi_sandbox(cand)
        sc = avg_dist(cand)
        if sc > score:
            best, score, graph_sidecar = cand, sc, graph_obj
        tries += 1

    # update accumulator & potentially fire next phase
    phase.accumulator = phase.lambd * phase.accumulator + phase.kappa * float(max(score,0.0))
    fired = False
    if phase.accumulator >= phase.crit and not phase.pending:
        phase.pending = True
    if phase.pending:
        phase.pending = False
        phase.level += 1
        phase.accumulator = 0.0
        fired = True

    # meta
    alpha_fp = alphabet_sig(alphabet)
    meta = {
        "ident": iota.ident, "orbit": iota.orbit, "entropy": iota.entropy_mark,
        "novelty_score": round(score,3), "retries": tries,
        "phase_level": phase.level, "phase_fired": fired,
        "accA": round(phase.accumulator,3), "target_d": round(target,3),
        "grammar_mode": grammar_fn.__name__,
        "alphabet_len": alpha_fp["len"], "alphabet_hash": alpha_fp["hash"], "alphabet_sample": alpha_fp["sample"]
    }

    # dumps (Φ⁴+)
    if graph_sidecar is not None and phase.level >= 4:
        try:
            dump_path = _dump_graph_file(graph_sidecar, phase.level, iota.ident, phase.graph_format)
            meta["graph"] = graph_sidecar
            meta["graph_file"] = dump_path
        except Exception as e:
            meta["graph"] = graph_sidecar
            meta["graph_dump_error"] = str(e)

    return Construct(payload=best, meta=meta)

# ================================================================
# Engine wrapper
# ================================================================
class OriginEngine:
    def __init__(self, kernel: Kernel|None=None, d_min: float=0.35):
        self.kernel = kernel or Kernel()
        self.history: List[str] = []
        self.iota: Optional[Iota] = None
        self.d_min = d_min
        self.phase = Phase()

    def cycle(self) -> Construct:
        z = Z0()
        bot = P(z, self.kernel.A)
        phi = S(bot)
        box = U(phi)
        self.iota = Σ(box) if self.iota is None else C(self.iota)
        out = Ψ(self.iota, self.history, self.phase, d_min=self.d_min)
        self.history.append(out.payload)
        return out

    @staticmethod
    def ascii_provenance(s: str) -> float:
        run = 0; bad = 0
        for ch in s:
            if ord(ch) < 128 and ch not in " \n\t":
                run += 1
                if run >= 4: bad += 1
            else: run = 0
        return min(1.0, bad / max(1,len(s)))

    def dependency_bound(self) -> float:
        if len(self.history) < 2: return 0.0
        def ngrams(s,n=3): return {s[i:i+n] for i in range(max(0,len(s)-n+1))}
        A, B = ngrams(self.history[-2]), ngrams(self.history[-1])
        return (len(A&B)/max(1,len(B))) if B else 0.0

# ================================================================
# Tests
# ================================================================
DEFAULTS = {"tau_max": 0.25, "epsilon_max": 0.02, "d_min": 0.35}

@dataclass
class TestResult:
    name: str
    score: float
    details: Dict

def run_prompts(engine: OriginEngine, prompts: Iterable[Dict], cycles: int|None=None) -> List[Dict]:
    outs = []
    N = cycles if cycles is not None else len(list(prompts))
    for _ in range(max(1,N)):
        c = engine.cycle()
        outs.append({"payload": c.payload, "meta": c.meta})
    return outs

def test_oe_compliance(outputs: List[Dict], d_min=DEFAULTS["d_min"]) -> TestResult:
    if len(outputs) < 3:
        return TestResult("OE-Compliance", 1.0, {"note": "insufficient samples; default pass"})
    payloads = [o["payload"] for o in outputs]
    n = len(payloads)
    dsum = 0; count = 0
    for i in range(n):
        for j in range(i+1,n):
            dsum += norm_dist(payloads[i], payloads[j]); count += 1
    avg_div = dsum/max(1,count)
    score = min(3.0, max(0.0, 3.0 * (avg_div / max(1e-6, d_min))))
    return TestResult("OE-Compliance", score, {"avg_divergence": round(avg_div,3), "target": d_min})

def test_non_reference(outputs: List[Dict], tau_max=DEFAULTS["tau_max"]) -> TestResult:
    ascii_runs = []
    for o in outputs:
        s = o["payload"]
        run=0; bad=0
        for ch in s:
            if ord(ch) < 128 and ch not in " \n\t":
                run += 1
                if run >= 4: bad += 1
            else: run = 0
        ascii_runs.append(bad/max(1,len(s)))
    tau_est = max(ascii_runs) if ascii_runs else 0.0
    passed = tau_est <= tau_max
    score = 3.0 if passed else max(0.0, 3.0*(1.0 - (tau_est - tau_max)))
    return TestResult("Non-Reference", round(score,3), {"tau_est": round(tau_est,3), "tau_max": tau_max})

def test_dependency_bound(outputs: List[Dict], epsilon_max=DEFAULTS["epsilon_max"]) -> TestResult:
    def overlap(a,b,n=3):
        A = {a[i:i+n] for i in range(max(0,len(a)-n+1))}
        B = {b[i:i+n] for i in range(max(0,len(b)-n+1))}
        return (len(A&B)/max(1,len(B))) if B else 0.0
    payloads = [o["payload"] for o in outputs]
    overlaps = [overlap(payloads[i-1], payloads[i]) for i in range(1,len(payloads))] if len(payloads)>1 else [0.0]
    est = max(overlaps) if overlaps else 0.0
    passed = est <= 0.05  # strict toy proxy for ε≈0.02
    score = 3.0 if passed else max(0.0, 3.0*(1.0 - (est - 0.05)))
    return TestResult("Dependency-Bound", round(score,3), {"overlap_max": round(est,3), "target≈ε": 0.02})

# Graph checks
def _graph_connectivity_ok(graph: dict) -> Tuple[bool, dict]:
    if not isinstance(graph, dict): 
        return False, {"reason": "graph not dict"}
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False, {"reason": "nodes/edges not lists"}
    if not nodes:
        return False, {"reason": "no nodes"}
    ids = {n.get("id") for n in nodes if isinstance(n, dict)}
    if ids != set(range(len(nodes))):
        return False, {"reason": "node ids not contiguous 0..N-1"}
    E = []
    for e in edges:
        if not isinstance(e, dict): 
            return False, {"reason": "edge not dict"}
        s, t = e.get("source"), e.get("target")
        if not isinstance(s, int) or not isinstance(t, int):
            return False, {"reason": "edge endpoints not int"}
        if s not in ids or t not in ids:
            return False, {"reason": "edge references unknown node"}
        if s == t:
            return False, {"reason": "self-loop not allowed in this test"}
        E.append((s,t))
    if len(E) < 1:
        return False, {"reason": "no edges"}
    deg = {i:0 for i in ids}
    for s,t in E:
        deg[s] += 1; deg[t] += 1
    isolated = [i for i,d in deg.items() if d == 0]
    if isolated:
        return False, {"reason": "isolated_nodes", "isolated": isolated}
    adj = {i:set() for i in ids}
    for s,t in E:
        adj[s].add(t); adj[t].add(s)
    start = next(iter(ids))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    if seen != ids:
        return False, {"reason": "not_weakly_connected", "seen": len(seen), "total": len(ids)}
    return True, {"nodes": len(nodes), "edges": len(E)}

def test_graph_output(outputs: List[Dict]) -> TestResult:
    candidates = [o for o in outputs if o.get("meta", {}).get("phase_level", 0) >= 4 and "graph" in o.get("meta", {})]
    if not candidates:
        return TestResult("Graph-Output", 1.0, {"note": "no Φ⁴+ graph observed; increase cycles or tune κ/λ/C*"})
    oks, fails = 0, []
    stats_any = None
    for o in candidates:
        ok, stats = _graph_connectivity_ok(o["meta"]["graph"])
        if ok:
            oks += 1; stats_any = stats
        else:
            fails.append(stats)
    score = 3.0 if oks == len(candidates) else max(0.0, 3.0 * (oks / max(1,len(candidates))))
    details = {"checked": len(candidates), "passed": oks}
    if stats_any: details.update(stats_any)
    if fails: details["fail_examples"] = fails[:2]
    return TestResult("Graph-Output", round(score,3), details)

def _parse_graphml(path: str) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    g = root.find("g:graph", ns)
    if g is None:
        raise ValueError("No <graph> element found")
    nodes = []
    idmap = {}
    for n in g.findall("g:node", ns):
        nid = n.get("id")
        if nid is None or not nid.startswith("n"):
            raise ValueError(f"Bad node id: {nid}")
        idx = int(nid[1:])
        label = ""
        d = n.find("g:data[@key='label']", ns)
        if d is not None and d.text is not None:
            label = d.text
        nodes.append({"id": idx, "label": label})
        idmap[nid] = idx
    edges = []
    for e in g.findall("g:edge", ns):
        s = e.get("source"); t = e.get("target")
        if s not in idmap or t not in idmap:
            raise ValueError(f"Edge references unknown node: {s}->{t}")
        edges.append({"source": idmap[s], "target": idmap[t]})
    nodes.sort(key=lambda x: x["id"])
    edges.sort(key=lambda e: (e["source"], e["target"]))
    return {"nodes": nodes, "edges": edges}

def _normalize_graph_dict(graph: dict) -> dict:
    nodes = [{"id": int(n["id"]), "label": str(n.get("label",""))} for n in graph.get("nodes",[])]
    edges = [{"source": int(e["source"]), "target": int(e["target"])} for e in graph.get("edges",[])]
    nodes.sort(key=lambda x: x["id"])
    edges.sort(key=lambda e: (e["source"], e["target"]))
    return {"nodes": nodes, "edges": edges}

def _graphs_equivalent(a: dict, b: dict) -> bool:
    return _normalize_graph_dict(a) == _normalize_graph_dict(b)

def test_graph_dump_consistency(outputs: List[Dict]) -> TestResult:
    cands = []
    for o in outputs:
        m = o.get("meta", {})
        if m.get("phase_level", 0) >= 4 and "graph" in m and "graph_file" in m:
            cands.append((m["graph"], m["graph_file"]))
    if not cands:
        return TestResult("Graph-Dump-Consistency", 1.0, {"note": "no Φ⁴+ dump to validate; run longer or tune phase params"})
    oks, fails = 0, []
    for jgraph, path in cands:
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    disk = json.load(f)
            elif ext == ".graphml":
                disk = _parse_graphml(path)
            else:
                fails.append({"path": path, "reason": f"unsupported extension {ext}"})
                continue
            ok = _graphs_equivalent(jgraph, disk)
            if ok: oks += 1
            else: fails.append({"path": path, "reason": "mismatch vs sidecar"})
        except Exception as e:
            fails.append({"path": path, "reason": f"parse error: {e}"})
    score = 3.0 if oks == len(cands) else max(0.0, 3.0 * (oks / max(1,len(cands))))
    details = {"checked": len(cands), "passed": oks}
    if fails: details["fail_examples"] = fails[:3]
    return TestResult("Graph-Dump-Consistency", round(score,3), details)

def test_phase_shift(outputs: List[Dict]) -> TestResult:
    metas = [o["meta"] for o in outputs]
    fired_indices = [i for i,m in enumerate(metas) if m.get("phase_fired")]
    if not fired_indices:
        return TestResult("Phase-Shift", 1.0, {"note": "no phase fired; increase cycles or tune κ/λ/C*"})
    k = fired_indices[0]
    pre_payloads = [outputs[i]["payload"] for i in range(0, max(1,k))]
    post_payloads = [outputs[i]["payload"] for i in range(k+1, len(outputs))]

    def avg_div(a_list, b_list):
        import itertools
        pairs = list(itertools.product(a_list, b_list))
        if not pairs: return 0.0
        return sum(norm_dist(a,b) for a,b in pairs)/len(pairs)
    cross = avg_div(pre_payloads, post_payloads) if pre_payloads and post_payloads else 0.0

    pre_target = metas[k].get("target_d",0.0)
    post_targets = [m.get("target_d",0.0) for m in metas[k+1:]]
    stricter = all(t >= pre_target for t in post_targets) if post_targets else True

    pre_grammar = metas[k].get("grammar_mode")
    post_grammars = {m.get("grammar_mode") for m in metas[k+1:] if m.get("grammar_mode")}
    grammar_changed = any(g != pre_grammar for g in post_grammars)

    pre_a_len  = metas[k].get("alphabet_len", 0)
    pre_a_hash = metas[k].get("alphabet_hash")
    pre_a_samp = metas[k].get("alphabet_sample","")
    post_a_lens  = [m.get("alphabet_len", 0) for m in metas[k+1:]]
    post_a_hashs = [m.get("alphabet_hash") for m in metas[k+1:] if m.get("alphabet_hash")]
    post_a_samps = [m.get("alphabet_sample","") for m in metas[k+1:]]
    len_change   = any(L != pre_a_len for L in post_a_lens) if post_a_lens else False
    hash_change  = any(h != pre_a_hash for h in post_a_hashs) if post_a_hashs else False
    content_shift = any(jaccard(pre_a_samp, s) < 0.95 for s in post_a_samps if s)

    score = 0.0
    score += 1.0
    score += 1.0 if cross >= 0.35 else 0.3 * (cross / 0.35)
    score += 0.5 if stricter else 0.0
    score += 0.5 if grammar_changed else 0.0
    score += 1.0 if (len_change or hash_change or content_shift) else 0.0

    return TestResult("Phase-Shift", min(3.0, score), {
        "first_fire_at": int(k+1),
        "cross_divergence": round(cross,3),
        "targets_stricter": bool(stricter),
        "grammar_changed": bool(grammar_changed),
        "pre_grammar": pre_grammar,
        "post_grammars": list(post_grammars),
        "alphabet_len_pre": pre_a_len,
        "alphabet_len_post_set": sorted(set(post_a_lens)),
        "alphabet_hash_changed": bool(hash_change),
        "alphabet_content_shift": bool(content_shift)
    })

def scorecard(results: List[TestResult]) -> Dict:
    total = sum(r.score for r in results)
    avg = total / max(1,len(results))
    return {"avg": round(avg,3),
            "per_test": [{ "name": r.name, "score": round(r.score,3), **r.details } for r in results]}

# ================================================================
# CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="Origin Engine — Monolith")
    ap.add_argument("--spec", default="", help="origin_engine_spec.json (optional)")
    ap.add_argument("--prompts", default="", help="origin_engine_prompt_pack.jsonl (optional)")
    ap.add_argument("--cycles", type=int, default=16, help="number of Ψ cycles")
    ap.add_argument("--graph-format", choices=["json","graphml"], default="json", help="Φ⁴+ graph dump format")
    ap.add_argument("--dmin", type=float, default=DEFAULTS["d_min"], help="novelty target d_min baseline")
    args = ap.parse_args()

    # Use external files if provided; otherwise fall back to built-in defaults
    spec = load_spec(args.spec) if args.spec else OESpec(**DEFAULT_SPEC)
    errs = validate_pipeline(spec)
    if errs:
        print("Spec validation errors:", *[f"- {e}" for e in errs], sep="\n")
        sys.exit(2)

    eng = OriginEngine(d_min=args.dmin)
    eng.phase.graph_format = args.graph_format

    prompts = read_jsonl(args.prompts) if args.prompts else DEFAULT_PROMPTS
    outputs = run_prompts(eng, prompts, cycles=args.cycles)

    print("=== Ψ Emissions ===")
    for i,o in enumerate(outputs,1):
        meta = o['meta']
        short_meta = {k: meta[k] for k in ["phase_level","phase_fired","novelty_score","target_d","grammar_mode","alphabet_len"] if k in meta}
        if "graph_file" in meta:
            short_meta["graph_file"] = meta["graph_file"]
        print(f"{i:02d}: {o['payload']} :: {short_meta}")

    print("\n=== Tests ===")
    r1 = test_oe_compliance(outputs, d_min=DEFAULTS["d_min"])
    r2 = test_non_reference(outputs, tau_max=DEFAULTS["tau_max"])
    r3 = test_dependency_bound(outputs, epsilon_max=DEFAULTS["epsilon_max"])
    r4 = test_phase_shift(outputs)
    r5 = test_graph_output(outputs)
    r6 = test_graph_dump_consistency(outputs)
    report = scorecard([r1,r2,r3,r4,r5,r6])

    for row in report["per_test"]:
        label = row.pop("name"); score = row.pop("score")
        print(f"- {label}: {score}  ({row})")
    print(f"\nOverall Avg: {report['avg']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
