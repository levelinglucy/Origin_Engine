#!/usr/bin/env python3
"""Exhaustive topology baseline for Origin Engine's five-node Φ⁴ graph.

Source inspected: origin_engine_monolith.py, blob 502af09e00b713724bf4dd3302891d29c4eac4a0.
In grammar_phase4plus_with_graph, each of the ten possible undirected edges
is independently included when active_rng.random() < 0.5. This enumerates
all 2**10 edge-selection outcomes at equal ideal-model probability; it does
NOT import the engine, model its novelty-based candidate selection, validate
its graph writer, or estimate frequencies across concrete PRNG seeds.

Usage: python3 experiments/graph_connectivity_baseline.py
"""

import json
from collections import Counter

NODES = 5
EDGES = tuple((u, v) for u in range(NODES) for v in range(u + 1, NODES))


def connected(edge_mask, forced_edges=()):
    """Check whole-graph connectivity, including isolated vertices."""
    adjacency = [set() for _ in range(NODES)]
    for i, (u, v) in enumerate(EDGES):
        if (edge_mask >> i) & 1 or (u, v) in forced_edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
    visited, frontier = {0}, [0]
    while frontier:
        u = frontier.pop()
        for v in adjacency[u] - visited:
            visited.add(v)
            frontier.append(v)
    return len(visited) == NODES, any(not neighbours for neighbours in adjacency)


def baseline():
    """Count exactly the outcomes under the documented independent-edge rule."""
    if len(EDGES) != 10:
        raise AssertionError('Expected exactly ten candidate edges')
    counts = Counter()
    edge_total = 0
    for mask in range(1 << len(EDGES)):
        is_connected, has_isolated = connected(mask)
        edge_total += mask.bit_count()
        counts['connected' if is_connected else 'disconnected'] += 1
        if not is_connected:
            counts['disconnected_with_isolate' if has_isolated
                   else 'disconnected_without_isolate'] += 1
    # Explicit regression checks make this a reproducible result rather than
    # a speculative statement about sampled outcomes.
    assert counts['connected'] == 728
    assert counts['disconnected'] == 296
    assert counts['disconnected_with_isolate'] == 256
    assert counts['disconnected_without_isolate'] == 40
    assert edge_total == 5120  # 1024 outcomes * 5 mean selected edges
    return {
        'assumption': 'five labelled vertices, 10 independent fair candidate edges',
        'outcomes': 1 << len(EDGES),
        'connected': counts['connected'],
        'disconnected': counts['disconnected'],
        'disconnected_with_isolate': counts['disconnected_with_isolate'],
        'disconnected_without_isolate': counts['disconnected_without_isolate'],
        'ideal_connectivity_fraction': f"{counts['connected']}/1024",
        'ideal_mean_edges': edge_total / 1024,
    }


def spanning_tree_control():
    """Exhibit a contrasting design, NOT a change to Origin Engine."""
    tree = {(u, u + 1) for u in range(NODES - 1)}
    optional = tuple(edge for edge in EDGES if edge not in tree)
    connected_count = 0
    edge_total = 0
    for mask in range(1 << len(optional)):
        extra = {edge for i, edge in enumerate(optional) if (mask >> i) & 1}
        is_connected, _ = connected(0, tree | extra)
        connected_count += is_connected
        edge_total += len(tree) + len(extra)
    assert len(optional) == 6
    assert connected_count == 64
    assert edge_total == 448
    return {'outcomes': 64, 'connected': connected_count,
            'ideal_mean_edges': edge_total / 64}


if __name__ == '__main__':
    print(json.dumps({'raw_edge_baseline': baseline(),
                      'forced_tree_contrast': spanning_tree_control()},
                     indent=2, sort_keys=True))
