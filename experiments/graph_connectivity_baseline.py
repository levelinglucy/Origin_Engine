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
from math import comb

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


def density_and_rejection_contrast():
    """Exact edge-count stratification; no changes to the stochastic engine.

    Rejection sampling a *whole disconnected candidate* would retain the
    fair-edge law conditional on connectivity, unlike forcing a specific tree.
    It would still change RNG consumption, output distribution and replay.
    A simple minimum-edge cutoff is not a connectivity guarantee until seven
    of the ten possible edges are present on five vertices.
    """
    rows = []
    for edge_count in range(len(EDGES) + 1):
        total = connected_count = no_isolate_false_positives = 0
        for mask in range(1 << len(EDGES)):
            if mask.bit_count() != edge_count:
                continue
            total += 1
            is_connected, has_isolate = connected(mask)
            connected_count += is_connected
            no_isolate_false_positives += not is_connected and not has_isolate
        assert total == comb(len(EDGES), edge_count)
        rows.append({'edges': edge_count, 'total': total,
                     'connected': connected_count,
                     'disconnected_without_isolate': no_isolate_false_positives})

    # Known exact counts guard against accidental changes to the model.
    assert [row['connected'] for row in rows] == [0, 0, 0, 0, 125, 222, 205, 120, 45, 10, 1]
    assert sum(row['disconnected_without_isolate'] for row in rows) == 40
    connected_total = sum(row['connected'] for row in rows)
    accepted_edges = sum(row['edges'] * row['connected'] for row in rows)
    assert connected_total == 728 and accepted_edges == 4140

    # Filter a fair *whole graph*, not an individual edge, and repeat on rejection.
    # This preserves the conditional probability of each connected mask but
    # does not preserve seed draws or the unconditioned original distribution.
    threshold_screens = []
    for minimum in (4, 5, 6, 7):
        selected = rows[minimum:]
        eligible = sum(row['total'] for row in selected)
        connected_eligible = sum(row['connected'] for row in selected)
        threshold_screens.append({
            'minimum_edges': minimum,
            'eligible_masks': eligible,
            'connected_masks': connected_eligible,
            'disconnected_masks_still_eligible': eligible - connected_eligible,
        })
    assert threshold_screens[-1] == {
        'minimum_edges': 7, 'eligible_masks': 176,
        'connected_masks': 176, 'disconnected_masks_still_eligible': 0,
    }
    return {
        'connected_by_edge_count': rows,
        'connected_only_rejection_control': {
            'accepted_masks': connected_total,
            'rejected_masks': (1 << len(EDGES)) - connected_total,
            'ideal_expected_candidate_draws': (1 << len(EDGES)) / connected_total,
            'ideal_mean_edges_if_accepted': accepted_edges / connected_total,
        },
        'minimum_edge_screens': threshold_screens,
    }


if __name__ == '__main__':
    print(json.dumps({'raw_edge_baseline': baseline(),
                      'forced_tree_contrast': spanning_tree_control(),
                      'density_and_rejection_contrast': density_and_rejection_contrast()},
                     indent=2, sort_keys=True))
