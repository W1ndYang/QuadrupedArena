"""Breach Arena: fixed-goal smoke / feint game between an LLM-guided quadruped and a human.

Self-contained and isolated from the repository's earlier games (historical scoring
trial in ``scripts/`` and fixed-goal v0.4.1 / v0.5 under ``benchmarks/``). Nothing in
this package imports from those directories, and their rules do not apply here.
"""

GAME = 'breach-arena'
RULES_VERSION = 'breach-v1.0'
PROTOCOL_VERSION = 'breach-protocol-v1'
MAP_SCHEMA = 'breach-map-v1'
