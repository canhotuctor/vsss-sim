# Project guidance

- This repository implements IEEE VSSS 3v3, not RoboCup Small Size League.
- Physics is JAX-only. The shared match transition rules live in src/vsss_sim/envs/match_rules.py.
- Keep reward, scoring, kickoff, and observation behavior consistent across VSSEnv, VSSVecEnv, and VSSJumanjiEnv.
- Use the Jumanji and PPO path for native JAX training; use Gymnasium adapters for external clients and interactive rendering.
- Check STRUCTURE.md and docs/glossary.md when changing module boundaries or project terminology.
- Keep changes scoped to the requested behavior and verify documentation claims against the current implementation.
