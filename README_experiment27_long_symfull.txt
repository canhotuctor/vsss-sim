experiment27_long_symfull completed run bundle
Source: exp27-wave-training at 23ca9c98b80e9c45047f40cd716a1eedd6f60bae
Config: 10,000 generations; 1 lineage; 1024 envs/lineage; full symmetry, mirror augmentation off.
Completed 10000 generations and 1,310,720,000 env steps in 18219.2 seconds.
Final scheduled evaluation: 100 samples vs opponent generation 3670; mean goal delta +0.45, goals for/against 0.57/0.12.

Files:
  small_experiment27_long_symfull.tgz: config, source path, run manifest, performance log, compressed metrics, and launcher log.
  snapshots/experiment27_long_symfull_part*.tgz: all 205 snapshot JSON/weights pairs, partitioned into independently extractable archives. Each includes config.json.

Restore metadata from the small bundle under a temporary directory, copy small/experiment27_long_symfull/ into outputs/experiments/experiment27_long_symfull/, then extract every snapshot part into outputs/experiments/. The parts share the same run/lineage/snapshots paths and combine.
