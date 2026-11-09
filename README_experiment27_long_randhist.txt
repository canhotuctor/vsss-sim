experiment27_long_randhist completed run bundle
Source: exp27-wave-training at ff8f5345f8b98fc82287b1095744bf06ea9fee94
Config: 10,000 generations; 1 lineage; 1024 envs/lineage; random-history opponents; full symmetry augmentation; simplified referee.
Completed 10000 generations and 1,310,720,000 env steps in 28968.7 seconds (45,246.1 env steps/s overall).

Files:
  small_experiment27_long_randhist.tgz: config, source path, run manifest, performance log, compressed metrics, and launcher log.
  snapshots/experiment27_long_randhist_part*.tgz: all 201 snapshot JSON/weights pairs, partitioned into independently extractable archives. Each includes config.json.
  SHA256SUMS_experiment27_long_randhist.txt: SHA-256 checksums for the small bundle and every snapshot part.

No separate final evaluation report was produced in this run output.
Restore the small bundle in a temporary directory, copy small/experiment27_long_randhist/ into outputs/experiments/experiment27_long_randhist/, gunzip metrics.jsonl.gz to metrics.jsonl, then extract every snapshot part into outputs/experiments/. The parts share the same run/lineage/snapshots paths and combine.
