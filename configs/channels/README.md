# Hardware channels

A channel is one (cluster, GPU model, environment) triple. `configs/opd_qwen3.yaml` holds the
SCIENCE — steps, lr, global_batch, epochs, tau, seed, the data slice. A channel file holds
only what the hardware forces, and `scripts/train.py --channel <name>` merges it on top.

**A channel file may not change the science.** `apply_channel` enforces an allowlist: any key
outside it is refused, loudly, before the run starts. Without that the two files would be two
places a run's meaning could be set, and the second one would win quietly.

**One channel per experiment.** A run's precompute and BOTH its arms must use the same
channel. Different channels have different teacher engine settings, so their BackendConfig
fingerprints differ and gate 1 refuses a cross-channel cache — but the rule is broader than
what gate 1 can see, and the reasoning is in `README.md`.

The channel name is recorded in every diagnostic record and manifest, so a number can always
be traced to the hardware that produced it.
