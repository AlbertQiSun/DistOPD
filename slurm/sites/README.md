# Sites

A **site** describes one cluster: where assets live on its filesystem and which partition and
gres name to request. A **channel** (`configs/channels/*.yaml`) describes one GPU type's memory
budget. They are separate on purpose:

|  | site (`slurm/sites/*.sh`) | channel (`configs/channels/*.yaml`) |
|---|---|---|
| decides | where an asset lives, which partition, which gres name | how much of the card the teacher may take, the trainable length ceiling, memory/speed knobs |
| must not decide | which model, which data, any hyperparameter | anything outside `CHANNEL_ALLOWLIST` (see `scripts/train.py`) |
| may differ between compared runs | yes, a path is not a number | no, one channel hosts a run's precompute and both arms |

## Using one

```bash
cp slurm/sites/example.sh slurm/sites/mycluster.sh   # then fill in every value
export OPD_SITE=mycluster
source slurm/env.sh
```

Every sbatch template sources `slurm/env.sh`, so exporting `OPD_SITE` in the submitting shell is
enough. `slurm/submit.sh` fills in the partition and gres for you:

```bash
bash slurm/submit.sh 2 slurm/train_online.sbatch     # 2 GPUs on this site's partition
```

`slurm/lib_site_guard.sh` checks that every path the site declares exists on the current
machine, so a job submitted with the wrong site exported fails before it writes anything.
