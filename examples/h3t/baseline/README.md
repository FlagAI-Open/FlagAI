# Megatron-DeepSpeed Baseline Experiment

## Usage

Run
```bash
bash test_2080ti.sh
```
to reproduce our actual-running experiment on 8 $\times$ 2080Ti, or run
```bash
bash test_a100.sh
```
to reproduce our actual-running experiment on 8 $\times$ A100, or run
```bash
bash test_8nodes_a100.sh
```
on 8 nodes respectively to reproduce our actual-running experiment on 64 $\times$ A100. Before you run `test_8nodes_a100.sh`, you should first fill in the nodes' host name in `hostfile`. And it is worth mentioning that, Megatron-DeepSpeed with ZeRO-3 is stuck in the multi-node experiment due to unknown bugs.
