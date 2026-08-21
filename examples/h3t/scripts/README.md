# Actual-running Experiments Scripts

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
bash test_8nodes_a100.sh $node_rank
```
on 8 nodes respectively to reproduce our actual-running experiment on 64 $\times$ A100, while `$node_rank` is the rank of the node. Besides, before you run `test_8nodes_a100.sh`, you should first set `$addr` as the IP address of node 0 in `_run_8nodes_a100.sh`.
