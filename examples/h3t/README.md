# H3T: Efficient Integration of Memory Optimization and Parallelism for High-Throughput Transformer Training

## Reproduce Actual-running Experiments in the Paper

See `scripts/README.md` and `baseline/README.md`.

We package the conda environments we used for our actual-running experiments. You can download them [here](https://cloud.tsinghua.edu.cn/d/5ca3763ffc4c44ccaaaf/). Here `env_baseline.tar.gz` is used for baseline experiment, and `env_h3t.tar.gz` is used for H3T.

## Usage

We maximize compatibility with the usage of BMTrain, except for its initialization way:
```python
# If you do not want to use the H3T solver, you can manually specify the optimization switches globally. Here `zero_level` can be 2 or 3, while `offload_parameter`, `checkpointing`, and `offload_hidden_state` can be True or False
bmt.init_distributed(
    seed = seed,
    nvlink_available = nvlink_available,
    zero_level = zero_level,
    offload_parameter = offload_parameter,
    checkpointing = checkpointing,
    offload_hidden_state = offload_hidden_state,
    tbl_auto_optimization = False, # disable H3T solver
)
```
or
```python
# If you want to use the H3T solver, you should specify the solver algorithm and the memory limit. Here `alg_name` can be "random", "greedy", or "dp", while the unit of the memory limit is byte. We suggest the `memory_limit` should be less than the memory capacity of the device.
bmt.init_distributed(
    seed=0,
    nvlink_available = nvlink_available,
    tbl_auto_optimization = True, # enable H3T solver
    tbl_auto_optimization = alg_name,
    tbl_memory_limit = memory_limit,
)
```

Please refer to `README-BMTrain.md` or `README-BMTrain-ZH.md` for the usage of BMTrain. We will also merge these features into the main branch and release the latest version as soon as possible.

## Citation

```bibtex
@inproceedings{wang2023h3t,
    title={H3T: Efficient Integration of Memory Optimization and Parallelism for High-Throughput Transformer Training},
    author={Wang, Yuzhong and Han, Xu and Zhao, Weilin and Zeng, Guoyang and Liu, Zhiyuan and Sun, Maosong}
    booktitle={Proceedings of NeurIPS},
    year={2023}
}
```