
(acceleration)=
# OpenDelta + BMTrain

- [BMTrain](https://github.com/OpenBMB/BMTrain) is an efficient large model training toolkit that can be used to train large models with tens of billions of parameters. It can train models in a distributed manner while keeping the code as simple as stand-alone training.
- [ModelCenter](https://github.com/OpenBMB/ModelCenter) implements pre-trained language models (PLMs) based on the backend OpenBMB/BMTrain. ModelCenter supports Efficient, Low-Resource, Extendable model usage and distributed training.

Now we have the LoraModel, AdapterModel, CompacterModel, ParallelAdapterModel, LowRankAdapterModel fully supported the distributed training with BMTrain and ModelCenter. 

Pass `backend='bmt'` in config or delta model initialization to enable `bmtrain`.


