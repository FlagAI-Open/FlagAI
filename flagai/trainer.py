# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# Arguments for training
try:
    import deepspeed.utils
    import deepspeed
except:
    pass
try:
    from flagai import mpu
except Exception:
    pass
from re import I
import torch
import argparse
import os
import random
import numpy as np
import torch.distributed as dist
from flagai.logger import log_dist
from torch.utils.tensorboard import SummaryWriter
from flagai.utils import load_checkpoint, save_checkpoint
from flagai.schedulers import AnnealingLR
from flagai.optimizers import get_optimizer, get_optimizer_param_groups
from flagai.fp16 import FP16_Module
from flagai.utils import Timers
from flagai.launch import launch_dist
from torch.nn.parallel import DistributedDataParallel as DDP
from flagai.fp16 import DynamicLossScaler
"""
The Trainer class, to easily train a pytorh model on a new task.
"""


class Trainer():
    """
    Trainer is a simple but unifed training and eval loop for PyTorch/Deepspeed/Megatron-LM.
    Args:
        model (`torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions.
        args ([`env_type`]):
            The enviroment type for training. Will default to 'pytorch'.
            env_type: `pytorch`, `pytorchDDP`, `deepspeed`, `deepspeed+mpu`
                        pytorch: single node cpu/gpu
                        pytorchDDP: single-/multi- node gpu <data parallel>
                        deepspeed: single-/multi- node gpu <data/pipline parallel>
                        deepspeed+mpu: single-/multi- node gpu <data parallel + model parallel>
        train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.DataLoader`, *optional*):
            The dataset to use for training.
            If it is an `Dataset`, we will create a `DataLoader` with the provided `Dataset` and `collate_fn' for the selected `env_type`.
            `Dataset` is prefred to iterally return a sample as followings,
            >>> {'text': 'I like big model.', 'label': 'positive'}
            If it is an `DataLoader`, we will directly use it.
            Important: Columns not accepted by the `model.forward()` method are automatically droped.
        eval_dataset (`torch.utils.data.Dataset` or `torch.utils.data.DataLoader`, *optional*):
             The dataset to use for evaluation. Similar to `train_dataset`.
        collate_fn (`DataCollator` or `function`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`.
        metrics (`function`, *optional*):
            The function that will be used to compute metrics at evaluation. Must return
            a dictionary string to metric values.
        optimizers (`torch.optim.Optimizer`, *optional*): A optimizer to use. Will default to an instance of
             [`AdamW`] on your model.
        lr_scheduler (`torch.optim.lr_scheduler`,  *optional*): A lr_scheduler to use. Will default to an instance of
             [`AnnealingLR`].
    Examples:
        Usage for Trainer:
            If we write the training pipline using Trainer in `train.py`.
            We only have to run it. And the rest works is done by the Trainer.
            >>> python train.py
            Done! The model is runing under the setted enviroment.
        Example settings for `pytorch`:
            >>> trainer = Trainer(env_type='pytorch', pytorch_device='cpu') # pytorch_device could be 'cuda' or 'cpu'
        Example settings for `pyrochDDP`:
            >>> trainer = Trainer(env_type='pytorchDDP',
            >>>                        master_ip='127.0.0.1',
            >>>                        master_port=17750,
            >>>                        num_nodes=1,   # nodes
            >>>                        num_gpus=2,    # gpus for each nodes
            >>>                        hostfile='./hostfile',
            >>>                        training_script=__file__)
            >>> cat ./hostfile
            >>> 127.0.0.1 slots=2
        Example settings for `deepspeed`:
            >>> trainer = Trainer(env_type='pytorchDDP',
            >>>                        master_ip='127.0.0.1',
            >>>                        master_port=17750,
            >>>                        num_nodes=1,   # nodes
            >>>                        num_gpus=2,    # gpus for each nodes
            >>>                        hostfile='./hostfile',
            >>>                        deepspeed_config='./deepspeed.json',
            >>>                        training_script=__file__)
            The detail settings for deepspeed.json refer to https://www.deepspeed.ai/docs/config-json/
        Example settins for `deepspeed+mpu`:
            The model must be build by megatron-lm!!!
            >>> trainer = Trainer(env_type='pytorchDDP',
            >>>                        master_ip='127.0.0.1',
            >>>                        master_port=17750,
            >>>                        num_nodes=1,   # nodes
            >>>                        num_gpus=2,    # gpus for each nodes
            >>>                        hostfile='./hostfile',
            >>>                        deepspeed_config='./deepspeed.json',
            >>>                        model_parallel_size=2, # mp_size ==1
            >>>                        training_script=__file__)

    """
    def __init__(
        self,
        timers=None,
        env_type='pytorch',
        pytorch_device='cpu',  # "cuda:0" etc.
        experiment_name="glm_large",  # "The experiment name for summary and checkpoint"
        batch_size=4,  # 'Data Loader batch size'
        gradient_accumulation_steps=1,  # 'Data Loader batch size'
        weight_decay=0.0,  # 'weight decay coefficient for L2 regularization'
        lr=1e-3,
        epochs=0,  # 'Number of finetunning epochs. Zero results in evaluation only.'
        save_epoch=1,  # 'number of epochs between saves')
        eval_interval=1,
        log_interval=1,
        seed=1234,  # 'random seed'
        fp16=False,
        clip_grad=1.0,
        checkpoint_activations=False,

        # model checkpointing
        save_dir='checkpoints',  # 'Output directory to save checkpoints to.')
        save_optim=False,  # save current optimizer.')
        save_rng=False,  # save current rng state.')
        load_dir='checkpoints/99',  # Path to a directory containing a model checkpoint.')
        load_optim=False,  # not load optimizer when loading checkpoint.')
        # ' not load rng state when loading checkpoint.')):
        load_rng=False,
        tensorboard_dir="tensorboard_summary",

        # distribute settings
        deepspeed_activation_checkpointing=False,
        num_checkpoints=1,
        master_ip='localhost',
        master_port=17750,
        num_nodes=1,
        num_gpus=1,
        hostfile=None,
        deepspeed_config=None,
        model_parallel_size=1,
        training_script="train.py",
    ):

        if timers is not None:
            self.timers = timers
        else:
            self.timers = Timers()
        self.env_type = env_type
        if env_type not in set(["deepspeed", 'pytorch', 'pytorchDDP', 'deepspeed+mpu']):
            raise Exception("Not supported env_type!!!!")
        os.environ["ENV_TYPE"] = env_type
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.clip_grad = clip_grad
        self.seed = seed
        self.fp16 = fp16

        self.log_interval = log_interval
        self.eval_interval = eval_interval

        # model checkpointing
        self.save_dir = save_dir
        self.save_epoch = save_epoch
        self.save_optim = save_optim
        self.save_rng = save_rng
        self.load_dir = load_dir
        self.load_optim = load_optim
        self.load_rng = load_rng
        self.tb_writer = SummaryWriter(
            os.path.join(tensorboard_dir, experiment_name))

        # distribute settings
        self.pytorch_device = pytorch_device
        self.checkpoint_activations = checkpoint_activations
        self.deepspeed_activation_checkpointing = deepspeed_activation_checkpointing
        self.num_checkpoints = num_checkpoints
        self.env_type = env_type
        self.not_call_launch = True
        self.deepspeed_config = deepspeed_config
        self.model_parallel_size = model_parallel_size
        if 'deepspeed' in env_type or env_type == 'pytorchDDP':
            # Implement for AutoLaunch
            # >>> python train.py # will call get_dist_args()
            # `--not_call_launch` is default 'False'
            # So, if `env_type` is `pytorch`, the `Trainer` will not call lanch_dist()
            # Otherwise, the lanch_dist() is called to launch 'train.py' with `--not_call_launch`
            self.get_dist_args()
            if not self.not_call_launch:
                launch_dist(launcher='distributed_deepspeed' if 'deepspeed'
                            in env_type else 'distributed_torch',
                            num_nodes=num_nodes,
                            gpus_per_node=num_gpus,
                            master_addr=master_ip,
                            master_port=master_port,
                            hostfile=hostfile,
                            training_script=training_script)
                os._exit(1)
            self.initialize_distributed()

    def get_dist_args(self):
        """
        Get required parameters for initialize the distributed enviroment, e.g., rank, local_rank & world_size...
        Important: --not_call_launch, default False, will not call launch_dist
        Returns: None
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank',
                            type=int,
                            default=0,
                            help="local_rank")
        parser.add_argument('--not_call_launch',
                            action='store_true',
                            help="not call launch!")
        ds_args = parser.parse_args()
        self.local_rank = ds_args.local_rank
        self.not_call_launch = ds_args.not_call_launch
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        self.master_port = os.environ.get('MASTER_PORT', '17500')
        log_dist("not_call_launch: {}".format(ds_args.not_call_launch))

    def set_seed(self, seed=1234):
        """Set random seed for reproducability."""
        if seed is not None and seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.env_type == 'deepspeed+mpu':
                mpu.model_parallel_cuda_manual_seed(seed)

    def initialize_distributed(self):
        """Initialize torch.distributed."""
        if self.env_type == 'pytorch':
            log_dist('No need to initialize')
            return
        if self.env_type in ['deepspeed', 'deepspeed+mpu', 'pytorchDDP']:
            torch.backends.cudnn.enabled = False
            # Manually set the device ids.
            device = self.rank % torch.cuda.device_count()
            if self.local_rank is not None:
                device = self.local_rank
            torch.cuda.set_device(device)
            # Call the init process
            init_method = 'tcp://'
            self.master_ip = os.getenv('MASTER_ADDR', 'localhost')
            self.master_port = os.getenv('MASTER_PORT', '6000')

            init_method += self.master_ip + ':' + self.master_port
            log_dist(
                "init method {}, rank {}, device {}, local_rank {}.".format(
                    init_method, self.rank, device, self.local_rank))
            torch.distributed.init_process_group(backend='nccl',
                                                 world_size=self.world_size,
                                                 rank=self.rank,
                                                 init_method=init_method)
        # Set the model-parallel / data-parallel communicators.
        if self.env_type == 'deepspeed+mpu':
            os.environ["MODEL_PARALLEL_SIZE"] = str(self.model_parallel_size)
            try:
                mpu.initialize_model_parallel(self.model_parallel_size)
                if 'deepspeed' in self.env_type and self.deepspeed_activation_checkpointing:
                    deepspeed.checkpointing.configure(
                        mpu,
                        deepspeed_config=self.deepspeed_config,
                        num_checkpoints=self.num_checkpoints)
                    mpu.checkpoint = deepspeed.checkpointing.checkpoint
                    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
                    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed
            except Exception as e:
                log_dist(e)
                log_dist("No mpu is installed! No model parallel is used")
            log_dist("initialize eviroments succesed")
        self.set_seed(self.seed)

    def get_dataloader(self, dataset, collate_fn, shuffle=False):
        """ initilize the dataloader"""
        if dataset is None:
            return None
        if self.env_type == 'pytorch':
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=self.batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=shuffle)
        else:
            if self.env_type == 'deepspeed+mpu':
                num_replicas = self.world_size // mpu.get_model_parallel_world_size(
                )
                rank = self.rank // mpu.get_model_parallel_world_size()
            else:
                num_replicas = self.world_size
                rank = self.rank
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=self.batch_size,
                                               sampler=sampler,
                                               num_workers=1,
                                               drop_last=False,
                                               pin_memory=False,
                                               collate_fn=collate_fn)

    def train(self,
              model=None,
              model_input_keys=None,
              optimizer=None,
              lr_scheduler=None,
              train_dataset=None,
              valid_dataset=None,
              metric_methods=[],
              collate_fn=None):
        """Training Loops"""
        """
       Trainer is a simple but unifed training and eval loop for PyTorch/Deepspeed/Megatron-LM.
       Args:
           model (`torch.nn.Module`, *optional*):
               The model to train, evaluate or use for predictions.
           args ([`env_type`]):
               The enviroment type for training. Will default to 'pytorch'.
               env_type: `pytorch`, `pytorchDDP`, `deepspeed`, `deepspeed+mpu`
                           pytorch: single node cpu/gpu
                           pytorchDDP: single-/multi- node gpu <data parallel>
                           deepspeed: single-/multi- node gpu <data/pipline parallel>
                           deepspeed+mpu: single-/multi- node gpu <data parallel + model parallel>
           train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.DataLoader`, *optional*):
               The dataset to use for training.
               If it is an `Dataset`, we will create a `DataLoader` with the provided `Dataset` and `collate_fn' for the selected `env_type`.
               `Dataset` is prefred to iterally return a sample as followings,
               >>> {'text': 'I like big model.', 'label': 'positive'}
               If it is an `DataLoader`, we will directly use it.
               Important: Columns not accepted by the `model.forward()` method are automatically droped.
           eval_dataset (`torch.utils.data.Dataset` or `torch.utils.data.DataLoader`, *optional*):
                The dataset to use for evaluation. Similar to `train_dataset`.
           collate_fn (`DataCollator` or `function`, *optional*):
               The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`.
           metrics (`function`, *optional*):
               The function that will be used to compute metrics at evaluation. Must return
               a dictionary string to metric values.
           optimizers (`torch.optim.Optimizer`, *optional*): A optimizer to use. Will default to an instance of
                [`AdamW`] on your model.
           lr_scheduler (`torch.optim.lr_scheduler`,  *optional*): A lr_scheduler to use. Will default to an instance of
                [`AnnealingLR`].
           """
        if not isinstance(train_dataset, torch.utils.data.DataLoader):
            train_dataloader = self.get_dataloader(train_dataset, collate_fn,
                                                   True)
        else:
            train_dataloader = train_dataset

        if not isinstance(valid_dataset, torch.utils.data.DataLoader):

            valid_dataloader = self.get_dataloader(valid_dataset, collate_fn,
                                                   False)
        else:
            valid_dataloader = valid_dataset

        if self.fp16:
            model.half()
        if self.env_type == 'pytorchDDP':
            model.to(torch.device('cuda', self.local_rank))
            model = DDP(model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=True)
        elif self.env_type == 'pytorch':
            model.to(self.pytorch_device)
        else:
            model.cuda(torch.device('cuda', self.local_rank))
        """Train the model."""
        # Turn on training mode which enables dropout.
        if self.fp16:
            model = FP16_Module(model)  
  
        model.train()
        param_groups = get_optimizer_param_groups(model)

        if hasattr(param_groups[0], 'params'):
            # for T5 Model
            param_groups = param_groups[0]['params']

        if optimizer is None and 'deepspeed' not in self.env_type:
            optimizer = get_optimizer(param_groups=param_groups,
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      cpu_optimizer=False,
                                      cpu_torch_adam=False,
                                      fp16=self.fp16)

        # if lr_scheduler == None:
        #     lr_scheduler = AnnealingLR(
        #         optimizer,
        #         start_lr=self.lr,
        #         warmup_iter=int(0.2* self.epochs * len(train_dataloader)),
        #         decay_style='linear',
        #         num_iters=self.epochs * len(train_dataloader))

        if 'deepspeed' in self.env_type:
            # initialize the deepspeed
            model, optimizer, __, ___ = deepspeed.initialize(
                model=model,
                # if huggingface t5: param_groups[0]['params']
                model_parameters=param_groups,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                mpu=mpu if self.env_type == 'deepspeed+mpu' else None,
                config=self.deepspeed_config,
                dist_init_required=True)

        # Tracking loss.
        total_lm_loss = 0.0
        self.iteration = 0
        best_iteration = 0
        # For each remaining epoch
        self.timers('interval time').start()
        # self.eval_metrics = eval_metrics
        # self.do_eval = valid_dataset!=None
        self.metric_methods = metric_methods
        log_dist("loading checkpoints form {}".format(self.load_dir))
        if self.load_dir:
            load_checkpoint(model,
                            optimizer,
                            lr_scheduler,
                            self.load_dir,
                            use_deepspeed=self.env_type != 'pytorch',
                            load_optim=self.load_optim,
                            load_rng=self.load_rng)
        for epoch in range(self.epochs):
            log_dist('working on epoch {} ...'.format(epoch), [0])
            # Set the data loader epoch to shuffle the index iterator.
            if self.env_type == 'deepspeed+mpu':
                if mpu.get_model_parallel_rank(
                ) == 0:  # TODO using env to decide
                    train_dataloader.sampler.set_epoch(self.seed + epoch)
            elif self.env_type == 'deepspeed':
                if dist.get_rank() == 0:
                    train_dataloader.sampler.set_epoch(self.seed + epoch)
            # For all the batches in the dataset.
            for iteration_, batch in enumerate(train_dataloader):
                # Train for one step.
                if 'deepspeed' in self.env_type or self.env_type == 'pytorchDDP':
                    batch = {
                        x: batch[x].to(torch.device('cuda', self.local_rank))
                        for x in batch if x not in ['uid', 'meta', 'mode']
                    }
                elif 'pytorch' == self.env_type:
                    batch = {
                        x: batch[x].to(torch.device(self.pytorch_device))
                        for x in batch if x not in ['uid', 'meta', 'mode']
                    }
                lm_loss, skipped_iter, _ = self.train_step(batch,
                                                           model,
                                                           model_input_keys,
                                                           optimizer,
                                                           lr_scheduler,
                                                           single_step=True)
                total_lm_loss += lm_loss.data.detach().float()

                # Logging.
                if (self.iteration + 1) % self.log_interval == 0:
                    if optimizer is not None:
                        learning_rate = optimizer.param_groups[0]['lr']
                    else:
                        learning_rate = model.optimizer.param_groups[0]['lr']
                    avg_lm_loss = total_lm_loss.item() / self.log_interval
                    elapsed_time = self.timers('interval time').elapsed()
                    self.report_iteration_metrics(
                        optimizer, learning_rate, avg_lm_loss,
                        elapsed_time * 1000.0 / self.log_interval,
                        self.iteration + 1,
                        self.epochs * len(train_dataloader))
                    self.tb_writer.add_scalar('train/loss', avg_lm_loss,
                                              self.iteration + 1)
                    self.tb_writer.add_scalar('lr', learning_rate,
                                              self.iteration + 1)
                    total_lm_loss = 0.0
                # Evaluation #todo add train_args

                if self.eval_interval and (
                        self.iteration + 1
                ) % self.eval_interval == 0 and valid_dataloader is not None:
                    self.timers.log(
                        ['forward', 'backward', 'allreduce', 'optimizer'],
                        normalizer=self.eval_interval)
                    prefix = 'epoch {}'.format(epoch)
                    eval_dict = self.evaluate_and_print_results(
                        prefix=prefix,
                        data_loader=valid_dataloader,
                        model=model,
                        forward_step_func=self.forward_step,
                        verbose=False)
                    if eval_dict is not None:
                        eval_loss = eval_dict.get("loss", 0.0)
                        self.tb_writer.add_scalar('eval/loss', eval_loss,
                                                  self.iteration + 1)
                        for i in range(len(self.metric_methods)):
                            name = self.metric_methods[i][0]
                            score = eval_dict.get(name, 0)
                            self.tb_writer.add_scalar(
                                'eval_metrics/%s' % (name), score,
                                self.iteration + 1)
                    # if eval_score > best_score:
                    #     save_checkpoint(self.iteration, model, optimizer, lr_scheduler, only_changed_parameters=True,
                    #                     save_dir=self.save_dir, use_deepspeed='pytorch' != self.env_type,
                    #                     save_optim=self.save_optim, save_rng=self.save_rng)
                self.iteration += 1

            # Checkpointing at the end of each epoch.
            if self.save_dir and (epoch + 1) % self.save_epoch == 0:
                if 'deepspeed' == self.env_type and dist.get_rank() != 0:
                    pass
                if 'deepspeed+mpu' == self.env_type and mpu.get_model_parallel_group(
                ) != 0:
                    pass
                if 'pytorchDDP' == self.env_type and dist.get_rank() != 0:
                    pass
                else:
                    save_checkpoint(self.iteration,
                                    model,
                                    optimizer,
                                    lr_scheduler,
                                    use_deepspeed='pytorch'
                                    not in self.env_type,
                                    save_optim=self.save_optim,
                                    save_dir=self.save_dir,
                                    save_rng=self.save_rng)

        # Evaluation #todo add train_args
        if self.eval_interval and (
                self.iteration
        ) % self.eval_interval != 0 and valid_dataloader is not None:
            prefix = 'final evaluate'
            self.evaluate_and_print_results(
                prefix=prefix,
                data_loader=valid_dataloader,
                model=model,
                forward_step_func=self.forward_step,
                verbose=False)
        return best_iteration

    def train_step(self,
                   data,
                   model,
                   model_input_keys,
                   optimizer,
                   lr_scheduler,
                   mems=None,
                   single_step=False):
        """Single training step."""
        lm_loss_total, count = 0.0, 0
        mems = [] if mems is None else mems
        # if not train_args.deepspeed:
        if 'deepspeed' not in self.env_type:
            optimizer.zero_grad()
        while True:
            skipped_iter, complete = 0, False
            # Forward model for one step.
            self.timers('forward').start()
            step_output = self.forward_step(data, model, mems)
            self.timers('forward').stop()
            lm_loss = step_output['loss']
            if 'deepspeed' not in self.env_type:
                lm_loss /= self.gradient_accumulation_steps

            reduced_loss = lm_loss.detach().clone().view(1)
            if self.env_type == 'deepspeed+mpu':
                torch.distributed.all_reduce(
                    reduced_loss.data, group=mpu.get_data_parallel_group())
            elif self.env_type == 'deepspeed':
                torch.distributed.all_reduce(
                    reduced_loss.data
                )  # group=deepspeed.utils.get_data_parallel_group())
            if 'deepspeed' in self.env_type:
                reduced_loss.data = reduced_loss.data / \
                    (self.world_size / self.model_parallel_size)
            if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
                lm_loss_total += reduced_loss
                count += 1
                # Calculate gradients, reduce across processes, and clip.
                self.timers('backward').start()
                self.backward_step(optimizer, model, lm_loss)
                self.timers('backward').stop()
                # Update parameters.
                self.timers('optimizer').start()
                # if train_args.deepspeed:
                if 'deepspeed' in self.env_type:
                    if model.is_gradient_accumulation_boundary():
                        model.step()
                        complete = True
                        if not (self.fp16 and optimizer.overflow):
                            if lr_scheduler:
                                lr_scheduler.step()
                        else:
                            skipped_iter = 1
                    else:
                        model.step()
                else:
                    if count == self.gradient_accumulation_steps:
                        optimizer.step()
                        complete = True
                        # Update learning rate.
                        if not (self.fp16 and hasattr(optimizer, 'overflow')
                                and optimizer.overflow):
                            if lr_scheduler:
                                lr_scheduler.step()
                        else:
                            skipped_iter = 1
                self.timers('optimizer').stop()
                if complete:
                    break
            else:
                log_dist("Found NaN loss, skip backward", [0])
                del lm_loss, reduced_loss
                mems = []
            if single_step:
                break

        # if train_args.deepspeed:
        if 'deepspeed' in self.env_type:
            lm_loss_total = lm_loss_total / count
        return lm_loss_total, skipped_iter, mems

    def forward_step(self, data, model, mems):
        """Simple forward step. """
        data['mems'] = mems
        model_output = model(**data)
        logits = model_output['logits']
        loss = model_output['loss']
        hidden_states = None
        if 'hidden_states' in model_output:
            hidden_states = model_output['hidden_states']
        elif 'encoder_hidden_states' in model_output:
            hidden_states = model_output['encoder_hidden_states']

        return {
            'loss': loss,
            'hidden_states': hidden_states,
            'logits': logits.contiguous().float()
        }

    def backward_step(self, optimizer, model, lm_loss):
        """Backward step."""

        # Total loss.
        loss = lm_loss
        # Backward pass.
        # if self.train_args.deepspeed:
        if 'deepspeed' in self.env_type:
            model.backward(loss)
        else:
            # optimizer.zero_grad()
            if hasattr(optimizer, 'backward'):
                optimizer.backward(loss, update_master_grads=False)
            else:
                loss.backward()

        # if self.train_args.deepspeed or self.train_args.DDP_impl == 'torch':
        self.timers('allreduce').reset()
        if self.env_type == 'pytorch':
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
        return lm_loss

    def evaluate(self,
                 data_loader=None,
                 model=None,
                 forward_step_func=None,
                 verbose=False):
        """Evaluation."""
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_lm_loss = 0.
        total_samples = 0.
        # metric = 0.
        mems = []
        metrics = [0. for _ in range(len(self.metric_methods))]
        with torch.no_grad():
            assert data_loader is not None, "val loader is not None."
            for data_iterator in data_loader:
                # Forward evaluation.

                meta = data_iterator.get('meta', None)

                if 'deepspeed' in self.env_type:
                    data_iterator = {
                        x: data_iterator[x].to(
                            torch.device('cuda', self.local_rank))
                        for x in data_iterator
                        if x not in ['uid', 'meta', 'mode']
                    }
                elif torch.cuda.is_available():

                    data_iterator = {
                        x:
                        data_iterator[x].to(torch.device(self.pytorch_device))
                        for x in data_iterator
                        if x not in ['uid', 'meta', 'mode']
                    }
                step_output = forward_step_func(data_iterator,
                                                model,
                                                mems=mems)
                lm_loss= step_output['loss']
                # mems = step_output['hidden_states']
                '''when contiguous memory optimizations are enabled, the buffers
                allocated by the optimizations are deallocated during backward pass
                in the absence of backward pass the buffers should be reset after each
                forward pass'''
                if 'deepspeed' in self.env_type and self.deepspeed_activation_checkpointing:
                    deepspeed.checkpointing.reset()
                logits = step_output['logits']
                batch_size = data_iterator['input_ids'].size()[0]
                lm_loss = lm_loss.data.detach().float().item()
                total_lm_loss += lm_loss
                total_samples += batch_size

                for i in range(len(self.metric_methods)):
                    eval_method = self.metric_methods[i][1]
                    metrics[i] += eval_method(logits,
                                              data_iterator['labels'],
                                              meta=meta)

        # Move model back to the train mode.
        model.train()
        if torch.cuda.is_available():
            loss_data = torch.cuda.FloatTensor([total_lm_loss, total_samples] +
                                               metrics)
        else:
            loss_data = torch.FloatTensor([total_lm_loss, total_samples] +
                                          metrics)
        if self.env_type == 'deepspeed+mpu':
            torch.distributed.all_reduce(loss_data,
                                         group=mpu.get_data_parallel_group())
        elif self.env_type == 'deepspeed':
            torch.distributed.all_reduce(
                loss_data, group=deepspeed.utils.get_data_parallel_group())
        elif self.env_type == 'pytorchDDP':
            torch.distributed.all_reduce(loss_data)
        loss_data = loss_data.tolist()
        total_lm_loss = loss_data[0] / loss_data[1]
        metric_dct = {}

        for i in range(len(self.metric_methods)):
            metric_name = self.metric_methods[i][0]
            metric = loss_data[2 + i] * 100. / loss_data[1]
            metric_dct.update({metric_name: metric})
        metric_dct.update({"loss": total_lm_loss})
        # return {"loss": total_lm_loss}.update(metric_dct)
        return metric_dct

    def report_iteration_metrics(self, optimizer, lr, loss, elapsed_time, step,
                                 total_step):
        log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time)
        log_string += ' learning rate {:.3E} |'.format(lr)
        log_string += ' loss {:.6E} |'.format(loss)
        if self.fp16:
            log_string += ' loss scale {:.1f} |'.format(
                optimizer.cur_scale if 'deepspeed' in self.env_type else
                hasattr(optimizer, 'loss_scale') and optimizer.loss_scale)
        log_dist(log_string, [0])

    def report_evaluate_metrics(self, prefix, loss, ppl, gpt_loss, bert_loss,
                                sent_loss, multi_loss, step):
        string = ' validation loss at {}'.format(prefix)
        string += ' | LM loss: {:.6E}'.format(loss)
        string += ' | LM PPL: {:.6E}'.format(ppl)
        length = len(string) + 1
        log_dist('-' * 100, [0])
        log_dist('-' * length, [0])
        log_dist(string, [0])
        log_dist('-' * length, [0])

    def evaluate_and_print_results(
        self,
        prefix=None,
        forward_step_func=None,
        data_loader=None,
        model=None,
        verbose=False,
    ):
        """Helper function to evaluate and dump results on screen."""
        eval_dict = self.evaluate(forward_step_func=forward_step_func,
                                  data_loader=data_loader,
                                  model=model,
                                  verbose=verbose)
        if eval_dict.get("loss", None) is not None:
            string = ' validation loss at {} | {:.4f}, '.format(
                prefix, eval_dict["loss"])

        if self.metric_methods is None:
            return eval_dict

        for i in range(len(self.metric_methods)):
            name = self.metric_methods[i][0]
            string += ", {} {:.2f}".format(name, eval_dict[name])
        # string = ' validation loss at {} | {:.4f},  Acc {:.2f}'.format(
        #     prefix, eval_dict["loss"], eval_dict["metrics"])
        length = len(string) + 1
        log_dist('-' * length, [0])
        log_dist(string, [0])
        log_dist('-' * length, [0])
        return eval_dict
