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

import torch
import argparse
import os
import random
import numpy as np
import torch.distributed as dist
from flagai.logger import log_dist
from torch.utils.tensorboard import SummaryWriter
from flagai.utils import load_checkpoint, save_checkpoint, load_optim, load_rng
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
def save_best(best_score, eval_dict):
    return best_score if best_score < eval_dict['loss'] else eval_dict['loss']

def get_args_list(env_args):
    not_need_to_launch_args = ["not_call_launch", "local_rank", "master_port", "master_ip", "hostfile", "num_gpus", "num_nodes"]
    args_list = []
    args = dir(env_args)
    for arg in args:
        if not arg.startswith("__") and not arg.startswith("_") and arg not in not_need_to_launch_args:
            args_list.append(f"--{arg}")
            args_list.append(str(getattr(env_args, arg)))

    print(f"args list is {args_list}")
    return args_list

class EnvTrainer():
    def __init__(self,
                 env_args,
    ):
        self.timers = Timers()
        self.env_type = env_args.env_type
        if self.env_type not in set(
            ["deepspeed", 'pytorch', 'pytorchDDP', 'deepspeed+mpu']):
            raise Exception("Not supported env_type!!!!")
        os.environ["ENV_TYPE"] = env_args.env_type
        self.experiment_name = env_args.experiment_name
        self.batch_size = env_args.batch_size
        self.gradient_accumulation_steps = env_args.gradient_accumulation_steps
        self.lr = env_args.lr
        self.weight_decay = env_args.weight_decay
        self.epochs = env_args.epochs
        self.clip_grad = env_args.clip_grad
        self.seed = env_args.seed
        self.fp16 = env_args.fp16
        self.warm_up = env_args.warm_up

        self.log_interval = env_args.log_interval
        self.eval_interval = env_args.eval_interval

        # model checkpointing
        self.save_dir = env_args.save_dir
        self.save_interval = env_args.save_interval
        self.save_optim = env_args.save_optim
        self.save_rng = env_args.save_rng
        self.save_best = save_best
        self.load_dir = env_args.load_dir
        self.load_type = env_args.load_type
        self.load_optim = env_args.load_optim
        self.load_rng = env_args.load_rng
        self.tb_writer = SummaryWriter(
            os.path.join(env_args.tensorboard_dir, env_args.experiment_name))

        # distribute settings
        self.pytorch_device = env_args.pytorch_device
        self.checkpoint_activations = env_args.checkpoint_activations
        self.deepspeed_activation_checkpointing = env_args.deepspeed_activation_checkpointing
        self.num_checkpoints = env_args.num_checkpoints
        self.env_type = env_args.env_type
        self.not_call_launch = env_args.not_call_launch
        self.deepspeed_config = env_args.deepspeed_config
        self.model_parallel_size = env_args.model_parallel_size
        self.num_nodes = env_args.num_nodes
        self.num_gpus = env_args.num_gpus
        self.master_ip = env_args.master_ip
        self.master_port = env_args.master_port
        self.hostfile = env_args.hostfile
        self.training_script = env_args.training_script

        if 'deepspeed' in self.env_type or self.env_type == 'pytorchDDP':
            training_paras = get_args_list(env_args)
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.local_rank = env_args.local_rank
            log_dist("not_call_launch: {}".format(self.not_call_launch))
            # Implement for AutoLaunch
            # >>> python train.py # will call get_dist_args()
            # `--not_call_launch` is default 'False'
            # So, if `env_type` is `pytorch`, the `Trainer` will not call lanch_dist()
            # Otherwise, the lanch_dist() is called to launch 'train.py' with `--not_call_launch`
            if not self.not_call_launch:
                launch_dist(launcher='distributed_deepspeed' if 'deepspeed'
                            in self.env_type else 'distributed_torch',
                            num_nodes=self.num_nodes,
                            gpus_per_node=self.num_gpus,
                            master_addr=self.master_ip,
                            master_port=self.master_port,
                            hostfile=self.hostfile,
                            training_script=self.training_script,
                            training_paras=training_paras)
                os._exit(1)
            self.initialize_distributed()

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
            torch.distributed.init_process_group(
                backend='nccl',  # gloo
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
                                               num_workers=4,
                                               prefetch_factor=4,
                                               pin_memory=True,
                                               drop_last=False,
                                               shuffle=shuffle)
        else:
            if self.env_type == 'deepspeed+mpu':
                # num_replicas = self.world_size // mpu.get_model_parallel_world_size(
                # )
                # rank = self.rank // mpu.get_model_parallel_world_size()
                # rank = mpu.get_model_parallel_rank()
                rank = mpu.get_model_parallel_src_rank()
                print("*"*80)
                print("local rank",self.rank, "model rank", rank)
                print("*"*80)
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    # num_replicas=num_replicas,
                    rank=rank,
                    shuffle=shuffle)
            else:
                num_replicas = self.world_size
                rank = self.rank
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, rank=rank, shuffle=shuffle)
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=self.batch_size,
                                               sampler=sampler,
                                               num_workers=4,
                                               drop_last=False,
                                               pin_memory=False,
                                               prefetch_factor=4,
                                               collate_fn=collate_fn)

    def train(self,
              model=None,
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

        if self.load_dir:
            log_dist("loading checkpoints form {}".format(self.load_dir))
            sd = load_checkpoint(model,
                                 load_dir=self.load_dir,
                                 load_type=self.load_type)
        """Train the model."""
        # Turn on training mode which enables dropout.
        model.train()
        if self.fp16 and self.env_type == 'pytorchDDP':
            log_dist(
                "Warning: The pytorchDDP plus FP16 may not working togather!!!"
            )
        if self.fp16:
            model.half()
        if self.checkpoint_activations:
            model.config[
                'checkpoint_activations'] = self.checkpoint_activations

        if self.env_type == 'pytorchDDP':
            model.to(torch.device('cuda', self.local_rank))
            model = DDP(model,
                        device_ids=[self.local_rank],
                        find_unused_parameters=True)

        elif self.env_type == 'pytorch':
            model.to(self.pytorch_device)
        else:
            model.cuda(torch.device('cuda', self.local_rank))
        if self.fp16:
            model = FP16_Module(model)

        param_groups = get_optimizer_param_groups(model)

        if hasattr(param_groups[0], 'params'):
            # for T5 Model
            param_groups = param_groups[0]['params']

        if optimizer is None and 'deepspeed' not in self.env_type and self.epochs > 0:
            optimizer = get_optimizer(
                param_groups=param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay,
                cpu_optimizer=False,
                cpu_torch_adam=False,
                fp16=self.fp16,
                optimizer='adam')  # if not self.fp16 else 'adafactor')

        if lr_scheduler == None and optimizer != None and self.warm_up > 0 and 'deepspeed' not in self.env_type and self.epochs > 0:

            lr_scheduler = AnnealingLR(
                optimizer,
                start_lr=self.lr,
                warmup_iter=int(self.warm_up * self.epochs *
                                len(train_dataloader)),
                decay_style='linear',
                num_iters=self.epochs * len(train_dataloader))

        if 'deepspeed' in self.env_type:
            # initialize the deepspeed
            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                # if huggingface t5: param_groups[0]['params']
                model_parameters=param_groups,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                mpu=mpu if self.env_type == 'deepspeed+mpu' else None,
                config=self.deepspeed_config,
                dist_init_required=True)
        if self.load_optim:
            print(self.load_optim)
            print(type(self.load_optim))
            load_optim(optimizer, lr_scheduler, sd)
        if self.load_rng:
            load_rng(sd)
        # Tracking loss.
        total_lm_loss = 0.0
        self.iteration = 0
        self.accumulate_count = 0
        best_iteration = 0
        best_loss = float('inf')
        # For each remaining epoch
        self.timers('interval time').start()
        # self.eval_metrics = eval_metrics
        # self.do_eval = valid_dataset!=None
        self.metric_methods = metric_methods
        best_score = float('inf')
        if len(self.metric_methods) > 0:
            best_score = -best_score

        for epoch in range(self.epochs):
            # log_dist('working on epoch {} ...'.format(epoch), [0])
            # Set the data loader epoch to shuffle the index iterator.
            # if self.env_type == 'deepspeed+mpu':
            #     if mpu.get_model_parallel_rank() == 0:
            #         train_dataloader.sampler.set_epoch(epoch + self.world_size)
            if self.env_type != 'pytorch':
                train_dataloader.sampler.set_epoch(epoch + self.world_size)

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
                if self.env_type == 'pytorchDDP':
                    lm_loss, _ = self.train_step_pytorchDDP(
                        batch, model, optimizer, lr_scheduler)
                    dist.barrier()

                elif self.env_type == 'pytorch':
                    lm_loss, _ = self.train_step_pytorch(
                        batch, model, optimizer, lr_scheduler)
                else:
                    lm_loss, _ = self.train_step_deepspeed(batch,
                                                           model,
                                                           optimizer,
                                                           lr_scheduler,
                                                           single_step=True)
                    dist.barrier()
                if lm_loss is not None:
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
                    self.timers.log(['forward', 'backward', 'optimizer'],
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
                                
                        if self.save_best is not None and self.save_best(best_score, eval_dict) != best_score:
                            best_score = self.save_best(best_score, eval_dict)
                            log_dist("saving best model with score {:.4f}".format(best_score))
                            best_iteration = self.iteration
                            save_checkpoint(self.iteration+1,
                                            best_iteration+1,

                                            model,
                                            optimizer,
                                            lr_scheduler,
                                            save_optim=self.save_optim,
                                            save_dir=self.save_dir,
                                            save_rng=self.save_rng)
                if self.save_dir and (self.iteration + 1) % self.save_interval == 0 and \
                        self.iteration != best_iteration:
                    save_checkpoint(self.iteration+1,
                                    best_iteration+1,
                                    model,
                                    optimizer,
                                    lr_scheduler,
                                    save_optim=self.save_optim,
                                    save_dir=self.save_dir,
                                    save_rng=self.save_rng)
                self.iteration += 1

                # Checkpointing at the end of each epoch.

        # Evaluation #todo add train_args
        if ((self.epochs == 0) or (self.eval_interval and
                                   (self.iteration ) % self.eval_interval != 0)
            ) and valid_dataloader is not None:
            prefix = 'final evaluate'
            self.evaluate_and_print_results(
                prefix=prefix,
                data_loader=valid_dataloader,
                model=model,
                forward_step_func=self.forward_step,
                verbose=False)

    def train_step_pytorch(self,
                           data,
                           model,
                           optimizer,
                           lr_scheduler,
                           mems=None):
        """Single training step."""
        # Forward model for one step.
        self.timers('forward').start()
        step_output = self.forward_step(data, model, mems)
        self.timers('forward').stop()
        # accumulate gradients
        lm_loss = step_output['loss']
        lm_loss /= self.gradient_accumulation_steps
        reduced_loss = lm_loss.detach().clone().view(1)
        # skip the iter while loss has NAN
        if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
            # Calculate gradients, reduce across processes, and clip.
            self.timers('backward').start()
            if self.fp16 and hasattr(optimizer, 'backward'):
                optimizer.backward(lm_loss,
                                   update_master_grads=False,
                                   retain_graph=True)
            else:
                lm_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            self.timers('backward').stop()

            # Update parameters.
            self.timers('optimizer').start()
            if (self.accumulate_count +
                    1) % self.gradient_accumulation_steps == 0:
                if self.fp16:
                    # optimizer.update_master_grads()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    # optimizer.zero_grad()
                self.accumulate_count = 0
            else:
                self.accumulate_count += 1
            if lr_scheduler:
                lr_scheduler.step()
            self.timers('optimizer').stop()

        else:
            log_dist("Found NaN loss, skip backward", [0])
            del lm_loss, reduced_loss
            mems = None
            reduced_loss = None
        return reduced_loss, mems

    def train_step_pytorchDDP(self,
                              data,
                              model,
                              optimizer,
                              lr_scheduler,
                              mems=None):
        """Single training step."""

        from contextlib import nullcontext
        if self.fp16:
            no_sync = model.module.no_sync
        else:
            no_sync = model.no_sync

        mycontext = no_sync if (
            self.accumulate_count +
            1) != self.gradient_accumulation_steps else nullcontext

        with mycontext():
            # Forward model for one step.
            self.timers('forward').start()
            step_output = self.forward_step(data, model, mems)
            self.timers('forward').stop()

            # accumulate gradients
            lm_loss = step_output['loss']
            lm_loss /= self.gradient_accumulation_steps
            # reduce sum of losses
            reduced_loss = lm_loss.detach().clone().view(1)
            # dist.all_reduce(reduced_loss.data)
            # reduced_loss.data = reduced_loss.data / self.world_size

            # skip the iter while loss has NAN
            if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
                # Calculate gradients, reduce across processes, and clip.
                self.timers('backward').start()

                if self.fp16 and hasattr(optimizer, 'backward'):
                    log_dist("The optimizer has backward function")
                    optimizer.backward(lm_loss,
                                       update_master_grads=False,
                                       retain_graph=True)
                else:
                    lm_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.module.parameters(),
                                               self.clip_grad)
                self.timers('backward').stop()

                # Update parameters.
                self.timers('optimizer').start()
                if (self.accumulate_count +
                        1) % self.gradient_accumulation_steps == 0:
                    if self.fp16:
                        optimizer.update_master_grads()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        # model.zero_grad()

                    self.accumulate_count = 0
                else:
                    self.accumulate_count += 1
                if lr_scheduler:
                    lr_scheduler.step()
                self.timers('optimizer').stop()
                dist.barrier()

            else:
                log_dist("Found NaN loss, skip backward", [0])
                del lm_loss, reduced_loss
                mems = None
                reduced_loss = None
        return reduced_loss, mems

    def train_step_deepspeed(self,
                             data,
                             model,
                             optimizer,
                             lr_scheduler,
                             mems=None,
                             single_step=False):
        """Single training step."""

        # Forward model for one step.
        if (self.accumulate_count + 1) % self.gradient_accumulation_steps == 0:
            model.set_gradient_accumulation_boundary(True)
        else:
            model.set_gradient_accumulation_boundary(False)
        self.timers('forward').start()
        step_output = self.forward_step(data, model, mems)
        self.timers('forward').stop()
        lm_loss = step_output['loss']
        reduced_loss = lm_loss.detach().clone().view(1)

        if self.env_type == 'deepspeed+mpu':
            torch.distributed.all_reduce(reduced_loss.data,
                                         group=mpu.get_data_parallel_group())
        elif self.env_type == 'deepspeed':
            torch.distributed.all_reduce(reduced_loss.data)
        if 'deepspeed' in self.env_type:
            reduced_loss.data = reduced_loss.data / \
                (self.world_size / self.model_parallel_size)
        if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
            # Calculate gradients, reduce across processes, and clip.
            self.timers('backward').start()
            model.backward(lm_loss)
            self.timers('backward').stop()
            # Update parameters.
            self.timers('optimizer').start()
            model.step()
            if lr_scheduler:
                lr_scheduler.step()
            self.timers('optimizer').stop()
            if (self.accumulate_count +
                    1) % self.gradient_accumulation_steps == 0:
                self.accumulate_count = 0
            else:
                self.accumulate_count += 1
            dist.barrier()
        else:
            log_dist("Found NaN loss, skip backward", [0])
            del lm_loss, reduced_loss
            mems = []
            reduced_loss = None
        return reduced_loss, mems

    def forward_step(self, data, model, mems=None):
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
                if self.env_type == 'pytorchDDP':
                    optimizer.step()

        # if self.train_args.deepspeed or self.train_args.DDP_impl == 'torch':
        self.timers('allreduce').reset()
        if self.env_type == 'pytorch':
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
        return lm_loss

    def _gather_all(self, input_):

        # Bypass the function if we are using only 1 GPU.
        if torch.distributed.get_world_size() == 1:
            return input_
        # Size and dimension.
        last_dim = input_.dim() - 1
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        tensor_list = [
            torch.empty_like(input_, device=input_.device)
            for _ in range(world_size)
        ]
        tensor_list[rank] = input_

        torch.distributed.all_gather(tensor_list, input_)

        # Note: torch.cat already creates a contiguous tensor.
        if last_dim >= 0:
            output = torch.cat(tensor_list, dim=0).contiguous()
        else:
            output = torch.mean(torch.FloatTensor(tensor_list))

        return output

    def _gather_all_mpu(self, input_):
        group = mpu.get_model_parallel_group()

        # Bypass the function if we are using only 1 GPU.
        if torch.distributed.get_world_size(group=group) == 1:
            return input_
        # Size and dimension.
        last_dim = input_.dim() - 1
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)

        tensor_list = [
            torch.empty_like(input_, device=input_.device)
            for _ in range(world_size)
        ]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_, group=group)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output

    def evaluate(self,
                 data_loader=None,
                 model=None,
                 forward_step_func=None,
                 verbose=False):
        """Evaluation."""

        # Turn off checkpoint_activations
        tmp_checkpoint_activations = None
        tmp_model = model
        while hasattr(tmp_model, 'module'):
            tmp_model = tmp_model.module
        # Turn on evaluation mode which disables dropout.
        tmp_model.eval()
        if hasattr(tmp_model,
                   'config') and 'checkpoint_activations' in tmp_model.config:
            tmp_checkpoint_activations = tmp_model.config[
                'checkpoint_activations']
            tmp_model.config['checkpoint_activations'] = False

        mems = None
        metrics = [0. for _ in range(len(self.metric_methods))]

        with torch.no_grad():
            assert data_loader is not None, "val loader is not None."
            all_logits = []
            all_labels = []
            all_losses = []
            for data_iterator in data_loader:
                # Forward evaluation.

                meta = data_iterator.get('meta', None)

                if 'deepspeed' in self.env_type or 'DDP' in self.env_type:
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
                step_output = forward_step_func(data_iterator, model, mems)
                '''when contiguous memory optimizations are enabled, the buffers
                allocated by the optimizations are deallocated during backward pass
                in the absence of backward pass the buffers should be reset after each
                forward pass'''
                if 'deepspeed' in self.env_type and self.deepspeed_activation_checkpointing:
                    deepspeed.checkpointing.reset()
                logits = step_output['logits']
                lm_loss = step_output['loss']

                if 'labels' in data_iterator:
                    labels = data_iterator['labels']
                else:
                    labels = data_iterator['target_ids']

                all_logits.append(logits)
                all_labels.append(labels)
                all_losses.append(lm_loss.view(1))

            if len(self.metric_methods) != 0:
                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

            all_losses = torch.cat(all_losses, dim=0)

            if self.env_type == 'pytorchDDP' or self.env_type == 'deepspeed':
                if len(self.metric_methods) != 0:
                    all_logits = self._gather_all(all_logits)
                    all_labels = self._gather_all(all_labels)
                all_losses = self._gather_all(all_losses)

            elif self.env_type == 'deepspeed+mpu':
                if len(self.metric_methods) != 0:
                    all_logits = self._gather_all_mpu(all_logits)
                    all_labels = self._gather_all_mpu(all_labels)
                all_losses = self._gather_all_mpu(all_losses)

            if all_losses.device != torch.device('cpu'):
                all_losses = all_losses.cpu().detach().numpy()[0]

            for i in range(len(self.metric_methods)):
                eval_method = self.metric_methods[i][1]
                metrics[i] += eval_method(all_logits, all_labels, meta=meta)

        # Move model back to the train mode.

        # model.train()
        tmp_model.train()
        # recover the settings for checkpoint_activations
        if hasattr(tmp_model,
                   'config') and 'checkpoint_activations' in tmp_model.config:
            tmp_model.config[
                'checkpoint_activations'] = tmp_checkpoint_activations
        metric_dct = {}
        for i in range(len(self.metric_methods)):
            metric_name = self.metric_methods[i][0]
            metric_dct.update({metric_name: metrics[i]})
        metric_dct.update({"loss": all_losses})
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
        # log_string += ' gradient_accumulation {}/{}'.format(self.accumulate_count, self.gradient_accumulation_steps)
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
        # with open("results.txt", "a") as myfile:
        #     myfile.write(string)
        if self.metric_methods is None:
            return eval_dict

        for i in range(len(self.metric_methods)):
            name = self.metric_methods[i][0]
            string += ", {} {:.3f}".format(name, eval_dict[name])
        # string = ' validation loss at {} | {:.4f},  Acc {:.2f}'.format(
        #     prefix, eval_dict["loss"], eval_dict["metrics"])
        length = len(string) + 1
        log_dist('-' * length, [0])
        log_dist(string, [0])
        log_dist('-' * length, [0])
        return eval_dict