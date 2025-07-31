import types
import inspect
import subprocess
import functools
import bmtrain as bmt
from typing import Optional, List
from collections import OrderedDict
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from packaging.version import parse as parse_version

from .distilling import BMDistill
from .pruning import BMPrune
from .moe import BMMoE
from .quant import BMQuant
from .utils.config import ConfigParser

def version_checker():
    """
    compatible with different ModelCenter version. (to help adjust the CookTrainer setup)

    return True if the ModelCenter version is earlier than 0.1.3
    """
    result = subprocess.check_output('pip show model-center', shell=True)
    version = parse_version(str(result).split('\\n')[1].split(' ')[-1])

    return version <= parse_version('0.1.3')

def remove_checkpointblock(model):
    """remove CheckpointBlock in bmtrain, to get access to the forward func"""
    for _, v in model.named_modules():

        if isinstance(v, bmt.TransformerBlockList):

            def new_func(list_self, hidden_states, *args):
                for i in range(len(list_self._modules)):
                    hidden_states = list_self._modules[str(i)](hidden_states, *args)
                return hidden_states

            v.forward = types.MethodType(new_func, v)

            for k in v._modules.keys():
                state_dict = v._modules[k].state_dict()
                for kk, vv in v._modules[k]._module.named_modules():
                    if kk+'.weight' in state_dict:
                        vv.weight.data = state_dict[kk+'.weight'].clone().cuda()
                    if kk+'.bias' in state_dict:
                        vv.bias.data = state_dict[kk+'.bias'].clone().cuda()
                v._modules[k] = v._modules[k]._module
    return model


class CookTrainer:
    r"""A basic training manager of BMCook.

    In BMCook, the distillation and pruning will introduce distill_loss, lagrangian_loss and sparsity. 
    They are necessary for gradient backpropagation. CookTrainer will combine the original model outputs
    and these necessary variables and return to users together.

    Args:
        _is_old_modelcenter(`bool`): the model center version checker. After the version 0.1.5 of model-center,
        the output is reformed as a  :class:`ModelOutput`. So CookTrainer will first check the model-center
        version, then modify the outputs.
    
    Example::
    >>> # setup the forward
    >>> CookTrainer.set_forward(your_cook_config, your_model, your_optimizer, your_teacher_model)
    >>> ...
    >>> # use the forward
    >>> outputs = CookTrainer.forward(your_model, your_inputs, your_loss_func)
    """

    _is_old_modelcenter: bool = version_checker()

    @staticmethod
    def forward(model: Module, loss_func: Module, targets: Tensor, model_args: List, model_kwargs: OrderedDict) -> List:
        r"""The core method in :class:`CookTrainer`.

        it combines the original model.forward and loss calculation.

        it should be rewrite in :method:`CookTrainer.set_forward`, if not an AttributeError will be raised.

        Args:
            model: target compressing model, basically from ModelCenter PLMs.
            loss_func: the loss function. supports the format: loss_func(input: Tensor, target: Tensor) -> Tensor
            targets: the target used to calculate loss.
            model_args: args of :method:`model.forward`
            model_kwargs: kwargs of :method:`model.forward`

        Return:
            `[loss, model_outputs, lag_loss, sparsity, d_loss, moe_hidden]`
        """
        raise AttributeError("The staticmethod forward() should be defined in :method:`set_forward`.")

    @classmethod
    def set_compression(cls, cook_config: ConfigParser, model: Optional[Module] = None, optimizer: Optional[Optimizer] = None, teacher: Optional[Module] = None):
        r"""Define the :method:`forward`, and set up :class:`BMPrune`, :class:`BMDistill`, :class:`BMQuant`
        and :class:`BMMoE`.

        :class:`BMPrune` should be ahead of :class:`BMDistill`, because the :method:`forward` is 
        chenged in both :class:`BMPrune` and :class:`BMDistill`.

        The output format: `[loss, logits, lag_loss, sparsity, distill_loss]`

        Args:
            cook_config: BMCook config. You can set the compression strategy in this config and use 
                `bmcook.utils.config.ConfigParse` to parse it.
            model: target compressing model, basically from ModelCenter PLMs.
            optimizer: optimizer used to train model.
            teacher: teacher model used to distillation, basically from ModelCenter PLMs.
        """
        model_args_list = inspect.getfullargspec(model.forward).args
        teacher_args_list = inspect.getfullargspec(teacher.forward).args
        if teacher_args_list != model_args_list:
            raise ValueError("the techer forward func differs from model forward func.")
        
        if 'return_logits' in model_args_list:
            model.forward = functools.partial(model.forward, return_logits=True)
        elif 'output_logits' in model_args_list:
            model.forward = functools.partial(model.forward, output_logits=True)

        if cls._is_old_modelcenter:
            def forward(model, loss_func, targets, *model_args, **model_kwargs):
                outputs = model(
                    *model_args, **model_kwargs)
                logits = outputs
                batch, seq_len, vocab_out_size = logits.size()

                loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
                
                ret = [loss, outputs, 0, 0, 0, None]
                return ret
        else:
            def forward(model, loss_func, targets, *model_args, **model_kwargs):
                outputs = model(
                    *model_args, **model_kwargs)
                logits = outputs.logits
                batch, seq_len, vocab_out_size = logits.size()

                loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

                ret = [loss, outputs, 0, 0, 0, None]
                return ret

        forward_doc = cls.forward.__doc__
        cls.forward = forward
        cls.forward.__doc__ = forward_doc

        # remove CheckpointBlock
        model = remove_checkpointblock(model)

        # for pruning
        BMPrune.version = cls._is_old_modelcenter
        BMPrune.compute_mask(model, cook_config)
        cls.forward = BMPrune.set_forward_sprune(cls.forward)
        BMPrune.set_optim_for_pruning(optimizer)

        # for distillation
        BMDistill.version = cls._is_old_modelcenter
        cls.forward = BMDistill.set_forward(model, teacher, cls.forward, cook_config)

        # for quantization
        BMQuant.quantize(model, cook_config)

        # for moefication
        cls.forward = BMMoE.get_hidden(model, cook_config, cls.forward)

        bmt.synchronize()


class CPMAntTrainer:
    r"""CookTrainer for CPM-Ant"""
    
    _is_old_modelcenter = version_checker()

    @staticmethod
    def forward(model, loss_func, targets, *model_args, **model_kwargs):
        raise AttributeError("The staticmethod forward() should be defined in :method:`set_forward`.")
    
    @classmethod
    def set_compression(cls, cook_config, model, optimizer, teacher):
        # remove CheckpointBlock

        def forward(model, loss_func, targets, *model_args, **model_kwargs):
            outputs = model(
                *model_args, **model_kwargs)
            logits = outputs[0]
            batch, seq_len, vocab_out_size = logits.size()

            loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

            ret = [loss, outputs, 0, 0, 0, None]
            
            return ret
        forward_doc = cls.forward.__doc__
        cls.forward = forward
        cls.forward.__doc__ = forward_doc

        # for pruning
        BMPrune.version = cls._is_old_modelcenter
        BMPrune.compute_mask(model, cook_config)
        cls.forward = BMPrune.set_forward_sprune(cls.forward)
        BMPrune.set_optim_for_pruning(optimizer)

        # remove CheckpointBlock
        model = remove_checkpointblock(model)

        # for distillation
        BMDistill.version = cls._is_old_modelcenter
        cls.forward = BMDistill.set_forward(model, teacher, cls.forward, cook_config)

        # for quantization
        BMQuant.quantize(model, cook_config)

        # for moefication
        cls.forward = BMMoE.get_hidden(model, cook_config, cls.forward)

        bmt.synchronize()