from packaging import version
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainner
from examples_prompt.trainers.trainer import BaseTrainer 

    # if is_sagemaker_mp_enabled():
#     import smdistributed.modelparallel.torch as smp

# from transformers.trainer_utils import ShardedDDPOption

# if is_fairscale_available():
#     dep_version_check("fairscale")
#     import fairscale
#     from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
#     from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
#     from fairscale.nn.wrap import auto_wrap
#     from fairscale.optim import OSS
#     from fairscale.optim.grad_scaler import ShardedGradScaler

from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names, is_sagemaker_mp_enabled
from transformers.integrations import is_fairscale_available



if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


class Seq2SeqTrainer(HfSeq2SeqTrainner, BaseTrainer):
    def __init__(self, train_dataset_sizes=None, delta_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset_sizes = train_dataset_sizes
        self.delta_args = delta_args

    def evaluate(
        self,
        eval_dataset: Optional[Dict[str, Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        # TODO: this also needs to be set per dataset
        self._max_length = max_length
        self._num_beams = num_beams
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        }
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    
    
    
    
