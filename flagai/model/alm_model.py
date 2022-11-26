from .glm_model import GLMModel
from flagai.model.base_model import BaseModel
from torch import nn


class ALMModel(GLMModel):
    pass
    

class ALMForSeq2Seq(BaseModel):

    def __init__(self, config, take_softmax=True, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.model = ALMModel(config)
        self.model.output_predict = True
        self.take_softmax = take_softmax

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # [h.remove() for h in self.hook_handles]
        sd = self.model.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def load_weights(self, checkpoint_path):
        self.model.load_weights_glm(checkpoint_path)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def forward(self,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                loss_mask=None,
                target_ids=None,
                logit_mask=None,
                prompt_pos=None,
                **kwargs):
        '''
        input_ids: 4 x 768
        target_ids: 4 x 768
        position_ids: 4 x 2 x 768
        attention_mask: 16
        '''
        if target_ids is None:
            return self.model(input_ids, position_ids, attention_mask)
        assert len(input_ids.shape) == 2
        model_out = self.model(input_ids,
                               position_ids,
                               attention_mask,
                               prompt_pos=prompt_pos)
        outputs, mems = model_out['logits'], model_out['hidden_states']
        vocab_size = outputs.size()[-1]
        target_ids = target_ids.view(-1)
        loss_mask = loss_mask.view(-1).float()
        Loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        logits = outputs.view(-1, vocab_size)

        loss = (Loss(logits, target_ids) * loss_mask).sum() / loss_mask.sum()
        return {"loss": loss, "hidden_states": mems, "logits": logits}