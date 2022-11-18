from transformers import PreTrainedModel,CLIPTextModel,CLIPVisionModel
from transformers.models.clip.modeling_clip import contrastive_loss
import datasets
import torch 
from typing import Optional,List
import torch.nn as nn

import torch.distributed as dist
from .modeling_xlmr import RobertaSeriesModelWithTransformation
from .configuration_altclip import RobertaSeriesConfig
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    student_loss = contrastive_loss(similarity)
    return student_loss

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )
allgather = AllGather.apply

class KDmodel(PreTrainedModel):
    def __init__(self,config,):
        super().__init__(config,)
        # init student and teacher
        self.teacher = CLIPTextModel.from_pretrained(config.teacher_model)
        # self.vision_encoder = CLIPVisionModel.from_pretrained(config.teacher_model)
        student_config = RobertaSeriesConfig.from_pretrained(config.student_model)
        student_config.project_dim = self.teacher.config.hidden_size
        student_config.pooler_fn = config.pooler_fn
        # turn True when we need to do mlm and mse.
        student_config.add_lm_task = True
        # no Dropout
        if config.loss_fn != 'cl':
            student_config.hidden_dropout_prob = 0.
            student_config.attention_probs_dropout_prob=0. 

        self.student = RobertaSeriesModelWithTransformation.from_pretrained(config.student_model,config=student_config)
        self.student_config = self.student.config
        self.teacher_config = self.teacher.config
        self.loss_fn =config.loss_fn
        self.kd_type =config.kd_type
        
        self.logit_scale_init_value = 1.5
        # up to rob space
        if self.loss_fn == 'cl':
            self.logit_scale = nn.Parameter(torch.ones([]) * self.logit_scale_init_value)
            
            
        if 'prekd' in self.kd_type:
            self.up_sampler = nn.Linear(self.teacher.config.hidden_size,student_config.hidden_size)
            self._init_weights(self.up_sampler)
        
        # freeze teacher and init weights
        self.freeze()
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.student_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def freeze(self):
        for _,m in self.teacher.named_parameters():
            m.requires_grad_(False)
        # # freeze the word embeddings in student model
        # for n,m in self.student.named_parameters():
        #     if 'embeddings.word_embeddings.weight' in n:
        #         print(n)
        #         m.requires_grad_(False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_input_ids = None,
        teacher_attention_mask = None,
        ## inputs of mlm task.
        mlm_input_ids: Optional[torch.Tensor] = None,
        mlm_attention_mask: Optional[torch.Tensor] = None,
        mlm_token_type_ids: Optional[torch.Tensor] = None,
        mlm_position_ids: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
    ) :
        # last_hidden_state 
        # pooler_output ## EOS's embedding
        # hidden_states ## layer embedding
        # attentions
        teacher_outputs = self.teacher(
            teacher_input_ids,
            teacher_attention_mask,
            output_hidden_states = True
        )
        if 'prekd' in self.kd_type:
            if self.kd_type == 'prekd_embed':
                embeds = self.teacher.get_input_embeddings()
                inputs_embeds = embeds(teacher_input_ids)
                inputs_embeds = self.up_sampler(inputs_embeds)
            elif self.kd_type == 'prekd_word':
                embeds = self.student.get_input_embeddings()
                inputs_embeds = embeds(input_ids)
            student_outputs = self.student(
                inputs_embeds=inputs_embeds,
                return_dict=return_dict,
                output_hidden_states=False,
            )
        else:
            student_outputs = self.student(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
                return_dict=return_dict,
                output_hidden_states=False,
                mode='kd',
            )
            mlm_loss = None
            if mlm_labels is not None:
                mlm_loss = self.student(
                    input_ids=mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    token_type_ids=mlm_token_type_ids,
                    position_ids=mlm_position_ids,
                    labels=mlm_labels,
                    output_hidden_states=False,
                    mode='lm',
                )['mlm_loss']
        # learn pooler
        student_embeds,teacher_embeds = student_outputs['pooler_output'],teacher_outputs.pooler_output

        # loss 
        if self.loss_fn=='cl':
            # normalized features
            teacher_embeds = teacher_embeds / teacher_embeds.norm(p=2, dim=-1, keepdim=True)
            student_embeds = student_embeds / student_embeds.norm(p=2, dim=-1, keepdim=True)
            teacher_embeds_all = allgather(student_embeds, torch.distributed.get_rank(), torch.distributed.get_world_size())
            student_embeds_all = allgather(teacher_embeds, torch.distributed.get_rank(), torch.distributed.get_world_size())
            logit_scale = self.logit_scale.exp()
            logits_per_student = torch.matmul(student_embeds_all, teacher_embeds_all.t()) * logit_scale
            loss = clip_loss(logits_per_student)
        elif self.loss_fn=='mse':
            loss = 0.
            loss_fn = torch.nn.MSELoss()
            mse_loss = loss_fn(student_embeds,teacher_embeds)
            loss += mse_loss
            if mlm_loss is not None: loss += mlm_loss
        elif self.loss_fn=='cosine':     
            from functools import partial
            loss_fn = torch.nn.CosineEmbeddingLoss()
            # partial for reduce redundant parameter 
            loss_fn = partial(loss_fn,target=torch.tensor([1.],device='cuda' if torch.cuda.is_available() else 'cpu'))
            teacher_embeds = teacher_embeds / teacher_embeds.norm(p=2, dim=-1, keepdim=True)
            student_embeds = student_embeds / student_embeds.norm(p=2, dim=-1, keepdim=True)
            loss = loss_fn(student_embeds,teacher_embeds)
        
        return {
            'loss':loss,
            'student_pooler_output':student_embeds,
            'teacher_pooler_putput':teacher_embeds,
            'mse_loss':mse_loss,
            'mlm_loss':mlm_loss,
        }