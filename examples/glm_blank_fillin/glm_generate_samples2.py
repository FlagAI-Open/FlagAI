# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""
import sys
sys.path.append('/data/wang/models/GLMgeneration')
sys.path.append('/data/wang/models/FlagAI')
import os
import torch
import torch.nn.functional as F
import time
from datetime import datetime
# from arguments import get_args
import random
from utils import load_checkpoint
import numpy as np
# from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import GLMLargeChTokenizer


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


# from model import GLMModel



def setup_model(args):
    """Setup model and optimizer."""
    # model = GLMModel.from_pretrain(model_name='glm_large_ch')

    from model import GLMModel
    model =  GLMModel(num_layers=args.num_layers,
                         vocab_size=args.vocab_size,
                         hidden_size=args.hidden_size,
                         num_attention_heads=args.num_attention_heads,
                         embedding_dropout_prob=args.hidden_dropout,
                         attention_dropout_prob=args.attention_dropout,
                         output_dropout_prob=args.hidden_dropout,
                         max_sequence_length=args.max_position_embeddings,
                         max_memory_length=args.mem_length,
                         checkpoint_activations=args.checkpoint_activations,
                         checkpoint_num_layers=args.checkpoint_num_layers,
                         parallel_output=False,
                         relative_encoding=False,
                         block_position_encoding=args.block_lm and not args.masked_lm,
                         output_predict=True,
                         spell_length=None,
                         spell_func='lstm',
                         attention_scale=1.0)


    args.no_load_optim = True

    _ = load_checkpoint(
        model, None, None, args)
    model.cuda(torch.cuda.current_device())
    return model



def glm_sample_sequence(model, tokenizer, context_tokens, context_length,
                    mems=None, end_tokens=None,out_seq_length=512,temperature=0.9,
                    top_k=40):
    tokens = context_tokens.new_full((1, 1), tokenizer.get_command('sop').Id)
    counter = 0
    if mems is None:
        mems = []

    last_beam_num = 1
    while counter < out_seq_length:
        position_ids = context_tokens.new_ones(last_beam_num, 2, 1)
        position_ids[:, 0] = context_length
        position_ids[:, 1] = counter + 1
        attention_mask = context_tokens.new_zeros([1], device=context_tokens.device, dtype=torch.long)
        last_token = tokens[:, -1:]
        next_token_logits, *mems = model(last_token, position_ids, attention_mask, *mems)
        next_token_logits = next_token_logits[:, -1]
        next_token_logits /= temperature
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_logits[indices_to_remove] = -float('Inf')
        log_probs = F.softmax(next_token_logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)[0]
        is_end = prev.item() in end_tokens
        if is_end:
            break
        prev = prev.view(1, 1)
        tokens = prev if tokens is None else torch.cat((tokens, prev), dim=1)
        counter += 1
    return torch.cat((context_tokens, tokens), dim=1), mems



def glm_generate_sample(model, tokenizer,text, top_k=40,seq_length=512,out_seq_length=512,
                     eod_token=50000,temperature=0.9, ):
    device=torch.cuda.current_device()
    model.eval()
    generation_mask = '[gMASK]'
    if 'MASK]' not in text:
        text += ' ' + generation_mask
    context_tokens = tokenizer.EncodeAsIds(text)
    context_tokens = [tokenizer.get_command('ENC').Id] + context_tokens
    if not text.endswith('[gMASK]'):
        context_tokens = context_tokens + [tokenizer.get_command('eos').Id]
    context_length = len(context_tokens)
    context_length_tensor = torch.cuda.LongTensor([context_length])
    context_length = context_length_tensor[0].item()
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    text = tokenizer.DecodeIds(context_tokens_tensor.tolist())

    start_time = time.time()
    mems = []
    tokens = context_tokens_tensor
    tokens = tokens.view(1, -1).contiguous()
    tokens = tokens.to(device)
    attention_mask = torch.tensor([tokens.size(1)], device=device, dtype=torch.long)
    position_ids = torch.arange(tokens.size(1), device=device, dtype=torch.long)
    block_position_ids = torch.zeros(tokens.size(1), device=device, dtype=torch.long)
    position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    position_ids = position_ids.unsqueeze(0)
    mask_tokens = ['MASK', 'sMASK', 'gMASK']
    mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
    end_tokens = [tokenizer.get_command('eop').Id, eod_token]
    mask_positions = []
    for token in mask_tokens:
        mask_positions += (context_tokens_tensor == token).nonzero(as_tuple=True)[0].tolist()
    mask_positions.sort()
    _, *mems = model(tokens, position_ids, attention_mask, *mems)
    for mask_position in mask_positions:
        position = mask_position
        tokens, mems = glm_sample_sequence(model, tokenizer, tokens, position,
                                         mems=mems, end_tokens=end_tokens,
                                       out_seq_length=out_seq_length,temperature=temperature, top_k=top_k)
    output_tokens_list = tokens.view(-1).contiguous()
    os.system('clear')
    print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
    print("\nContext:", text, flush=True)
    decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
    trim_decode_tokens = decode_tokens
    print("\nGLM:", trim_decode_tokens, flush=True)


from argparse import ArgumentParser
from flagai.model.predictor.utils import glm_generate_sample
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples')
    # Arguments.
    parser =  ArgumentParser(description='PyTorch BERT Model')
    args = parser.parse_args([])
    args.seed=1234
    args.num_layers=24
    args.vocab_size=50048
    args.hidden_size=1024
    args.num_attention_heads=16
    args.hidden_dropout=0.1
    args.attention_dropout=0.1
    args.hidden_dropout=0.1
    args.max_position_embeddings=1024
    args.mem_length = 511
    args.checkpoint_activations = False
    args.checkpoint_num_layers = 1
    args.block_lm=True
    args.masked_lm=False
    args.load='/mnt/model_save/glm-large-ch-300M/186000/'
    args.deepspeed=False
    args.finetune=False
    args.no_load_rng=False


    # Random seeds for reproducability.
    set_random_seed(443)
    tokenizer = GLMLargeChTokenizer(vocab_path='./checkpoints/glm-large-ch/cog-pretrain.model',
                                    add_block_symbols=True,
                                    add_task_mask=True,
                                    add_decoder_mask=False,
                                    fix_command_token=False)



    # Model, optimizer, and learning rate.
    model = setup_model(args)
    predictor = Predictor(model, tokenizer)
    # generate samples
    text = '问题：吃什么东西养胃？回答：[gMASK]'
    glm_generate_sample(model, tokenizer,text )

    # text = '凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。'
    # glm_generate_sample(model, tokenizer, text )
    #
    # text = '工业互联网（Industrial Internet）是新一代信息通信技术与工业经济深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK]它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成平台化设计、智能化制造、网络化协同、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。'
    # glm_generate_sample(model, tokenizer, text)



