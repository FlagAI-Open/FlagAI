# coding=utf-8
# Copyright 2022 paper "A Contrastive Framework for Neural Text Generation"
# code repository https://github.com/yxuansu/SimCTG
import sys
import os
import operator
import torch
from torch import nn
import random
import numpy as np
import torch.nn.functional as F

def ranking(context_hidden, next_hidden, next_top_k_ids, next_top_k_probs, alpha):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
    '''
    beam_width, context_len, embed_dim = context_hidden.size()
    assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
    assert cosine_matrix.size() == torch.Size([beam_width, context_len])
    scores, _ = torch.max(cosine_matrix, dim = -1)
    assert scores.size() == torch.Size([beam_width])
    next_top_k_probs = next_top_k_probs.view(-1)
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
    _, selected_idx = torch.topk(scores, k = 1)
    assert selected_idx.size() == torch.Size([1])
    selected_idx = selected_idx.unsqueeze(0)
    assert selected_idx.size() == torch.Size([1,1])
    next_id = torch.gather(next_top_k_ids, dim = 0, index=selected_idx)
    assert next_id.size() == torch.Size([1,1])
    return next_id

def ContrastiveDecodingOneStep(model, input_ids, beam_width, alpha):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''
    device = next(model.parameters()).device
    with torch.no_grad():
        input_ids = torch.as_tensor(input_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
        outputs = model(**{"input_ids": input_ids, "output_hidden_states": True})
        logits = outputs["logits"]
        prev_hidden_states = outputs["logits"][-1]

    _, _, vocab_size = logits.size()
    seqlen, embed_dim = prev_hidden_states.size()
    p = random.uniform(0, 1)

    logit_for_next_step = logits[:,-1,:]
    assert logit_for_next_step.size() == torch.Size([1, vocab_size])

    next_probs = F.softmax(logit_for_next_step, dim = -1)
    assert next_probs.size() == logit_for_next_step.size()

    _, top_k_ids = torch.topk(logit_for_next_step, dim = -1, k = beam_width)
    assert top_k_ids.size() == torch.Size([1, beam_width])
        
    top_k_probs = torch.gather(next_probs, dim = 1, index=top_k_ids)

    assert top_k_probs.size() == top_k_ids.size()
    # compute new hidden 
    expanded_context = [input_ids for _ in range(beam_width)]
    expanded_context = torch.cat(expanded_context, dim = 0)
    assert expanded_context.size() == torch.Size([beam_width, seqlen])
    top_k_ids = top_k_ids.view(beam_width, 1)
    next_input_ids = torch.cat([expanded_context, top_k_ids], dim = -1)
    assert next_input_ids.size() == torch.Size([beam_width, seqlen+1])
    #new_hidden_states, next_logits = model.compute_logits_and_hidden_states(next_input_ids)
    with torch.no_grad():
        #next_input_ids = torch.tensor(next_input_ids, device=device)
        if next_input_ids.ndim == 1:
            next_input_ids = next_input_ids.view(1, -1)
        outputs = model(**{"input_ids": next_input_ids, "output_hidden_states": True})
        next_logits = outputs["logits"]
        new_hidden_states = outputs["logits"]
    assert new_hidden_states.size() == torch.Size([beam_width, seqlen+1, embed_dim])
    context_hidden = new_hidden_states[:,:seqlen,:]
    assert context_hidden.size() == torch.Size([beam_width, seqlen, embed_dim])
    next_hidden = new_hidden_states[:,seqlen:,:]
    assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])

    next_id = ranking(context_hidden, next_hidden, top_k_ids, top_k_probs, alpha)       

    next_input_ids = torch.cat([input_ids, next_id], dim = -1)
    assert next_input_ids.size() == torch.Size([1, seqlen+1])
    return next_input_ids

def contrastive_search(model, tokenizer, text, input_max_length, out_max_length,
                       beam_size, alpha=0.6, end_of_sequence_token_id=None):
    '''
       model: model instance
       tokenizer: tokenizer instance
       text: text input
       out_max_length: how many tokens to generate
       beam_size: size of candidate pool during decoding
       alpha: regulates importance of model confidence and degeneration penalty
    '''
    assert alpha >= 0. and alpha <= 1.0

    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    input_ids = tokenizer_out["input_ids"][:-1]
    input_ids = np.array(input_ids).reshape(1, -1)
    input_ids_len = input_ids.shape[1]
    for step in range(out_max_length):
        input_ids = ContrastiveDecodingOneStep(model, input_ids, beam_size, alpha)
    output = tokenizer.decode(input_ids[0].cpu().numpy())
    output = output[input_ids_len:]
    end_pos = len(output)
    if end_of_sequence_token_id is not None:
        for idx in range(len(output)):
            if output[idx] == end_of_sequence_token_id:
                end_pos = idx+1;
                break
    return output[:end_pos]
