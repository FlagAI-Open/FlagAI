from typing import List
import torch



def aquila_generate(
        tokenizer,
        model,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.95,
        prompts_tokens: List[List[int]] = None,
    ) -> List[str]:
        # token_end_id depends
        token_end_id = tokenizer.get_command_id('sep')
        token_unk = tokenizer.get_command_id('unk')

        if prompts_tokens is not None:
            bsz = len(prompts_tokens)
            prompt_tokens = [torch.LongTensor(x) for x in prompts_tokens]
        else:
            bsz = len(prompts)
            prompt_tokens = [torch.LongTensor(tokenizer.encode(x)) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(2048, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = t.clone().detach().long()
        input_text_mask = tokens != 0
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)["logits"]
            #print(logits.shape)
            if temperature > 0:
                logits /= temperature
                indices_to_remove = logits < torch.topk(
                    logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
                probs = torch.softmax(logits, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            if token_end_id == next_token.item() or token_unk == next_token.item():
                break
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            #t = t[: len(prompt_tokens[i]) + max_gen_len]
            t = t[len(prompt_tokens[i]):max_gen_len]
            tt = []
            for j in t:
                if token_end_id == j or token_unk == j:
                    break
                tt.append(j)
            decoded.append(tokenizer.decode(tt))
        return decoded[0]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    probs_sort.float()
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

