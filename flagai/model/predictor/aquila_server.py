from typing import List
import torch

def aquila_generate_by_ids_stream(
            model,
            tokenizer,
            input_ids,
            out_max_length: int = 200,
            top_k: int = 30,
            top_p: float = 1.0,
            temperature: float = 1.0,
            seed=1234,
            device="cuda:0"
    ) -> List[str]:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        device = next(model.parameters()).device
        bsz = 1
        max_gen_len = out_max_length

        prompt_tokens = [torch.LongTensor(input_ids)]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(2048, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != 0
        start_pos = min_prompt_size
        prev_pos = 0
        probs_re = []
        tokens_re = []

        with torch.no_grad():

            if max_gen_len == 0:
                logits = model.forward(tokens[:, 0:start_pos], 0, labels=tokens)["logits"]
                logits = logits.softmax(dim=-1)

                probs = []
                for index in range(1, len(tokens[0])):
                    probs.append(logits[0, index-1, tokens[0, index]].cpu().item())

                return probs, prompt_tokens[0].numpy().tolist(), probs

            next_token_list = []
            res_list = ""
            for cur_pos in range(start_pos, total_len):
                logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)["logits"]

                if temperature > 0:
                    logits /= temperature

                    indices_to_remove = logits < torch.topk(
                    logits, top_k)[0][..., -1, None]

                    logits[indices_to_remove] = -float('Inf')

                    probs = torch.softmax(logits, dim=-1)
                    next_token = sample_top_p(probs, top_p, generator=generator)

                    probs_re.append(probs[0][next_token[0]].item())
                    tokens_re.append(next_token[0].item())

                else:
                    next_token = torch.argmax(logits, dim=-1)
                    probs_re.append(0.0)
                    tokens_re.append(next_token[0].item())

                next_token = next_token.reshape(-1)
                # only replace token if prompt has already been generated
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )


                if "###" in tokenizer.decode(next_token.tolist()) or "</" in tokenizer.decode(next_token.tolist()):
                    yield res_list
                    raise StopIteration  

                if len(next_token_list) == 0:
                    tmp = tokenizer.decode(next_token.tolist())
                else :
                    next_token_list.append(next_token.cpu().numpy()[0])
                    tmp = tokenizer.decode(next_token_list)

                if '�' in tmp and len(next_token_list) < 5:
                    if len(next_token_list) == 0:
                        next_token_list.append(next_token.cpu().numpy()[0])
                else:
                    tmp.replace("�", "")
                    next_token_list = []
                    res_list += tmp
                    if len(res_list) >= 10:
                        print(res_list)
                        yield res_list
                        res_list = ""

                tokens[:, cur_pos] = next_token
                prev_pos = cur_pos

def aquila_generate_by_ids(
            model,
            tokenizer,
            input_ids,
            out_max_length: int = 200,
            top_k: int = 30,
            top_p: float = 1.0,
            temperature: float = 1.0,
            seed = 1234,
            device="cuda:0"
    ) -> List[str]:

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        device = next(model.parameters()).device
        bsz = 1
        input_size = len(input_ids)
        max_gen_len = out_max_length

        prompt_tokens = [torch.LongTensor(input_ids)]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(2048, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != 0
        start_pos = min_prompt_size
        prev_pos = 0
        probs_re = []
        tokens_re = []

        if max_gen_len == 0:
            ## 计算每个token的预测概率，而不需要预测下一个了。
            with torch.no_grad():
                logits = model.forward(tokens[:, 0:start_pos], 0, labels=tokens)["logits"]
            logits = logits.softmax(dim=-1)
            # print(logits.shape)

            probs = []
            for index in range(1, len(tokens[0])):
                probs.append(logits[0, index-1, tokens[0, index]].cpu().item())

            return probs, prompt_tokens[0].numpy().tolist(), probs

        for cur_pos in range(start_pos, total_len):
            with torch.no_grad():
                logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)["logits"]

            if temperature > 0:
                logits /= temperature

                indices_to_remove = logits < torch.topk(
                logits, top_k)[0][..., -1, None]

                logits[indices_to_remove] = -float('Inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = sample_top_p(probs, top_p, generator=generator)

                probs_re.append(probs[0][next_token[0]].item())
                tokens_re.append(next_token[0].item())

            else:
                next_token = torch.argmax(logits, dim=-1)
                probs_re.append(0.0)
                tokens_re.append(next_token[0].item())

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[input_size: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(100007)]
            except ValueError:
                pass
            decoded.append(tokenizer.decode(t))


        return decoded[0], tokens_re, probs_re

def sample_top_p(probs, p, generator):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1, generator=generator).long()
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
