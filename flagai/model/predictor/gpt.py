from flagai.model.predictor.utils import RepetitionPenaltyLogitsProcessor, TemperatureLogitsProcessor, TopPLogitsProcessor, TopKLogitsProcessor, ListProcessor
import torch
import torch.nn.functional as F


def gpt_random_sample_use_cache(model, tokenizer, text, input_max_length, out_max_length,
                      top_k, top_p, repetition_penalty, temperature, device):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    token_ids = tokenizer_out["input_ids"]
    token_end_id = tokenizer.token_end_id
    if token_ids[-1] == token_end_id:
        token_ids = token_ids[:-1]

    lp = [
        RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
        TemperatureLogitsProcessor(temperature=temperature),
        TopKLogitsProcessor(top_k=top_k),
        TopPLogitsProcessor(top_p=top_p),
    ]
    list_processor = ListProcessor(lp)

    token_ids = torch.tensor(token_ids, device=device,
                             dtype=torch.long).view(1, -1)
    output_ids = []
    sep_id = token_end_id
    outputs = model(**{"input_ids": token_ids, "use_cache": True})
    scores = outputs["logits"]
    past_key_values = outputs["hidden_states"]

    logit_score = torch.log_softmax(scores[:, -1], dim=-1)
    logit_score[:, tokenizer.get_command_id('unk')] = -float('Inf')

    filtered_logits = list_processor(token_ids, logit_score)
    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1),
                                   num_samples=1)
    token_ids = torch.cat([token_ids, next_token.long()], dim=1)

    with torch.no_grad():
        for step in range(out_max_length - 1):
            outputs = model(**{"input_ids": next_token, "use_cache": True, "past_key_values": past_key_values})
            scores = outputs["logits"]
            past_key_values = outputs["hidden_states"]

            logit_score = torch.log_softmax(scores[:, -1], dim=-1)
            logit_score[:, tokenizer.get_command_id('unk')] = -float('Inf')

            filtered_logits = list_processor(token_ids, logit_score)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1),
                                           num_samples=1)
            if sep_id == next_token.item():
                break
            output_ids.append(next_token.item())
            token_ids = torch.cat((token_ids, next_token.long()), dim=1)

    return tokenizer.decode(output_ids)