import torch
import torch.nn.functional as F
from .generation_utils import BeamHypotheses, apply_repetition_penalty, top_k_top_p_filtering, pad

class T5Generation:
    def __init__(self, model, tokenizer):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.end_tokens = [self.tokenizer.eos_token_id]

    def _convert_to_tensors(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text, max_length=512, truncation=True)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])
        rest = 4 - len(input_ids) % 4 if len(input_ids) % 4 != 0 else 0
        model_inputs["input_ids"] += [0] * rest
        model_inputs["attention_mask"] += [0] * rest
        model_inputs["decoder_input_ids"] = [self.tokenizer.pad_token_id] # t5 use pad_token_id as decode start
        model_inputs["decoder_attention_mask"] = [1] * len(model_inputs["decoder_input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs

    def _process_texts(self, text_list):
        input_tensors = list(map(self._convert_to_tensors, text_list))
        keys = set(input_tensors[0].keys())
        padded = {}
        for key in keys:
            padded[key] = pad(input_tensors, key, padding_side='right').cuda()
        return padded

    def generate(self, text_list, **kwargs):
        model_inputs = self._process_texts(text_list)
        with torch.inference_mode():
            result = self._decode(model_inputs, **kwargs)
        return result

    def _decode(self, model_inputs, **kwargs):
        raise NotImplementedError("_decode is not implemented.")


class T5BeamSearch(T5Generation):
    def _decode(
        self,
        model_inputs,
        beam_size=3,
        max_length=32,
        repetition_penalty=1.0,
        repetition_window=None,
        **kwargs
    ):
        """
        Beam search
        Args:
            model_inputs (dict): input ids.
            beam_size (int, optional, defaults to 3): beam size of beam search.
            generate_length (int, optional, defaults to 100): maximum generation length.
            repetition_penalty (float, optional, defaults to 1.0): repetition penalty coefficient, 1.0 means no penalty.
            repetition_window (int, optional, defaults to None): window size of repetition penalty, None means that all output tokens are penalized.
        """   # noqa: E501
        # generate_length + 1 for EOS token
        max_length += 1

        # expand dimmension
        batch_size = model_inputs["input_ids"].size(0)
        input_ids = model_inputs["input_ids"].unsqueeze(1).expand(batch_size, beam_size, -1).contiguous().view(batch_size * beam_size, -1)
        attention_mask = model_inputs["attention_mask"].unsqueeze(1).expand(batch_size, beam_size, -1).contiguous().view(batch_size * beam_size, -1)
        decoder_input_ids = model_inputs["decoder_input_ids"].unsqueeze(1).expand(batch_size, beam_size, -1).contiguous().view(batch_size * beam_size, -1)
        decoder_attention_mask = model_inputs["decoder_attention_mask"].unsqueeze(1).expand(batch_size, beam_size, -1).contiguous().view(batch_size * beam_size, -1)

        done = [False for _ in range(batch_size)]

        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(beam_size, max_length, length_penalty=1, early_stopping=False)
            for _ in range(batch_size)
        ]

        pred_start_index = decoder_input_ids.size(-1)
        encoder_outputs = None
        for i in range(max_length + 1):
            if i == 0:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict = True,
                )
                logits, encoder_outputs = out.logits, out.encoder_last_hidden_state
            else:
                out = self.model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict = True,
                )
                logits = out.logits

            # skip all steps when we are done with each sentence
            if all(done):
                break

            # (batch * beam, seqlen, model_dim)
            logits = logits[:, -1, :]

            for token in self.tokenizer.additional_special_tokens_ids:
                if token not in self.end_tokens:
                    logits[:, token] = -float("inf")
            if i == 0:
                for end_token in self.end_tokens: 
                    logits[:, end_token] = -float("inf")

            apply_repetition_penalty(
                logits,
                batch_size,
                beam_size,
                decoder_input_ids,
                repetition_penalty,
                pred_start_index,
                decoder_input_ids.size(-1) - 1,
                repetition_window,
            )
            scores = F.log_softmax(logits, dim=-1)

            next_scores = scores + beam_scores[:, None].expand_as(
                scores
            )  # (batch_size * beam_size, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
            next_scores, next_words = torch.topk(
                next_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )

            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)
            next_batch_beam = []

            for sent_id in range(batch_size):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item(), i
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, 0, 0)] * beam_size
                    )  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = torch.div(idx, scores.size(-1), rounding_mode="floor")
                    word_id = idx % scores.size(-1)

                    # end of sentence, or next word
                    if word_id in self.end_tokens or i == max_length:
                        generated_hyps[sent_id].add(
                            decoder_input_ids[sent_id * beam_size + beam_id, pred_start_index:]
                            .clone()
                            .cpu()
                            .tolist(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if i == max_length else beam_size
                if len(next_sent_beam) < beam_size:
                    next_sent_beam.extend([(0, 0, 0)] * (beam_size-len(next_sent_beam)))  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # we have reached the last step
            if i == max_length:
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = decoder_input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = decoder_input_ids.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            decoder_input_ids = decoder_input_ids[beam_idx, :]

            # update input ids
            decoder_input_ids = torch.cat([decoder_input_ids, beam_words.unsqueeze(1)], dim=-1)
            decoder_attention_mask = torch.cat(
                [decoder_attention_mask, torch.ones((decoder_attention_mask.size(0), 1), dtype=torch.int, device=decoder_attention_mask.device)],
                dim=-1,
            )

        # select the best hypotheses
        results = []
        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            results.append(best_hyp)

        result_text = list(map(self.tokenizer.decode, results))
        return result_text

