# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import json
import os
from typing import List
import torch
import numpy as np
import torch.nn.functional as F
import time
from PIL import Image
from itertools import islice
from transformers import AutoFeatureExtractor
import math

join = os.path.join


def get_safety_checker():
    # load safety model
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
        safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        safety_model_id)
    return safety_checker, safety_feature_extractor


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize(
            (hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def load_config(config_path):
    with open(config_path) as f:
        j = json.load(f)

    return j


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize(
            (hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(safety_checker, safety_feature_extractor, x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image),
                                                    return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:

        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty,
                            score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores


class TemperatureLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsWarper` for temperature (exponential scaling output probability distribution).
    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(
                f"`temperature` has to be a strictly positive float, but is {temperature}"
            )

        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor,
                 scores: torch.Tensor) -> torch.FloatTensor:
        scores = scores / self.temperature
        return scores


class TopPLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.
    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to top_p or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self,
                 top_p: float,
                 filter_value: float = -float("Inf"),
                 min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(
                f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        # print(sorted_logits.softmax(dim=-1))
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopKLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self,
                 top_k: int,
                 filter_value: float = -float("Inf"),
                 min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"`top_k` has to be a strictly positive integer, but is {top_k}"
            )

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep),
                    scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1,
                                                                  None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class ListProcessor(LogitsProcessor):

    def __init__(self, list_processor: List[LogitsProcessor]) -> None:
        super().__init__()
        self.list_processor = list_processor

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:

        for processor in self.list_processor:

            scores = processor(input_ids, scores)

        return scores


class BeamHypotheses:
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, mems=None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (max(hyp.shape[-1], 1) ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, mems))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def viterbi_decode(nodes, trans):
    """
    nodes: (seq_len, target_size)
    trans: (target_size, target_size)
    """
    scores = nodes[0]
    scores[1:] -= 100000  # 刚开始标签肯定是"O"
    target_size = nodes.shape[1]
    seq_len = nodes.shape[0]
    labels = torch.arange(0, target_size).view(1, -1)
    path = labels
    for pos_t in range(1, seq_len):
        scores = scores.view(-1, 1)
        M = scores + trans + nodes[pos_t].view(1, -1)
        scores, ids = M.max(0)
        path = torch.cat((path[:, ids], labels), dim=0)

    return path[:, scores.argmax()]


def decode_labels(labels, target):
    entities = []
    starting = False
    for i, label in enumerate(labels):
        if label > 0:
            label_name = target[label]

            if label_name[0] == "B":
                starting = True
                entities.append([[i], label_name[2:]])
            elif starting:
                entities[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return entities


def bert_predict_generate(model, input_ids, token_type_ids):
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids = torch.tensor(input_ids, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
            token_type_ids = token_type_ids.view(1, -1)
        score = model(**{
            "input_ids": input_ids,
            "segment_ids": token_type_ids
        })["logits"]
    return score


def gpt_predict_generate(model, input_ids):
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids = torch.tensor(input_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
        score = model(**{"input_ids": input_ids})["logits"]
    return score


def bert_beam_search(model,
                     token_ids,
                     word2ix,
                     token_type_ids=None,
                     beam_size=1,
                     out_max_length=50):
    """
    beam-search operation
    """
    sep_id = word2ix["[SEP]"]
    if token_type_ids is None:
        token_type_ids = np.zeros_like(token_ids).astype(np.int64)

    output_ids = None

    with torch.no_grad():
        output_scores = np.zeros([1])
        new_token_type_ids = token_type_ids
        new_input_ids = token_ids
        for step in range(out_max_length):
            if step == 0:
                scores = bert_predict_generate(model, token_ids,
                                               token_type_ids)
                #repeat beam-size times:
                token_ids = np.tile(token_ids.reshape([1, -1]), [beam_size, 1])
                token_type_ids = np.tile(token_type_ids.reshape([1, -1]),
                                         [beam_size, 1])
            else:
                scores = bert_predict_generate(model, new_input_ids,
                                               new_token_type_ids)

            logit_score = F.log_softmax(scores[:, -1], dim=-1).cpu().numpy()

            logit_score = output_scores.reshape(
                [-1, 1]) + logit_score  #cumulate logit score
            logit_score = logit_score.reshape([-1])  #flatten
            hype_pos = np.argpartition(logit_score, -beam_size,
                                       axis=-1)[-beam_size:]
            hype_score = logit_score[hype_pos]
            indice1 = (hype_pos // scores.shape[-1]).reshape([-1])  #row index
            indice2 = (hype_pos % scores.shape[-1]).astype(np.int64).reshape(
                [-1, 1])  #col index

            output_scores = hype_score
            if output_ids is None:
                output_ids = indice2.reshape([beam_size, 1])
            else:
                output_ids = np.concatenate([output_ids[indice1], indice2],
                                            axis=1).astype(np.int64)

            new_input_ids = np.concatenate([token_ids, output_ids], axis=1)
            new_token_type_ids = np.concatenate(
                [token_type_ids, np.ones_like(output_ids)], axis=1)

            end_counts = (output_ids == sep_id).sum(
                1)  #Binary vector dim:beamsize
            best_one = output_scores.argmax()
            if end_counts[best_one] == 1:
                #there is end_id in the highest score output sequence
                return output_ids[best_one][:-1]
            else:
                flag = (end_counts < 1)  #flag=False if there is no end_id
                if not flag.all():  #remove the finished ones(have end_id)
                    token_ids = token_ids[flag]
                    token_type_ids = token_type_ids[flag]
                    new_input_ids = new_input_ids[flag]
                    new_token_type_ids = new_token_type_ids[flag]
                    output_ids = output_ids[flag]
                    output_scores = output_scores[flag]
                    beam_size = flag.sum()

        return output_ids[output_scores.argmax()]

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Optional, Tuple, List, Iterable

class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PretrainedModel.beam_search` and
    :meth:`~transformers.PretrainedModel.beam_sample`.
    """

    @abstractmethod
    def process(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            **kwargs
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def finalize(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            **kwargs
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    r"""
    :class:`transformers.BeamScorer` implementing standard beam search decoding.

    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        device (:obj:`torch.device`):
            Defines the device type (*e.g.*, :obj:`"cpu"` or :obj:`"cuda"`) on which this instance of
            :obj:`BeamSearchScorer` will be allocated.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
    """

    def __init__(
            self,
            batch_size: int,
            max_length: int,
            num_beams: int,
            device: torch.device,
            length_penalty: Optional[float] = 1.0,
            do_early_stopping: Optional[bool] = False,
            num_beam_hyps_to_keep: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            mems=None
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.num_beams)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        device = next_scores.device
        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                        len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.num_beams + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        mems=[mem[[next_index.item()]] for mem in mems] if mems else None
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.num_beams:
                    break

            if beam_idx < self.num_beams:
                raise ValueError(
                    f"At most {self.num_beams} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
            self,
            input_ids: torch.LongTensor,
            final_beam_scores: torch.FloatTensor,
            final_beam_tokens: torch.LongTensor,
            final_beam_indices: torch.LongTensor,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            mems=None
    ) -> Tuple[torch.LongTensor, List[torch.Tensor]]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score, mems=[mem[[batch_beam_idx]] for mem in mems] if mems else None)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                score, best_hyp, mems = sorted_hyps.pop()
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append((best_hyp, mems, score))

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item(), self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        scores = final_beam_scores.new(batch_size * self.num_beam_hyps_to_keep)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        mems = []
        for i, (hypo, mem, score) in enumerate(best):
            scores[i] = score
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id
            mems.append(mem)
        mems = [torch.cat([mem[i] for mem in mems], dim=0) for i in range(len(mems[0]))] if mems and mems[0] else None
        return decoded, mems, scores


def alm_beam_search(model,
                    tokenizer,
                    context_tokens,
                    context_length,
                    mems=None,
                    end_tokens=[50007, 50000],
                    out_max_length=512,
                    beam_size=40,
                    top_k=40):
    tokens = context_tokens.new_full((1, 1), tokenizer.get_command_id('sop'))
    counter = 0
    if mems is None:
        mems = []

    last_beam_num = 1
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        max_length=out_max_length,
        num_beams=beam_size,
        device=context_tokens.device,
        length_penalty=0.0,
        do_early_stopping=False,
    )
    beam_scores = torch.zeros(1, dtype=torch.float, device=context_tokens.device)
    while counter < out_max_length:
        position_ids = context_tokens.new_ones(last_beam_num, 2, 1)
        position_ids[:, 0] = context_length
        position_ids[:, 1] = counter + 1
        attention_mask = context_tokens.new_zeros([1],
                                                  device=context_tokens.device,
                                                  dtype=torch.long)

        last_token = tokens[:, -1:]
        
        output_ = model(last_token,
                        position_ids,
                        attention_mask,
                        mems=mems,
                        return_memory=True)

        mems = output_['hidden_states']
        next_token_logits = output_['logits']
        next_token_logits = next_token_logits[:, -1]


        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(1, last_beam_num * vocab_size)

        probs = F.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=2 * beam_size)
        next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
        next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
        next_tokens = torch.gather(next_tokens, -1, _indices)

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size
        # stateless
        tokens = tokens.expand((beam_size, -1))
        beam_outputs = beam_scorer.process(
            tokens,
            next_token_scores,
            next_tokens,
            next_indices,
            eos_token_id=end_tokens,
            mems=mems
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        beam_next_tokens = beam_next_tokens.unsqueeze(-1)
        tokens = torch.cat([tokens[beam_idx, :], beam_next_tokens], dim=-1)
        mems = [mem[beam_idx] for mem in mems] if mems else None
        if beam_scorer.is_done:
            break
        last_beam_num = beam_size
        counter += 1 

    tokens, mems, _ = beam_scorer.finalize(tokens, beam_scores, next_tokens, next_indices, eos_token_id=50000,
                                                mems=mems)
    return torch.cat((context_tokens, tokens), dim=1), mems


def glm_beam_search(model,
                    tokenizer,
                    input_ids,
                    position_ids,
                    attention_mask,
                    beam_size=1,
                    out_max_length=50):

    device = next(model.parameters()).device
    end_id = tokenizer.command_name_map['eop'].Id
    output_ids = None
    lp = [RepetitionPenaltyLogitsProcessor(penalty=10.0)
          ]  #example of adding logits processor in beam search
    list_processor = ListProcessor(lp)
    with torch.no_grad():
        output_scores = np.zeros([1])
        new_input_ids = input_ids
        for step in range(out_max_length):
            if step == 0:
                scores = model(
                    input_ids=input_ids,  #GLMModel
                    position_ids=position_ids,
                    attention_mask=attention_mask)[
                        "logits"]  #scores size:(1,max_src_length,vocab_size)
                input_ids = torch.tile(input_ids.reshape([1, -1]),
                                       (beam_size, 1))
                position_ids = torch.tile(position_ids, (beam_size, 1, 1))
                attention_mask = torch.tile(attention_mask, (beam_size, ))

                logit_score = F.log_softmax(scores[:, -1],
                                            dim=-1)  #(1,vocab_size)
                logit_score[:,
                            tokenizer.CommandTokenIds(
                                exception=["eop", "gMASK"])] = -float(
                                    'Inf')  #Don't generate special tokens
            else:
                scores = model(
                    input_ids=new_input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask
                )["logits"]  #scores size:(beam_size,max_src_length,vocab_size)
                logit_score = F.log_softmax(scores[:, -1],
                                            dim=-1)  #(1,vocab_size)
                logit_score[:,
                            tokenizer.CommandTokenIds(
                                exception=["eop", "gMASK"])] = -float(
                                    'Inf')  #Don't generate special tokens
                #logits process:
                # logit_score = list_processor(
                #     torch.tensor(output_ids, device=device, dtype=torch.long),
                #     logit_score)

            logit_score = output_scores.reshape(
                [-1, 1]) + logit_score.cpu().numpy()  #(beam_size,vocab_size)
            logit_score = logit_score.reshape([-1])
            hype_pos = np.argpartition(
                logit_score, -beam_size,
                axis=-1)[-beam_size:]  #Output the top beam_size largest id
            hype_score = logit_score[hype_pos]
            indice1 = (hype_pos // scores.shape[-1]).reshape([-1])
            indice2 = (hype_pos % scores.shape[-1]).astype(np.int64).reshape(
                [-1, 1])

            output_scores = hype_score
            if output_ids is None:
                output_ids = indice2.reshape([beam_size, 1])
            else:
                output_ids = np.concatenate(
                    [output_ids[indice1], indice2],
                    axis=1).astype(np.int64)  #(beam_size,n) n grows up
            new_input_ids = torch.cat(
                (input_ids,
                 torch.tensor(output_ids, device=device, dtype=torch.long)),
                dim=1)
            add_pos_ids = torch.tile(
                torch.tensor([[[position_ids.size()[-1]], [1]]],
                             device=device), (beam_size, 1, 1))
            position_ids = torch.cat([position_ids, add_pos_ids], dim=2)  #

            end_counts = (output_ids == end_id).sum(
                1)  # Binary vector dim:beamsize
            best_one = output_scores.argmax()
            if end_counts[best_one] == 1:
                #there is end_id in the highest score output sequence
                return output_ids[best_one][:-1]
            else:
                flag = (end_counts < 1)  #flag=False if there is no end_id
                if not flag.all():  #remove the finished ones(have end_id)
                    input_ids = input_ids[flag]
                    position_ids = position_ids[flag]
                    attention_mask = attention_mask[flag]
                    new_input_ids = new_input_ids[flag]
                    output_ids = output_ids[flag]
                    output_scores = output_scores[flag]
                    beam_size = flag.sum()  #beam_size reduce
        if not output_scores.any():
            return None
        return output_ids[output_scores.argmax()].tolist()


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim(
    ) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def t5_random_sample(model, tokenizer, text, input_max_length, out_max_length,
                     top_k, top_p, repetition_penalty, temperature, device):
    token_ids = tokenizer.encode_plus(text,
                                      max_length=input_max_length)["input_ids"]
    token_ids = torch.tensor(token_ids, device=device,
                             dtype=torch.long).view(1, -1)
    output_ids = []
    input_decoder_ids = torch.tensor(tokenizer.get_command_id('cls'),
                                     device=device,
                                     dtype=torch.long).view(1, -1)
    lp = [
        RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
        TemperatureLogitsProcessor(temperature=temperature),
        TopKLogitsProcessor(top_k=top_k),
        TopPLogitsProcessor(top_p=top_p),
    ]
    list_processor = ListProcessor(lp)
    with torch.no_grad():
        for step in range(out_max_length):
            scores = model(**{
                "input_ids": token_ids,
                "decoder_input_ids": input_decoder_ids
            })["logits"]
            logit_score = torch.log_softmax(scores[:, -1], dim=-1)
            logit_score[:, tokenizer.get_command_id('unk')] = -float('Inf')
            # filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
            filtered_logits = list_processor(input_decoder_ids, logit_score)

            filterd_logits_prob = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(filterd_logits_prob, num_samples=1)
            if tokenizer.get_command_id('eos') == next_token.item():
                break
            output_ids.append(next_token.item())
            input_decoder_ids = torch.cat(
                (input_decoder_ids, next_token.long()), dim=1)
    return tokenizer.decode(output_ids)


def bert_random_sample(model, tokenizer, text, input_max_length,
                       out_max_length, top_k, top_p, repetition_penalty,
                       temperature, device):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    token_ids = tokenizer_out["input_ids"]
    token_type_ids = tokenizer_out["token_type_ids"]
    token_ids = torch.tensor(token_ids, device=device,
                             dtype=torch.long).view(1, -1)
    token_type_ids = torch.tensor(token_type_ids,
                                  device=device,
                                  dtype=torch.long).view(1, -1)

    output_ids = []
    lp = [
        RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
        TemperatureLogitsProcessor(temperature=temperature),
        TopKLogitsProcessor(top_k=top_k),
        TopPLogitsProcessor(top_p=top_p),
    ]
    list_processor = ListProcessor(lp)
    with torch.no_grad():
        for step in range(out_max_length):
            scores = model(**{
                "input_ids": token_ids,
                "segment_ids": token_type_ids
            })["logits"]
            logit_score = torch.log_softmax(scores[:, -1], dim=-1)
            logit_score[:, tokenizer.get_command_id('unk')] = -float('Inf')
            filtered_logits = list_processor(token_ids, logit_score)

            filterd_logits_prob = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(filterd_logits_prob, num_samples=1)
            if tokenizer.get_command_id('eos') == next_token.item():
                break
            output_ids.append(next_token.item())
            token_ids = torch.cat((token_ids, next_token.long()), dim=1)
            token_type_ids = torch.cat(
                (token_type_ids,
                 torch.tensor([[1]], device=device, dtype=torch.long)),
                dim=1)
    return tokenizer.decode(output_ids)


def gpt_random_sample(model, tokenizer, text, input_max_length, out_max_length,
                      top_k, top_p, repetition_penalty, temperature, device):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    token_ids = tokenizer_out["input_ids"]
    token_end_id = tokenizer.get_command_id('eos')
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
    sep_id = tokenizer.get_command_id('eos')
    with torch.no_grad():
        for step in range(out_max_length):
            scores = model(**{"input_ids": token_ids})["logits"]
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


def alm_random_sample(model, tokenizer, text, out_max_length, top_k, top_p,
                      repetition_penalty, temperature, device):
    return glm_random_sample(model, tokenizer, text, out_max_length, top_k, top_p,
                      repetition_penalty, temperature, device)


def glm_random_sample(model, tokenizer, text, out_max_length, top_k, top_p,
                      repetition_penalty, temperature, device):
    if 'MASK]' in text:
        return glm_generate_sample(model,
                                   tokenizer,
                                   text,
                                   out_seq_length=out_max_length,
                                   top_k=top_k,
                                   temperature=temperature)

    else:
        #tokenizer:GLMChineseSPTokenizer
        #model input:
        data = tokenizer.encode_plus(text)
        input_ids = torch.tensor([data['input_ids']],
                                 device=device,
                                 dtype=torch.long)
        position_ids = torch.tensor([data['position_ids']],
                                    device=device,
                                    dtype=torch.long)
        attention_mask = torch.tensor(data['attention_mask'],
                                      device=device,
                                      dtype=torch.long)
        #choosabel processor to overlay:
        lp = [
            TemperatureLogitsProcessor(temperature=temperature),
            TopPLogitsProcessor(top_p=top_p),
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
            TopKLogitsProcessor(top_k=top_k)
        ]
        list_processor = ListProcessor(lp)

        output_ids = []
        end_id = tokenizer.command_name_map['eop'].Id
        with torch.no_grad():
            for step in range(out_max_length):
                scores = model(
                    input_ids=input_ids,  #GLMModel
                    position_ids=position_ids,
                    attention_mask=attention_mask)[
                        "logits"]  #[1,max_src_length,vocab_size]
                logit_score = torch.log_softmax(scores[:, -1],
                                                dim=-1)  #[1,vocab_size]
                logit_score[:,
                            tokenizer.CommandTokenIds(
                                exception=["eop", "gMASK"])] = -float('Inf')

                filtered_logits = list_processor(input_ids, logit_score)
                prob = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(prob, num_samples=1)
                if end_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.long()), dim=1)
                new_position_ids = torch.tensor(
                    [[[position_ids.size()[2]], [0]]],
                    device=device,
                    dtype=torch.long)
                position_ids = torch.cat((position_ids, new_position_ids),
                                         dim=2)  #shape:[1,2,inputsize]
        if not output_ids:
            print("None")
        return tokenizer.DecodeIds(output_ids)



def alm_beamsearch(model, tokenizer, text, out_max_length, beam_size, eod_token=50000):  
    device = next(model.parameters()).device 
    model.eval()

    generation_mask = '[gMASK]'
    if 'MASK]' not in text:
        text += ' ' + generation_mask
    context_tokens = tokenizer.EncodeAsIds(text)
    context_tokens = [tokenizer.get_command_id('cls')] + context_tokens
    if not text.endswith('[gMASK]'):
        context_tokens = context_tokens + [tokenizer.get_command_id('eos')]
    context_length = len(context_tokens)
    context_length_tensor = torch.LongTensor([context_length])
    context_length = context_length_tensor[0].item()
    context_tokens_tensor = torch.LongTensor(context_tokens)
    text = tokenizer.DecodeIds(context_tokens_tensor.tolist())

    start_time = time.time()
    mems = []
    tokens = context_tokens_tensor
    tokens = tokens.view(1, -1).contiguous()
    tokens = tokens.to(device)
    attention_mask = torch.tensor([tokens.size(1)],
                                  device=device,
                                  dtype=torch.long)
    position_ids = torch.arange(tokens.size(1),
                                device=device,
                                dtype=torch.long)
    block_position_ids = torch.zeros(tokens.size(1),
                                     device=device,
                                     dtype=torch.long)
    position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    position_ids = position_ids.unsqueeze(0)
    mask_tokens = ['MASK', 'sMASK', 'gMASK']
    mask_tokens = [tokenizer.get_command_id(token) for token in mask_tokens]
    end_tokens = [tokenizer.get_command_id('eop'), eod_token]
    mask_positions = []
    for token in mask_tokens:
        mask_positions += (context_tokens_tensor == token).nonzero(
            as_tuple=True)[0].tolist()
    mask_positions.sort()
    output_ = model(tokens, position_ids, attention_mask, return_memory=True)
    mems = output_['hidden_states']
    for mask_position in mask_positions:
        position = mask_position
        tokens, mems = alm_beam_search(model,
                                           tokenizer,
                                           tokens,
                                           position,
                                           mems=mems,
                                           beam_size=beam_size,
                                           out_max_length=out_max_length)
    output_tokens_list = tokens.view(-1).contiguous()
    decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())

    return decode_tokens

def glm_beamsearch(model, tokenizer, text, out_max_length, beam_size):  #
    #tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    device = next(model.parameters()).device
    data = tokenizer.encode_plus(text)
    input_ids = torch.tensor([data['input_ids']],
                             device=device,
                             dtype=torch.long)
    position_ids = torch.tensor([data['position_ids']],
                                device=device,
                                dtype=torch.long)
    attention_mask = torch.tensor([0], device=device, dtype=torch.long)

    out_puts_ids = glm_beam_search(model,
                                   tokenizer,
                                   input_ids,
                                   position_ids,
                                   attention_mask,
                                   beam_size=beam_size,
                                   out_max_length=out_max_length)
    output = tokenizer.DecodeIds(out_puts_ids)
    return output


def bert_beamsearch(model, tokenizer, text, input_max_length, out_max_length,
                    beam_size):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    vocab = tokenizer.get_vocab()
    token_ids = tokenizer_out["input_ids"]
    token_ids = np.array(token_ids).reshape(1, -1)
    out_puts_ids = bert_beam_search(model,
                                    token_ids,
                                    word2ix=vocab,
                                    beam_size=beam_size,
                                    out_max_length=out_max_length)
    output = tokenizer.decode(out_puts_ids)
    return output


def t5_beamsearch(model, tokenizer, text, input_max_length, out_max_length,
                  beam_size):
    tokenizer_out = tokenizer.encode_plus(
        text,
        max_length=input_max_length,
    )
    token_ids = tokenizer_out["input_ids"]
    token_ids = np.array(token_ids).reshape(1, -1)
    out_puts_ids = t5_beam_search(model,
                                  token_ids,
                                  tokenizer,
                                  beam_size=beam_size,
                                  out_max_length=out_max_length)
    output = tokenizer.decode(out_puts_ids)
    return output


def gpt_beamsearch(model, tokenizer, text, input_max_length, out_max_length,
                   beam_size):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    token_ids = tokenizer_out["input_ids"][:-1]
    token_ids = np.array(token_ids).reshape(1, -1)
    out_puts_ids = gpt_beam_search(model,
                                   token_ids,
                                   tokenizer,
                                   beam_size=beam_size,
                                   out_max_length=out_max_length)
    output = tokenizer.decode(out_puts_ids)
    return output


def cpm_beamsearch(model, tokenizer, text, input_max_length, out_max_length,
                   beam_size):
    # tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    # token_ids = tokenizer_out["input_ids"][:-1]
    # token_ids = np.array(token_ids).reshape(1, -1)
    out_puts_ids = cpm_beam_search(model,
                                   tokenizer,
                                   text,
                                   beam_size=beam_size,
                                   out_max_length=out_max_length)
    output = tokenizer.decode(out_puts_ids)
    return output


def t5_predict_generate(model,
                        input_ids=None,
                        encoder_hidden_state=None,
                        decoder_input_ids=None):

    with torch.no_grad():
        device = next(model.parameters()).device
        decoder_input_ids = torch.tensor(decoder_input_ids,
                                         device=device,
                                         dtype=torch.long)
        if input_ids is not None:
            input_ids = torch.tensor(input_ids,
                                     device=device,
                                     dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.view(1, -1)

            scores = model(**{
                "input_ids": input_ids,
                "decoder_input_ids": decoder_input_ids
            })
        else:
            encoder_hidden_state = torch.from_numpy(encoder_hidden_state).to(
                device)
            scores = model(
                **{
                    "encoder_outputs": [encoder_hidden_state],
                    "decoder_input_ids": decoder_input_ids
                })

    return scores


def t5_beam_search(model,
                   token_ids,
                   tokenizer,
                   beam_size=1,
                   out_max_length=50):

    sep_id = tokenizer.get_command_id('eos')
    decoder_input_ids = np.array(tokenizer.get_command_id('cls'),
                                 dtype=np.int64).reshape(1, -1)

    output_ids = None
    with torch.no_grad():
        output_scores = np.zeros([1])
        for step in range(out_max_length):
            if step == 0:
                pred_out = t5_predict_generate(
                    model,
                    input_ids=token_ids,
                    decoder_input_ids=decoder_input_ids)
                encoder_hidden_state = pred_out[
                    "encoder_last_hidden_state"].cpu().numpy()
                scores = pred_out["logits"]
                decoder_input_ids = np.tile(decoder_input_ids.reshape([1, -1]),
                                            [beam_size, 1])
                encoder_hidden_state = np.tile(encoder_hidden_state,
                                               [beam_size, 1, 1])
            else:
                scores = t5_predict_generate(
                    model,
                    encoder_hidden_state=encoder_hidden_state,
                    decoder_input_ids=decoder_input_ids)["logits"]

            logit_score = F.log_softmax(scores[:, -1], dim=-1).cpu().numpy()

            logit_score = output_scores.reshape([-1, 1]) + logit_score
            logit_score = logit_score.reshape([-1])
            hype_pos = np.argpartition(logit_score, -beam_size,
                                       axis=-1)[-beam_size:]
            hype_score = logit_score[hype_pos]
            indice1 = (hype_pos // scores.shape[-1]).reshape([-1])
            indice2 = (hype_pos % scores.shape[-1]).astype(np.int64).reshape(
                [-1, 1])

            output_scores = hype_score
            if output_ids is None:
                output_ids = indice2.reshape([beam_size, 1])
            else:
                output_ids = np.concatenate([output_ids[indice1], indice2],
                                            axis=1).astype(np.int64)

            decoder_input_ids = np.concatenate([decoder_input_ids, indice2],
                                               axis=1)

            end_counts = (output_ids == sep_id).sum(1)
            best_one = output_scores.argmax()
            if end_counts[best_one] == 1:
                return output_ids[best_one][:-1]
            else:
                flag = (end_counts < 1)
                if not flag.all():
                    decoder_input_ids = decoder_input_ids[flag]
                    output_ids = output_ids[flag]
                    output_scores = output_scores[flag]
                    beam_size = flag.sum()
                    encoder_hidden_state = encoder_hidden_state[flag]

        return output_ids[output_scores.argmax()]


def glm_sample_sequence(model,
                        tokenizer,
                        context_tokens,
                        context_length,
                        mems=None,
                        end_tokens=None,
                        out_seq_length=512,
                        temperature=0.9,
                        top_k=40):
    tokens = context_tokens.new_full((1, 1), tokenizer.get_command_id('sop'))
    counter = 0
    if mems is None:
        mems = []

    last_beam_num = 1

    while counter < out_seq_length:
        position_ids = context_tokens.new_ones(last_beam_num, 2, 1)
        position_ids[:, 0] = context_length
        position_ids[:, 1] = counter + 1
        attention_mask = context_tokens.new_zeros([1],
                                                  device=context_tokens.device,
                                                  dtype=torch.long)

        last_token = tokens[:, -1:]
        
        output_ = model(last_token,
                        position_ids,
                        attention_mask,
                        mems=mems,
                        return_memory=True)

        mems = output_['hidden_states']
        next_token_logits = output_['logits']
        next_token_logits = next_token_logits[:, -1]
        next_token_logits /= temperature
        indices_to_remove = next_token_logits < torch.topk(
            next_token_logits, top_k)[0][..., -1, None]
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


def glm_generate_sample(
    model,
    tokenizer,
    text,
    top_k=40,
    seq_length=512,
    out_seq_length=512,
    eod_token=50000,
    temperature=0.9,
):
    device = next(model.parameters()).device 
    model.eval()

    generation_mask = '[gMASK]'
    if 'MASK]' not in text:
        text += ' ' + generation_mask
    context_tokens = tokenizer.EncodeAsIds(text)
    context_tokens = [tokenizer.get_command_id('cls')] + context_tokens
    if not text.endswith('[gMASK]'):
        context_tokens = context_tokens + [tokenizer.get_command_id('eos')]
    context_length = len(context_tokens)
    context_length_tensor = torch.LongTensor([context_length])
    context_length = context_length_tensor[0].item()
    context_tokens_tensor = torch.LongTensor(context_tokens)
    text = tokenizer.DecodeIds(context_tokens_tensor.tolist())

    start_time = time.time()
    mems = []
    tokens = context_tokens_tensor
    tokens = tokens.view(1, -1).contiguous()
    tokens = tokens.to(device)
    attention_mask = torch.tensor([tokens.size(1)],
                                  device=device,
                                  dtype=torch.long)
    position_ids = torch.arange(tokens.size(1),
                                device=device,
                                dtype=torch.long)
    block_position_ids = torch.zeros(tokens.size(1),
                                     device=device,
                                     dtype=torch.long)
    position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    position_ids = position_ids.unsqueeze(0)
    mask_tokens = ['MASK', 'sMASK', 'gMASK']
    mask_tokens = [tokenizer.get_command_id(token) for token in mask_tokens]
    end_tokens = [tokenizer.get_command_id('eop'), eod_token]
    mask_positions = []
    for token in mask_tokens:
        mask_positions += (context_tokens_tensor == token).nonzero(
            as_tuple=True)[0].tolist()
    mask_positions.sort()
    output_ = model(tokens, position_ids, attention_mask, return_memory=True)
    mems = output_['hidden_states']
    for mask_position in mask_positions:
        position = mask_position
        tokens, mems = glm_sample_sequence(model,
                                           tokenizer,
                                           tokens,
                                           position,
                                           mems=mems,
                                           end_tokens=end_tokens,
                                           out_seq_length=out_seq_length,
                                           temperature=temperature,
                                           top_k=top_k)
    output_tokens_list = tokens.view(-1).contiguous()

    decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())

    return decode_tokens


def gpt_beam_search(model,
                    token_ids,
                    tokenizer,
                    beam_size=1,
                    out_max_length=50):

    sep_id = tokenizer.get_command_id('sep')

    output_ids = None
    with torch.no_grad():
        output_scores = np.zeros([1])
        for step in range(out_max_length):
            if step == 0:
                scores = gpt_predict_generate(model, input_ids=token_ids)
                token_ids = np.tile(token_ids, [beam_size, 1])
            else:
                scores = gpt_predict_generate(model, input_ids=token_ids)

            logit_score = F.log_softmax(scores[:, -1], dim=-1).cpu().numpy()
            logit_score = output_scores.reshape([-1, 1]) + logit_score
            logit_score = logit_score.reshape([-1])
            hype_pos = np.argpartition(logit_score, -beam_size,
                                       axis=-1)[-beam_size:]
            hype_score = logit_score[hype_pos]
            indice1 = (hype_pos // scores.shape[-1]).reshape([-1])
            indice2 = (hype_pos % scores.shape[-1]).astype(np.int64).reshape(
                [-1, 1])

            output_scores = hype_score
            if output_ids is None:
                output_ids = indice2.reshape([beam_size, 1])
            else:
                output_ids = np.concatenate([output_ids[indice1], indice2],
                                            axis=1).astype(np.int64)

            token_ids = np.concatenate([token_ids, indice2], axis=1)

            end_counts = (output_ids == sep_id).sum(
                1)  # Binary vector dim:beamsize
            best_one = output_scores.argmax()
            if end_counts[best_one] == 1:
                #there is end_id in the highest score output sequence
                return output_ids[best_one][:-1]
            else:
                flag = (end_counts < 1)  #flag=False if there is no end_id
                if not flag.all():  #remove the finished ones(have end_id)
                    output_ids = output_ids[flag]
                    output_scores = output_scores[flag]
                    beam_size = flag.sum()  #beam_size reduce
                    token_ids = token_ids[flag]

        return output_ids[output_scores.argmax()]


def convert_to_ids(tokenizer, text):
    ids = tokenizer.encode(text)
    ids = [j for j in ids if j != tokenizer.unk_id]
    return ids


def get_control(control, tokenizer, task):
    sep_id1 = 30665
    sep_id2 = 30666
    keywords = []
    if 'keywords' in control and control['keywords'] != []:
        keywords_set = set()
        for i, keyword in enumerate(control['keywords']):
            if keyword not in keywords_set:
                keywords_set.add(keyword)
                keywords += convert_to_ids(tokenizer, keyword)
            if i != len(control['keywords']) - 1:
                keywords += [sep_id1]
        keywords = [tokenizer.begin_of_keyword_id
                    ] + keywords + [tokenizer.end_of_keyword_id]

    if 'genre' in control and control['genre'] != "" and control[
            'genre'] not in [
                '新闻', '学术', '公文', '诗歌', '武侠仙侠', '玄幻奇幻', '科幻灵异', '军事历史', '言情'
            ]:
        style = convert_to_ids(tokenizer, control['genre'])
        style = [tokenizer.begin_of_style_id
                 ] + style + [tokenizer.end_of_style_id]
    else:
        style = []

    relations = []
    if 'relations' in control and control['relations'] != []:
        relation_set = set()
        for items in control['relations']:
            relation_join = "/".join(items)
            if relation_join not in relation_set:
                relation_set.add(relation_join)
                relation = []
                for i, item in enumerate(items):
                    relation += convert_to_ids(tokenizer, item)
                    if i != len(items) - 1:
                        relation += [sep_id1]
                ids = [tokenizer.begin_of_relation_id
                       ] + relation + [tokenizer.end_of_relation_id]
                relations = relations + ids

    events = []
    if 'events' in control and control['events'] != []:
        events_set = set()
        for i in control['events']:
            event_join = "/".join(
                [":".join(x) for x in sorted(i.items(), key=lambda x: x[0])])
            if event_join not in events_set:
                events_set.add(event_join)
                event = []
                for idx, j in enumerate(i):
                    event += convert_to_ids(tokenizer,
                                            j) + [sep_id2] + convert_to_ids(
                                                tokenizer, i[j])
                    if idx != len(i) - 1:
                        event += [sep_id1]
                ids = [tokenizer.begin_of_event_id
                       ] + event + [tokenizer.end_of_event_id]
                events = events + ids

    if task == 0:
        # lm
        res = keywords + relations + events
    elif task == 1:
        # compress parser
        res = []
    elif task == 2:
        # expand parser
        res = keywords
    elif task == 3:
        # rewrite
        res = style + keywords
    elif task == 4:
        # rewrite_s
        res = []
    elif task == 5:
        # compress_para
        res = keywords
    elif task == 6:
        # expand_para
        res = keywords + relations + events
    else:
        raise ValueError("task id error")

    return res


def encode(tokenizer, i, target_span_len=100, use_target=False):
    task_ids = {
        'lm': 0,
        'compress': 1,
        'expand': 2,
        'rewrite': 3,
        'rewrite_s': 4,
        'compress_para': 5,
        'expand_para': 6,
    }

    task = task_ids[i['mode']]

    ids = []
    info = [task]

    if 'control' in i:
        control = get_control(i['control'], tokenizer, task)
    else:
        control = []

    ids += control
    info.append(len(control))

    assert len(i['source']) <= 2
    src = i['source'][0]

    src_ids = convert_to_ids(tokenizer, src)
    src_ids = [tokenizer.bos_id] + src_ids
    if task != 0:
        src_ids += [tokenizer.eos_id]
    ids += src_ids
    info.append(len(src_ids))
    if not use_target:
        tgt_ids = [0] * target_span_len
    else:
        tgt_ids = convert_to_ids(tokenizer, i['target'])
    if task == 0:
        # 续写（只预测single span)
        if len(i['source']) > 1 and len(i['source'][1]) > 0:
            # 生成中间，可以看见后面的context和eos
            end_ids = convert_to_ids(tokenizer, i['source'][1])
            end_ids += [tokenizer.eos_id]
            ids += tgt_ids + end_ids
            info.extend([len(tgt_ids), len(end_ids)])
        else:
            # 生成结尾
            tgt_ids += [tokenizer.eos_id]
            ids += tgt_ids
            # 选项一：模型生成eos
            info.extend([len(tgt_ids), 0])
            # 选项二：指定长度，给定eos
            # info.extend([len(tgt_ids)-1, 1])
    elif task == 1 or task == 2 or task == 5 or task == 6:
        # 压缩、扩写（句子级、段落级）
        tgt_ids = [tokenizer.bos_id] + tgt_ids + [tokenizer.eos_id]
        ids += tgt_ids
        # 选项一：模型生成eos
        info.extend([len(tgt_ids), 0])
        # 选项二：指定长度，给定eos
        # info.extend([len(tgt_ids)-1, 1])
    else:
        # 改写和改错
        tgt_ids = [tokenizer.bos_id] + tgt_ids + [tokenizer.eos_id]
        ids += tgt_ids
        # 不控制长度，eos自己生成
        info.extend([len(tgt_ids), 0])

    info = info[:1] + np.cumsum(info[1:]).tolist()

    assert len(ids) == info[-1]
    assert len(info) % 2 == 1  # task, control, src, tgt, src, tgt, ...., end
    return ids, info


def make_input_cpm3(ctx, info, prompt_length):
    task = info[0]
    len_ctx = len(ctx)
    inp = np.arange(
        (prompt_length + len_ctx), dtype=np.int64) + prompt_length * task
    inp[prompt_length:] = ctx[:len_ctx]
    len_inp = len(inp)

    info = [x + prompt_length for x in info[1:]]
    context_inp = np.full(len_inp, True)
    # 保证end一定能看见
    for i in range(1, len(info) - 1, 2):
        context_inp[info[i]:info[i + 1]] = False

    tgt = np.full((len_inp), -100, dtype=np.int64)
    tgt[:-1] = np.where(context_inp[1:], -100, inp[1:])

    position_inp = np.arange((len_inp), dtype=np.float32) / prompt_length
    segment_inp = np.zeros((len_inp), dtype=np.int64)

    if task == 0:
        arr = [(2, info[0]), (1, 0), (1, info[-1])]
    else:
        arr = [(2, info[0]), (2 + task, info[1]), (1, info[-1])]

    last = prompt_length
    for (typ, end) in arr:
        if end > last:
            segment_inp[last:end] = typ
            position_inp[last:end] = np.arange(end - last) / (end - last)
            last = end
    assert last == len_inp

    max_length = (len_inp + 2 - 1) // 2 * 2

    _ctx = torch.zeros((max_length, ), dtype=torch.long)
    _ctx[:len_inp] = torch.from_numpy(inp)[:len_inp].long()
    _context = torch.full((max_length, ), False, dtype=torch.bool)
    _context[:len_inp] = torch.from_numpy(context_inp)[:len_inp].bool()
    _position = torch.full((max_length, ), False, dtype=torch.float)
    _position[:len_inp] = torch.from_numpy(position_inp)[:len_inp].float()
    _segment = torch.full((max_length, ), False, dtype=torch.long)
    _segment[:len_inp] = torch.from_numpy(segment_inp)[:len_inp].long()
    _tgt = torch.full((max_length, ), -100, dtype=torch.long)
    _tgt[:len_inp] = torch.from_numpy(tgt)[:len_inp].long()

    _span = torch.zeros((max_length + 1, ), dtype=torch.long)
    _span[len_inp] = 1  # 每个拼接的句子结尾的后一位是1
    _span = torch.cumsum(_span, dim=-1)[:-1]

    len_cxt = torch.LongTensor([len_inp])

    return _ctx.unsqueeze(0), len_cxt, _context.unsqueeze(0),\
           _position.unsqueeze(0), _segment.unsqueeze(0), _span.unsqueeze(0), _tgt.unsqueeze(0)


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits = logits.view(batch_size, -1).contiguous()
        for index in range(len(logits)):

            sorted_logits, sorted_indices = torch.sort(logits[index].view(-1),
                                                       descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                            dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[index][indices_to_remove] = filter_value

        logits = logits.view(batch_size, -1).contiguous()

    return logits


def enforce_repetition_penalty_(tokenizer,
                                lprobs,
                                batch_size,
                                num_beams,
                                prev_output_tokens,
                                repetition_penalty,
                                start_idx=None,
                                end_idx=None,
                                window_size=None):
    assert repetition_penalty >= 1, "repetition penalty coefficient should >= 1"
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    for i in range(batch_size * num_beams):
        if start_idx is None or end_idx is None:
            output_tokens = prev_output_tokens[i].tolist()
        else:
            if end_idx >= start_idx:
                if window_size:
                    output_tokens = prev_output_tokens[
                        i][max(start_idx, end_idx + 1 - window_size):end_idx +
                           1].tolist()
                else:
                    output_tokens = prev_output_tokens[i][start_idx:end_idx +
                                                          1].tolist()
            else:
                output_tokens = []
        #print(output_tokens)
        for previous_token in set(output_tokens):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())

    return banned_ngrams.get(ngram_idx, [])


def calc_banned_ngram_tokens(prev_input_ids: torch.Tensor,
                             num_hypos: int,
                             ngram_size: int,
                             start_idx=None,
                             end_idx=None,
                             window_size=None,
                             tokenizer=None):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if start_idx is not None and end_idx is not None:
        if window_size:
            prev_input_ids = prev_input_ids[:,
                                            max(start_idx, end_idx + 1 -
                                                window_size):end_idx + 1]
        else:
            prev_input_ids = prev_input_ids[:, start_idx:end_idx + 1]

    cur_len = prev_input_ids.size(1)

    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx],
                              prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]

    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids,
                              bad_words_ids,
                              start_idx=None,
                              end_idx=None):
    if start_idx is not None and end_idx is not None:
        prev_input_ids = prev_input_ids[:, start_idx:end_idx + 1]

    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens):] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(
                banned_token_seq
            ) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids)

            if _tokens_match(prev_input_ids_slice.tolist(),
                             banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue
            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def min_length_constraint(logits, cur_len, min_len, tokenizer):
    # This enforcing a min-length by setting EOS probability to 0.
    if cur_len <= min_len:
        logits[:, tokenizer.eos_id] = -float("inf")


def postprocess_next_token_scores(tokenizer,
                                  scores,
                                  input_ids,
                                  no_repeat_ngram_size,
                                  bad_words_ids,
                                  repetition_penalty,
                                  batch_size,
                                  num_beams,
                                  start_idx=None,
                                  end_idx=None,
                                  window_size=None,
                                  min_len=None):

    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(tokenizer, scores, batch_size, num_beams,
                                    input_ids, repetition_penalty, start_idx,
                                    end_idx, window_size)

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(input_ids,
                                                       num_batch_hypotheses,
                                                       no_repeat_ngram_size,
                                                       start_idx, end_idx,
                                                       window_size, tokenizer)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids,
                                                  start_idx, end_idx)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    # 允许生成eos和bos
    scores[:, [0, 1, 2, 3, 4, 5] + [x for x in range(8, 20)]] = -float("inf")

    if start_idx is not None and end_idx is not None and min_len is not None:
        min_length_constraint(scores, end_idx - start_idx + 2, min_len,
                              tokenizer)

    return scores


def cpm_beam_search(model,
                    tokenizer,
                    instance,
                    target_span_len=100,
                    beam_size=3,
                    temperature=.9,
                    top_k=0,
                    top_p=0.9,
                    no_repeat_ngram_size=0,
                    repetition_penalty=1.2,
                    random_sample=False,
                    min_len=None,
                    **kwags):
    print('tokenizer is', tokenizer)
    device = next(model.parameters()).device  
    vocab_size = tokenizer.vocab_size

    ids, info = encode(tokenizer, instance, target_span_len)

    prompt_length = 64
    input_tokens, input_length, context_input, position_input, segment_input, span_input, _ = make_input_cpm3(
        ids, info, prompt_length)

    # (batch, max_length)
    max_length = input_tokens.size(-1)
    batch_size = input_tokens.size(0)

    # (batch, beam_size, max_length)
    input_tokens = input_tokens.unsqueeze(1).expand(batch_size, beam_size,
                                                    max_length)
    input_length = input_length.unsqueeze(1).expand(batch_size, beam_size)
    span_input = span_input.unsqueeze(1).expand(batch_size, beam_size,
                                                max_length)
    context_input = context_input.unsqueeze(1).expand(batch_size, beam_size,
                                                      max_length)
    position_input = position_input.unsqueeze(1).expand(
        batch_size, beam_size, max_length)
    segment_input = segment_input.unsqueeze(1).expand(batch_size, beam_size,
                                                      max_length)
    # (batch * beam_size, max_length)
    input_tokens = input_tokens.contiguous().view(batch_size * beam_size,
                                                  max_length)
    input_length = input_length.contiguous().view(batch_size * beam_size, )
    span_input = span_input.contiguous().view(batch_size * beam_size,
                                              max_length)
    context_input = context_input.contiguous().view(batch_size * beam_size,
                                                    max_length)
    position_input = position_input.contiguous().view(batch_size * beam_size,
                                                      max_length)
    segment_input = segment_input.contiguous().view(batch_size * beam_size,
                                                    max_length)

    input_tokens = input_tokens.int().to(device)
    input_length = input_length.int().to(device)
    context_input = context_input.bool().to(device)
    position_input = position_input.float().to(device)
    segment_input = segment_input.int().to(device)
    span_input = span_input.int().to(device)

    done = [False for _ in range(batch_size)]
    # (batch_size * beam_size, 0)

    beam_scores = torch.zeros((batch_size, beam_size),
                              dtype=torch.float,
                              device=input_tokens.device)
    beam_scores[:, 1:] = -1e9  # 确保第一次只在一个vocab大小里选取
    beam_scores = beam_scores.view(-1)

    # current position
    cur_len = 0

    lef = info[2] + prompt_length
    rig = info[3] + prompt_length

    span_length = rig - lef

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(beam_size,
                       span_length,
                       length_penalty=1,
                       early_stopping=False,
                       tokenizer=tokenizer) for _ in range(batch_size)
    ]

    with torch.inference_mode():
        past_key_values = None
        cached_attn_mask_pos_bias = None
        for i in range(lef - 1, rig):
            # skip all steps when we are done with each sentence
            if all(done):
                break  # Note: break not supports multi-GPUs

            if i == lef - 1:
                # for the first time step, we will move the right context to the beginning inside model
                logits, _, past_key_values, cached_attn_mask_pos_bias = model(
                    input_tokens, input_length, context_input, position_input,
                    segment_input, span_input, past_key_values, rig, i,
                    cached_attn_mask_pos_bias)
            else:
                logits, _, past_key_values, cached_attn_mask_pos_bias = model(
                    input_tokens[:, i:i + 1], input_length, context_input,
                    position_input, segment_input, span_input, past_key_values,
                    rig, i, cached_attn_mask_pos_bias)

            logits = logits[:, -1, :]

            logits = postprocess_next_token_scores(
                tokenizer=tokenizer,
                scores=logits,
                input_ids=input_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=[[0]],
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=beam_size,
                start_idx=lef,
                end_idx=i,
                window_size=None,
                min_len=min_len)
            scores = F.log_softmax(logits, dim=-1)

            if random_sample:
                assert temperature != 0, "temperature should not be zero!"
                scores = scores - math.log(temperature)
                _scores = scores + beam_scores[:, None].expand_as(scores)

                _scores = top_k_logits(_scores, top_k=top_k, top_p=top_p)
                _scores = _scores.contiguous().view(batch_size,
                                                    beam_size * vocab_size)
                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_words = torch.multinomial(
                    probs,
                    num_samples=2 * beam_size)  # (batch_size, beam_size * 2)
                # Compute next scores
                next_scores = torch.gather(
                    _scores, -1, next_words)  # (batch_size, beam_size * 2)
                # sort the sampled vector to make sure that the first beam_size samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores,
                                                              descending=True,
                                                              dim=1)
                next_words = torch.gather(
                    next_words, -1,
                    next_scores_indices)  # (batch_size, beam_size * 2)
            else:
                next_scores = scores + beam_scores[:, None].expand_as(
                    scores)  # (batch_size * beam_size, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, beam_size *
                    vocab_size)  # (batch_size, beam_size * vocab_size)

                next_scores, next_words = torch.topk(next_scores,
                                                     2 * beam_size,
                                                     dim=1,
                                                     largest=True,
                                                     sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size,
                                                               2 * beam_size)
            # next batch beam content
            next_batch_beam = []

            for sent_id in range(batch_size):

                # if we are done with this sentence
                done[sent_id] = done[
                    sent_id] or generated_hyps[sent_id].is_done(
                        next_scores[sent_id].max().item(), cur_len)
                if done[sent_id]:
                    next_batch_beam.extend([(0, tokenizer.pad_id, 0)] *
                                           beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id],
                                      next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id == tokenizer.eos_id or cur_len == span_length:
                        if cur_len > 0:
                            generated_hyps[sent_id].add(
                                input_tokens[sent_id * beam_size + beam_id,
                                             lef:lef + cur_len].clone(),
                                value.item())
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam
                           ) == 0 if cur_len == span_length else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, tokenizer.pad_id, 0)
                                      ] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # At the last step, we should not add the token to the next position
            if i == rig - 1:
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_tokens.new([x[1] for x in next_batch_beam])
            beam_idx = input_length.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            input_tokens = input_tokens[beam_idx, :]
            input_tokens[:, lef + cur_len] = beam_words

            for key_value_layer in past_key_values:
                key_value_layer[0] = key_value_layer[0][beam_idx]
                key_value_layer[1] = key_value_layer[1][beam_idx]

            # update current length
            cur_len = cur_len + 1

        # select the best hypotheses
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            best.append(best_hyp)

        if instance['mode'] == 'lm':
            decode_start_idx = 0
        else:
            decode_start_idx = 1
        return best[0][decode_start_idx:].cpu().numpy()
        # for id in best[0][decode_start_idx:].cpu().numpy():
        #     token = tokenizer.decode([id])

        #     yield token
