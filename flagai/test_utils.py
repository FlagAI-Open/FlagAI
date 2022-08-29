# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
def build_input_from_ids(text_a_ids=None,
                         text_b_ids=None,
                         answer_ids=None,
                         max_seq_length=16,
                         tokenizer=None,
                         args=None,
                         add_cls=True,
                         add_sep=False,
                         add_piece=False,
                         add_eos=True,
                         mask_id=None,
                         masked_lm=False):
    if mask_id is None:
        mask_id = tokenizer.get_command_id('MASK')
    eos_id = tokenizer.get_command_id('eos')
    cls_id = tokenizer.get_command_id('cls')
    sep_id = tokenizer.get_command_id('sep')
    ids = []
    types = []
    paddings = []
    # CLS
    if add_cls:
        ids.append(cls_id)
        types.append(0)
        paddings.append(1)
    # A
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    types.extend([0] * len_text_a)
    paddings.extend([1] * len_text_a)
    # B
    if text_b_ids is not None:
        # SEP
        if add_sep:
            ids.append(sep_id)
            types.append(0)
            paddings.append(1)
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        types.extend([1] * len_text_b)
        paddings.extend([1] * len_text_b)
    eos_length = 1 if add_eos else 0
    # Cap the size.
    if len(ids) >= max_seq_length - eos_length:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        types = types[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]
    end_type = 0 if text_b_ids is None else 1
    if add_eos:
        ids.append(eos_id)
        types.append(end_type)
        paddings.append(1)
    sep = len(ids)
    target_ids = [0] * len(ids)
    loss_masks = [0] * len(ids)
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)
    # Piece
    if add_piece or answer_ids is not None:
        sop_id = tokenizer.get_command_id('sop')
        mask_position = ids.index(
            mask_id
        ) if not args.sentinel_token else args.max_position_embeddings
        ids.append(sop_id)
        types.append(end_type)
        paddings.append(1)
        position_ids.append(mask_position)
        block_position_ids.append(1)
        if answer_ids is not None:
            len_answer = len(answer_ids)
            ids.extend(answer_ids[:-1])
            types.extend([end_type] * (len_answer - 1))
            paddings.extend([1] * (len_answer - 1))
            position_ids.extend([mask_position] * (len_answer - 1))
            if not args.no_block_position:
                block_position_ids.extend(range(2, len(answer_ids) + 1))
            else:
                block_position_ids.extend([1] * (len(answer_ids) - 1))
            target_ids.extend(answer_ids)
            loss_masks.extend([1] * len(answer_ids))
        else:
            target_ids.append(0)
            loss_masks.append(1)
    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        types.extend([eos_id] * padding_length)
        paddings.extend([0] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    if masked_lm:
        position_ids = [position_ids]
    return ids, types, paddings, position_ids, sep, target_ids, loss_masks


class CollateArguments:

    def __init__(self):
        self.cloze_eval = True
        self.pretrained_bert = False

        # these values could be wrong
        self.task_mask = True
        self.continuous_prompt = False
        self.prefix_prompt = 0

        self.sentinel_token = False
        self.max_position_embeddings = 1024
        self.no_block_position = False
        self.masked_lm = False
        self.pattern_id = 0
        self.seq_length = 256
        self.num_prompt_tokens = 0
        self.multi_token = False
        self.segment_length = 0
        self.fast_decode = False

        self.few_superglue = False
        self.pattern_text = False


class Seq2SeqCollateArguments:

    def __init__(self):
        self.cloze_eval = True
        self.pretrained_bert = False

        # these values could be wrong
        self.task_mask = True
        self.continuous_prompt = False
        self.prefix_prompt = 0

        self.max_src_length = 464
        self.max_tgt_length = 48
        self.min_tgt_length = 0
        self.no_block_position = True

        # self.sentinel_token = False
        # self.max_position_embeddings = 1024
        # self.no_block_position = False
        # self.masked_lm = False
        # self.pattern_id = 0
        # self.seq_length = 256
        # self.num_prompt_tokens = 0
        # self.multi_token = False
        # self.segment_length = 0
        # self.fast_decode = False
        #
        # self.few_superglue = False
        # self.pattern_text = False


class PretrainDatasetArguments:

    def __init__(self):
        self.task_mask = True  # Distinguished the generation and gap-sentence mask
        self.block_mask_prob = 0.1
        self.block_lm = True  # Whether do masking
        self.masked_lm = False  # Whether do simple masking (same symbol among masks)

        self.pre_tokenize = True
        self.no_lazy_loader = True
        self.half_lazy_loader = False
        self.sentinel_token = False
