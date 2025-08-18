from ... import C
from ...llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM

import torch
from ..tree_drafter import *
import time
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
import torch.nn.functional as F

class W4A16GPTQMarlinLLM_with_tree_drafter(W4A16GPTQMarlinLLM):
    def __init__(self,
                 drafter_type, drafter_path, base_path,
                 tree_size,
                 use_rope: bool=False,
                 temperature: float=0.0,
                 **kwargs):
        super().__init__(base_path, **kwargs)

        self.drafter_type = drafter_type
        self.drafter_path = drafter_path
        self.base_path = base_path
        self.use_rope = use_rope

        self.tree_size = tree_size
        self.tree_draft_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_gt_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_attn_mask = torch.empty((tree_size), dtype=torch.uint64, device="cuda")
        self.tree_parent = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.temperature = temperature

        self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")

    def load_from_hf(self):
        with torch.no_grad():
            self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)

            if self.use_rope:
                if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
                    rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
                else:
                    rope_type = "default"
                # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
                inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, "cpu", seq_len=self.max_total_length)
                # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
                self._load(f"{self.drafter_type}.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
                # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)

            super().load_from_hf()

    def generate(self, input_ids, generation_length=100, teminators=[], use_stream=False):
        """
        Generate text with optional streaming output for quantized tree drafter.
        Returns (tokens, accept_lengths, decode_time, prefill_time) if use_stream=False, or generator yielding {'token', 'text', 'is_finished', 'accept_length', 'prefill_time', 'decode_time'} if use_stream=True.
        """
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        # Check if input length exceeds maximum supported length
        if prefix_length > self.max_total_length:
            raise ValueError(f"Input token count ({prefix_length}) exceeds maximum supported length ({self.max_total_length}) under current memory limit")
        
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        
        # Set progress flag before prefill for stream mode
        if use_stream:
            self._show_prefill_progress = True
        else:
            self._show_prefill_progress = False
        
        # Measure prefill time
        if self.use_enter and use_stream:
            # In use_enter mode, timing will be handled inside prefill method
            logits = self.prefill(input_ids, position_ids)
            prefill_time = getattr(self, '_last_prefill_time', 0.0)  # Get actual prefill time
        else:
            torch.cuda.synchronize()
            prefill_start = time.time()
            logits = self.prefill(input_ids, position_ids)
            torch.cuda.synchronize()
            prefill_time = time.time() - prefill_start
        
        if self.temperature > 0.0:
            self.tree_draft_ids[:1].copy_(torch.multinomial(F.softmax(logits[0]/self.temperature, dim=-1), num_samples=1, generator=self.generator))
        else:
            self.tree_draft_ids[:1].copy_(logits[0].argmax(dim=-1))

        # Wait for user input before decode phase if use_decode_enter is enabled
        if self.use_decode_enter:
            if use_stream and self.use_enter:
                # In stream mode with use_enter, we already showed [Decoding], just wait for input
                print("Please Press Enter to Start Decoding...", end="", flush=True)
                input()  # Wait for Enter key
                print("\r" + " " * 50 + "\r", end="", flush=True)  # Clear the prompt without showing [Decoding] again
            else:
                # In other modes, show prompt and wait
                print("Please Press Enter to Start Decoding...", end="", flush=True)
                input()  # Wait for Enter key
                print("\r" + " " * 50 + "\r[Decoding]", flush=True)  # Show [Decoding] only when use_enter is not enabled

        if use_stream:
            # Stream generation for quantized tree drafter (optimized)
            def _stream_generator():
                # Keep minimal context for correct spacing
                prev_token = None
                
                # yield first token
                token = self.tree_draft_ids[0].item()
                text = self.tokenizer.decode([token], skip_special_tokens=False)
                prev_token = token
                
                yield {
                    'token': token,
                    'text': text,
                    'is_finished': token in teminators,
                    'accept_length': 1,
                    'prefill_time': prefill_time,
                    'decode_time': 0.0  # First token comes from prefill
                }
                
                if token in teminators:
                    return

                decode_start_time = time.time()
                
                i = 0
                while i < generation_length-1:
                    self.cache_length[0] = prefix_length + i

                    # draft step
                    C.draft(self.tree_draft_ids.data_ptr(), self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(), self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr())

                    logits = self.decode(self.tree_draft_ids, self.tree_position_ids, self.cache_length, mask_2d=self.tree_attn_mask)
                    if self.temperature > 0.0:
                        self.tree_gt_ids.copy_(torch.multinomial(F.softmax(logits/self.temperature, dim=-1), num_samples=1, generator=self.generator).squeeze(-1))
                    else:
                        self.tree_gt_ids.copy_(logits.argmax(dim=-1))

                    # verify step
                    accept_length = C.verify_and_fix(
                        self.tree_draft_ids.numel(), self.tree_draft_ids.data_ptr(), self.tree_gt_ids.data_ptr(),
                        self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(),
                        self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr()
                    )

                    # yield accepted tokens (optimized with minimal context)
                    if accept_length > 0:
                        accepted_tokens = self.tree_draft_ids[:accept_length].tolist()
                        
                        # For correct spacing, decode with previous token context
                        if prev_token is not None:
                            context_tokens = [prev_token] + accepted_tokens
                            context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=False)
                            prev_text = self.tokenizer.decode([prev_token], skip_special_tokens=False)
                            new_text = context_text[len(prev_text):]
                        else:
                            new_text = self.tokenizer.decode(accepted_tokens, skip_special_tokens=False)
                        
                        # Yield tokens with batch text for first token, empty for others
                        for j in range(accept_length):
                            if i + j >= generation_length - 1:
                                break
                                
                            token = accepted_tokens[j]
                            
                            # Give all new text to first token, empty to others
                            if j == 0:
                                text = new_text
                            else:
                                text = ""
                            
                            terminal = token in teminators
                            is_finished = terminal or (i + j == generation_length - 2)
                            
                            # Only calculate time for the last token in the batch to reduce overhead
                            decode_time = time.time() - decode_start_time if j == accept_length - 1 else 0.0
                            
                            yield {
                                'token': token,
                                'text': text,
                                'is_finished': is_finished,
                                'accept_length': accept_length if j == 0 else 0,  # only report accept_length for first token in batch
                                'prefill_time': 0.0,  # Only report prefill_time for first token
                                'decode_time': decode_time
                            }
                            
                            if terminal:
                                return
                        
                        # Update prev_token to the last accepted token
                        prev_token = accepted_tokens[-1]

                    self.tree_draft_ids[0] = self.tree_draft_ids[accept_length - 1]
                    i += accept_length
                    
            return _stream_generator()
        else:
            # Original batch generation
            tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
            tokens[0].copy_(self.tree_draft_ids[0])
            accept_lengths = []
            i = 0
            terminal = False
            torch.cuda.synchronize()
            decode_start = time.time()
            while i < generation_length-1 and not terminal:
                self.cache_length[0] = prefix_length + i

                # torch.cuda.nvtx.range_push(f"draft")
                C.draft(self.tree_draft_ids.data_ptr(), self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(), self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr())
                # torch.cuda.nvtx.range_pop()

                logits = self.decode(self.tree_draft_ids, self.tree_position_ids, self.cache_length, mask_2d=self.tree_attn_mask)
                if self.temperature > 0.0:
                    self.tree_gt_ids.copy_(torch.multinomial(F.softmax(logits/self.temperature, dim=-1), num_samples=1, generator=self.generator).squeeze(-1))
                else:
                    self.tree_gt_ids.copy_(logits.argmax(dim=-1))

                # torch.cuda.nvtx.range_push(f"verify")
                accept_length = C.verify_and_fix(
                    self.tree_draft_ids.numel(), self.tree_draft_ids.data_ptr(), self.tree_gt_ids.data_ptr(),
                    self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(),
                    self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr()
                )
                # torch.cuda.nvtx.range_pop()

                accept_lengths.append(accept_length)
                for temin in teminators:
                    if temin in self.tree_draft_ids[:accept_length]:
                        terminal = True
                append_length = min(accept_length, generation_length - 1 - i)
                tokens[1+i:1+i+append_length].copy_(self.tree_draft_ids[:append_length])
                self.tree_draft_ids[0] = self.tree_draft_ids[accept_length - 1]
                i += accept_length
            torch.cuda.synchronize()
            decode_time = time.time() - decode_start
            tokens = tokens[:1+i].tolist()

            return tokens, accept_lengths, decode_time, prefill_time