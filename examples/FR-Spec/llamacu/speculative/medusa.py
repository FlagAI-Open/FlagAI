from .. import C
from .tree_drafter import LLM_with_tree_drafter, pack_mask

import torch
from transformers import PretrainedConfig

class MedusaConfig(PretrainedConfig):
    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        assert self.medusa_num_layers == 1, "Currently only supports 1 layer"

class LLM_with_medusa(LLM_with_tree_drafter):
    def __init__(self,
                 medusa_path,
                 base_path,
                 medusa_num_heads=4,
                 medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]],
                 medusa_topk=10,
                 V=15000,
                 **kwargs):
        super().__init__(
            "medusa", medusa_path, base_path,
            tree_size = len(medusa_choices) + 1,
            **kwargs
        )

        self.medusa_config = MedusaConfig.from_pretrained(medusa_path, medusa_num_heads=medusa_num_heads)

        # init static tree
        medusa_buffers = self._generate_medusa_buffers(medusa_choices, medusa_topk)
        self._init_static_tree(medusa_buffers["medusa_attn_mask"][0][0].to(torch.int32))

        self.medusa_position_ids = medusa_buffers["medusa_position_ids"].to(torch.int32)
        self.medusa_tree_indices = (medusa_buffers["tree_indices"][1:] - 1).to(torch.int32)
        C.init_medusa_model(
            self.medusa_config.medusa_num_heads,
            self.medusa_config.medusa_num_layers,
            medusa_topk,
            self.tree_size,
            self.medusa_tree_indices.data_ptr(),
            self.medusa_position_ids.data_ptr(),
            V,
            self.dtype_int,
        )

    def _init_static_tree(self, mask_2d):
        self.tree_attn_mask.copy_(pack_mask(mask_2d))
        for i in range(1, self.tree_size):
            for j in reversed(range(i)):
                if mask_2d[i][j] == 1:
                    self.tree_parent[i] = j
                    break
            else:
                assert False, f"No parent found for {i}"

    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.drafter_type:
            if name == "token_id_remap":
                C.load_model(f"{cls}.{name}", param.data_ptr())
                return
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous().to(dtype)
            if int(name.split(".")[0]) < self.medusa_config.medusa_num_heads:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)

    def _generate_medusa_buffers(self, medusa_choices, medusa_topk, device="cuda"):
        """
        /////////// COMES FROM https://github.com/FasterDecoding/Medusa ///////////
        Generate buffers for the Medusa structure based on the provided choices.
        
        Parameters:
        - medusa_choices (list): A nested list representing tree in the Medusa structure.
        - device (str): Device to which the tensors should be moved. Default is "cuda".
        
        Returns:
        - dict: A dictionary containing buffers related to the Medusa structure.
        """
        def pad_path(path, length, pad_value=-2):
            return path + [pad_value] * (length - len(path))

        # Sort the medusa_choices based on their lengths and then their values
        sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
        medusa_len = len(sorted_medusa_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_medusa_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth
        
        # Create the attention mask for Medusa
        medusa_attn_mask = torch.eye(medusa_len, medusa_len)
        medusa_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                # retrieve ancestor position
                if len(cur_medusa_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_medusa_choice) - 1):
                    ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
                medusa_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        # Generate tree indices for the Medusa structure
        medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
        medusa_tree_indices[0] = 0
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + medusa_topk * i + 1
            start += depth_counts[i]

        # Generate position IDs for the Medusa structure
        medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        # Generate retrieval indices for Medusa structure verification
        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_medusa_choices)):
            cur_medusa_choice = sorted_medusa_choices[-i-1]
            retrieve_indice = []
            if cur_medusa_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_medusa_choice)):
                    retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                    retrieve_paths.append(cur_medusa_choice[:c+1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

        # Aggregate the generated buffers into a dictionary
        medusa_buffers = {
            "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
            "tree_indices": medusa_tree_indices,
            "medusa_position_ids": medusa_position_ids,
            "retrieve_indices": retrieve_indices,
            }
        
        # Move the tensors in the dictionary to the specified device
        medusa_buffers = {
            k: v.clone().to(device)
            if isinstance(v, torch.Tensor)
            else torch.tensor(v,  device=device)
            for k, v in medusa_buffers.items()
        }
        return medusa_buffers