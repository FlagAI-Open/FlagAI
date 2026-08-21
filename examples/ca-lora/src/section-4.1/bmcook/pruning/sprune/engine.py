import torch
import bmtrain as bmt
from typing import Dict
from torch.nn.parameter import Parameter
from .func import determinate_mask, sample, binarize
from .plugin import SPrunePlugin
from .utils import get_params_from_block


class SPruneStrategy:
    def __init__(self, config: Dict) -> None:
        self.criterion = config['criterion']
        assert self.criterion == 'l0', "BMCook sprune do not support other criterions besides l0 yet."
        self.fixed_mask_path = config['fixed_mask_path']
        self.training_mask = config['training_mask']
        self.mask_mode = config['mask_mode']
        self.target_mode = config['target_mode']
        self.target_sparsity = config['target_sparsity']


class SPruneEngine:
    r"""
    SPruneEngine is used for the mask computation and update of SPrunePlugin.

    The engine design is based on L0 regularization method and a lagrangian term. For L0 regularization details, see paper
        "Learning Sparse Neural Networks through L_0 Regularization" <https://openreview.net/forum?id=H1Y8hhg0b>.
        For lagrangian term in PLM structure pruning, see paper "Structured Pruning of Large Language Models" 
        <https://arxiv.org/abs/1910.04732>.
    """
    def __init__(self, config: Dict, plugin: SPrunePlugin) -> None:
        r"""Init the SpruneEngine from a SPrunePlugin. It will initilize all the :class:`torch.nn.Parameter`
        used for learning the sprune mask, and create the optimizer for l0 regularization.

        Args:
            config: `(Dict)`, the sprune config.
            plugin: `(SPrunePlugin)`, the SPrunePlugin.
        """
        super().__init__()
        self.strategy = SPruneStrategy(config)
        self.target_sparsity = self.strategy.target_sparsity
        self.plugin = plugin
        self.training = True

        self.lambda_1 = Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
        self.lambda_2 = Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
        self.training_loga = {}
        for mask in self.strategy.training_mask:
            shape = self.plugin.info_to_engine['shape'][mask]
            self.training_loga[mask+'_loga'] = Parameter(torch.empty(shape[0], dtype=torch.float, device='cuda').normal_(0., 1e-2))

        self.create_sprune_optimizer()

    def create_sprune_optimizer(self):
        r"""Create the sprune optimizer and lagrangian optimizer, making the learning of loga and 
        lagrangian terms to be an adversarial game.
        
        sprune optimizer will manage the loga parameters.

        lagrangian optimizer will manage the lagrangian terms.
        """
        l0_params = [{
                        "params": [p for _, p in self.training_loga.items()],
                        "weight_decay": 0.0,
                        "lr": 0.1
                        }]
        self.sp_optimizer = torch.optim.AdamW(l0_params)

        lagrangian_params = [{
                    "params": [self.lambda_1, self.lambda_2],
                    "weight_decay": 0.0,
                    "lr": -0.1
                }]
        self.lagrangian_optimizer = torch.optim.AdamW(lagrangian_params)
    
    def update(self):
        r"""
        update the sprune parameters and lagrangian parameters.
        """
        if self.training:
            info_list = self.update_plugin_mask(training=True)
            loss, sparsity = self.loss(info_list)
            if torch.abs(sparsity - self.target_sparsity) < 5e-5:
                bmt.print_rank("binarize the mask and begin finetune...")
                info_list = self.update_plugin_mask(training=False)
                self.lambda_1.requires_grad_(False)
                self.lambda_2.requires_grad_(False)
                for v in self.training_loga.values():
                    v.requires_grad_(False)
                self.training = False
        else:
            info_list = self.update_plugin_mask(training=False)
            loss, sparsity = self.loss(info_list)
        return loss, sparsity

    def step(self):
        r"""run :method:`.step()` of sprune optimizer and lagrangian optimizer"""
        self.sp_optimizer.step()
        self.lagrangian_optimizer.step()
    
    def zero_grad(self):
        r"""run :method:`.zero_grad()` of sprune optimizer and lagrangian optimizer"""
        self.sp_optimizer.zero_grad()
        self.lagrangian_optimizer.zero_grad()

    def loss(self, info_list):
        r"""calculate the lagrangian loss. It can be calculated in sparsity(`float`) or dimension(`int`)"""
        if self.strategy.target_mode == 'sparsity':
            return self.lagrangian_loss_sparsity(info_list, layer_constraint=False)
        elif self.strategy.target_mode == 'dimension':
            return self.lagrangian_loss_dimension()

    def update_plugin_mask(self, training: bool = True):
        r"""update the mask managed in plugin"""
        info_list = {}
        for k, v in self.training_loga.items():
            module = k.split('_loga')[0]

            mask = sample(v) if training is True else binarize(v)
            train_mask = determinate_mask(v)
            assert mask.size(0) == train_mask.size(0)
            
            for index in range(mask.size(0)):
                self.plugin.__dict__[module][index]['mask'] = mask[index].clone().detach()
                
                param = self.plugin.__dict__[module][index]['param']
                index_all = self.plugin.__dict__[module][index]['index']

                if index_all not in info_list:
                    info_list[index_all] = {'module': [module], 'param': [param], 'score': [train_mask[index]]}
                else:
                    if module in info_list[index_all]['module']:
                        module_correct = 'cross_' + module
                    else:
                        module_correct = module
                    info_list[index_all]['module'].append(module_correct)
                    info_list[index_all]['param'].append(param)
                    info_list[index_all]['score'].append(train_mask[index])

        return info_list

    def lagrangian_loss_sparsity(self, info_list, layer_constraint: bool = False):
        r"""The func 'lagrangian_loss_sparsity' is to calculate the lagrangian loss to get the target sparsity"""
        expected_sparsity = get_params_from_block(info_list)
        loss_sparsity = expected_sparsity - self.target_sparsity
        if layer_constraint:
            loss_sparsity = torch.mean(torch.abs(loss_sparsity))
            expected_sparsity = torch.mean(expected_sparsity)

        lagrangian_loss = self.lambda_1 * loss_sparsity + self.lambda_2 * (loss_sparsity ** 2)

        return lagrangian_loss, expected_sparsity

    def lagrangian_loss_dimension(self):
        r"""calculate the lagrangian loss to get the target dimension"""
        dimension_score = determinate_mask(self.training_loga)
        all_dimension = dimension_score.size(1)
        
        expected_dimension = torch.sum(dimension_score, -1)
        loss_dimension = torch.sum((self.target_dimension - expected_dimension) / all_dimension)
        
        lagrangian_loss = self.lambda_1 * loss_dimension + self.lambda_2 * (loss_dimension ** 2)
        
        return lagrangian_loss, expected_dimension

    def get_model_sparsity(self, layer_constraint: bool = False):
        r"""calculate the current sparsity to calculate lagrangian loss."""
        if not layer_constraint:
            total_res = 0
            num_res = 0
            for k, v in self.training_loga.items():
                # layer-wise
                for layer_index in range(v.size(0)):
                    param = self.plugin.__dict__[k.split('_loga')[0]][layer_index]['param']
                    v_cur = v[layer_index]

                    total_res +=  torch.sum(determinate_mask(v_cur)) * param * 3
                    num_res   +=  v_cur.numel() * param * 3
            ratio = total_res / num_res
        else:
            ratio = []
            for k, v in self.training_loga.items():
                # layer-wise
                for layer_index in range(v.size(0)):
                    param = self.plugin.__dict__[k.split('_loga')[0]][layer_index]['param']
                    v_cur = v[layer_index]
                    ratio.append(torch.sum(determinate_mask(v_cur)) / v_cur.numel())
                ratio = torch.stack(ratio)
        return 1 - ratio  # return sparsity
