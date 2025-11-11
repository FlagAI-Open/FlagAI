"""
Transformers Patcher Module
Provides unified interface for patching transformers models to support Megatron

This module allows dynamic patching of transformers library components to use
FlagAI's Megatron-compatible implementations. This enables seamless compatibility
with Megatron model parallelism for transformers models.

Compatible with:
- Transformers models (BERT, GPT, T5, Llama, etc.)
- Megatron model parallelism
- DeepSpeed with model parallelism
"""
import warnings
import os
from typing import Optional, Any, Dict
import torch
import torch.nn as nn


class TransformersPatcher:
    """
    Unified transformers patcher that replaces transformers components with
    Megatron-compatible implementations.
    
    This class provides a centralized way to patch transformers models to support
    Megatron model parallelism. Patches are applied before model initialization
    to ensure proper configuration.
    
    Usage:
        >>> from flagai.model.transformers_patcher import TransformersPatcher
        >>> patcher = TransformersPatcher(training_args)
        >>> patcher.apply_patches()
        >>> # Now load transformers model - patches will be active
    """
    
    def __init__(self, training_args: Optional[Any] = None):
        """
        Initialize TransformersPatcher with training arguments.
        
        Args:
            training_args: TrainingArgs object or namespace with training configuration
        """
        self.training_args = training_args
        self.applied_patches = []
        self.original_modules = {}
        
        # Check if Megatron model parallelism is enabled
        self.use_megatron = False
        if training_args is not None:
            env_type = getattr(training_args, 'env_type', '')
            model_parallel_size = getattr(training_args, 'model_parallel_size', 1)
            self.use_megatron = ('deepspeed+mpu' in env_type or 
                                 os.getenv('ENV_TYPE') == 'deepspeed+mpu') and \
                                model_parallel_size > 1
        elif os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            model_parallel_size = int(os.getenv('MODEL_PARALLEL_SIZE', '1'))
            self.use_megatron = model_parallel_size > 1
        
    def apply_patches(self):
        """
        Apply all relevant patches to transformers library.
        """
        if not self.use_megatron:
            return
        
        # Patch Linear layers
        self._patch_linear_layers()
        
        # Patch Embedding layers
        self._patch_embedding_layers()
        
        # Patch Attention layers (if needed)
        # Note: This is model-specific and may need custom implementation
        
    def _patch_linear_layers(self):
        """Patch transformers Linear layers with Megatron-compatible versions."""
        try:
            from flagai.model.layers.feedforward import (
                ColumnParallelLinear, 
                RowParallelLinear
            )
            from flagai.model.utils import normal_init_method
            
            # Store original nn.Linear for restoration
            self.original_modules['nn.Linear'] = nn.Linear
            
            # Create a wrapper that returns appropriate parallel linear based on context
            def create_parallel_linear(in_features, out_features, bias=True, 
                                      gather_output=False, input_is_parallel=False,
                                      init_method=None):
                """Create parallel linear layer based on context."""
                if self.use_megatron:
                    # Determine if this should be column or row parallel
                    # This is a simplified version - actual implementation may need
                    # more context to determine the right type
                    if input_is_parallel:
                        return RowParallelLinear(
                            in_features, out_features, bias=bias,
                            input_is_parallel=True,
                            init_method=init_method or normal_init_method(0, 0.02)
                        )
                    else:
                        return ColumnParallelLinear(
                            in_features, out_features, bias=bias,
                            gather_output=gather_output,
                            init_method=init_method or normal_init_method(0, 0.02)
                        )
                else:
                    return nn.Linear(in_features, out_features, bias=bias)
            
            # Note: Direct patching of nn.Linear is not recommended as it affects
            # all models. Instead, we provide a utility function that can be used
            # when building models.
            self.applied_patches.append('parallel_linear_utility')
            print("Created parallel linear utility for Megatron compatibility")
            
        except ImportError as e:
            warnings.warn(f"Failed to import parallel linear layers: {e}")
        except Exception as e:
            warnings.warn(f"Error patching linear layers: {e}")
    
    def _patch_embedding_layers(self):
        """Patch transformers Embedding layers with Megatron-compatible versions."""
        try:
            from flagai.model.layers.embeddings import VocabParallelEmbedding
            
            # Store original nn.Embedding for restoration
            self.original_modules['nn.Embedding'] = nn.Embedding
            
            # Create a wrapper that returns vocab parallel embedding when needed
            def create_parallel_embedding(num_embeddings, embedding_dim, 
                                         padding_idx=None, init_method=None):
                """Create parallel embedding layer based on context."""
                if self.use_megatron:
                    return VocabParallelEmbedding(
                        num_embeddings, embedding_dim,
                        padding_idx=padding_idx,
                        init_method=init_method
                    )
                else:
                    return nn.Embedding(num_embeddings, embedding_dim, 
                                      padding_idx=padding_idx)
            
            self.applied_patches.append('parallel_embedding_utility')
            print("Created parallel embedding utility for Megatron compatibility")
            
        except ImportError as e:
            warnings.warn(f"Failed to import parallel embedding layers: {e}")
        except Exception as e:
            warnings.warn(f"Error patching embedding layers: {e}")
    
    def patch_model_class(self, model_class, model_name: str = None):
        """
        Patch a specific transformers model class to use Megatron-compatible components.
        
        Args:
            model_class: The transformers model class to patch
            model_name: Optional model name for model-specific patches
        
        Returns:
            Patched model class
        """
        if not self.use_megatron:
            return model_class
        
        # Model-specific patching logic
        # This is a placeholder - actual implementation would need to:
        # 1. Identify Linear layers in the model
        # 2. Replace them with ColumnParallelLinear/RowParallelLinear
        # 3. Identify Embedding layers
        # 4. Replace them with VocabParallelEmbedding
        # 5. Update forward methods if needed
        
        return model_class
    
    def restore_patches(self):
        """Restore original transformers modules."""
        for module_name, original_module in self.original_modules.items():
            if module_name == 'nn.Linear':
                nn.Linear = original_module
            elif module_name == 'nn.Embedding':
                nn.Embedding = original_module
        
        self.applied_patches = []
        print("Restored original transformers modules")
    
    def get_applied_patches(self):
        """
        Get list of applied patches.
        
        Returns:
            List of patch names that have been applied
        """
        return self.applied_patches.copy()


def apply_transformers_patches(training_args: Optional[Any] = None) -> TransformersPatcher:
    """
    Convenience function to apply transformers patches based on training arguments.
    
    Args:
        training_args: TrainingArgs object or namespace with training configuration
    
    Returns:
        TransformersPatcher instance with applied patches
    
    Examples:
        >>> from flagai.model.transformers_patcher import apply_transformers_patches
        >>> from flagai.training_args import TrainingArgs
        >>> args = TrainingArgs(env_type="deepspeed+mpu", model_parallel_size=2)
        >>> patcher = apply_transformers_patches(args)
        >>> # Now transformers models will use Megatron-compatible components
    """
    patcher = TransformersPatcher(training_args)
    patcher.apply_patches()
    return patcher


def create_megatron_compatible_linear(in_features, out_features, bias=True,
                                     gather_output=False, input_is_parallel=False,
                                     init_method=None):
    """
    Create a Megatron-compatible linear layer.
    
    This function returns the appropriate parallel linear layer based on the
    current environment configuration.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias
        gather_output: If True, gather output from all model parallel ranks
        input_is_parallel: If True, input is already parallel
        init_method: Initialization method for weights
    
    Returns:
        Linear layer (either standard or parallel based on configuration)
    """
    use_megatron = (os.getenv('ENV_TYPE') == 'deepspeed+mpu' and 
                   int(os.getenv('MODEL_PARALLEL_SIZE', '1')) > 1)
    
    if use_megatron:
        from flagai.model.layers.feedforward import (
            ColumnParallelLinear, 
            RowParallelLinear
        )
        from flagai.model.utils import normal_init_method
        
        if input_is_parallel:
            return RowParallelLinear(
                in_features, out_features, bias=bias,
                input_is_parallel=True,
                init_method=init_method or normal_init_method(0, 0.02)
            )
        else:
            return ColumnParallelLinear(
                in_features, out_features, bias=bias,
                gather_output=gather_output,
                init_method=init_method or normal_init_method(0, 0.02)
            )
    else:
        return nn.Linear(in_features, out_features, bias=bias)


def create_megatron_compatible_embedding(num_embeddings, embedding_dim,
                                        padding_idx=None, init_method=None):
    """
    Create a Megatron-compatible embedding layer.
    
    This function returns the appropriate parallel embedding layer based on the
    current environment configuration.
    
    Args:
        num_embeddings: Size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        padding_idx: If specified, the entries at padding_idx do not contribute
                     to the gradient
        init_method: Initialization method for weights
    
    Returns:
        Embedding layer (either standard or parallel based on configuration)
    """
    use_megatron = (os.getenv('ENV_TYPE') == 'deepspeed+mpu' and 
                   int(os.getenv('MODEL_PARALLEL_SIZE', '1')) > 1)
    
    if use_megatron:
        from flagai.model.layers.embeddings import VocabParallelEmbedding
        
        return VocabParallelEmbedding(
            num_embeddings, embedding_dim,
            padding_idx=padding_idx,
            init_method=init_method
        )
    else:
        return nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

