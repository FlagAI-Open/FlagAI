"""
Model Patcher Module
Provides unified interface for applying model patches based on training configuration

This module allows dynamic patching of model components (e.g., attention mechanisms,
embeddings, etc.) based on training arguments. Patches are applied before model
initialization to ensure proper configuration.

Compatible with:
- Flash Attention models
- Condense rotary embeddings
- BMTrain optimizations
- Custom model modifications
- Transformers models with Megatron compatibility
"""
import warnings
import os
from typing import Optional, Dict, Any
import torch


class ModelPatcher:
    """
    Unified model patcher that applies patches based on training configuration.
    
    This class provides a centralized way to apply model patches (monkey patches)
    based on training arguments. Patches are applied before model initialization
    to ensure proper configuration.
    
    Usage:
        >>> from flagai.model.model_patcher import ModelPatcher
        >>> patcher = ModelPatcher(training_args)
        >>> patcher.apply_patches()
        >>> # Now load model - patches will be active
    """
    
    def __init__(self, training_args: Optional[Any] = None):
        """
        Initialize ModelPatcher with training arguments.
        
        Args:
            training_args: TrainingArgs object or namespace with training configuration
        """
        self.training_args = training_args
        self.applied_patches = []
        
    def apply_patches(self, model_name: Optional[str] = None):
        """
        Apply all relevant patches based on training configuration.
        
        Args:
            model_name: Optional model name to determine which patches to apply
        """
        if self.training_args is None:
            return
        
        # Get model name from training_args if not provided
        if model_name is None:
            model_name = getattr(self.training_args, 'model_name', '').lower()
        
        # Apply Flash Attention patches if enabled
        if getattr(self.training_args, 'enable_flash_attn_models', False):
            self._apply_flash_attention_patches(model_name)
        
        # Apply other patches based on configuration
        # Add more patch types here as needed
        
    def _apply_flash_attention_patches(self, model_name: str):
        """
        Apply Flash Attention patches for supported models.
        
        Args:
            model_name: Model name (lowercase)
        """
        if 'aquila' in model_name or 'aquila2' in model_name:
            self._patch_aquila_flash_attention()
        elif 'llama' in model_name:
            self._patch_llama_flash_attention()
        # Add more model-specific patches here
        
    def _patch_aquila_flash_attention(self):
        """Apply Flash Attention patch for Aquila models."""
        try:
            from flagai.model.aquila2.aquila2_flash_attn_monkey_patch import replace_aquila_attn_with_flash_attn
            replace_aquila_attn_with_flash_attn()
            self.applied_patches.append('aquila_flash_attention')
            print("Applied Flash Attention patch for Aquila model")
        except ImportError as e:
            warnings.warn(f"Failed to apply Aquila Flash Attention patch: {e}")
        except Exception as e:
            warnings.warn(f"Error applying Aquila Flash Attention patch: {e}")
    
    def _patch_llama_flash_attention(self):
        """Apply Flash Attention patch for Llama models."""
        try:
            # Check if llama patch exists in examples
            import importlib.util
            import os
            # Try to find the patch file in examples directory
            patch_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'examples', 'Llama', 'llama_bmt_monkey_patch.py')
            if os.path.exists(patch_file):
                import sys
                sys.path.insert(0, os.path.dirname(patch_file))
                from llama_bmt_monkey_patch import replace_llama_attn_with_bmt
                replace_llama_attn_with_bmt()
                self.applied_patches.append('llama_bmt_attention')
                print("Applied BMTrain attention patch for Llama model")
        except ImportError:
            # Patch file may not exist, skip silently
            pass
        except Exception as e:
            warnings.warn(f"Error applying Llama attention patch: {e}")
    
    def apply_condense_patch(self, ratio: float = 4.0):
        """
        Apply condense rotary embedding patch for Aquila models.
        
        Args:
            ratio: Condense ratio (default: 4.0)
        """
        try:
            from flagai.model.aquila2.aquila_condense_monkey_patch import replace_aquila_with_condense
            replace_aquila_with_condense(ratio)
            self.applied_patches.append(f'aquila_condense_ratio_{ratio}')
            print(f"Applied Condense Rotary Embedding patch with ratio {ratio}")
        except ImportError as e:
            warnings.warn(f"Failed to apply Condense patch: {e}")
        except Exception as e:
            warnings.warn(f"Error applying Condense patch: {e}")
    
    def get_applied_patches(self):
        """
        Get list of applied patches.
        
        Returns:
            List of patch names that have been applied
        """
        return self.applied_patches.copy()


def apply_model_patches(training_args: Optional[Any] = None, 
                        model_name: Optional[str] = None,
                        **kwargs) -> ModelPatcher:
    """
    Convenience function to apply model patches based on training arguments.
    
    Args:
        training_args: TrainingArgs object or namespace with training configuration
        model_name: Optional model name
        **kwargs: Additional patch configuration (e.g., condense_ratio)
    
    Returns:
        ModelPatcher instance with applied patches
    
    Examples:
        >>> from flagai.model.model_patcher import apply_model_patches
        >>> patcher = apply_model_patches(training_args, model_name="aquila2-7b")
        >>> # Now load model
    """
    patcher = ModelPatcher(training_args)
    patcher.apply_patches(model_name=model_name)
    
    # Apply additional patches from kwargs
    if 'condense_ratio' in kwargs:
        patcher.apply_condense_patch(ratio=kwargs['condense_ratio'])
    
    # Apply transformers patches for Megatron compatibility
    if training_args is not None:
        env_type = getattr(training_args, 'env_type', '')
        model_parallel_size = getattr(training_args, 'model_parallel_size', 1)
        if ('deepspeed+mpu' in env_type or os.getenv('ENV_TYPE') == 'deepspeed+mpu') and \
           model_parallel_size > 1:
            try:
                from flagai.model.transformers_patcher import apply_transformers_patches
                transformers_patcher = apply_transformers_patches(training_args)
                patcher.applied_patches.extend(transformers_patcher.get_applied_patches())
                print("Applied transformers patches for Megatron compatibility")
            except Exception as e:
                warnings.warn(f"Failed to apply transformers patches: {e}")
    
    return patcher

