# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
Unit tests for transformers_patcher module
"""
import unittest
import os
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from flagai.model.transformers_patcher import (
    TransformersPatcher,
    apply_transformers_patches,
    create_megatron_compatible_linear,
    create_megatron_compatible_embedding
)
from flagai.training_args import TrainingArgs


class TestTransformersPatcher(unittest.TestCase):
    """Test cases for TransformersPatcher class"""

    def setUp(self):
        """Set up test fixtures"""
        self.training_args_megatron = TrainingArgs(
            env_type="deepspeed+mpu",
            model_parallel_size=4
        )
        self.training_args_no_megatron = TrainingArgs(
            env_type="pytorch",
            model_parallel_size=1
        )

    def test_transformers_patcher_initialization_with_megatron(self):
        """Test TransformersPatcher initialization with Megatron enabled"""
        patcher = TransformersPatcher(self.training_args_megatron)
        self.assertIsNotNone(patcher)
        self.assertTrue(patcher.use_megatron)
        self.assertEqual(patcher.applied_patches, [])

    def test_transformers_patcher_initialization_without_megatron(self):
        """Test TransformersPatcher initialization without Megatron"""
        patcher = TransformersPatcher(self.training_args_no_megatron)
        self.assertIsNotNone(patcher)
        self.assertFalse(patcher.use_megatron)

    def test_transformers_patcher_initialization_without_args(self):
        """Test TransformersPatcher initialization without training_args"""
        patcher = TransformersPatcher(None)
        self.assertIsNotNone(patcher)
        self.assertFalse(patcher.use_megatron)

    @patch.dict(os.environ, {'ENV_TYPE': 'deepspeed+mpu', 'MODEL_PARALLEL_SIZE': '4'})
    def test_transformers_patcher_initialization_from_env(self):
        """Test TransformersPatcher initialization from environment variables"""
        patcher = TransformersPatcher(None)
        self.assertTrue(patcher.use_megatron)

    def test_transformers_patcher_apply_patches_without_megatron(self):
        """Test apply_patches when Megatron is not enabled"""
        patcher = TransformersPatcher(self.training_args_no_megatron)
        patcher.apply_patches()
        # Should return without applying patches
        self.assertEqual(len(patcher.applied_patches), 0)

    @patch('flagai.model.layers.feedforward.ColumnParallelLinear')
    @patch('flagai.model.layers.feedforward.RowParallelLinear')
    @patch('flagai.model.utils.normal_init_method')
    def test_transformers_patcher_apply_patches_with_megatron(self, mock_init, mock_row, mock_col):
        """Test apply_patches when Megatron is enabled"""
        patcher = TransformersPatcher(self.training_args_megatron)
        patcher.apply_patches()
        # Should apply patches
        self.assertGreater(len(patcher.applied_patches), 0)

    def test_transformers_patcher_patch_linear_layers(self):
        """Test _patch_linear_layers method"""
        patcher = TransformersPatcher(self.training_args_megatron)
        with patch('flagai.model.layers.feedforward.ColumnParallelLinear'):
            with patch('flagai.model.layers.feedforward.RowParallelLinear'):
                with patch('flagai.model.utils.normal_init_method'):
                    patcher._patch_linear_layers()
                    # Should add patch to applied_patches
                    self.assertIn('parallel_linear_utility', patcher.applied_patches)

    def test_transformers_patcher_patch_embedding_layers(self):
        """Test _patch_embedding_layers method"""
        patcher = TransformersPatcher(self.training_args_megatron)
        with patch('flagai.model.layers.embeddings.VocabParallelEmbedding'):
            patcher._patch_embedding_layers()
            # Should add patch to applied_patches
            self.assertIn('parallel_embedding_utility', patcher.applied_patches)

    def test_transformers_patcher_patch_model_class(self):
        """Test patch_model_class method"""
        patcher = TransformersPatcher(self.training_args_megatron)
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
        
        model_class = TestModel
        result = patcher.patch_model_class(model_class)
        # Should return model class (placeholder implementation)
        self.assertEqual(result, model_class)

    def test_transformers_patcher_patch_model_class_without_megatron(self):
        """Test patch_model_class when Megatron is not enabled"""
        patcher = TransformersPatcher(self.training_args_no_megatron)
        
        class TestModel(nn.Module):
            pass
        
        model_class = TestModel
        result = patcher.patch_model_class(model_class)
        # Should return original model class
        self.assertEqual(result, model_class)

    def test_transformers_patcher_restore_patches(self):
        """Test restore_patches method"""
        patcher = TransformersPatcher(self.training_args_megatron)
        patcher.original_modules['nn.Linear'] = nn.Linear
        patcher.applied_patches = ['test_patch']
        
        patcher.restore_patches()
        # Should clear applied_patches
        self.assertEqual(len(patcher.applied_patches), 0)

    def test_transformers_patcher_get_applied_patches(self):
        """Test get_applied_patches method"""
        patcher = TransformersPatcher(self.training_args_megatron)
        patcher.applied_patches = ['patch1', 'patch2']
        
        patches = patcher.get_applied_patches()
        self.assertEqual(patches, ['patch1', 'patch2'])
        # Should return a copy
        patches.append('patch3')
        self.assertEqual(len(patcher.applied_patches), 2)

    def test_apply_transformers_patches_function(self):
        """Test apply_transformers_patches convenience function"""
        patcher = apply_transformers_patches(self.training_args_megatron)
        self.assertIsInstance(patcher, TransformersPatcher)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""

    def setUp(self):
        """Set up test fixtures"""
        # Clear environment variables
        if 'ENV_TYPE' in os.environ:
            del os.environ['ENV_TYPE']
        if 'MODEL_PARALLEL_SIZE' in os.environ:
            del os.environ['MODEL_PARALLEL_SIZE']

    def test_create_megatron_compatible_linear_without_megatron(self):
        """Test create_megatron_compatible_linear without Megatron"""
        linear = create_megatron_compatible_linear(10, 5)
        self.assertIsInstance(linear, nn.Linear)
        self.assertEqual(linear.in_features, 10)
        self.assertEqual(linear.out_features, 5)

    @patch.dict(os.environ, {'ENV_TYPE': 'deepspeed+mpu', 'MODEL_PARALLEL_SIZE': '4'})
    @patch('flagai.model.transformers_patcher.ColumnParallelLinear')
    def test_create_megatron_compatible_linear_with_megatron(self, mock_col_linear):
        """Test create_megatron_compatible_linear with Megatron"""
        mock_instance = MagicMock()
        mock_col_linear.return_value = mock_instance
        
        linear = create_megatron_compatible_linear(10, 5, gather_output=True)
        # Should create ColumnParallelLinear
        mock_col_linear.assert_called_once()

    @patch.dict(os.environ, {'ENV_TYPE': 'deepspeed+mpu', 'MODEL_PARALLEL_SIZE': '4'})
    @patch('flagai.model.transformers_patcher.RowParallelLinear')
    def test_create_megatron_compatible_linear_row_parallel(self, mock_row_linear):
        """Test create_megatron_compatible_linear with input_is_parallel=True"""
        mock_instance = MagicMock()
        mock_row_linear.return_value = mock_instance
        
        linear = create_megatron_compatible_linear(10, 5, input_is_parallel=True)
        # Should create RowParallelLinear
        mock_row_linear.assert_called_once()

    def test_create_megatron_compatible_embedding_without_megatron(self):
        """Test create_megatron_compatible_embedding without Megatron"""
        embedding = create_megatron_compatible_embedding(1000, 128)
        self.assertIsInstance(embedding, nn.Embedding)
        self.assertEqual(embedding.num_embeddings, 1000)
        self.assertEqual(embedding.embedding_dim, 128)

    @patch.dict(os.environ, {'ENV_TYPE': 'deepspeed+mpu', 'MODEL_PARALLEL_SIZE': '4'})
    @patch('flagai.model.transformers_patcher.VocabParallelEmbedding')
    def test_create_megatron_compatible_embedding_with_megatron(self, mock_vocab_embedding):
        """Test create_megatron_compatible_embedding with Megatron"""
        mock_instance = MagicMock()
        mock_vocab_embedding.return_value = mock_instance
        
        embedding = create_megatron_compatible_embedding(1000, 128)
        # Should create VocabParallelEmbedding
        mock_vocab_embedding.assert_called_once()

    def test_create_megatron_compatible_linear_with_bias(self):
        """Test create_megatron_compatible_linear with bias parameter"""
        linear = create_megatron_compatible_linear(10, 5, bias=False)
        self.assertIsInstance(linear, nn.Linear)
        self.assertIsNone(linear.bias)

    def test_create_megatron_compatible_embedding_with_padding_idx(self):
        """Test create_megatron_compatible_embedding with padding_idx"""
        embedding = create_megatron_compatible_embedding(1000, 128, padding_idx=0)
        self.assertIsInstance(embedding, nn.Embedding)
        self.assertEqual(embedding.padding_idx, 0)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTransformersPatcher))
    suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

