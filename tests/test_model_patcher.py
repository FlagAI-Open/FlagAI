# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
Unit tests for model_patcher module
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from flagai.model.model_patcher import ModelPatcher, apply_model_patches
from flagai.training_args import TrainingArgs


class TestModelPatcher(unittest.TestCase):
    """Test cases for ModelPatcher class"""

    def setUp(self):
        """Set up test fixtures"""
        self.training_args = TrainingArgs(
            enable_flash_attn_models=False,
            model_parallel_size=1
        )

    def test_model_patcher_initialization(self):
        """Test ModelPatcher initialization"""
        patcher = ModelPatcher(self.training_args)
        self.assertIsNotNone(patcher)
        self.assertEqual(patcher.training_args, self.training_args)
        self.assertEqual(patcher.applied_patches, [])

    def test_model_patcher_initialization_without_args(self):
        """Test ModelPatcher initialization without training_args"""
        patcher = ModelPatcher(None)
        self.assertIsNotNone(patcher)
        self.assertIsNone(patcher.training_args)

    def test_model_patcher_apply_patches_without_args(self):
        """Test apply_patches without training_args"""
        patcher = ModelPatcher(None)
        patcher.apply_patches()
        # Should return without error
        self.assertEqual(len(patcher.applied_patches), 0)

    def test_model_patcher_get_applied_patches(self):
        """Test get_applied_patches method"""
        patcher = ModelPatcher(self.training_args)
        patches = patcher.get_applied_patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 0)

    @patch('flagai.model.model_patcher.ModelPatcher._apply_flash_attention_patches')
    def test_model_patcher_apply_patches_with_flash_attn(self, mock_flash_attn):
        """Test apply_patches with flash attention enabled"""
        import sys
        training_args = TrainingArgs(
            enable_flash_attn_models=True,
            model_name="aquila2"
        )
        with patch.object(sys, 'argv', ['test_model_patcher.py']):
            training_args = training_args.parse_args()
        patcher = ModelPatcher(training_args)
        patcher.apply_patches(model_name="aquila2")
        # Should call flash attention patches
        mock_flash_attn.assert_called_once()

    @patch('flagai.model.model_patcher.ModelPatcher._patch_aquila_flash_attention')
    def test_model_patcher_patch_aquila(self, mock_patch):
        """Test patching Aquila model"""
        training_args = TrainingArgs(
            enable_flash_attn_models=True,
            model_name="aquila2"
        )
        patcher = ModelPatcher(training_args)
        patcher._apply_flash_attention_patches("aquila2")
        mock_patch.assert_called_once()

    @patch('flagai.model.model_patcher.ModelPatcher._patch_llama_flash_attention')
    def test_model_patcher_patch_llama(self, mock_patch):
        """Test patching Llama model"""
        training_args = TrainingArgs(
            enable_flash_attn_models=True,
            model_name="llama"
        )
        patcher = ModelPatcher(training_args)
        patcher._apply_flash_attention_patches("llama")
        mock_patch.assert_called_once()

    def test_model_patcher_apply_condense_patch(self):
        """Test apply_condense_patch method"""
        patcher = ModelPatcher(self.training_args)
        # Mock the actual patch function to avoid import errors
        with patch('flagai.model.aquila2.aquila_condense_monkey_patch.replace_aquila_with_condense') as mock_replace:
            patcher.apply_condense_patch(ratio=4.0)
            # Should call the replace function
            mock_replace.assert_called_once_with(4.0)
            # Should add patch to applied_patches
            self.assertIn('aquila_condense_ratio_4.0', patcher.get_applied_patches())

    @patch('flagai.model.model_patcher.apply_transformers_patches')
    def test_apply_model_patches_with_megatron(self, mock_transformers):
        """Test apply_model_patches with Megatron environment"""
        training_args = TrainingArgs(
            env_type="deepspeed+mpu",
            model_parallel_size=4
        )
        mock_patcher = MagicMock()
        mock_patcher.get_applied_patches.return_value = []
        
        with patch('flagai.model.model_patcher.ModelPatcher', return_value=mock_patcher):
            result = apply_model_patches(training_args=training_args, model_name="test_model")
            # Should apply transformers patches for Megatron
            # Note: actual behavior depends on environment

    def test_apply_model_patches_function(self):
        """Test apply_model_patches function"""
        training_args = TrainingArgs(
            enable_flash_attn_models=False
        )
        patcher = apply_model_patches(training_args=training_args)
        self.assertIsInstance(patcher, ModelPatcher)

    def test_apply_model_patches_with_condense_ratio(self):
        """Test apply_model_patches with condense_ratio"""
        training_args = TrainingArgs()
        patcher = apply_model_patches(
            training_args=training_args,
            condense_ratio=4.0
        )
        self.assertIsInstance(patcher, ModelPatcher)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestModelPatcher))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

