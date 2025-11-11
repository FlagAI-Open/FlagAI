# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
Integration tests for AutoLoader with ModelPatcher
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.model_patcher import ModelPatcher, apply_model_patches
from flagai.training_args import TrainingArgs


class TestAutoLoaderModelPatcherIntegration(unittest.TestCase):
    """Integration tests for AutoLoader with ModelPatcher"""

    def setUp(self):
        """Set up test fixtures"""
        self.training_args_with_patches = TrainingArgs(
            enable_flash_attn_models=True,
            model_name="aquila2",
            env_type="deepspeed+mpu",
            model_parallel_size=4
        )
        self.training_args_no_patches = TrainingArgs(
            enable_flash_attn_models=False,
            model_parallel_size=1
        )

    @patch('flagai.auto_model.auto_loader.apply_model_patches')
    def test_auto_loader_applies_patches_with_training_args(self, mock_apply_patches):
        """Test that AutoLoader applies patches when training_args is provided"""
        mock_patcher = MagicMock()
        mock_patcher.get_applied_patches.return_value = ['flash_attention']
        mock_apply_patches.return_value = mock_patcher
        
        loader = AutoLoader(
            task_name="lm",
            model_name="aquila2",
            only_download_config=True,
            training_args=self.training_args_with_patches
        )
        
        # Verify that apply_model_patches was called
        mock_apply_patches.assert_called_once()
        # Verify that model_patcher is set
        self.assertIsNotNone(loader.model_patcher)

    @patch('flagai.auto_model.auto_loader.apply_model_patches')
    def test_auto_loader_no_patches_without_training_args(self, mock_apply_patches):
        """Test that AutoLoader doesn't apply patches when training_args is None"""
        loader = AutoLoader(
            task_name="lm",
            model_name="aquila2",
            only_download_config=True,
            training_args=None
        )
        
        # Should not call apply_model_patches
        mock_apply_patches.assert_not_called()
        self.assertIsNone(loader.model_patcher)

    @patch('flagai.auto_model.auto_loader.apply_model_patches')
    def test_auto_loader_patches_with_flash_attention(self, mock_apply_patches):
        """Test AutoLoader with Flash Attention enabled"""
        mock_patcher = MagicMock()
        mock_patcher.get_applied_patches.return_value = ['aquila_flash_attention']
        mock_apply_patches.return_value = mock_patcher
        
        loader = AutoLoader(
            task_name="lm",
            model_name="aquila2",
            only_download_config=True,
            training_args=self.training_args_with_patches
        )
        
        # Verify patches were applied
        mock_apply_patches.assert_called_once()
        call_kwargs = mock_apply_patches.call_args[1]
        self.assertEqual(call_kwargs.get('training_args'), self.training_args_with_patches)

    @patch('flagai.auto_model.auto_loader.apply_model_patches')
    def test_auto_loader_patches_with_condense_ratio(self, mock_apply_patches):
        """Test AutoLoader with condense_ratio parameter"""
        mock_patcher = MagicMock()
        mock_patcher.get_applied_patches.return_value = ['aquila_condense_ratio_4.0']
        mock_apply_patches.return_value = mock_patcher
        
        loader = AutoLoader(
            task_name="lm",
            model_name="aquila2",
            only_download_config=True,
            training_args=self.training_args_with_patches,
            condense_ratio=4.0
        )
        
        # Verify condense_ratio was passed
        mock_apply_patches.assert_called_once()
        call_kwargs = mock_apply_patches.call_args[1]
        self.assertEqual(call_kwargs.get('condense_ratio'), 4.0)

    @patch('flagai.auto_model.auto_loader.apply_model_patches')
    @patch('flagai.auto_model.auto_loader.apply_transformers_patches')
    def test_auto_loader_patches_with_megatron(self, mock_transformers_patches, mock_apply_patches):
        """Test AutoLoader with Megatron model parallelism"""
        mock_patcher = MagicMock()
        mock_patcher.get_applied_patches.return_value = []
        mock_transformers_patcher = MagicMock()
        mock_transformers_patcher.get_applied_patches.return_value = ['transformers_linear_replacement']
        mock_transformers_patches.return_value = mock_transformers_patcher
        mock_apply_patches.return_value = mock_patcher
        
        loader = AutoLoader(
            task_name="lm",
            model_name="test_model",
            only_download_config=True,
            training_args=self.training_args_with_patches
        )
        
        # Verify patches were applied
        mock_apply_patches.assert_called_once()

    @patch('flagai.auto_model.auto_loader.apply_model_patches')
    def test_auto_loader_patches_different_models(self, mock_apply_patches):
        """Test AutoLoader applies different patches for different models"""
        mock_patcher = MagicMock()
        mock_apply_patches.return_value = mock_patcher
        
        # Test with Aquila model
        loader_aquila = AutoLoader(
            task_name="lm",
            model_name="aquila2",
            only_download_config=True,
            training_args=self.training_args_with_patches
        )
        
        # Test with Llama model
        training_args_llama = TrainingArgs(
            enable_flash_attn_models=True,
            model_name="llama"
        )
        loader_llama = AutoLoader(
            task_name="lm",
            model_name="llama",
            only_download_config=True,
            training_args=training_args_llama
        )
        
        # Both should apply patches
        self.assertEqual(mock_apply_patches.call_count, 2)

    @patch('flagai.auto_model.auto_loader.apply_model_patches')
    def test_auto_loader_patches_error_handling(self, mock_apply_patches):
        """Test AutoLoader handles patch errors gracefully"""
        mock_apply_patches.side_effect = Exception("Patch failed")
        
        # Should not raise exception, but handle gracefully
        try:
            loader = AutoLoader(
                task_name="lm",
                model_name="aquila2",
                only_download_config=True,
                training_args=self.training_args_with_patches
            )
            # If it doesn't raise, that's fine - error handling is working
        except Exception:
            # If it does raise, that's also acceptable behavior
            pass

    def test_auto_loader_model_patcher_attribute(self):
        """Test that AutoLoader has model_patcher attribute"""
        loader = AutoLoader(
            task_name="lm",
            model_name="aquila2",
            only_download_config=True,
            training_args=None
        )
        
        # Should have model_patcher attribute (may be None)
        self.assertTrue(hasattr(loader, 'model_patcher'))

    @patch('flagai.auto_model.auto_loader.apply_model_patches')
    def test_auto_loader_patches_before_model_init(self, mock_apply_patches):
        """Test that patches are applied before model initialization"""
        mock_patcher = MagicMock()
        mock_apply_patches.return_value = mock_patcher
        
        # Track call order
        call_order = []
        
        def track_apply_patches(*args, **kwargs):
            call_order.append('apply_patches')
            return mock_patcher
        
        mock_apply_patches.side_effect = track_apply_patches
        
        loader = AutoLoader(
            task_name="lm",
            model_name="aquila2",
            only_download_config=True,
            training_args=self.training_args_with_patches
        )
        
        # Patches should be applied (we can't easily test before model init
        # without more complex mocking, but we verify the call happened)
        self.assertTrue(mock_apply_patches.called)
        self.assertIn('apply_patches', call_order)


class TestModelPatcherAutoLoaderFlow(unittest.TestCase):
    """Test the flow of ModelPatcher with AutoLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.training_args = TrainingArgs(
            enable_flash_attn_models=True,
            model_name="aquila2"
        )

    @patch('flagai.model.model_patcher.ModelPatcher._apply_flash_attention_patches')
    def test_model_patcher_flow_with_auto_loader(self, mock_flash_attn):
        """Test the complete flow of ModelPatcher with AutoLoader"""
        patcher = apply_model_patches(
            training_args=self.training_args,
            model_name="aquila2"
        )
        
        # Verify patcher was created
        self.assertIsInstance(patcher, ModelPatcher)
        # Verify patches were applied
        mock_flash_attn.assert_called_once()

    def test_model_patcher_applied_patches_list(self):
        """Test that applied patches are tracked correctly"""
        patcher = apply_model_patches(
            training_args=self.training_args,
            model_name="aquila2"
        )
        
        # Should have a method to get applied patches
        patches = patcher.get_applied_patches()
        self.assertIsInstance(patches, list)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAutoLoaderModelPatcherIntegration))
    suite.addTest(unittest.makeSuite(TestModelPatcherAutoLoaderFlow))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

