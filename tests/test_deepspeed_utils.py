# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
Unit tests for deepspeed_utils module
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn


class TestDeepSpeedUtils(unittest.TestCase):
    """Test cases for deepspeed_utils module"""

    def setUp(self):
        """Set up test fixtures"""
        self.simple_model = nn.Linear(10, 5)
        self.optimizer = torch.optim.Adam(self.simple_model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

    def test_get_deepspeed_version_without_deepspeed(self):
        """Test get_deepspeed_version when DeepSpeed is not installed"""
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', False):
            from flagai.deepspeed_utils import get_deepspeed_version
            result = get_deepspeed_version()
            self.assertIsNone(result)

    def test_is_deepspeed_version_without_deepspeed(self):
        """Test is_deepspeed_version when DeepSpeed is not installed"""
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', False):
            from flagai.deepspeed_utils import is_deepspeed_version
            result = is_deepspeed_version("0.9.0")
            self.assertFalse(result)

    def test_is_activation_checkpointing_configured_without_deepspeed(self):
        """Test is_activation_checkpointing_configured when DeepSpeed is not installed"""
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', False):
            from flagai.deepspeed_utils import is_activation_checkpointing_configured
            result = is_activation_checkpointing_configured()
            self.assertFalse(result)

    def test_get_checkpointing_functions_without_deepspeed(self):
        """Test get_checkpointing_functions when DeepSpeed is not installed"""
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', False):
            from flagai.deepspeed_utils import get_checkpointing_functions
            checkpoint, get_cuda_rng_tracker, model_parallel_cuda_manual_seed = get_checkpointing_functions()
            self.assertIsNone(checkpoint)
            self.assertIsNone(get_cuda_rng_tracker)
            self.assertIsNone(model_parallel_cuda_manual_seed)

    def test_configure_activation_checkpointing_without_deepspeed(self):
        """Test configure_activation_checkpointing when DeepSpeed is not installed"""
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', False):
            from flagai.deepspeed_utils import configure_activation_checkpointing
            mpu = Mock()
            configure_activation_checkpointing(mpu=mpu)
            # Should return without error

    def test_configure_activation_checkpointing_with_none_mpu(self):
        """Test configure_activation_checkpointing with None mpu"""
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', True):
            from flagai.deepspeed_utils import configure_activation_checkpointing
            configure_activation_checkpointing(mpu=None)
            # Should return without error

    def test_reset_activation_checkpointing_without_deepspeed(self):
        """Test reset_activation_checkpointing when DeepSpeed is not installed"""
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', False):
            from flagai.deepspeed_utils import reset_activation_checkpointing
            reset_activation_checkpointing()
            # Should return without error

    def test_initialize_deepspeed_without_deepspeed_installed(self):
        """Test initialize_deepspeed raises ImportError when DeepSpeed is not installed"""
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', False):
            with patch('flagai.deepspeed_utils.DS_VERSION', None):
                # Reload module to get updated HAS_DEEPSPEED
                import importlib
                import flagai.deepspeed_utils
                importlib.reload(flagai.deepspeed_utils)
                from flagai.deepspeed_utils import initialize_deepspeed
                with self.assertRaises(ImportError):
                    initialize_deepspeed(model=self.simple_model)

    @patch('flagai.deepspeed_utils.is_deepspeed_version')
    def test_initialize_deepspeed_new_version(self, mock_is_version):
        """Test initialize_deepspeed with new DeepSpeed version"""
        mock_is_version.return_value = True
        mock_engine = MagicMock()
        mock_engine.optimizer = self.optimizer
        mock_engine.lr_scheduler = self.lr_scheduler
        
        # Mock deepspeed module
        mock_ds = MagicMock()
        mock_ds.initialize.return_value = mock_engine
        
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', True):
            with patch.dict('sys.modules', {'deepspeed': mock_ds}):
                # Import after patching
                import importlib
                import flagai.deepspeed_utils
                importlib.reload(flagai.deepspeed_utils)
                flagai.deepspeed_utils.deepspeed = mock_ds
                from flagai.deepspeed_utils import initialize_deepspeed
                result = initialize_deepspeed(
                    model=self.simple_model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    config={}
                )
                self.assertIsNotNone(result)
                mock_ds.initialize.assert_called_once()

    @patch('flagai.deepspeed_utils.is_deepspeed_version')
    def test_initialize_deepspeed_old_version(self, mock_is_version):
        """Test initialize_deepspeed with old DeepSpeed version"""
        mock_is_version.return_value = False
        
        # Mock deepspeed module
        mock_ds = MagicMock()
        mock_ds.initialize.return_value = (self.simple_model, self.optimizer, None, self.lr_scheduler)
        
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', True):
            with patch.dict('sys.modules', {'deepspeed': mock_ds}):
                # Import after patching
                import importlib
                import flagai.deepspeed_utils
                importlib.reload(flagai.deepspeed_utils)
                flagai.deepspeed_utils.deepspeed = mock_ds
                from flagai.deepspeed_utils import initialize_deepspeed
                result = initialize_deepspeed(
                    model=self.simple_model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    config={}
                )
                self.assertIsNotNone(result)
                mock_ds.initialize.assert_called_once()


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDeepSpeedUtils))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

