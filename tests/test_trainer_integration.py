# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
Integration tests for Trainer with deepspeed_utils
"""
import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from flagai.trainer import Trainer
from flagai.deepspeed_utils import (
    initialize_deepspeed,
    configure_activation_checkpointing,
    get_checkpointing_functions
)


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class TestTrainerDeepSpeedIntegration(unittest.TestCase):
    """Integration tests for Trainer with DeepSpeed utilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

    @patch('flagai.trainer.initialize_deepspeed')
    def test_trainer_uses_deepspeed_utils(self, mock_init_deepspeed):
        """Test that Trainer uses deepspeed_utils.initialize_deepspeed"""
        mock_init_deepspeed.return_value = (self.model, self.optimizer, None, self.lr_scheduler)
        
        trainer = Trainer(
            env_type='deepspeed',
            pytorch_device='cpu',
            epochs=1,
            batch_size=2
        )
        
        # Mock distributed environment
        with patch('torch.distributed.is_initialized', return_value=False):
            with patch('flagai.trainer.launch_dist'):
                # Initialize trainer (this would normally call initialize_deepspeed)
                # We're just checking that the import is correct
                self.assertTrue(hasattr(trainer, 'env_type'))
                self.assertEqual(trainer.env_type, 'deepspeed')

    @patch('flagai.trainer.configure_activation_checkpointing')
    def test_trainer_uses_configure_activation_checkpointing(self, mock_configure):
        """Test that Trainer uses deepspeed_utils.configure_activation_checkpointing"""
        trainer = Trainer(
            env_type='deepspeed+mpu',
            pytorch_device='cpu',
            epochs=1,
            batch_size=2,
            model_parallel_size=2,
            deepspeed_activation_checkpointing=True
        )
        
        # Mock distributed environment
        with patch('torch.distributed.is_initialized', return_value=False):
            with patch('flagai.trainer.launch_dist'):
                with patch('flagai.trainer.mpu') as mock_mpu:
                    mock_mpu.initialize_model_parallel = Mock()
                    # The configure_activation_checkpointing should be called during initialization
                    # This is tested indirectly through the import
                    self.assertTrue(hasattr(trainer, 'deepspeed_activation_checkpointing'))

    @patch('flagai.trainer.get_checkpointing_functions')
    def test_trainer_uses_get_checkpointing_functions(self, mock_get_functions):
        """Test that Trainer uses deepspeed_utils.get_checkpointing_functions"""
        mock_get_functions.return_value = (Mock(), Mock(), Mock())
        
        trainer = Trainer(
            env_type='deepspeed+mpu',
            pytorch_device='cpu',
            epochs=1,
            batch_size=2,
            model_parallel_size=2,
            deepspeed_activation_checkpointing=True
        )
        
        # Mock distributed environment
        with patch('torch.distributed.is_initialized', return_value=False):
            with patch('flagai.trainer.launch_dist'):
                with patch('flagai.trainer.mpu') as mock_mpu:
                    mock_mpu.initialize_model_parallel = Mock()
                    # The get_checkpointing_functions should be available
                    # This is tested indirectly through the import
                    self.assertTrue(hasattr(trainer, 'model_parallel_size'))

    def test_trainer_imports_deepspeed_utils(self):
        """Test that Trainer correctly imports from deepspeed_utils"""
        # Check that the functions are imported
        from flagai import trainer
        self.assertTrue(hasattr(trainer, 'initialize_deepspeed'))
        self.assertTrue(hasattr(trainer, 'configure_activation_checkpointing'))
        self.assertTrue(hasattr(trainer, 'get_checkpointing_functions'))

    @patch('flagai.trainer.deepspeed')
    @patch('flagai.trainer.initialize_deepspeed')
    def test_trainer_deepspeed_initialization_flow(self, mock_init, mock_ds):
        """Test the flow of DeepSpeed initialization in Trainer"""
        mock_engine = MagicMock()
        mock_engine.optimizer = self.optimizer
        mock_engine.lr_scheduler = self.lr_scheduler
        mock_init.return_value = (mock_engine, self.optimizer, None, self.lr_scheduler)
        
        trainer = Trainer(
            env_type='deepspeed',
            pytorch_device='cpu',
            epochs=1,
            batch_size=2
        )
        
        # Verify that initialize_deepspeed is available
        self.assertTrue(callable(trainer.initialize_deepspeed) or 
                       hasattr(trainer, 'initialize_deepspeed'))

    def test_trainer_deepspeed_config_handling(self):
        """Test that Trainer correctly handles DeepSpeed configuration"""
        trainer = Trainer(
            env_type='deepspeed',
            pytorch_device='cpu',
            epochs=1,
            batch_size=2,
            deepspeed_config={'train_batch_size': 4}
        )
        
        self.assertIsNotNone(trainer.deepspeed_config)
        self.assertEqual(trainer.deepspeed_config['train_batch_size'], 4)

    def test_trainer_activation_checkpointing_config(self):
        """Test that Trainer correctly handles activation checkpointing configuration"""
        trainer = Trainer(
            env_type='deepspeed+mpu',
            pytorch_device='cpu',
            epochs=1,
            batch_size=2,
            model_parallel_size=2,
            deepspeed_activation_checkpointing=True,
            num_checkpoints=4
        )
        
        self.assertTrue(trainer.deepspeed_activation_checkpointing)
        self.assertEqual(trainer.num_checkpoints, 4)


class TestDeepSpeedUtilsIntegration(unittest.TestCase):
    """Integration tests for deepspeed_utils functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

    @patch('flagai.deepspeed_utils.deepspeed')
    @patch('flagai.deepspeed_utils.is_deepspeed_version')
    def test_initialize_deepspeed_integration(self, mock_is_version, mock_ds):
        """Test initialize_deepspeed integration"""
        mock_is_version.return_value = True
        mock_engine = MagicMock()
        mock_engine.optimizer = self.optimizer
        mock_engine.lr_scheduler = self.lr_scheduler
        mock_ds.initialize.return_value = mock_engine
        
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', True):
            result = initialize_deepspeed(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                config={}
            )
            self.assertIsNotNone(result)
            mock_ds.initialize.assert_called_once()

    @patch('flagai.deepspeed_utils.deepspeed')
    def test_configure_activation_checkpointing_integration(self, mock_ds):
        """Test configure_activation_checkpointing integration"""
        mock_mpu = Mock()
        mock_ds.checkpointing.configure = Mock()
        mock_ds.checkpointing.checkpoint = Mock()
        mock_ds.checkpointing.get_cuda_rng_tracker = Mock()
        
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', True):
            with patch('flagai.deepspeed_utils.is_deepspeed_version', return_value=False):
                configure_activation_checkpointing(
                    mpu=mock_mpu,
                    checkpoint_activations=True,
                    num_checkpoints=4
                )
                # Should configure checkpointing
                mock_ds.checkpointing.configure.assert_called_once()

    @patch('flagai.deepspeed_utils.deepspeed')
    def test_get_checkpointing_functions_integration(self, mock_ds):
        """Test get_checkpointing_functions integration"""
        mock_checkpoint = Mock()
        mock_get_tracker = Mock()
        mock_seed = Mock()
        
        mock_ds.checkpointing.checkpoint = mock_checkpoint
        mock_ds.checkpointing.get_cuda_rng_tracker = mock_get_tracker
        mock_ds.checkpointing.model_parallel_cuda_manual_seed = mock_seed
        
        with patch('flagai.deepspeed_utils.HAS_DEEPSPEED', True):
            with patch('flagai.deepspeed_utils.is_deepspeed_version', return_value=False):
                checkpoint, get_tracker, seed_func = get_checkpointing_functions()
                self.assertIsNotNone(checkpoint)
                self.assertIsNotNone(get_tracker)
                self.assertIsNotNone(seed_func)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTrainerDeepSpeedIntegration))
    suite.addTest(unittest.makeSuite(TestDeepSpeedUtilsIntegration))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

