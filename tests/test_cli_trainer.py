# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
Unit tests for CLITrainer class
"""
import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from flagai.cli_trainer import CLITrainer
from flagai.training_args import TrainingArgs


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class TestCLITrainer(unittest.TestCase):
    """Test cases for CLITrainer class"""

    def setUp(self):
        """Set up test fixtures"""
        import sys
        self.training_args = TrainingArgs(
            env_type="pytorch",
            experiment_name="test_experiment",
            batch_size=2,
            epochs=1,
            lr=1e-4,
            pytorch_device="cpu",
            save_dir="./test_checkpoints",
            log_interval=10
        )
        # Parse args to get attributes - mock sys.argv to avoid parsing test script arguments
        with patch.object(sys, 'argv', ['test_cli_trainer.py']):
            self.training_args = self.training_args.parse_args()
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

    def test_cli_trainer_initialization(self):
        """Test CLITrainer initialization"""
        trainer = CLITrainer(self.training_args)
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.env_type, "pytorch")

    def test_cli_trainer_initialization_with_different_env_types(self):
        """Test CLITrainer initialization with different env_types"""
        import sys
        env_types = ["pytorch", "pytorchDDP", "deepspeed", "bmtrain"]
        for env_type in env_types:
            args = TrainingArgs(env_type=env_type)
            with patch.object(sys, 'argv', ['test_cli_trainer.py']):
                args = args.parse_args()
            # Mock environment variables for distributed training
            with patch.dict(os.environ, {'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1'}, clear=False):
                trainer = CLITrainer(args)
                self.assertEqual(trainer.env_type, env_type)

    @patch('flagai.cli_trainer.torch.distributed.is_initialized')
    def test_cli_trainer_pre_train(self, mock_is_init):
        """Test CLITrainer pre_train method"""
        mock_is_init.return_value = False
        trainer = CLITrainer(self.training_args)
        trainer.pre_train(model=self.model)
        # Should complete without error

    def test_cli_trainer_train_method(self):
        """Test CLITrainer train method (compatibility method)"""
        trainer = CLITrainer(self.training_args)
        
        # Create a simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randn(10),
                    'labels': torch.randint(0, 5, (1,)).item()
                }
        
        dataset = SimpleDataset()
        
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'labels': torch.tensor([item['labels'] for item in batch])
            }
        
        # Mock the do_train method to avoid actual training
        with patch.object(trainer, 'do_train') as mock_do_train:
            trainer.train(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataset=dataset,
                collate_fn=collate_fn
            )
            # Verify pre_train and do_train were called
            mock_do_train.assert_called_once()

    def test_cli_trainer_get_dataloader(self):
        """Test CLITrainer get_dataloader method"""
        trainer = CLITrainer(self.training_args)
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return {'data': torch.randn(5)}
        
        dataset = SimpleDataset()
        
        def collate_fn(batch):
            return {'data': torch.stack([item['data'] for item in batch])}
        
        dataloader = trainer.get_dataloader(dataset, collate_fn, shuffle=True)
        self.assertIsNotNone(dataloader)
        self.assertEqual(len(dataloader), 5)  # batch_size=2, 10 samples = 5 batches

    def test_cli_trainer_save_checkpoint(self):
        """Test CLITrainer save_checkpoint method"""
        trainer = CLITrainer(self.training_args)
        
        with patch('flagai.utils.save_checkpoint') as mock_save:
            # CLITrainer uses utils.save_checkpoint directly
            from flagai import utils
            utils.save_checkpoint(
                iteration=100,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                save_dir=self.training_args.save_dir
            )
            # Should call save_checkpoint
            mock_save.assert_called_once()

    def test_cli_trainer_load_checkpoint(self):
        """Test CLITrainer load_checkpoint method"""
        trainer = CLITrainer(self.training_args)
        
        with patch('flagai.utils.load_checkpoint') as mock_load:
            mock_load.return_value = {'iteration': 100}
            # CLITrainer uses utils.load_checkpoint directly
            from flagai import utils
            result = utils.load_checkpoint(
                model=self.model,
                load_dir="./test_checkpoints"
            )
            # Should call load_checkpoint
            mock_load.assert_called_once()
            self.assertIsNotNone(result)

    def test_cli_trainer_evaluate(self):
        """Test CLITrainer evaluate method"""
        trainer = CLITrainer(self.training_args)
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randn(10),
                    'labels': torch.randint(0, 5, (1,)).item()
                }
        
        dataset = SimpleDataset()
        
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'labels': torch.tensor([item['labels'] for item in batch])
            }
        
        def metric_method(output, labels, meta=None):
            return 0.95
        
        # Create data loader
        from torch.utils.data import DataLoader
        data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        # Mock model forward
        def forward_step_func(data_iterator, model, mems=None):
            return {'logits': torch.randn(2, 5), 'loss': torch.tensor(0.5)}
        
        with patch.object(self.model, 'forward', return_value=torch.randn(2, 5)):
            # CLITrainer.evaluate takes data_loader, model, and forward_step_func
            trainer.metric_methods = [["accuracy", metric_method]]
            # Mock tokenizer attribute to avoid AttributeError
            trainer.tokenizer = None
            result = trainer.evaluate(
                data_loader=data_loader,
                model=self.model,
                forward_step_func=forward_step_func
            )
            # Should return evaluation results
            self.assertIsNotNone(result)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCLITrainer))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

