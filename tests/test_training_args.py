# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
Unit tests for training_args module
"""
import unittest
import sys
from unittest.mock import patch
from flagai.training_args import TrainingArgs, str2bool, save_best


class TestTrainingArgs(unittest.TestCase):
    """Test cases for TrainingArgs class"""

    def setUp(self):
        """Set up test fixtures"""
        self.default_args = TrainingArgs()

    def test_training_args_initialization(self):
        """Test TrainingArgs initialization with default values"""
        import sys
        args = TrainingArgs()
        # TrainingArgs stores values in parser, need to parse to get attributes
        # Mock sys.argv to avoid parsing test script arguments
        with patch.object(sys, 'argv', ['test_training_args.py']):
            parsed_args = args.parse_args()
            self.assertEqual(parsed_args.env_type, "pytorch")
            self.assertEqual(parsed_args.experiment_name, "test_experiment")
            self.assertEqual(parsed_args.batch_size, 1)
            self.assertEqual(parsed_args.lr, 1e-5)
            self.assertEqual(parsed_args.epochs, 1)

    def test_training_args_custom_values(self):
        """Test TrainingArgs initialization with custom values"""
        import sys
        args = TrainingArgs(
            env_type="pytorchDDP",
            experiment_name="my_experiment",
            batch_size=64,
            lr=1e-4,
            epochs=10,
            num_gpus=4
        )
        # TrainingArgs stores values in parser, need to parse to get attributes
        # Mock sys.argv to avoid parsing test script arguments
        with patch.object(sys, 'argv', ['test_training_args.py']):
            parsed_args = args.parse_args()
            self.assertEqual(parsed_args.env_type, "pytorchDDP")
            self.assertEqual(parsed_args.experiment_name, "my_experiment")
            self.assertEqual(parsed_args.batch_size, 64)
            self.assertEqual(parsed_args.lr, 1e-4)
            self.assertEqual(parsed_args.epochs, 10)
            self.assertEqual(parsed_args.num_gpus, 4)

    def test_training_args_parse_args(self):
        """Test TrainingArgs parse_args method"""
        test_args = ['--batch_size=32', '--lr=2e-4', '--epochs=5']
        with patch.object(sys, 'argv', ['test_training_args.py'] + test_args):
            args = TrainingArgs()
            parsed_args = args.parse_args()
            self.assertEqual(parsed_args.batch_size, 32)
            self.assertEqual(parsed_args.lr, 2e-4)
            self.assertEqual(parsed_args.epochs, 5)

    def test_training_args_add_arg(self):
        """Test TrainingArgs add_arg method"""
        args = TrainingArgs()
        args.add_arg("custom_param", default=100, type=int, help="Custom parameter")
        test_args = ['--custom_param=200']
        with patch.object(sys, 'argv', ['test_training_args.py'] + test_args):
            parsed_args = args.parse_args()
            self.assertEqual(parsed_args.custom_param, 200)

    def test_training_args_add_arg_store_true(self):
        """Test TrainingArgs add_arg with store_true"""
        args = TrainingArgs()
        args.add_arg("enable_feature", default=False, store_true=True, help="Enable feature")
        test_args = ['--enable_feature']
        with patch.object(sys, 'argv', ['test_training_args.py'] + test_args):
            parsed_args = args.parse_args()
            self.assertTrue(parsed_args.enable_feature)

    def test_training_args_deepspeed_config(self):
        """Test TrainingArgs with DeepSpeed configuration"""
        import sys
        args = TrainingArgs(
            env_type="deepspeed",
            deepspeed_config="./deepspeed.json",
            deepspeed_activation_checkpointing=True,
            num_checkpoints=4
        )
        with patch.object(sys, 'argv', ['test_training_args.py']):
            parsed_args = args.parse_args()
            self.assertEqual(parsed_args.env_type, "deepspeed")
            self.assertEqual(parsed_args.deepspeed_config, "./deepspeed.json")
            self.assertTrue(parsed_args.deepspeed_activation_checkpointing)
            self.assertEqual(parsed_args.num_checkpoints, 4)

    def test_training_args_model_parallel(self):
        """Test TrainingArgs with model parallel configuration"""
        import sys
        args = TrainingArgs(
            env_type="deepspeed+mpu",
            model_parallel_size=4,
            num_gpus=8
        )
        with patch.object(sys, 'argv', ['test_training_args.py']):
            parsed_args = args.parse_args()
            self.assertEqual(parsed_args.env_type, "deepspeed+mpu")
            self.assertEqual(parsed_args.model_parallel_size, 4)
            self.assertEqual(parsed_args.num_gpus, 8)

    def test_training_args_lora_config(self):
        """Test TrainingArgs with LoRA configuration"""
        import sys
        args = TrainingArgs(
            lora=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            lora_target_modules=["q_proj", "v_proj"]
        )
        with patch.object(sys, 'argv', ['test_training_args.py']):
            parsed_args = args.parse_args()
            self.assertTrue(parsed_args.lora)
            self.assertEqual(parsed_args.lora_r, 16)
            self.assertEqual(parsed_args.lora_alpha, 32)
            self.assertEqual(parsed_args.lora_dropout, 0.1)
            self.assertEqual(parsed_args.lora_target_modules, ["q_proj", "v_proj"])

    def test_training_args_fp16(self):
        """Test TrainingArgs with FP16 configuration"""
        import sys
        args = TrainingArgs(
            fp16=True,
            pytorch_device="cuda"
        )
        with patch.object(sys, 'argv', ['test_training_args.py']):
            parsed_args = args.parse_args()
            self.assertTrue(parsed_args.fp16)
            self.assertEqual(parsed_args.pytorch_device, "cuda")

    def test_training_args_wandb_config(self):
        """Test TrainingArgs with Weights & Biases configuration"""
        import sys
        args = TrainingArgs(
            wandb=True,
            wandb_dir="./wandb_logs",
            wandb_key="test_key"
        )
        with patch.object(sys, 'argv', ['test_training_args.py']):
            parsed_args = args.parse_args()
            self.assertTrue(parsed_args.wandb)
            self.assertEqual(parsed_args.wandb_dir, "./wandb_logs")
            self.assertEqual(parsed_args.wandb_key, "test_key")


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""

    def test_str2bool_true_values(self):
        """Test str2bool with true values"""
        self.assertTrue(str2bool(True))
        self.assertTrue(str2bool("true"))
        self.assertTrue(str2bool("True"))
        self.assertTrue(str2bool("1"))
        self.assertTrue(str2bool("yes"))
        self.assertTrue(str2bool("on"))

    def test_str2bool_false_values(self):
        """Test str2bool with false values"""
        self.assertFalse(str2bool(False))
        self.assertFalse(str2bool("false"))
        self.assertFalse(str2bool("False"))
        self.assertFalse(str2bool("0"))
        self.assertFalse(str2bool("no"))
        self.assertFalse(str2bool("off"))

    def test_str2bool_invalid_value(self):
        """Test str2bool with invalid value"""
        with self.assertRaises(ValueError):
            str2bool("invalid")

    def test_save_best(self):
        """Test save_best function"""
        best_score = 0.5
        eval_dict = {'loss': 0.3}
        result = save_best(best_score, eval_dict)
        self.assertEqual(result, 0.3)

        best_score = 0.2
        eval_dict = {'loss': 0.5}
        result = save_best(best_score, eval_dict)
        self.assertEqual(result, 0.2)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTrainingArgs))
    suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

