# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-config', type=str, 
                       help='model configuration file')
    return parser

def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--base-path', type=str, default=None,
                       help='Path to the project base directory.')
    group.add_argument('--dataset_name', type=str, default=None,
                       help='Name of the dataset')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-name', type=str, default=None,
                       help='Output filename to save checkpoints to.')
    group.add_argument('--save-iters', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument('--inspect-iters', type=int, default=1000,
                       help='number of inspecting')
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--train-iters', type=int, default=1000000,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--max-length', type=int, default=512,
                       help='max length of input')
    group.add_argument('--max-encoder-length', type=int, default=512,
                       help='max length of encoder input')
    group.add_argument('--max-decoder-length', type=int, default=256,
                       help='max length of decoder input')
    group.add_argument('--start-step', type=int, default=0,
                       help='step to start or continue training')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')

    group.add_argument('--epochs', type=int, default=1,
                       help='total number of epochs to train over all training runs')

    # Learning rate.
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')

    group.add_argument('--warmup-iters', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    
    # Experiment type
    group.add_argument('--pet', type=str, default="")
    group.add_argument('--comp-type', type=str, default='none',
                       choices=['quant', 'moe', 'pr', 'spr', 'mix', 'none'])
    group.add_argument('--pet-init-type', type=str, default='random',
                       choices=['random', 'inherit'])
    group.add_argument('--recover', type=str, default="")
    group.add_argument('--distill', type=str, default="")
    group.add_argument('--quant-ckpt-path', type=str, default='')
    group.add_argument('--quant-config-path', type=str, default='')
    group.add_argument('--moe-ckpt-path', type=str, default='')
    group.add_argument('--pr-ckpt-path', type=str, default='')
    group.add_argument('--pr-config-path', type=str, default='')
    group.add_argument('--spr-config-path', type=str, default='')
    group.add_argument('--spr-ckpt-path', type=str, default='')
    group.add_argument('--model-ckpt-path', type=str, default='')
    group.add_argument('--mix-ckpt-path', type=str, default='')
    group.add_argument('--inherit-ckpt-path', type=str, default='')
    group.add_argument('--mix-layer-ckpt-path', type=str, default='')

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    
    args = parser.parse_args()
    return args
