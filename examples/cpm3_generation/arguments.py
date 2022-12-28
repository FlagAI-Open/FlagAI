import argparse

def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-config', type=str, 
                       help='model configuration file')
    group.add_argument('--vocab-file', type=str, 
                       help='model vocab file')
    return parser

def add_inference_args(parser: argparse.ArgumentParser):
    """inference arguments"""

    group = parser.add_argument_group('infer', 'inference configuration')
    group.add_argument('--output-file', type=str, default=None,
                       help='output file')
    group.add_argument('--input-file', type=str, default=None,
                       help='input file')
    group.add_argument('--span-length', type=int, default=150,
                       help='span length')
    group.add_argument('--beam-size', type=int, default=1,
                       help='beam size')
    group.add_argument('--top-k', type=int, default=0,
                       help='top k')
    group.add_argument('--top-p', type=float, default=0.0,
                       help='top p')
    group.add_argument('--temperature', type=float, default=0.9,
                       help='temperature')
    group.add_argument('--no-repeat-ngram-size', type=int, default=0,
                       help='no repeat ngram size')
    group.add_argument('--repetition-penalty', type=float, default=1.2,
                       help='repetition penalty')
    group.add_argument('--random-sample', default=False, action='store_true',
                       help='use random sample strategy')
    group.add_argument('--use-contrastive-search', default=False, action='store_true',
                       help='whether to use contrastive search')
    return parser

def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

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
                       help='Data Loader batch size while training')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--train-iters', type=int, default=1000000,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--max-length', type=int, default=840,
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
    group.add_argument('--eval-step', type=int, default=5,
                       help='eval steps while training')
    group.add_argument('--eval-batch-size', type=int, default=16,
                       help='batch size in evaluation')
    group.add_argument('--dataset', type=str, default=None,
                       help='dataset name')
    group.add_argument('--task', type=str, default=None,
                       help='task name, choices: lm, compress, expand, rewrite, rewrite_s')
    group.add_argument('--log-dir', type=str, default=None,
                       help='tensorboard log directory')

    return parser



def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    parser = add_inference_args(parser)
    
    args = parser.parse_args()
    return args
