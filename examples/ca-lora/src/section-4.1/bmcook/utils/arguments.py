import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--save-dir', required=True, help='Directory to save result to')
    p.add_argument('--log-interval', type=int, default=1, help='Log loss, LR, scale every n steps')
    p.add_argument('--save-interval', type=int, default=1000, help='Interval steps between checkpoints')
    p.add_argument('--start-lr', type=float, default=1e-4, help='Start learning rate')
    p.add_argument('--data-path', default='openwebtext_document', help='Path to dataset')
    p.add_argument('--cook-config', type=str, help='Path to BMCook config file')

    return p.parse_args()

