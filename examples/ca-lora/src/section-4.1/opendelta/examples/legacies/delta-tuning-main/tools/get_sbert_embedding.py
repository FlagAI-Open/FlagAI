from sentence_transformers import SentenceTransformer, util
import argparse
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_sentence(task, line):
    if task in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
        # Text classification tasks
        if line[1] is None or pd.isna(line[1]):
            return ''
        else:
            return line[1]
    else:
        # GLUE tasks
        line = line.strip().split('\t')
        if task == 'CoLA':
            return line[-1]
        elif task == 'MNLI':
            return line[8] + ' ' + line[9]
        elif task == 'MRPC':
            return line[-2] + ' ' + line[-1]
        elif task == 'QNLI':
            return line[1] + ' ' + line[2]
        elif task == 'QQP':
            return line[3] + ' ' + line[4]
        elif task == 'RTE':
            return line[1] + ' ' + line[2]
        elif task == 'SNLI':
            return line[7] + ' ' + line[8]
        elif task == 'SST-2':
            return line[0]
        elif task == 'STS-B':
            return line[-3] + ' ' + line[-2]
        elif task == 'WNLI':
            return line[1] + ' ' + line[2]
        else:
            raise NotImplementedError

def split_header(task, lines):
    """Returns if the task file has a header or not."""
    if task in ["CoLA"]:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")

def load_datasets(data_dir, task, do_test=False):
    dataset = {}
    if task == "MNLI":
        splits = ["train", "dev_matched"]
        if do_test:
            splits += ['test_matched', 'test_mismatched'] 
    else:
        splits = ["train", "dev"]
        if do_test:
            splits.append('test')
    for split in splits:
        if task in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
            filename = os.path.join(data_dir, f"{split}.csv")
            dataset[split] = pd.read_csv(filename, header=None).values.tolist()
        else:
            filename = os.path.join(data_dir, f"{split}.tsv")
            with open(filename, "r") as f:
                lines = f.readlines()
                header, content = split_header(task, lines)
            dataset[split] = content
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_test", action='store_true', help="Generate embeddings for test splits (test set is usually large, so we don't want to repeatedly generate embeddings for them)")
    parser.add_argument("--sbert_model", type=str, default='roberta-large', help="Sentence BERT model name")

    parser.add_argument("--k", type=int, help="Number of training instances per label", default=16)
    parser.add_argument("--data_dir", type=str, default="data/k-shot", help="Path to few-shot data")
    parser.add_argument("--seed", type=int, nargs="+", default=[42, 13, 21, 87, 100], help="Seeds for data splits")
    parser.add_argument("--task", type=str, nargs="+", default=["SST-2", "sst-5", "mr", "cr", "mpqa", "subj", "trec", "CoLA", "MRPC", "QQP", "STS-B", "MNLI", "SNLI", "QNLI", "RTE"], help="Tasks")

    args = parser.parse_args()

    model = SentenceTransformer('{}-nli-stsb-mean-tokens'.format(args.sbert_model))
    model = model.cuda()

    for task in args.task:
        for seed in args.seed:
            folder = os.path.join(args.data_dir, task, '{}-{}'.format(args.k, seed))
            dataset = load_datasets(folder, task, do_test=args.do_test)
            for split in dataset:
                print('{}-{}-{}-{}'.format(task, args.k, seed, split))
                lines = dataset[split]
                embeddings = []
                for line_id, line in tqdm(enumerate(lines)):
                    sent = get_sentence(task, line)
                    if line_id == 0:
                        print('|', sent)
                    emb = model.encode(sent)
                    embeddings.append(emb)
                embeddings = np.stack(embeddings)
                np.save(os.path.join(folder, "{}_sbert-{}.npy".format(split, args.sbert_model)), embeddings)

if __name__ == '__main__':
    main()
