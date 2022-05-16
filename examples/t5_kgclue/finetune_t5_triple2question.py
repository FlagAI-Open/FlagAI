import os
import argparse
import torch
from flagai.data.tokenizer.t5.t5_pegasus_tokenizer import T5BatchPegasusTokenizer
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from torch.utils.data.distributed import DistributedSampler
# from utils.tool_utils import set_random_seed
import re
import json
import pandas as pd
from torch.utils.data import Dataset
from rouge import Rouge
from sklearn import metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from flagai.trainer import Trainer


def get_rouge(pred, target):
    scores = Rouge().get_scores(hyps=pred, refs=target)
    rouge_1 = scores[0]['rouge-1']['f']
    rouge_2 = scores[0]['rouge-2']['f']
    rouge_l = scores[0]['rouge-l']['f']
    return rouge_1, rouge_2, rouge_l


def get_bleu(pred, target):
    return sentence_bleu(references=[target.split(' ')],
                         hypothesis=pred.split(' '),
                         smoothing_function=SmoothingFunction().method1)


class DataSet(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, target_len,
                 source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = source["input_ids"].squeeze()
        attention_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()
        decoder_input_ids = target_ids[:-1]
        labels = target_ids[1:].clone().detach()
        labels[target_ids[1:] == tokenizer.pad_token_id]

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long),
            "decoder_input_ids": decoder_input_ids.to(dtype=torch.long),
            "labels": labels.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
        }


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    batch_tuple = tuple(map(torch.stack, zip(*batch)))
    batch_lens = torch.sum(batch_tuple[1], dim=-1, keepdim=False)
    max_len = batch_lens.max().item()
    results = ()
    for item in batch_tuple:
        if item.dim() >= 2:
            results += (item[:, :max_len], )
        else:
            results += (item, )
    return results


def train_data_process_kgclue_cgrm_seq2seq(file):

    def question_clean(text):
        text = re.sub(r'你能告诉我', '', text)
        text = re.sub(r'能够告诉我', '', text)
        text = re.sub(r'告诉我一下', '', text)
        text = re.sub(r'大家知道', '', text)
        text = re.sub(r'我想知道', '', text)
        text = re.sub(r'请告诉我', '', text)
        text = re.sub(r'有人知道', '', text)
        text = re.sub(r'大家了解', '', text)
        text = re.sub(r'我很疑惑', '', text)
        text = re.sub(r'我很好奇', '', text)
        text = re.sub(r'查一下', '', text)
        text = re.sub(r'你知道', '', text)
        text = re.sub(r'谁知道', '', text)
        text = re.sub(r'请问', '', text)
        text = re.sub(r'告诉我', '', text)
        text = re.sub(r'你了解', '', text)
        text = re.sub(r'谁了解', '', text)
        text = re.sub(r'你记得', '', text)
        text = re.sub(r'你觉得', '', text)
        text = re.sub(r'谁是', '', text)
        return text

    triplet, question = [], []
    with open(file) as f:
        for line in f.readlines():
            line = json.loads(line)
            qus = line['question'].strip()
            question.append(question_clean(qus))
            entity1, relation, entity2 = line['answer'].split('|||')
            entity1 = re.sub(r'（.*?（.*?）.*?）', '', entity1)
            entity1 = re.sub(r'（.*?）', '', entity1)
            entity1 = re.sub(r'\(.*?\)', '', entity1).strip()
            triplet.append(entity1 + ' ' + relation + ' ' + '[MASK]')
    return pd.DataFrame(data={'seq1': triplet, 'seq2': question})


def validate(tokenizer, model, loader, max_length_target):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    total = 0
    Rouge_1, Rouge_2, Rouge_l, Bleu = 0, 0, 0, 0
    device = torch.device(0)

    with torch.no_grad():
        for n, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['input_ids'].to(device, dtype=torch.long)
            generated_ids = model.generate(
                ids,
                decoder_start_token_id=tokenizer.cls_token_id,
                eos_token_id=tokenizer.sep_token_id,
                top_k=1,
                num_beams=1,  # num_returned_sequences=1,
                max_length=max_length_target).cpu().numpy()

            targets = [
                tokenizer.decode(item.cpu().numpy(),
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
                for item in y
            ]
            preds = [
                tokenizer.decode(generated_id,
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
                for generated_id in generated_ids
            ]

            assert len(preds) == len(
                targets), "len(x)=%d, len(y)=%d, " % (len(preds), len(targets))
            for pred, target in zip(preds, targets):
                total += 1
                if pred:
                    tem = get_rouge(pred, target)
                    Rouge_1 += tem[0]
                    Rouge_2 += tem[1]
                    Rouge_l += tem[2]
                    Bleu += get_bleu(pred, target)
                if total % 200 == 0:
                    print(pred)
                    print(target)
                    print('++++++++++++++++++++++')
        Rouge_1 /= total
        Rouge_2 /= total
        Rouge_l /= total
        Bleu /= total
        res = (Rouge_l + Bleu) / 2.0

        return res


if __name__ == '__main__':

    train_path = '/mnt/datasets/t5_triple2ques_data/train.json'
    test_path = '/mnt/datasets/t5_triple2ques_data/dev.json'
    model_name = 'T5'
    batch_size = 64
    max_length_source = 64
    max_length_target = 64
    model_path = '/mnt/T5_JIEBA/'

    T5Trainer = Trainer(
        env_type="pytorch",
        experiment_name="roberta_ner",
        batch_size=64,
        lr=2e-4,
        weight_decay=1e-3,
        epochs=10,
        load_dir=None,
        save_dir="kgclue",
        save_epoch=1,
        eval_interval=False,
        hostfile='./hostfile',
        seed=0,
    )

    train_data = train_data_process_kgclue_cgrm_seq2seq(train_path)
    ttest_data = train_data_process_kgclue_cgrm_seq2seq(test_path)
    tokenizer = T5BatchPegasusTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    # model = model.to(args.device)
    train_dataset = DataSet(train_data[["seq1",
                                        "seq2"]], tokenizer, max_length_source,
                            max_length_target, "seq1", "seq2")
    valid_dataset = DataSet(ttest_data[["seq1",
                                        "seq2"]], tokenizer, max_length_source,
                            max_length_target, "seq1", "seq2")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=4,
    )

    # validate(tokenizer, model, loader, max_length_target)
    model_input_keys = [
        "input_ids", "attention_mask", "decoder_input_ids", "labels"
    ]

    T5Trainer.train(model=model,
                    train_dataset=train_dataloader,
                    model_input_keys=model_input_keys)
