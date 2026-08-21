# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import json
import tqdm
import os
from flagai.data.dataset.block.lazy_loader import LazyLoader
# from lazy_loader import LazyLoader
from multiprocessing import Queue, Process

NUM_PROCESSES = 100


def punctuation_standardization(string: str):
    punctuation_dict = {
        "\u201c": "\"",
        "\u201d": "\"",
        "\u2019": "'",
        "\u2018": "'",
        "\u2013": "-"
    }
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string


class DataReader:
    # PATH = None
    assert_str = None
    reserve_punct = False
    split_row = True
    TASK_QUEUE_LIMIT = 10000000
    DONE_QUEUE_LIMIT = 10000000

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        raise NotImplementedError

    def print_info(self, info):
        pass

    def __init__(self,
                 writers,
                 tokenizer=None,
                 tokenize=False,
                 path=None,
                 **kwargs):
        # print(self.PATH)
        if path is not None:
            self.PATH = path
        assert os.path.exists(self.PATH), self.assert_str
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        self.writers = writers

    def process(self, num_processes):
        if os.path.isdir(self.PATH):
            paths = [
                os.path.join(top, name) for top, _, names in os.walk(self.PATH)
                for name in names
            ]
            # paths = [entry.path for entry in os.scandir(self.PATH) if
            #          not entry.is_dir() and not entry.name.endswith("bz2")]
        else:
            paths = [self.PATH]
        task_queue, done_queue, info_queue = Queue(
            maxsize=self.TASK_QUEUE_LIMIT), Queue(
                maxsize=self.DONE_QUEUE_LIMIT), Queue()
        processes = []
        for i in range(num_processes):
            process = Process(target=self.tokenize_worker,
                              args=(task_queue, done_queue, info_queue,
                                    self.tokenizer, self.tokenize))
            process.start()
            processes.append(process)

        def read_input_to_queue():
            for path in paths:
                # if torch.cuda.is_available():print_rank_0(f"Start reading {path}")
                with open(path) as file:
                    if self.split_row:
                        for row in file:
                            task_queue.put(row)
                    else:
                        items = json.load(file)
                        for item in items["RECORDS"]:
                            task_queue.put(item)
            # if torch.cuda.is_available():print_rank_0("Read input complete")
            for i in range(len(processes)):
                task_queue.put('STOP')

        process = Process(target=read_input_to_queue)
        process.start()
        count = len(processes)
        progress_bar = tqdm.tqdm()
        while True:
            data = done_queue.get()
            if data == 'COMPLETE':
                count -= 1
                if count == 0:
                    break
            else:
                self.write_result(data, self.writers)
                progress_bar.update()
        progress_bar.close()
        self.print_info(info_queue)

    @staticmethod
    def write_result(data, writers):
        raise NotImplementedError

    @staticmethod
    def get_token_count(contents):
        return sum(map(len, contents))

    @classmethod
    def process_sample(cls, text, tokenizer, tokenize):
        if isinstance(text, str) and tokenize:
            if not cls.reserve_punct:
                text = punctuation_standardization(text)
            text = tokenizer.EncodeAsIds(text) if text else []
        return text

    @staticmethod
    def trim_field(content, max_length):
        if len(content) > max_length:
            content = content[:max_length]
            content += "......"
        return content

    def process_line(self, data, tokenizer, tokenize):
        raise NotImplementedError


class PromptReader(DataReader):
    is_json = True

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        for row in iter(input.get, 'STOP'):
            if row:
                if self.is_json:
                    row = row.rstrip()
                    row = json.loads(row)
                prompts, texts = self.process_line(row, tokenizer, tokenize)
                for prompt, text in zip(prompts, texts):
                    output.put((prompt, text))
        output.put("COMPLETE")

    @staticmethod
    def write_result(data, writers):
        prompt, text = data
        writers['prompt'].write(prompt)
        writers['text'].write(text)


from torch.utils.data import Dataset


class PromptDataset(Dataset):

    def __init__(self,
                 prompt_loader,
                 text_loader,
                 tokenizer=None,
                 to_tokenize=False,
                 **kwargs):
        self.prompts = prompt_loader
        self.texts = text_loader
        self.tokenizer = tokenizer
        self.to_tokenize = to_tokenize
        if isinstance(self.prompts, LazyLoader) and isinstance(
                self.texts, LazyLoader):
            self.prompt_lens = self.prompts.lens
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        return self.prompt_lens[idx] + self.text_lens[idx]

    def __getitem__(self, index):
        prompt = self.prompts[index]
        text = self.texts[index]
        if self.to_tokenize:
            prompt = self.tokenizer.EncodeAsIds(prompt)
            text = self.tokenizer.EncodeAsIds(text)
        return {
            "tokens": prompt + text,
            "loss_masks": [0] * len(prompt) + [1] * len(text)
        }

    def __len__(self):
        return len(self.prompts)


class WuDaoCorpus(PromptReader):
    is_json = False
    reserve_punct = True
    split_row = False

    def process_line(self, item, tokenizer, tokenize):
        prompts, texts = [], []
        text = ""
        title = item.get("title", None)
        content = item.get("content", None)
        if title:
            text += title.strip() + " "
        if content:
            text += content
        if len(text) > 100:
            prompt, text = self.process_sample("", tokenizer,
                                               tokenize), self.process_sample(
                                                   text, tokenizer, tokenize)
            prompts.append(prompt)
            texts.append(text)
        return prompts, texts
