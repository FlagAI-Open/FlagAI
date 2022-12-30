# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from time import sleep

import requests
import json
import os

import torch
from tqdm.auto import tqdm

try:
    import bmtrain as bmt
except:
    pass


def download_from_url(url, size=0, rank=0, to_path=None, file_pname=None):
    """
    url: file url
    file_pname: file save name
    chunk_size: chunk size
    resume_download: download from last chunk
    """
    try:
        requests.get(url, stream=True, verify=True)
    except Exception:
        raise ValueError('please check the download file names')
    total_size = size
    if to_path is None:
        to_path = './checkpoints/'
    if file_pname is None:
        file_path = os.path.join(to_path, url.split('/')[-1])
    else:
        file_path = os.path.join(to_path, file_pname)


    if (os.environ["ENV_TYPE"] != 'bmtrain' and (not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0)) or (os.environ["ENV_TYPE"] == 'bmtrain' and bmt.rank() == 0):
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        if os.path.exists(file_path):
            resume_size = os.path.getsize(file_path)
        else:
            resume_size = 0
        if resume_size == total_size:
            return
        headers = {'Range': 'bytes=%d-' % resume_size}
        res = requests.get(url, stream=True, verify=True, headers=headers)
        progress = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=total_size,
            initial=resume_size,
            desc="Downloading",
        )
        while 1:
            with open(file_path, "ab") as f:
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
                        f.flush()

            resume_size = os.path.getsize(file_path)
            if resume_size >= total_size:
                print('-----model dowloaded in ', os.getcwd() + to_path[1:])
                break
            else:
                headers = {'Range': 'bytes=%d-' % resume_size}
                res = requests.get(url,
                                    stream=True,
                                    verify=True,
                                    headers=headers)
    else:
        while not os.path.exists(
                file_path) or total_size != os.path.getsize(file_path):
            sleep(1)


def _get_config_path(download_path, config_name, model_id, rank=0):
    dic_download = {'model_id': model_id, 'checkpoint_name': config_name}
    config_requests = requests.post(
        'https://model.baai.ac.cn/api/downloadCode', json=dic_download)
    config_requests.encoding = "utf-8"
    if json.loads(config_requests.text)['code'] == '40002':
        file_list = json.loads(config_requests.text)['files']
        print('file {} not exist in {}'.format(config_name, file_list))
        return '40002'
    url = json.loads(config_requests.text)['url']
    size = json.loads(config_requests.text)['size']
    download_from_url(url,
                      size=size,
                      to_path=download_path,
                      file_pname=config_name,
                      rank=rank)

    return os.path.join(download_path, config_name)


def _get_vocab_path(download_path, vocab_name, model_id, rank=0):
    dic_download = {'model_id': model_id, 'checkpoint_name': vocab_name}
    vocab_requests = requests.post('https://model.baai.ac.cn/api/downloadCode',
                                   json=dic_download)

    vocab_requests.encoding = "utf-8"
    if json.loads(vocab_requests.text)['code'] == '40002':
        file_list = json.loads(vocab_requests.text)['files']
        print('file {} not exist in {}'.format(vocab_name, file_list))
        return '40002'
    url = json.loads(vocab_requests.text)['url']
    size = json.loads(vocab_requests.text)['size']

    download_from_url(url,
                      size=size,
                      to_path=download_path,
                      file_pname=vocab_name,
                      rank=rank)

    return os.path.join(download_path, vocab_name)


def _get_checkpoint_path(download_path, checkpoint_name, model_id, rank=0):
    dic_download = {'model_id': model_id, 'checkpoint_name': checkpoint_name}
    checkpoint_requests = requests.post(
        'https://model.baai.ac.cn/api/downloadCode', json=dic_download)
    checkpoint_requests.encoding = "utf-8"
    if json.loads(checkpoint_requests.text)['code'] == '40002':
        file_list = json.loads(checkpoint_requests.text)['files']
        print('file {} not exist in {}'.format(checkpoint_name, file_list))
        return '40002'
    url = json.loads(checkpoint_requests.text)['url']
    size = json.loads(checkpoint_requests.text)['size']
    download_from_url(url,
                      size=size,
                      to_path=download_path,
                      file_pname=checkpoint_name,
                      rank=rank)

    return os.path.join(download_path, checkpoint_name)


def _get_model_id(model_name):
    return requests.get('https://model.baai.ac.cn/api/searchModleByName', {
        'model_name': model_name
    }).text

def _get_model_files(model_name):
    return requests.get('https://model.baai.ac.cn/api/searchModelFileByName', {
        'model_name': model_name
    }).text

