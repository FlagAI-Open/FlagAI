# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from setuptools import find_packages, setup

setup(
    name="flagai",
    version="v1.6.3",
    description=
    "FlagAI aims to help researchers and developers to freely train and test large-scale models for NLP/CV/VL tasks.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="FlagAI-Open",
    author_email="open@baai.ac.cn",
    url="https://github.com/FlagAI-Open/FlagAI",
    packages=find_packages(exclude="tests"),  # same as name
    license="Apache 2.0",
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        'nltk==3.8.1',
        'sentencepiece==0.1.99',
        'boto3==1.26.144',
        'pandas==2.0.2',
        'jieba==0.42.1',
        'scikit-learn==1.2.2',
        'tensorboard==2.13.0',
        'transformers==4.29.2',
        'datasets==2.0.0',
        'setuptools==66.0.0',
        'protobuf==4.23.2',
        'ftfy == 6.1.1',
        'pillow == 9.5.0',
        'einops == 0.6.1',
        'diffusers == 0.16.1',
        'pytorch-lightning == 2.0.2',
        'taming-transformers-rom1504 == 0.0.6',
        'rouge-score == 0.1.2',
        'sacrebleu == 2.3.1',
    ])
