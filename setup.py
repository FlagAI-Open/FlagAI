from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="flagai",
    version="v1.0.0-beta3",
    description="FlagAI aims to help researchers and developers to freely train and test large-scale models for NLP tasks.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="BAAI Open",
    author_email="liuguang@baai.ac.cn",
    url="https://github.com/BAAI-Open/FlagAI",
    packages=find_packages(exclude="tests"),  # same as name
    license="Apache 2.0",
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.8",
)
