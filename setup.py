from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="flagai",
    version="0.10",
    description="A simple framework for state-of-the-art Big Models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Zhiyuan",
    author_email="zhiyuan@baai.ac.cn",
    url="https://github.com/Wudao/Sailing",
    packages=find_packages(exclude="tests"),  # same as name
    license="MIT",
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.7",
)
