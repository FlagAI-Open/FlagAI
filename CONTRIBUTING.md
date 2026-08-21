# Contributing to FlagAI

We are happy to accept your contributions to make `FlagAI` better and more awesome! To avoid unnecessary work on either
side, please stick to the following process:

1. Check if there is already [an issue](https://github.com/FlagAI-Open/FlagAI/issues) for your concern.
2. If there is not, open a new one to start a discussion. We hate to close finished PRs!
3. If we decide your concern needs code changes, we would be happy to accept a pull request. Please consider the
commit guidelines below.

## Sign the CLA

Before you can contribute to FlagAI, you will need to sign the [Contributor License Agreement](CLA.md).

## Git Commit Guidelines

If there is already a ticket, use this number at the start of your commit message.
Use meaningful commit messages that described what you did.

**Example:** `GH-42: Added new type of embeddings: DocumentEmbedding.`
**Example:** `ISSUE#123: Fix typo in README.`


## Developing locally

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit
tests for examples of how to call methods. Nearly all classes and methods are documented, so finding your way around
the code should hopefully be easy.

### setup

You can either use [Pipenv](https://pipenv.readthedocs.io/) for this:

or create a python environment of your preference and run
```bash
python setup.py install
```

### tests
Install `pytest` for testing
```
pip install pytest
```
To run all basic tests execute:
```bash
pytest
```

### code formatting

To ensure a standardized code style we use the formatter [yapf](https://github.com/google/yapf).
You can automatically format the code via `yapf flagai/ -i` in the flair root folder.
