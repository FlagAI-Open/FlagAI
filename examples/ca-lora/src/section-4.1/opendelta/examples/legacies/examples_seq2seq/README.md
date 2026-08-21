# Appling OpenDelta to GLUE/SuperGLUE tasks using Seq2Seq Paradigm


## install the repo
```bash
cd ../
python setup_seq2seq.py develop
```
This will add `examples_seq2seq` to the environment path of the python lib.

## Generating the json configuration file

```
python config_gen.py --job $job_name

```
The available job configuration (e.g., `--job lora_t5-base`) can be seen from `config_gen.py`. You can also
create your only configuration.


## Run the code

```
python run_seq2seq.py configs/$job_name/$dataset.json
```

## Possible Errors

1. 
```
ValueError: You must login to the Hugging Face hub on this computer by typing `transformers-cli login` and entering your credentials to use `use_auth_token=Tr
ue`. Alternatively, you can pass your own token as the `use_auth_token` argument.
```
- Solution 1: Please register an account on [HuggingFace](https://huggingface.co/) 
Then run transformers-cli login on your command line to enter the username and password.

- Solution 2: Disable push_to_hub by modifying in the config.json : "push_to_hub": False

2. 
```
OSError: Looks like you do not have git-lfs installed, please install. You can install from https://git-lfs.github.com/. Then run `git lfs install` (you only have to do this once).
```

- Solution 1:
```
wget -P ~ https://github.com/git-lfs/git-lfs/releases/download/v3.0.2/git-lfs-linux-amd64-v3.0.2.tar.gz
cd ~
tar -xvzf git-lfs-linux-amd64-v3.0.2.tar.gz
export PATH=~:$PATH
git-lfs install
```

- Solution 2: Disable push_to_hub by modifying in the config.json : "push_to_hub": False


3. dataset connection error

Solution 1: open a python console, running the error command again, may not be useful

Solution 2: download the dataset by yourself on a internect connected machine, saved to disk and transfer to your server, at last load_from_disk.


## Link to the original training scripts
This example repo is based on the [compacter training scripts](https://github.com/rabeehk/compacter), with compacter-related lines removed. Thanks to the authors of the original repo. In addition, in private correspondence with the authors, they shared the codes to create the json configs. Thanks again for their efforts. 
