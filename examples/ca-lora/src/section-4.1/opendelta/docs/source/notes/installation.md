
(installation)=
# Installation


The lasted version of OpenDelta is tested on on [Python 3.8](https://www.python.org/) and [Pytorch 1.12](<https://pytorch.org/>). Other versions are likely to be supported as well.


## install the lastest version
```bash
pip install git+https://github.com/thunlp/OpenDelta.git
```

## install the lastest pip version (more stable)
```bash
pip install opendelta
```

## build from source
```bash
git clone git@github.com:thunlp/OpenDelta.git
cd OpenDelta
```
then 
```
python setup.py install
```
or if you want to do some modifications on the code for your research:
```
python setup.py develop
```