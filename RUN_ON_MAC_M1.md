# Steps to get FlagAI run on Mac M1
I tried to get the code adapted to Mac M1 on my Mac Studio M1 Ultra. And Finally got it to work.

## Results of the example.

```
python3.10 generate_chat.py
```

![image](https://github.com/davideuler/FlagAI/assets/377983/167433fb-8261-4c3a-aec0-6c2f15828b60)


<img width="1466" alt="image" src="https://github.com/davideuler/FlagAI/assets/377983/cbf57d91-2458-4810-b604-a3315e243572">

## Install FlagAI from source code.

```
python3.10 setup.py install

```

## Update the generate.py

```
vim ~/workspace/FlagAI/examples/Aquila/Aquila-chat/generate_chat.py
```

Change the fp16 parameter to False.
```
fp16=False
```

Commen the line of model operation for CUDA:
```
#model.half()
#model.cuda()
```

Add new line:
```
model.to("cpu")
```

That's it for the generate.py.

## Then need to fix some code from install source. 

vim ~/.pyenv/versions/3.10.10/lib/python3.10/site-packages/flagai-1.7.2-py3.10.egg/flagai/model/predictor/aquila.py

Change the cuda() to cpu() as the following code:

```
        #tokens = torch.full((bsz, total_len), 0).cuda().long()
        tokens = torch.full((bsz, total_len), 0).cpu().long()
```

## Finally run the reference code

It should works. If got any errors, try to adapt the code for CPU.

```
python3.10 generate_chat.py
```

