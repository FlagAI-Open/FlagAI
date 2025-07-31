import numpy as np
import pandas as pd
import os

task_name = ['CoLA', 'SST-2', 'MRPC', 'STS-B', 'QQP', 'MNLI', 'QNLI', 'RTE']
config_name = ['N-none', 'N-adapter', 'N-bias', 'N-bias-adapter', 'N-prompt', 'N-prompt-adapter', 'N-prompt-bias', 'N-prompt-bias-adapter',
'Y-none', 'Y-adapter', 'Y-bias', 'Y-bias-adapter', 'Y-prompt', 'Y-prompt-adapter', 'Y-prompt-bias', 'Y-prompt-bias-adapter']

m = []
s = []
for task in task_name:
    s.append([])
    m.append([])
    for config in config_name:
        f = open(os.path.join('./result', task+'-fulldata', "{}-{}.out".format(task, config)), 'r')
        content = f.readlines()
        d = eval(content[-1])
        m[-1].append(d['mean'])
        s[-1].append(d['std'])

x = np.array(m)
y = np.array(s)
m.append([])
s.append([])
for i in range(len(config_name)):
    m[-1].append(np.mean(x[:, i]))
    s[-1].append(np.mean(y[:, i]))

res = []
for i in range(len(task_name)+1):
    res.append([])
    for j in range(len(config_name)):
        res[-1].append("{}$_{}$".format(round(m[i][j]*100, 1), '{'+str(round(s[i][j]*100, 1))+'}'))

frame = pd.DataFrame(res)
frame.to_csv("full-data.csv")
