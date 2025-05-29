import os
import torch
import matplotlib.pyplot as plt
import seaborn
import torch
from torch import device

param_to_dir = {}
param_to_acc = {}

log_file = open('../log', 'r')
log = log_file.readlines()

log = log[-60:-12]

for item in log:
    training_arg = eval(item)
    training_params = training_arg['training_params']
    output_dir = training_arg['output_dir']
    eval_acc = training_arg['sst-2_dev_eval_acc']
    tag = training_arg['tag'][-1]
    lr = training_arg['learning_rate']
    param_str = training_params[0] + '-' + training_params[1] + '-' + training_params[2] + '-' + tag + '-' + str(lr)
    param_to_dir[param_str] = output_dir
    param_to_acc[param_str] = eval_acc

label_to_name = ['adapter', 'bias', 'prompt']
label_to_color = ['r', 'c', 'b']

for i in range(3):
    for j in range(3):
        for k in range(3):
            if i == j or i == k or j == k: 
                continue
            a = label_to_name[i]
            b = label_to_name[j]
            c = label_to_name[k]
            for n in ['Y', 'N']:
                best_acc = 0
                best_lr = None
                for lr in ['0.01', '0.001', '0.0001', '1e-05']:
                    p = a+'-'+b+'-'+c+'-'+n+'-'+lr
                    if param_to_acc[p] > best_acc:
                        best_acc = param_to_acc[p]
                        best_lr = lr
                p = a+'-'+b+'-'+c+'-'+n+'-'+best_lr
                path = os.path.join('../', param_to_dir[p])

                log_a = torch.load(os.path.join(path, 'log_history_'+a+'.bin'))
                log_b = torch.load(os.path.join(path, 'log_history_'+b+'.bin'))
                log_c = torch.load(os.path.join(path, 'log_history_'+c+'.bin'))
                print(best_acc, best_lr)
                train_loss = []
                for it, item in enumerate(log_a):
                    if 'eval_acc' in item and 'eval_acc' not in log_a[it-1]:
                        train_loss.append(item['eval_acc'])
                for it, item in enumerate(log_b):
                    if 'eval_acc' in item and 'eval_acc' not in log_b[it-1]:
                        train_loss.append(item['eval_acc'])
                for it, item in enumerate(log_c):
                    if 'eval_acc' in item and 'eval_acc' not in log_c[it-1]:
                        train_loss.append(item['eval_acc'])
                
                print(a, b, c, n, train_loss[89])
                assert(len(train_loss) == 90)
                fig, ax = plt.subplots()
                ax.plot(range(200, 6200, 200), train_loss[:30], color=label_to_color[i], linewidth=2.0, label=label_to_name[i])
                ax.plot(range(6000, 12200, 200), train_loss[29:60], color=label_to_color[j], linewidth=2.0, label=label_to_name[j])
                ax.plot(range(12000, 18200, 200), train_loss[59:], color=label_to_color[k], linewidth=2.0, label=label_to_name[k])
                if a == 'prompt':
                    ax.set_ylim(0.5, 1.0)
                else:
                    ax.set_ylim(0.85, 1.0)
                ax.set_xlim(0)
                ax.legend()
                if n == 'Y':
                    plt.title("Evaluating Acc for "+a+'-'+b+'-'+c+" method with template")
                else:
                    plt.title("Evaluating Acc for "+a+'-'+b+'-'+c+" method without template")
                # path = os.path.join('../', 'result', 'SST-2-full-mul-'+a+'-'+b+'-'+c+'-'+n+'13')
                path = os.path.join('../', 'figure')
                if not os.path.isdir(path):
                    os.mkdir(path)
                plt.savefig(os.path.join(path, a+'-'+b+'-'+c+'-'+n+'.png'))
                


