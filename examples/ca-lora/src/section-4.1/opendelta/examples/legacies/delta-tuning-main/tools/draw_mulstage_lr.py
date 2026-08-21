import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch import device

mpl.rcParams['xtick.minor.size'] = 0
mpl.rcParams['xtick.minor.width'] = 0
mpl.rc('font', size=18)
xlabels = range(0, 18200, 2000)
xlabels_txt = [str(i) for i in xlabels]
param_to_dir = {}

log_file = open('../log', 'r')
log = log_file.readlines()

log = log[2051:2063]
print(len(log))

for item in log:
    training_arg = eval(item)
    training_params = training_arg['training_params']
    output_dir = training_arg['output_dir']
    tag = training_arg['tag'][-1]
    param_str = training_params[0] + '-' + training_params[1] + '-' + training_params[2] + '-' + tag
    param_to_dir[param_str] = output_dir

label_to_name = ['adapter', 'bias', 'prompt']
label2n = ['AP', 'BF', 'PT']
label_to_color = ['#fe5803', '#a000c6', '#0044fb']

for i in range(3):
    for j in range(3):
        for k in range(3):
            if i == j or i == k or j == k: 
                continue
            a = label_to_name[i]
            b = label_to_name[j]
            c = label_to_name[k]
            for n in ['Y', 'N']:
                p = a+'-'+b+'-'+c+'-'+n
                path = os.path.join('../', param_to_dir[p])

                log_a = torch.load(os.path.join(path, 'log_history_'+a+'.bin'))
                log_b = torch.load(os.path.join(path, 'log_history_'+b+'.bin'))
                log_c = torch.load(os.path.join(path, 'log_history_'+c+'.bin'))
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
                
                print(a, b, c, n, max(train_loss))
                assert(len(train_loss) == 90)
                fig, ax = plt.subplots()
                ax.plot(range(200, 6200, 200), train_loss[:30], '-', color=label_to_color[i], label=label2n[i])
                ax.plot(range(6000, 12200, 200), train_loss[29:60], '-',  color=label_to_color[j], label=label2n[j])
                ax.plot(range(12000, 18200, 200), train_loss[59:], '-',  color=label_to_color[k], label=label2n[k])
                ax.legend()
                ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                if n == 'Y':
                    ax.title.set_text('{}+{}+{}, with template'.format(label2n[i], label2n[j], label2n[k]))
                else:
                    ax.title.set_text('{}+{}+{}, without template'.format(label2n[i], label2n[j], label2n[k]))
                ax.set_ylabel('ACC')
                ax.set_xlabel('steps')
                ax.set_xticks(xlabels)
                fontdict = {'fontsize': 12, 
                'fontweight': mpl.rcParams['axes.titleweight'] }
                ax.set_xticklabels(xlabels_txt, fontdict=fontdict)
                for x0 in xlabels:
                    if x0 == 0:
                        continue
                    index = int(x0/200)-1
                    if index < 30:
                        ax.plot([x0], [train_loss[index]], '-d', color=label_to_color[i], label=label2n[i])
                    elif index < 60:
                        ax.plot([x0], [train_loss[index]], '-d', color=label_to_color[j], label=label2n[j])
                    else:
                        ax.plot([x0], [train_loss[index]], '-d', color=label_to_color[k], label=label2n[k])
                ax.set_ylim([0.92, 0.97])
                ylabel = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
                # ylabel = np.linspace(ymin, ymax, 5)
                plt.yticks(ylabel)
                ax.set_yticklabels(ylabel)
                for y0 in ylabel:
                    ax.axhline(y=y0, color='gray', linestyle='--', linewidth=0.5)
                # path = os.path.join('../', 'result', 'SST-2-full-mul-'+a+'-'+b+'-'+c+'-'+n+'13')
                l = ax.figure.subplotpars.left
                r = ax.figure.subplotpars.right
                t = ax.figure.subplotpars.top
                bt = ax.figure.subplotpars.bottom
                figw = 6.0/(r-l)
                figh = 4.0/(t-bt)
                ax.figure.set_size_inches(figw, figh)
                path = os.path.join('../', 'figure-lr')
                if not os.path.isdir(path):
                    os.mkdir(path)
                plt.savefig(os.path.join(path, a+'-'+b+'-'+c+'-'+n+'.pdf'), bbox_inches='tight', pad_inches=0.1)
                


