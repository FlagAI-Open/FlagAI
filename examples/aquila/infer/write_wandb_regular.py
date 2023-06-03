import sys
import wandb

target=sys.argv[1]
wandb.init(project="Aquila-7b-24n8g",name="Aquila-7b-24n8g-reload74-knowledge",dir="wandb",resume=True,id="s5gl3ctb")
with open("code_loss.log",'r') as f:
    for line in f.readlines():
        res = line.strip("\n").split(" ")
        if len(res)==2:
            continue
        ckpt, loss, acc = res
        if int(ckpt)==int(target):
            wandb.log({"code_loss":float(loss)}, step=int(ckpt))
            wandb.log({"code_acc":float(acc)}, step=int(ckpt))

with open("all_loss.log",'r') as f:
    for line in f.readlines():
        res = line.strip("\n").split(" ")
        if len(res)==2:
            continue
        ckpt, loss, acc = res
        if int(ckpt)==int(target):
            wandb.log({"all_loss":float(loss)}, step=int(ckpt))
            wandb.log({"all_acc":float(acc)}, step=int(ckpt))

with open("mmlu_loss.log",'r') as f:
    for line in f.readlines():
        res = line.strip("\n").split(" ")
        print(len(res))
        if len(res)==2:
            continue
        ckpt, loss, acc = res
        if int(ckpt)==int(target):
            wandb.log({"mmlu_loss":float(loss)}, step=int(ckpt))
            wandb.log({"mmlu_acc":float(acc)}, step=int(ckpt))

with open("sft_loss.log",'r') as f:
    for line in f.readlines():
        res = line.strip("\n").split(" ")
        if len(res)==2:
            continue
        ckpt, loss, acc = res
        if int(ckpt)==int(target):
            wandb.log({"sft_loss":float(loss)}, step=int(ckpt))
            wandb.log({"sft_acc":float(acc)}, step=int(ckpt))

wandb.finish()
