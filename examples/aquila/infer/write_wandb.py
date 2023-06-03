import sys
import wandb

wandb.init(project="Aquila-7b-64n8g",name="Aquila-7b-24n8g-reinit-knowledge",dir="wandb",resume=True,id="s5gl3ctb")
with open("code_loss.log",'r') as f:
    line = f.readlines()[-1].strip("\n")
    ckpt, loss, acc = line.split(" ")
    wandb.log({"code_loss":float(loss)}, step=int(ckpt))
    wandb.log({"code_acc":float(acc)}, step=int(ckpt))

with open("all_loss.log",'r') as f:
    line = f.readlines()[-1].strip("\n")
    ckpt, loss, acc = line.split(" ")
    wandb.log({"all_loss":float(loss)}, step=int(ckpt))
    wandb.log({"all_acc":float(acc)}, step=int(ckpt))

with open("mmlu_loss.log",'r') as f:
    line = f.readlines()[-1].strip("\n")
    ckpt, loss, acc = line.split(" ")
    wandb.log({"mmlu_loss":float(loss)}, step=int(ckpt))
    wandb.log({"mmlu_acc":float(acc)}, step=int(ckpt))

with open("sft_loss.log",'r') as f:
    line = f.readlines()[-1].strip("\n")
    ckpt, loss, acc = line.split(" ")
    wandb.log({"sft_loss":float(loss)}, step=int(ckpt))
    wandb.log({"sft_acc":float(acc)}, step=int(ckpt))

wandb.finish()
