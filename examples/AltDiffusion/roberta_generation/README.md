**文件结构**
---


# Introduction
采用seq2seq model，实现text generation task。
模型采用“RoBERTa-base”，实现框架采用FlagAI.

## dataset
dataset由input text和target text组成，数据集来自淘宝电商数据，input text 是关键词，如“红色,撞色,条纹,衬衫,立领”；target text是具体描述，如“对设计亮点立领，领口撞色绣线缝制，别致的造型鲜明，条纹元素是不过时的时髦，融入到衬衫中，赋予其时尚活力，衣袖精致的红色麻花编织纹，美观大方，装饰效果好，彰显穿衣品味。”。

数据集下载地址：https://www.dropbox.com/s/6i2jvzjlyic9y2m/data.zip?dl=0



## train & test
应用flagAI的内置函数Trainer.train()和Predictor函数完成模型的训练和测试过程。
