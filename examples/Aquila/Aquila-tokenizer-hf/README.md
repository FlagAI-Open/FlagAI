# 使用方法
```
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./Aquila-tokenizer-hf")
    input_text = 'There is something wrong, please query again'
    print(tokenizer.encode(input_text))
```

