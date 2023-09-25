from flagai.auto_model.auto_loader import AutoLoader

state_dict = "./checkpoints/"
model_name = 'Aquila2Chat-hf'

state_dict = "/data2/20230907/"
model_name = 'iter_0205000_hf'

autoloader = AutoLoader("aquila2",
                    model_dir=state_dict,
                    model_name=model_name,
                    qlora_dir="/data2/yzd/FastChat/checkpoints_out/30bhf_save/checkpoint-4200",)
                    # qlora_dir='/data2/yzd/FlagAI/examples/Aquila2/checkpoints/qlora/aquila2chat-hf')
                    # lora_dir='/data2/yzd/FlagAI/examples/Aquila2/checkpoints/lora/aquila2chat-hf')
                    # )

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()
# 

test_data = [
    "请介绍下北京有哪些景点。",
    "唾面自干是什么意思",
    "'我'字有几个笔划",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))

