from flagai.model.predictor.predictor import Predictor
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.llama import setup_model_parallel

# torchrun --nproc_per_node 1 llama_7b_generate.py

local_rank, world_size = setup_model_parallel()

loader = AutoLoader(task_name="lm",
                    model_name="llama-7b-en",
                    )
model = loader.get_model()
tokenizer = loader.get_tokenizer()
predictor = Predictor(model, tokenizer)

prompts = ["The capital of Germany is the city of", 
           "Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]
for text in prompts:
    result = predictor.predict_generate_randomsample(text, 
                                                  out_max_length=256, 
                                                  temperature=1.0, 
                                                  top_p=0.9)
    if local_rank == 0:
        print(result)
        print("\n==================================\n")

