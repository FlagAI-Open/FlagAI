
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

if __name__ == '__main__':

    text = '''默写古诗:
    白日依山尽，黄河入海流。
    床前明月光，'''

    loader = AutoLoader(task_name="lm",
                        model_name="CPM-large-ch-generation",
                        model_dir="./state_dict/")

    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    predictor = Predictor(model=model,
                          tokenizer=tokenizer,
                          )

    out = predictor.predict_generate_randomsample(text,
                                                  top_p=0.9,
                                                  out_max_length=50)

    print(out)
