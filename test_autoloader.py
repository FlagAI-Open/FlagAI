from flagai.auto_model.auto_loader import AutoLoader

auto_loader = AutoLoader(
    "seq2seq",
    model_name="RoBERTa-bas",
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()