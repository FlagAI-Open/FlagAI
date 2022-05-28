import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "../state_dict/"

# Note "./checkpoints_seq2seq/{}/mp_rank_00_model_states.pt", {} is a directory in the checkpoints_seq2seq.
model_save_path = "./checkpoints_seq2seq/7079/mp_rank_00_model_states.pt"
maxlen = 512
auto_loader = AutoLoader(
    "seq2seq",
    model_name="bert-base-uncased",
    model_dir=model_dir,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
predictor = Predictor(model, tokenizer)
model.load_state_dict(
    torch.load(model_save_path, map_location=device)["module"])
model.to(device)
model.eval()
test_data = [
    "Four minutes after the red card, Emerson Royal nodded a corner into the path of the unmarked Kane at the far post, who nudged the ball in for his 12th goal in 17 North London derby appearances. Arteta's misery was compounded two minutes after half-time when Kane held the ball up in front of goal and teed up Son to smash a shot beyond a crowd of defenders to make it 3-0.The goal moved the South Korea talisman a goal behind Premier League top scorer Mohamed Salah on 21 for the season, and he looked perturbed when he was hauled off with 18 minutes remaining, receiving words of consolation from Pierre-Emile Hojbjerg.Once his frustrations have eased, Son and Spurs will look ahead to two final games in which they only need a point more than Arsenal to finish fourth.",
]
for text in test_data:
    print(
        predictor.predict_generate_beamsearch(text,
                                              input_max_length=500,
                                              out_max_length=50,
                                              beam_size=3))
    print(
        predictor.predict_generate_randomsample(text,
                                                input_max_length=500,
                                                out_max_length=50,
                                                repetition_penalty=1.5,
                                                top_k=40,
                                                top_p=1.0,
                                                temperature=1.2))
