# construct data
# data from https://github.com/murufeng/ChineseNlpCorpus
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import pandas as pd
import numpy as np
from tqdm import tqdm
import collections
import faiss

faq_data_path = "./data/financezhidao_filter.csv"
answer_save_path = "./data/finance_fqa.json"
embeddings_save_path = "./data/finance_embeddings.json"

maxlen = 256
task_name = "embedding"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auto_loader = AutoLoader(
    task_name=task_name,
    model_name="RoBERTa-base-ch",
    load_pretrain_params=True,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
model.to(device)
predictor = Predictor(model, tokenizer=tokenizer)


def resave_data():
    answer = collections.OrderedDict()
    embeddings = []
    df = pd.read_csv(faq_data_path)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if type(row[0]) == str:
            if row[0] not in answer:
                answer[row[0]] = row[2]
                embeddings.append(
                    predictor.predict_embedding(row[0], maxlen=maxlen).numpy())

    embeddings = np.array(embeddings)
    torch.save(answer, answer_save_path)
    torch.save(embeddings, embeddings_save_path)

    print(
        f"data is saved successfully: {answer_save_path}, {embeddings_save_path}"
    )


if __name__ == '__main__':

    resave_data()
