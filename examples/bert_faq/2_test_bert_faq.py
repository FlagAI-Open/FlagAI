import faiss
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

faq_data_path = "./data/financezhidao_filter.csv"
answer_save_path = "./data/finance_fqa.json"
embeddings_save_path = "./data/finance_embeddings.json"

maxlen = 256
d = 768
nlist = 5
nprobe = 10

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


class Search:

    def __init__(self, training_vectors, d, nlist=10, nprobe=1):
        quantizer = faiss.IndexFlatIP(d)  # the other index
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not self.index.is_trained
        self.index.train(training_vectors)
        assert self.index.is_trained
        self.index.nprobe = nprobe  # default nprobe is 1, try a few more
        self.index.add(training_vectors)  # add may be a bit slower as well
        self.d = d

    def search(self, answer, query, k=10):
        query = query.numpy().reshape(-1, self.d)
        D, I = self.index.search(query, k)  # actual search
        result_question = []
        all_question = list(answer.keys())
        for s, i in zip(D[0], I[0]):
            if i != -1:
                result_question.append([all_question[i], s])

        print(result_question)
        best_quesiton = result_question[0][0]
        print(f"answer is {answer[best_quesiton]}")

        return result_question


if __name__ == '__main__':
    # load data from 1_construct_data.py
    answer = torch.load(answer_save_path)
    embeddings = torch.load(embeddings_save_path)

    method = Search(training_vectors=embeddings,
                    d=d,
                    nlist=nlist,
                    nprobe=nprobe)

    while True:
        question = input("please input a question")
        if question == "q":
            break
        question_embedding = predictor.predict_embedding(question,
                                                         maxlen=maxlen)
        method.search(answer, question_embedding, k=10)
