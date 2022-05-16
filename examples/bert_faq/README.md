# Bert for FAQ task

## Dataset
Data can be downloaded from https://github.com/murufeng/ChineseNlpCorpus .

The task consists of two steps: 1. construct database 2. index by the faiss package.

faiss package is : https://github.com/facebookresearch/faiss

## Construct database

In "1_construct_data.py", you need to set some parameters, such as "faq_data_path", "answer_save_path", "embedding_save_path".

1. faq_data_path is the source data downloaded from https://github.com/murufeng/ChineseNlpCorpus .
2. answer_save_path is the path to save the quesion-answer dictionary, we can index the answer by the question quickly.
3. embeddings_save_path is the path to save the question-embeddings, we can index the similar questions by the faiss.

## Run
In "2_test_bert_faq.py", we provide the search function to search the similar question quickly.
```python
while True:
    question = input("please input a question")
    if question == "q":
        break
    question_embedding = predictor.predict_embedding(question, maxlen=maxlen)
    method.search(answer, question_embedding, k=10)
```
K =10 means we want to find the 10 results that are closest to asking the question.