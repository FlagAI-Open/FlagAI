import jsonlines
jsonl_file = "./datasets/convo_samples.jsonl"
conversations = []
with jsonlines.open(jsonl_file) as reader:
    for line in reader:
        conversations.append(line)
import pdb;pdb.set_trace()