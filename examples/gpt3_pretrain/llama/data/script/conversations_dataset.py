import jsonlines
import samples_from_conversation as conversation_lib

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    source["chat_desc"] = header
    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": conversation_lib.default_conversation.roles[0],  # human role
        "gpt": conversation_lib.default_conversation.roles[1],  # gpt role
    }
    if "instruction" in source and source["instruction"] is not None and len(source["instruction"]) > 0:
        source["instruction"] = (
            BEGIN_SIGNAL
            + "System: "
            + source["instruction"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += source["instruction"]
    for sentence in source["conversations"]:
        sentence_from = sentence["from"].lower()
        sentence["value"] = (
            BEGIN_SIGNAL
            + roles.get(sentence_from, unknown_role)
            + ": "
            + sentence["value"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    return conversation

header = "A chat between a curious user and an artificial intelligence assistant. " \
         "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

input_file = 'conversations/sample_data_v0.6_5w_0420.jsonl'
output_file = 'conversations/sample_data_v0.6_5w_0420_dataset.jsonl'
input_file = '/share/project/ldwang/sft/datasets/conversations/merge_chat_clean.jsonl'
output_file = '/share/project/ldwang/sft/datasets/conversations/merge_chat_clean_convo_dataset.jsonl'
input_file = 'conversation_demo.jsonl'
output_file = 'conversation_dataset_demo.jsonl'
fo = jsonlines.open(output_file, mode='w')
with jsonlines.open(input_file) as reader:
    for obj in reader:
        conversation = _add_speaker_and_signal(header, obj, get_conversation=True)
        #print(conversation)
        fo.write(obj)
