import jsonlines
import re
import os

## Usage:
### 1. replace using vim or sed command
#### %s/\[Round \d\]//g
#### %s/\\n问：/ #用户#/g
#### %s/\\n答：/ #ai助手#/g

### 2. convo format converting
input_file = f'input.jsonl'
output_file = f'input_convo.jsonl'

SEP = ' '
HUMAN = SEP + '#用户#'
BOT = SEP + '#ai助手#'

fo = jsonlines.open(output_file, mode='w')
with jsonlines.open(input_file) as reader:
    for idx, input_obj in enumerate(reader):
        line = input_obj['prompt'] + input_obj['response']
        obj = dict()
        obj['id'] = f"{os.path.basename(input_file)}_%d" % idx
        obj['conversations'] = []
        obj['instruction'] = ''
        obj['raw'] = line

        role_str = HUMAN
        role = 'gpt'

        prev_start_idx = -1
        prev_end_idx = -1
        start_idx = 0
        first_human = True
        while True:
            start_idx = line.find(role_str, start_idx)
            if start_idx == -1:
                conversation = dict()
                conversation["from"] = role
                conversation["value"] = line[prev_end_idx:]
                obj['conversations'].append(conversation)
                break
            end_idx = start_idx + len(role_str)
            if role_str == HUMAN:
                if first_human:
                    pass
                else:
                    conversation = dict()
                    conversation["from"] = role
                    conversation["value"] = line[prev_end_idx:start_idx]
                    obj['conversations'].append(conversation)
                first_human = False
                role_str = BOT
                role = 'human'
            else:
                conversation = dict()
                conversation["from"] = role
                conversation["value"] = line[prev_end_idx:start_idx]
                obj['conversations'].append(conversation)
                role_str = HUMAN
                role = 'gpt'

            prev_start_idx = start_idx
            prev_end_idx = end_idx
            start_idx = end_idx + len(role_str)

        fo.write(obj)

