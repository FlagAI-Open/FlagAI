import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    instruction: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            if self.instruction is not None and len(self.instruction) > 0:
                ret += self.roles[2] + ": " + self.instruction + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            if self.instruction is not None and len(self.instruction) > 0:
                ret += self.roles[2] + ": " + self.instruction + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            instruction=self.instruction,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "instruction": self.instruction,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    instruction="",
    roles=("Human", "Assistant", "System"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    instruction="",
    roles=("Human", "Assistant", "System"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_bair_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    instruction="",
    roles=("USER", "GPT", "System"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


default_conversation = conv_v1_2
conv_templates = {
    "v1": conv_v1_2,
    "bair_v1": conv_bair_v1,
}


def covert_prompt_to_input_ids_with_history(text, history, tokenizer, max_token):
    conv = default_conversation.copy()

    conv.append_message(conv.roles[1], None)
    conv.append_message(conv.roles[0], text)

    example = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']

    while(len(history) > 0 and (len(example) < max_token)):
        tmp = history.pop()
        if tmp[0] == 'ASSISTANT':
            conv.append_message(conv.roles[1], tmp[1])
        else:
            conv.append_message(conv.roles[0], tmp[1])
        example = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']

    if len(example) >= max_token:
        conv.messages.pop()
    conv.messages = conv.messages[::-1]
    print('model in:', conv.get_prompt())
    example = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
    example = example[1:-1]

    return example

if __name__ == "__main__":
    print(default_conversation.get_prompt())

