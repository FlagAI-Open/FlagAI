# from openprompt.prompts import ManualTemplate

class BasePrompt(object):
    def __init__(self, template_id=0, verbalizer_id=0, generation=True):
        self.template = self.textual_templates[template_id]
        if generation:
            self.verbalizer = self.generation_verbalizers[verbalizer_id]
        else:
            self.verbalizer = self.mlmhead_verbalizers[verbalizer_id]



    def __call__(self, example):

        def eval_syntax(syntaxlist, example):
            composed = []
            for x in syntaxlist:
                if x.startswith("[_eval_]"):
                    t = eval(x[len("[_eval_]"):])
                else:
                    t = x
                composed.append(t)
            return composed
        src_texts = eval_syntax(self.template,example)

        tgt_texts = self.verbalizer[str(example['label'])]
        if isinstance(tgt_texts, list):
            tgt_texts = eval_syntax(tgt_texts, example)
        else:
            tgt_texts = [tgt_texts]
        return src_texts,  tgt_texts





class MRPCPrompt(BasePrompt):
    generation_verbalizers = [
        {
            "0": "different",
            "1": "same"
        },
        {
            "0": "not_equivalent",
            "1": "equivalent"
        }
    ]
    mlmhead_verbalizers = {
        "0": "different",
        "1": "same"
    }
    textual_templates = [
        ["sentence1:", """[_eval_]example['sentence1']""",
        "sentence2:", """[_eval_]example["sentence2"]""", "Meanings different of same? Answer: " ]
    ]

class BoolQPrompt(BasePrompt):
    generation_verbalizers = [
        {
            "0": "different",
            "1": "same"
        },
        {
            "0": "not_equivalent",
            "1": "equivalent"
        }
    ]
    mlmhead_verbalizers = {
        "0": "different",
        "1": "same"
    }
    textual_templates = [
        ["sentence1:", """[_eval_]example['sentence1']""",
        "sentence2:", """[_eval_]example["sentence2"]""", "Meanings different of same? Answer: " ]
    ]

class BoolQPrompt(BasePrompt):
    generation_verbalizers = [
        {
            "0": "no",
            "1": "yes"
        },
    ]
    mlmhead_verbalizers = {
        "0": "no",
        "1": "yes"
    }
    textual_templates = [
        ["hypothesis:", """[_eval_]example['hypothesis']""",
        "premise:", """[_eval_]example["premise"]""", "The answer was " ]
    ]

class COLAPrompt(BasePrompt):
    generation_verbalizers = [
        {
            "0": "No",
            "1": "Yes"
        },
    ]
    mlmhead_verbalizers = {
        "0": "No",
        "1": "Yes"
    }
    textual_templates = [
        ["sentence:", """[_eval_]example['sentence']""",
        "grammar correct? " ]
    ]


class RTEPrompt(BasePrompt):
    generation_verbalizers = [
        {
            "0": "yes",
            "1": "no"
        },
    ]
    mlmhead_verbalizers = {
        "0": "yes",
        "1": "no"
    }
    textual_templates = [
        ["sentence1:", """[_eval_]example['premise']""", "sentence2:",
        """[_eval_]example['hypothesis']""",
        "The answer was "]
    ]

class CBPrompt(BasePrompt):
    generation_verbalizers = [{
        "0": "yes",
        "1": "no",
        "2": "maybe"
    },
    ]
    mlmhead_verbalizers = [{
        "0": "yes",
        "1": "no",
        "2": "maybe"
    }]
    textual_templates = [
        ["hypothesis:", """[_eval_]example['hypothesis']""", "premise:",
        """[_eval_]example['premise']""",
        "The answer was " ]
    ]

PromptCollections = {
    "mrpc": MRPCPrompt,
    "cola": COLAPrompt,
    "rte": RTEPrompt,
    "superglue-boolq": BoolQPrompt,
    "cb": CBPrompt,
}


