#  Custom prompt-verbalizer pair(PVP)

## 1. Define your own prompt-verbalizer patterns
We provide api for users to create their own function to construct prompt-verbalizer patterns. Here is an example below:
```python
class RtePVP(PVP):
    # Verbalizer convert original labels to more meaningful ones
    VERBALIZER = {"not_entailment": [" No"], "entailment": [" Yes"]}

    @staticmethod
    def available_patterns():
        return [0, 1, 2]

    @property
    def spell_length(self):
        return self.num_prompt_tokens + self.prefix_prompt

    def get_parts(self, example: InputExample):
        """
        Construct patterns with input texts and mask, "None" here stands for places to insert continuous prompt tokens
        """
        text_a = example.text_a
        text_b = example.text_b.rstrip(string.punctuation)
        if self.pattern_id == 0:
            parts_a, parts_b = [None, '"',
                                self.shortenable(text_b), '" ?'], [
                                    None, [self.mask], ',', None, ' "',
                                    self.shortenable(text_a), '"'
                                ]
        elif self.pattern_id == 1:
            parts_a, parts_b = [None, self.shortenable(text_b), '?'], [
                None, [self.mask], ',', None,
                self.shortenable(" " + text_a)
            ]
        elif self.pattern_id == 2:
            parts_a, parts_b = [
                None,
                self.shortenable(text_a), None, ' question:',
                self.shortenable(" " + text_b), ' True or False?', None,
                ' answer:', [self.mask]
            ], []
        else:
            raise NotImplementedError(self.pattern_id)
        parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
        return parts_a, parts_b

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 4:
            return [' true'] if label == 'entailment' else [' false']
        return RtePVP.VERBALIZER[label]
```

## 2. Pass the user-defined class to the collate function
```python
collate_fn = ConstructSuperglueStrategy(cl_args,
                                        tokenizer,
                                        task_name=task_name,
                                        custom_pvp=RtePVP)
```
