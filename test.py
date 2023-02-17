# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# import unittest

# print('test syn')
# test_dir = './tests'
# test_report_path = './test_report'
# discover = unittest.defaultTestLoader.discover(test_dir, pattern='test_*.py')
# with open(test_report_path, "w") as report_file:
#     runner = unittest.TextTestRunner(stream=report_file, verbosity=2)
#     #runner=unittest.TextTestRunner()
#     runner.run(discover)
from dataclasses import dataclass, field
@dataclass(frozen=True, eq=True)
class AddedToken:
    """
    AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
    way it should behave.
    """

    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True

    def __getstate__(self):
        return self.__dict__
            
class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]    
    def __init__(self,  **kwargs):
        print(kwargs)
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        # self.verbose = verbose
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"special token {key} has to be either str or AddedToken but got: {type(value)}")

class Tokenizer(SpecialTokensMixin):
    def __init__(self,eos_token="</s>",
                        unk_token="<unk>",
                        pad_token="<pad>", 
                        additional_special_tokens=None,
                        **kwargs):
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            # extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            # sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )
tokenizer = Tokenizer()
import pdb;pdb.set_trace()