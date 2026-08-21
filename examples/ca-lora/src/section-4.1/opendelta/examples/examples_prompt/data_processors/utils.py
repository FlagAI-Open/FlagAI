import numpy as np
import re

def round_stsb_target(label):
    """STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    Args:
      label: original label.
    Returns:
      A preprocessed label.
    """
    return np.round((label * 5) / 5, decimals=1)


def pad_punctuation(text):
   """Re-implementation of _pad_punctuation in t5. This function adds spaces
   around punctuation. While this pads punctuation as expected, it has the
   unexpected effected of padding certain unicode characters with accents, with
   spaces as well. For instance: "François" becomes "Fran ç ois"""
   # Pad everything except for: underscores (_), whitespace (\s),
   # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
   text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', text)
   # Collapse consecutive whitespace into one space.
   text = re.sub(r'\s+', ' ', text)
   return text