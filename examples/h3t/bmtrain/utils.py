import torch
import sys
from typing import Any, Dict, Iterable, Optional
from .global_var import config

ALIGN = 4
ROW_WIDTH = 60

def round_up(x, d):
    return (x + d - 1) // d * d

def print_dict(title : str, content : Dict[str, Any], file=sys.stdout):
    max_kw_len = max([ len(kw) for kw in content.keys() ])
    max_kw_len = round_up(max_kw_len + 3, 4)

    raw_content = ""

    for kw, val in content.items():
        raw_content += kw + " :" + " " * (max_kw_len - len(kw) - 2)
        raw_val = "%s" % val
        
        len_val_row = ROW_WIDTH - max_kw_len
        st = 0
        if len(raw_val) == 0:
            raw_val = " "
        while st < len(raw_val):
            if st > 0:
                raw_content += " " * max_kw_len
            raw_content += raw_val[st: st + len_val_row] + "\n"
            st += len_val_row
    
    print_block(title, raw_content, file)


def print_block(title : str, content : Optional[str] = None, file=sys.stdout):
    left_title = (ROW_WIDTH - len(title) - 2) // 2
    right_title = ROW_WIDTH - len(title) - 2 - left_title
    
    print("=" * left_title + " " + title + " " + "=" * right_title, file=file)
    if content is not None:
        print(content, file=file)
    
def print_rank(*args, rank=0, **kwargs):
    """
    Prints the message only on the `rank` of the process.

    Args:
        *args: The arguments to be printed.
        rank (int): The rank id of the process to print.
        **kwargs: The keyword arguments to be printed.

    """
    if config["rank"] == rank:
        print(*args, **kwargs)

def see_memory(message, detail=False):
    """
    Outputs a message followed by GPU memory status summary on rank 0.
    At the end of the function, the starting point in tracking maximum GPU memory will be reset.

    Args:
        message (str): The message to be printed. It can be used to distinguish between other outputs.
        detail (bool): Whether to print memory status in a detailed way or in a concise way. Default to false.

    Example:
        >>> bmt.see_memory("before forward")
        >>> # forward_step()
        >>> bmt.see_memory("after forward")
    
    """
    print_rank(message)
    if detail:
        print_rank(torch.cuda.memory_summary())
    else:
        print_rank(f"""
        =======================================================================================
        memory_allocated {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB
        max_memory_allocated {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB
        =======================================================================================
        """)
    torch.cuda.reset_peak_memory_stats()

class AverageRecorder:
    """A utility class to record the average value of a quantity over time.

    Args:
        alpha (float): The decay factor of the average.
        start_value (float): The initial value of the average.
    
    Use `.value` to get the current average value.
    It is calculated as `alpha * old_value + (1 - alpha) * new_value`.
    
    """
    def __init__(self, alpha = 0.9, start_value = 0):
        self._value = start_value
        self.alpha = alpha
        self._steps = 0
    
    def record(self, v):
        """Records a new value.
        Args:
            v (float): The new value.
        """
        self._value = self._value * self.alpha + v * (1 - self.alpha)
        self._steps += 1
    
    @property
    def value(self):
        if self._steps <= 0:
            return self._value
        return self._value / (1 - pow(self.alpha, self._steps))
