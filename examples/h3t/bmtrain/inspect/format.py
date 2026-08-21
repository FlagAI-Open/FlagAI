from typing import Any, Dict, List

def align_str(s : str, align : int, left : bool) -> str:
    if left:
        return s + " " * (align - len(s))
    else:
        return " " * (align - len(s)) + s

def format_line(strs : List[str], length : List[int]):
    ret = ""
    for v, l in zip(strs, length):
        if len(v) + 1 > l:
            v = " " + v[:l - 1]
        else:
            v = " " + v
        ret += align_str(v, l, True)
    return ret

def item_formater(x) -> str:
    if isinstance(x, float):
        return "{:.4f}".format(x)
    else:
        return str(x)

def format_summary(summary : List[Dict[str, Any]]) -> str:
    """Format summary to string.

    Args:
        summary (List[Dict[str, Any]]): The summary to format.

    Returns:
        str: The formatted summary.

    """
    ret = []

    max_name_len = max([len("name")] + [len(item["name"]) for item in summary]) + 4
    headers = [
        "name",
        "shape",
        "max",
        "min",
        "std",
        "mean",
        "grad_std",
        "grad_mean",
    ]
    headers_length = [
        max_name_len,
        20,
        10,
        10,
        10,
        10,
        10,
        10
    ]
    ret.append( format_line(headers, headers_length) )
    for item in summary:
        values = [ item_formater(item[name]) for name in headers ]
        ret.append( format_line(values, headers_length) )
    return "\n".join(ret)

        