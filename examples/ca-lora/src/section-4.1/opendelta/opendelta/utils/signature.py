import inspect
from collections import namedtuple

def signature(f):
    r"""Get the function f 's input arguments. A useful gadget
    when some function slot might be instantiated into multiple functions.

    Args:
        f (:obj:`function`) : the function to get the input arguments.

    Returns:
        namedtuple : of args, default, varargs, keywords, respectively.s

    """
    sig = inspect.signature(f)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    keywords = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    keywords = keywords[0] if keywords else None
    defaults = [
        p.default for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is not p.empty
    ] or None
    argspec = namedtuple('Signature', ['args', 'defaults',
                                        'varargs', 'keywords'])
    return argspec(args, defaults, varargs, keywords)

def get_arg_names(f):
    r""" Get a functions argument name, remove the ``self`` argument
    """
    args = signature(f).args
    if args[0] == "self":
        args = args[1:]
    return args



def get_arg_names_inside_func(func):
    r""" Get the functions argument name inside the function itself. Remove ``self`` argument.
    """
    arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
    if arg_names[0] == "self":
        arg_names = arg_names[1:]
    return arg_names