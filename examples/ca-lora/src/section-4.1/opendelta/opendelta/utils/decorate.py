# copied and modified from decorator.decorate

import re
import sys
import inspect
import operator
import itertools
from contextlib import _GeneratorContextManager
from inspect import getfullargspec, iscoroutinefunction, isgeneratorfunction

def fix(args, kwargs, sig):
    """
    Fix args and kwargs to be consistent with the signature
    """
    ba = sig.bind(*args, **kwargs)
    ba.apply_defaults()  # needed for test_dan_schult
    return ba.args, ba.kwargs


def decorate(func, caller, extras=(), kwsyntax=False):
    """
    Decorates a function/generator/coroutine using a caller.
    If kwsyntax is True calling the decorated functions with keyword
    syntax will pass the named arguments inside the ``kw`` dictionary,
    even if such argument are positional, similarly to what functools.wraps
    does. By default kwsyntax is False and the the arguments are untouched.

    **The difference between this function and decorator.decorate is that
    is support nested decorate. 
    """
    sig = inspect.signature(func)
    if iscoroutinefunction(caller):
        async def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            return await caller(func, *(extras + args), **kw)
    elif isgeneratorfunction(caller):
        def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            for res in caller(func, *(extras + args), **kw):
                yield res
    else:
        def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            return caller(func, *(extras + args), **kw)
    fun.__name__ = func.__name__
    fun.__doc__ = func.__doc__
    __wrapped__ = func # support nested wrap 
    fun.__signature__ = sig
    fun.__qualname__ = func.__qualname__
    # builtin functions like defaultdict.__setitem__ lack many attributes
    try:
        fun.__defaults__ = func.__defaults__
    except AttributeError:
        pass
    try:
        fun.__kwdefaults__ = func.__kwdefaults__
    except AttributeError:
        pass
    try:
        fun.__annotations__ = func.__annotations__
    except AttributeError:
        pass
    try:
        fun.__module__ = func.__module__
    except AttributeError:
        pass
    try:
        fun.__dict__.update(func.__dict__)
    except AttributeError:
        pass
    fun.__wrapped__ = __wrapped__ # support nested wrap 
    return fun