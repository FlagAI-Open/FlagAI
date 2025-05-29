import torch

DEBUG_VARS = {}

def clear(key=None):
    global DEBUG_VARS
    if key is None:
        DEBUG_VARS = {}
    else:
        DEBUG_VARS.pop(key, None)

def set(key, value):
    global DEBUG_VARS
    if torch.is_tensor(value):
        value = value.detach().cpu()
    DEBUG_VARS[key] = value

def get(key, default=None):
    global DEBUG_VARS
    if key in DEBUG_VARS:
        return DEBUG_VARS[key]
    return default

def append(key, value):
    global DEBUG_VARS
    if key not in DEBUG_VARS:
        DEBUG_VARS[key] = []
    DEBUG_VARS[key].append(value)

def extend(key, value):
    global DEBUG_VARS
    if key not in DEBUG_VARS:
        DEBUG_VARS[key] = []
    DEBUG_VARS[key].extend(value)