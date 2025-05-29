# This utils is used to support Using pytorch's native DataParallel method,
# which create several backbone model inside DataParallel.
# DistributedDataParallel doesn't need this function.
from opendelta.utils.decorate import decorate
from collections import OrderedDict

def sequential_caller(_org_func, org_module, delta_name,  *args, **kwargs):
    args = args[1:] # the first argument here is ``self``
    delta_module = getattr(org_module, delta_name)
    if hasattr(delta_module, "pre_forward"):
        args, kwargs = delta_module.pre_forward(*args, **kwargs)
    ret = _org_func(*args, **kwargs)
    if hasattr(delta_module, "post_forward"):
        ret = delta_module.post_forward(ret)
    return ret

def before_caller(_org_func, org_module, delta_name,  *args, **kwargs):
    args = args[1:] # the first argument here is ``self``
    delta_module = getattr(org_module, delta_name)
    if hasattr(delta_module, "pre_forward"):
        args, kwargs = delta_module.pre_forward(*args, **kwargs)
    ret = _org_func(*args, **kwargs)
    return ret

def after_caller(_org_func, org_module, delta_name,  *args, **kwargs):
    args = args[1:] # the first argument here is ``self``
    delta_module = getattr(org_module, delta_name)
    ret = _org_func(*args, **kwargs)
    if hasattr(delta_module, "post_forward"):
        ret = delta_module.post_forward(ret)
    return ret

def parallel_caller(_org_func, org_module, delta_name, *args, **kwargs):
    args = args[1:] # the first argument here is ``self``
    delta_module = getattr(org_module, delta_name)
    ret_1 = _org_func(*args, **kwargs)
    ret_2 = delta_module.forward(*args, **kwargs)
    return ret_1 + ret_2

caller_map = {
    "sequential": sequential_caller,
    "parallel": parallel_caller,
    "before": before_caller,
    "after": after_caller,
}

def new_replicate_for_data_parallel(self):
    r""" self is the parent module. 
    """
    # rewrite the replicate in DataParallel.
    replica = self.__new__(type(self))
    org_forward = replica.forward
    replica.__dict__ = self.__dict__.copy()
    assert replica.forward != org_forward
    replica.__dict__['forward'] = org_forward


    for _delta_info in self._delta_infos:
        if _delta_info['state'] == 'on':
            if _delta_info['method'] in caller_map.keys():
                caller = caller_map[_delta_info['method']]
                new_forward = decorate(replica.forward, caller, extras=(replica, _delta_info['delta_name']), kwsyntax=True)
            else:
                raise NotImplementedError(f"data_parallel for _delta_info['method']=='{_delta_info['method']}' is not supported")
            replica.__dict__['forward'] = new_forward.__get__(replica, type(replica)) 
    
    # replicas do not have parameters themselves, the replicas reference the original
    # module.
    replica._parameters = OrderedDict()
    replica._buffers = replica._buffers.copy()
    replica._modules = replica._modules.copy()
    replica._is_replica = True

    return replica