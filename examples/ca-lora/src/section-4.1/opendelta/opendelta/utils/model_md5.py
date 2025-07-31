import hashlib
 
def gen_model_hash(model, with_parameters=True):
    r"""Get model hash (structure and parameter)
    """
    str_model_structure = str(model).replace("\n","").replace(" ","").replace("\t","").encode('utf-8')
    md5 = hashlib.md5(str_model_structure)

    if with_parameters:
        md5 = gen_parameter_hash(model.parameters(), md5=md5)
    
    md5_code = md5.hexdigest()
    return md5_code 


    
def gen_parameter_hash(generator, md5=None):
    r"""Get parameter hash. From https://zhuanlan.zhihu.com/p/392942816

    """
    if md5 is None:
        md5 = hashlib.md5()  
    for arg in generator:
        x = arg.data
        if hasattr(x, "cpu"):
            md5.update(x.cpu().numpy().data.tobytes())
        elif hasattr(x, "numpy"):
            md5.update(x.numpy().data.tobytes())
        elif hasattr(x, "data"):
            md5.update(x.data.tobytes())
        else:
            try:
                md5.update(x.encode("utf-8"))
            except:
                md5.update(str(x).encode("utf-8"))
    return md5