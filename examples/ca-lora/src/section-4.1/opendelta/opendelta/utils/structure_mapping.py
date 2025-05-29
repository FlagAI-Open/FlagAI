from typing import OrderedDict
import copy
import opendelta.utils.logging as logging
from bigmodelvis import Visualization
logger = logging.get_logger(__name__)


from opendelta.utils.common_structures import CoreMappings

MAPPINGERROR_MSG = f"Available Models with default configurations are {list(CoreMappings.keys())} . Please manually add the delta models by speicifying 'modified_modules' based on the visualization of your model structure. Refer to `https://opendelta.readthedocs.io/en/latest/notes/faq.html` for detail."


def transform(org_key, mapping, strict=True, warning=False, verbose=False):
    chain = org_key.split(".")
    query = ""
    node = mapping

    new_chain = []
    virtual_key, virtual_chain, in_virtual_order = None, None, None
    for elem in chain:
        query += elem
        if query in node:
            node = node[query]
            new_elem = node["__name__"]
            if new_elem == "":
                if strict:
                    if warning:
                        print(f"'{org_key}' has no common mapping.")
                    return
                else:
                    new_chain.append(query)
            else:
                splited_new_elem = new_elem.split(".")
                splited_new_elem = [e+"@" for e in splited_new_elem]
                special_token = '.'.join(splited_new_elem)
                if '__virtual__' in node:
                    virtual_chain = copy.deepcopy(new_chain)
                    virtual_chain.append(".".join([e+'@' for e in node["__virtual__"].split(".")]))
                    in_virtual_order = node['__order__']
                new_chain.append(special_token) # special token for transformed key
                

            query = ""
        elif "$" in node:
            node = node["$"]
            new_chain.append(query)
            query = ""
        else:
            query += "."
    if query!="":
        if strict:
            if warning:
                print("A part of the orginial key hasn't been matched!")
            return
        else:
            new_chain.append(query.strip(".")) # tailing query

    new_key = ".".join(new_chain)
    if verbose:
        print(f"{org_key} => {new_key}")
    if virtual_chain is not None:
        virtual_key = ".".join(virtual_chain)

    return new_key, virtual_key, in_virtual_order



class CommonStructureMap(object):
    r""" A loading structure map.
    """

    New_Mappings = CoreMappings

    SpecialModelInverseMaps = {
    }
    def __init__(self, backbone_model, strict=True, warning=False, visualize=True):
        self.matched_pairs = {}
        self.find_sub_common_structure(backbone_model, matched_pairs=self.matched_pairs)
        if len(self.matched_pairs) == 0:
            raise KeyError(MAPPINGERROR_MSG)


    def __repr__(self,):
        return self.mapping

    def transform(self, org_key, strict=True, warning=False):
        r'''Transform a key in the original model to the name convention in common structure. 
        '''
        new_key = org_key
        virtual_key, in_virtual_order = None, None

        for key in self.matched_pairs:
            left, right = org_key[:len(key)], org_key[len(key):].strip(".")
            if left == key and len(right) > 0:
                transformed_key, virtual_key, in_virtual_order = transform(right, self.matched_pairs[key], strict, warning)
                if len(left) > 0:
                    new_key = left + "." + transformed_key
                else:
                    new_key = transformed_key
                break
        return new_key, virtual_key, in_virtual_order

    def find_sub_common_structure(self, module, prefix='',matched_pairs = []):
        if module.__class__.__name__ in self.New_Mappings:
            if self.New_Mappings[module.__class__.__name__]:
                if callable(self.New_Mappings[module.__class__.__name__]):
                    mapping = self.New_Mappings[module.__class__.__name__](module)
                else:
                    mapping = self.New_Mappings[module.__class__.__name__]
            matched_pairs[prefix] =  mapping
            return
        for name, m in module.named_children():
            new_prefix = '.'.join([prefix, name]) if prefix != '' else name
            self.find_sub_common_structure(m, prefix=new_prefix, matched_pairs = matched_pairs)
            



