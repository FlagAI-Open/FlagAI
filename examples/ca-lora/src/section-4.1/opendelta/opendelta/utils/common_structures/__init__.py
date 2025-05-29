CoreMappings = {}

import importlib
import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_path)

filelist = os.listdir(cur_path)

for file in filelist:
    if not file.endswith(".py"):
        continue
    elif file.endswith("__init__.py"):
        continue
    else:
        filename = file[:-3]
        mappings = importlib.import_module(f".utils.common_structures.{filename}", "opendelta")
        CoreMappings.update(mappings.Mappings)
    


    