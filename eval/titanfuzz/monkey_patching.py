import re, os, sys
from utils.misc import map_torch_to_driver, generate_executible_snippet_from_str

def replace_function_invocation(script, old_function_name, new_function_name):
    pattern = rf"{old_function_name}\((.*?)\)"
    matches = re.finditer(pattern, script)
    for match in matches:
        old_args = match.group(1)
        new_invocation = f"{new_function_name}({old_function_name},'{old_function_name}',{old_args})"
        script = script.replace(match.group(), new_invocation)
    
    return script

def get_prefix():
    return """
import os, pickle, torch
import numpy as np

class monke:
    ind = 0
    def monkey_patcher(self, func, func_str, *args, **kwargs):
        arg_dict = {
            func_str: {
                'args': args,
                'kwargs': kwargs
            }
        }
        pkl_filename = func_str + '_' + str(self.ind) + '_' + os.path.basename(__file__)[:-2] + 'pkl'
        with open(func_str + '/' + pkl_filename, 'wb') as f:
            pickle.dump(arg_dict, f)

        self.ind += 1
        return func(*args, **kwargs)

a_monke = monke()

"""

def get_driver(api):
    return f"""
import sys, os, pickle, torch

dir = sys.argv[1]
for file in os.listdir(dir):
    if not file.endswith('.pkl'):
        continue
    
    with open(os.path.join(dir, file), 'rb') as f:
        input_dict = pickle.load(f)
        output = {api}(*input_dict['{api}']['args'], **input_dict['{api}']['kwargs'])
"""

def monkey_patch(code, apis):
    prefix = get_prefix()
    for api in apis:
        code = replace_function_invocation(code, api, "a_monke.monkey_patcher")
    
    return prefix + generate_executible_snippet_from_str(code)

def main():
    if len(sys.argv) < 4:
        print("Missing argument: input file, input_apis, output_dir")

    input_file = sys.argv[1]
    apis_file = sys.argv[2]
    out_dir = sys.argv[3]
    apis = []
    torch_to_driver, driver_to_torch = map_torch_to_driver()
    with open(apis_file, "r") as f_api:
        for line in f_api.readlines():
            apis.append(driver_to_torch[line.strip()])
    
    for api in apis:
        if not os.path.isdir(f"{out_dir}/{api}"):
            os.mkdir(f"{out_dir}/{api}")
            with open(f"{out_dir}/{api}/driver.py", "w") as f_driver:
                f_driver.write(get_driver(api))
    
    with open(input_file, "r") as f:
        modified_input = monkey_patch(f.read(), apis)

    with open(f"{out_dir}/{os.path.basename(input_file)}", "w") as f_m:
        f_m.write(modified_input)

if __name__ == "__main__":
    main()