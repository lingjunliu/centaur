import re, os, sys
from utils.misc import generate_executible_snippet_from_str

# TODO: import from oracle
# list_of_exceptions = [
#     "Segmentation fault",
#     "Aborted",
#     "Illegal instruction",
#     "Floating point exception",
#     "Bus error",
#     "Killed",
#     "Abort trap",
#     "Process killed",
#     "MemoryError",
#     "INTERNAL ASSERT ERROR",
#     "please report a bug",
#     "CUDA out of memory",
#     "CUDA error"
#     "Timeout"
#     # Add more crash-related strings as needed
# ]

def replace_function_invocation(script, old_function_name, new_function_name):
    pattern = rf"{old_function_name}\((.*?)\)"
    matches = re.finditer(pattern, script)
    for match in matches:
        old_args = match.group(1)
        new_invocation = f"{new_function_name}({old_function_name},'{old_function_name}',{old_args})"
        script = script.replace(match.group(), new_invocation)
    
    return script

def get_prefix(lib):
    return f"""
import os, pickle
{'import torch' if lib == 'torch' else 'import tensorflow as tf'}
import numpy as np

class monke:
    ind = 0
    def monkey_patcher(self, func, func_str, *args, **kwargs):
        arg_dict = {{
            func_str: {{
                'args': args,
                'kwargs': kwargs
            }}
        }}
        pkl_filename = func_str + '_' + str(self.ind) + '_' + os.path.basename(__file__)[:-2] + 'pkl'
        with open(func_str + '/' + pkl_filename, 'wb') as f:
            pickle.dump(arg_dict, f)

        self.ind += 1
        return func(*args, **kwargs)

a_monke = monke()

"""

def get_driver(api, lib):
    # valid,invalid,crash,exception,total,valid_prcnt
    return f"""
import sys, os, pickle
{'import torch' if lib == 'torch' else 'import tensorflow as tf'}

total = 0
valid = 0
invalid = 0
dir = sys.argv[1]
for file in os.listdir(dir):
    if not file.endswith('.pkl'):
        continue
    
    with open(os.path.join(dir, file), 'rb') as f:
        input_dict = pickle.load(f)
        try:
            output = {api}(*input_dict['{api}']['args'], **input_dict['{api}']['kwargs'])
            valid += 1
        except Exception as e:            
            invalid += 1
        total += 1

print('Total inputs: ', total)
print('Valid inputs: ', valid)
print('Invalid inputs: ', invalid)
"""

def monkey_patch(code, apis, lib):
    prefix = get_prefix(lib)
    for api in apis:
        code = replace_function_invocation(code, api, "a_monke.monkey_patcher")
    
    return prefix + generate_executible_snippet_from_str(code)

def main():
    if len(sys.argv) < 4:
        print("Missing argument: input file, input_apis, output_dir")

    input_file = sys.argv[1]
    apis_file = sys.argv[2]
    out_dir = sys.argv[3]
    lib = sys.argv[4]
    with open(apis_file, "r") as f:
        apis = [line.strip() for line in f.readlines() if line.strip()]
    
    for api in apis:
        os.makedirs(f"{out_dir}/{api}", exist_ok=True)
        with open(f"{out_dir}/{api}/driver.py", "w") as f_driver:
            f_driver.write(get_driver(api, lib))
    
    with open(input_file, "r") as f:
        modified_input = monkey_patch(f.read(), apis, lib)

    with open(f"{out_dir}/{os.path.basename(input_file)}", "w") as f_m:
        f_m.write(modified_input)

if __name__ == "__main__":
    main()