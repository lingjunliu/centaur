
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def set_warn_always_inputs():
    list_of_inputs = []

    list_of_inputs.append(copy.deepcopy({"warn_always": True}))
    list_of_inputs.append(copy.deepcopy({"warn_always": False}))
    list_of_inputs.append(copy.deepcopy({"warn_always": bool(1)}))
    list_of_inputs.append(copy.deepcopy({"warn_always": bool(0)}))
    list_of_inputs.append(copy.deepcopy({"warn_always": bool("x")}))
    list_of_inputs.append(copy.deepcopy({"warn_always": bool("")}))
    list_of_inputs.append(copy.deepcopy({"warn_always": bool([])}))
    list_of_inputs.append(copy.deepcopy({"warn_always": bool([0])}))
    list_of_inputs.append(copy.deepcopy({"warn_always": any([0, 0, 1])}))
    list_of_inputs.append(copy.deepcopy({"warn_always": all([1, True, 2])}))
    list_of_inputs.append(copy.deepcopy({"warn_always": all([])}))
    list_of_inputs.append(copy.deepcopy({"warn_always": any([])}))

    return list_of_inputs

generated_inputs["torch.set_warn_always"] = set_warn_always_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_warn_always' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_warn_always'.")


check_valid('torch.set_warn_always', generated_inputs['torch.set_warn_always'], lib="torch", suffix=0)
