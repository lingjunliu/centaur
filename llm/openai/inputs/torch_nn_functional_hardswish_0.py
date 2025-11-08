
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def hardswish_inputs():
    list_of_inputs = []

    input = np.array([-4., -3., -2., -1., 0., 1., 2., 3., 4.], dtype=np.float32)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = np.array([[-6.0, -2.5, 0.0],
                      [1.2, 2.8, 7.5]], dtype=np.float64)
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = np.linspace(-5, 5, num=24, dtype=np.float16).reshape(2, 3, 4)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = ((np.arange(24, dtype=np.float32) - 12.0) / 3.0).reshape(1, 2, 3, 4)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = np.array(2.5, dtype=np.float32)
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = np.array([], dtype=np.float32)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    input = base[:, ::2] - 3.0
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    arrC = np.arange(10, dtype=np.float64).reshape(2, 5)
    input = np.asfortranarray((arrC - 5.0) / 2.0)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = np.array([-np.inf, -10.0, -3.0, -0.0, np.nan, 3.0, 10.0, np.inf], dtype=np.float32)
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = np.linspace(-3.5, 3.5, num=24, dtype=np.float32).reshape(1, 2, 1, 3, 4)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = np.linspace(-4, 4, num=9, dtype=np.float64)[::-1]
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    input = np.full((2, 2), -3.5, dtype=np.float16)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "inplace": inplace}))

    return list_of_inputs

generated_inputs["torch.nn.functional.hardswish"] = hardswish_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hardswish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardswish'.")


check_valid('torch.nn.functional.hardswish', generated_inputs['torch.nn.functional.hardswish'], lib="torch", suffix=0)
