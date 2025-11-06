
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def logical_not_inputs():
    list_of_inputs = []
    
    input = torch.tensor([True, False, True, False, True], dtype=torch.bool).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[-1, 0, 1],
                          [2, 0, -3]], dtype=torch.int32).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[[0.0, 1.0, -2.5]],
                          [[3.3, 0.0, 4.4]]], dtype=torch.float32).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor(True, dtype=torch.bool).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[[[0, 1, 2],
                            [0, 0, 3]],
                           [[4, 0, 5],
                            [6, 0, 0]]]], dtype=torch.uint8).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0, 1.5], dtype=torch.float64).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int8).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([0, 10**12, -10**12, 5, 0], dtype=torch.int64).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.empty((0,), dtype=torch.float32).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([0+0j, 1+0j, 0+2j, -3-4j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[[[[True, False, True],
                             [False, False, True]]]],
                           [[[[False, True, False],
                              [True, True, False]]]]], dtype=torch.bool).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[0.0, -0.0, 2.0],
                          [-3.0, 0.0, 4.0],
                          [np.nextafter(0.0, 1.0), -5.5, 0.0]], dtype=torch.float32).numpy()
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.logical_not"] = logical_not_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_not'.")


check_valid('torch.logical_not', generated_inputs['torch.logical_not'], lib="torch", suffix=0)
