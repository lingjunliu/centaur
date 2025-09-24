
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def xlog1py_3_inputs():
    list_of_inputs = []
    
    input = torch.tensor([0.1, 1.5, 10.0], dtype=torch.float32).numpy()
    other = np.int32(2)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[-1.0, 0.0, 1.0],
                          [2.0, -3.0, 4.0]], dtype=torch.float64).numpy()
    other = np.int64(-1)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.arange(6, dtype=torch.float32).view(1, 2, 3).numpy()
    other = np.int16(-3)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor(3.1415926535, dtype=torch.float64).numpy()
    other = np.int8(5)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.arange(6, dtype=torch.float32).view(2, 1, 3, 1).numpy()
    other = np.int16(0)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.arange(10, dtype=torch.float64)[::2].numpy()
    other = np.int64(1)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([0.0, float('inf'), -float('inf'), float('nan')], dtype=torch.float32).numpy()
    other = np.int32(-1)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    other = np.int16(5)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.arange(12, dtype=torch.float64).view(3, 4).t().numpy()
    other = np.int32(10)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[[-1e-5, 2.0], [3.5, -4.2]]], dtype=torch.float32).numpy()
    other = np.int8(-2)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([1e-300, 1e300, -1e-100, -1e100], dtype=torch.float64).numpy()
    other = np.int64(1000000)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.empty((0,), dtype=torch.float64).numpy()
    other = np.int32(0)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py_3"] = xlog1py_3_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.xlog1py_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlog1py_3'.")


check_valid('torch.special.xlog1py', generated_inputs['torch.special.xlog1py_3'], lib="torch", suffix=3)
