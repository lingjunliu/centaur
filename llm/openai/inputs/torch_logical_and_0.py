
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def logical_and_inputs():
    list_of_inputs = []
    
    # Input 1: simple 1D bool arrays
    input_t = torch.tensor([True, False, True, False], dtype=torch.bool)
    other_t = torch.tensor([False, True, True, False], dtype=torch.bool)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    input = input_t.numpy()
    other = other_t.numpy()
    out = out_t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 2: 2D array with scalar broadcast
    input_t = torch.tensor([[True, False, True],
                            [False, True, False]], dtype=torch.bool)
    other_t = torch.tensor(True, dtype=torch.bool)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 3: 3D vs 1D broadcast
    input_t = (torch.arange(8).reshape(2, 1, 4) > 2)
    other_t = torch.tensor([True, False, True, False], dtype=torch.bool)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 4: noncontiguous transpose with scalar broadcast
    base = torch.arange(12).reshape(3, 4)
    input_t = (base > 5).t()
    other_t = torch.tensor(False, dtype=torch.bool)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 5: broadcast (1,3,1) with (2,1,5)
    input_t = torch.tensor([[[True], [False], [True]]], dtype=torch.bool)  # (1,3,1)
    other_t = (torch.arange(10).reshape(2, 1, 5) % 2 == 0)  # (2,1,5)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 6: empty dimension broadcast (0,3) with (1,3)
    input_t = torch.zeros((0, 3), dtype=torch.bool)
    other_t = torch.tensor([[True, False, True]], dtype=torch.bool)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 7: noncontiguous stride slicing
    base1d = torch.tensor([True, False, True, True, False, False, True, False, True, True], dtype=torch.bool)
    input_t = base1d[::2]
    other_t = torch.tensor([False, False, True, False, True], dtype=torch.bool)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 8: multiple empty dims broadcast
    input_t = torch.zeros((2, 0, 3), dtype=torch.bool)
    other_t = torch.tensor([[[True]], [[False]]], dtype=torch.bool).reshape(2, 1, 1)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 9: broadcast with random-generated booleans
    input_t = (torch.randn(3, 1, 4) > 0)
    other_t = (torch.linspace(-1, 1, 2).reshape(1, 2, 1) > 0)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 10: constructed from integer/float comparisons to bool
    input_t = (torch.tensor([0, -1, 2, 0, 5]) != 0)
    other_t = torch.tensor([True, True, False, False, True], dtype=torch.bool)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 11: 4D broadcast
    input_t = (torch.arange(2 * 3 * 1 * 4).reshape(2, 3, 1, 4) % 3 == 0)
    other_t = (torch.arange(3 * 5).reshape(1, 3, 5, 1) > 7)
    out_t = torch.empty(torch.broadcast_tensors(input_t, other_t)[0].shape, dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    # Input 12: 0-dim tensors
    input_t = torch.tensor(True, dtype=torch.bool)
    other_t = torch.tensor(False, dtype=torch.bool)
    out_t = torch.empty((), dtype=torch.bool)
    list_of_inputs.append(copy.deepcopy({"input": input_t.numpy(), "other": other_t.numpy(), "out": out_t.numpy()}))
    
    return list_of_inputs

generated_inputs["torch.logical_and"] = logical_and_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logical_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_and'.")


check_valid('torch.logical_and', generated_inputs['torch.logical_and'], lib="torch", suffix=0)
