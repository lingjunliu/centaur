
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def floor_divide_inputs():
    list_of_inputs = []
    
    # Input 1: 1D int64 with mixed signs
    input = torch.tensor([10, 20, -30, -40], dtype=torch.int64).numpy()
    other = torch.tensor([3, -7, 5, -6], dtype=torch.int64).numpy()
    out = torch.empty(0, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 2: 1D int64 negatives
    input = torch.tensor([-1, -2, -3, -4, -5], dtype=torch.int64).numpy()
    other = torch.tensor([2, 2, 2, 2, 2], dtype=torch.int64).numpy()
    out = torch.empty(0, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 3: 2D float32 and scalar float32
    input = torch.tensor([[1.5, -2.5, 3.5],
                          [4.2, -5.1, 6.0]], dtype=torch.float32).numpy()
    other = torch.tensor(2.0, dtype=torch.float32).numpy()
    out = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 4: int32 and int16 with broadcasting
    input = torch.tensor([[10, -20, 30],
                          [40, -50, 60]], dtype=torch.int32).numpy()
    other = torch.tensor([[1, -2, 3]], dtype=torch.int16).numpy()
    out = torch.empty(0, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 5: 3D float64 broadcasting
    input = torch.tensor([[[1.2, -4.5, 3.3]],
                          [[-2.2, 7.7, -8.8]]], dtype=torch.float64).numpy()
    other = torch.tensor([[[2.5],
                           [-3.5],
                           [4.0],
                           [5.5]]], dtype=torch.float64).numpy()
    out = torch.empty(0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 6: 0-dim int8 scalars
    input = torch.tensor(-127, dtype=torch.int8).numpy()
    other = torch.tensor(3, dtype=torch.int8).numpy()
    out = torch.empty(0, dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 7: int16 with (3,1) and (1,4) broadcasting
    input = torch.tensor([[5],
                          [-9],
                          [12]], dtype=torch.int16).numpy()
    other = torch.tensor([[2, -3, 4, -5]], dtype=torch.int16).numpy()
    out = torch.empty(0, dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 8: 1D float16 arrays
    input = torch.tensor([-1.5, 2.0, 3.5, -4.2, 5.1, -6.6], dtype=torch.float16).numpy()
    other = torch.tensor([2.0, -3.0, 1.5, -2.5, 4.0, -1.25], dtype=torch.float16).numpy()
    out = torch.empty(0, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 9: 4D int64 broadcasting
    input = (torch.arange(2*2*1*3, dtype=torch.int64).view(2, 2, 1, 3) + 1).numpy()
    other_t = torch.ones((1, 2, 4, 1), dtype=torch.int64) * 2
    other_t[:, 1, :, :] = -3
    other = other_t.numpy()
    out = torch.empty(0, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 10: Mixed float32 and int64
    input = torch.tensor([-10.5, 20.25, -30.75, 40.5, -50.0], dtype=torch.float32).numpy()
    other = torch.tensor([3, -4, 5, -6, 7], dtype=torch.int64).numpy()
    out = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 11: Non-contiguous int32 input via slicing
    base = torch.arange(1, 19, dtype=torch.int32).view(3, 6)
    input = base[:, ::2].numpy()
    other = torch.tensor([[2, -3, 4],
                          [5, -6, 7],
                          [8, -9, 10]], dtype=torch.int32).numpy()
    out = torch.empty(0, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 12: 3D int64 broadcasting to (2,3,4)
    input = torch.tensor([[[10], [-20], [30]],
                          [[-40], [50], [-60]]], dtype=torch.int64).numpy()
    other = torch.tensor([[[2, -3, 4, -5]]], dtype=torch.int64).numpy()
    out = torch.empty(0, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.floor_divide_1"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_divide_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_1'.")


check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_1'], lib="torch", suffix=1)
