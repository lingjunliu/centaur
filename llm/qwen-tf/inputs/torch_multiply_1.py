
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def multiply_inputs():
    list_of_inputs = []
    
    # 1
    input = torch.tensor([1.0, -2.0, 3.5, 0.0, -1.25], dtype=torch.float32).numpy()
    other = torch.tensor([0.5, 2.0, -4.0, 1.0, 3.0], dtype=torch.float32).numpy()
    out = torch.empty((5,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 2
    input = torch.tensor([[-1, 2, -3], [4, -5, 6]], dtype=torch.int64).numpy()
    other = torch.tensor([[7, -8, 9], [-10, 11, -12]], dtype=torch.int64).numpy()
    out = torch.empty((2, 3), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 3
    input = torch.tensor([[[1.0, -1.0], [2.0, -2.0]],
                          [[0.5, -0.5], [1.5, -1.5]]], dtype=torch.float16).numpy()
    other = torch.tensor([[[2.0, 3.0], [-1.0, 0.5]],
                          [[4.0, -2.0], [0.0, 1.0]]], dtype=torch.float16).numpy()
    out = torch.empty((2, 2, 2), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 4
    input = torch.tensor(3, dtype=torch.int32).numpy()
    other = torch.tensor(-2, dtype=torch.int32).numpy()
    out = torch.empty((), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 5
    input = torch.empty((0,), dtype=torch.float64).numpy()
    other = torch.empty((0,), dtype=torch.float64).numpy()
    out = torch.empty((0,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 6
    input = torch.tensor([[1.0], [-2.0], [3.5]], dtype=torch.float32).numpy()
    other = torch.tensor([[0.5, -1.0, 2.0, -3.0]], dtype=torch.float32).numpy()
    out = torch.empty((3, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 7
    input = torch.tensor([[[1], [-2]]], dtype=torch.int16).numpy()          # (1,2,1)
    other = torch.tensor([[[3, -1, 2]],
                          [[-4, 5, 0]],
                          [[1, 1, 1]],
                          [[2, -2, -2]]], dtype=torch.int16).numpy()        # (4,1,3)
    out = torch.empty((4, 2, 3), dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 8
    input = torch.tensor([[1+2j, -3+0.5j, 2-1j],
                          [-0.5-0.5j, 0+1j, -1-2j]], dtype=torch.complex64).numpy()
    other = torch.tensor([[2-1j, -1+3j, 0.5+0.5j],
                          [-2-2j, 1-1j, 3+0j]], dtype=torch.complex64).numpy()
    out = torch.empty((2, 3), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 9
    input = torch.randn((1, 2, 3, 4), dtype=torch.float32).numpy()
    other = torch.randn((1, 2, 3, 4), dtype=torch.float32).numpy()
    out = torch.empty((1, 2, 3, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 10
    input = torch.tensor([[1, 255], [128, 64]], dtype=torch.uint8).numpy()
    other = torch.tensor([[2, 2], [3, 4]], dtype=torch.uint8).numpy()
    out = torch.empty((2, 2), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.multiply_1"] = multiply_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.multiply_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.multiply_1'.")


check_valid('torch.multiply', generated_inputs['torch.multiply_1'], lib="torch", suffix=1)
