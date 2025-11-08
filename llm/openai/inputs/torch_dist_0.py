
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def dist_inputs():
    list_of_inputs = []
    
    # Input 1
    input = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    other = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 2 (broadcast with scalar)
    input = torch.arange(6, dtype=torch.float64).view(2, 3).numpy()
    other = torch.tensor(1.5, dtype=torch.float64).numpy()
    p = 1.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 3 (2D equal shapes, p=0)
    input = torch.tensor([[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[0.0, -1.0, 1.0], [2.0, -3.0, 4.0]], dtype=torch.float32).numpy()
    p = 0.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 4 (broadcast along first dim)
    input = torch.tensor([[1.0, -2.0, 3.0],
                          [4.0, -5.0, 6.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float32).numpy()
    p = 3.5
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 5 (3D broadcast)
    input = torch.randn(2, 1, 4, dtype=torch.float32).numpy()
    other = torch.randn(1, 3, 1, dtype=torch.float32).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 6 (empty tensors)
    input = torch.empty(0, dtype=torch.float32).numpy()
    other = torch.empty(0, dtype=torch.float32).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 7 (double with broadcast across leading dims, fractional p)
    input = torch.linspace(-2.0, 2.0, steps=10, dtype=torch.float64).reshape(2, 1, 5).numpy()
    other = torch.tensor([0.5, -0.5, 1.0, -1.0, 0.0], dtype=torch.float64).numpy()
    p = 0.5
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 8 (mixed dims broadcast)
    input = torch.randn(3, 1, 2, 1, dtype=torch.float32).numpy()
    other = torch.randn(1, 4, 1, 1, dtype=torch.float32).numpy()
    p = 1.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 9 (4D equal shapes, higher p)
    input = torch.tensor([[[[1.0, -2.0],
                            [3.0, -4.0]],
                           [[-5.0, 6.0],
                            [-7.0, 8.0]]],
                          [[[0.5, -1.5],
                            [2.5, -3.5]],
                           [[-4.5, 5.5],
                            [-6.5, 7.5]]]], dtype=torch.float32).numpy()
    other = torch.tensor([[[[0.0, 1.0],
                            [-1.0, 2.0]],
                           [[-2.0, 3.0],
                            [-3.0, 4.0]]],
                          [[[1.0, -1.0],
                            [1.0, -1.0]],
                           [[2.0, -2.0],
                            [2.0, -2.0]]]], dtype=torch.float32).numpy()
    p = 7.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 10 (1D, p=inf)
    input = torch.randn(8, dtype=torch.float64).numpy()
    other = torch.randn(8, dtype=torch.float64).numpy()
    p = float('inf')
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 11 (0-D scalars)
    input = torch.tensor(3.14, dtype=torch.float32).numpy()
    other = torch.tensor(-2.71, dtype=torch.float32).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    # Input 12 (3D equal shapes)
    input = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    other = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    p = 1.0
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "p": p}))
    
    return list_of_inputs

generated_inputs["torch.dist"] = dist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dist'.")


check_valid('torch.dist', generated_inputs['torch.dist'], lib="torch", suffix=0)
