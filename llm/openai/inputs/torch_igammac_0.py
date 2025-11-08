
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def igammac_inputs():
    list_of_inputs = []
    
    # Input 1: 1D float32
    input = torch.tensor([0.5, 1.0, 2.5], dtype=torch.float32).numpy()
    other = torch.tensor([0.1, 3.0, 5.0], dtype=torch.float32).numpy()
    out = np.zeros((3,), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 2: 2D float64
    input = (torch.rand((2, 3), dtype=torch.float64) + 0.1).numpy()
    other = (torch.rand((2, 3), dtype=torch.float64) * 10.0).numpy()
    out = np.zeros((2, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 3: broadcasting (3,1) and (1,4) -> (3,4), float32
    input = (torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float32)).numpy()
    other = (torch.tensor([[0.0, 0.5, 2.0, 4.0]], dtype=torch.float32)).numpy()
    out = np.zeros((3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 4: scalar input and 1D other, float64
    input = torch.tensor(1.5, dtype=torch.float64).numpy()
    other = torch.tensor([0.0, 0.1, 1.0, 2.0, 10.0], dtype=torch.float64).numpy()
    out = np.zeros((5,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 5: 1D input and scalar other, float64
    input = torch.tensor([0.3, 0.7, 1.2, 2.3, 5.0], dtype=torch.float64).numpy()
    other = torch.tensor(1.0, dtype=torch.float64).numpy()
    out = np.zeros((5,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 6: 3D broadcasting (2,1,3) and (1,4,1) -> (2,4,3), float32
    input = (torch.rand((2, 1, 3), dtype=torch.float32) + 0.5).numpy()
    other = (torch.rand((1, 4, 1), dtype=torch.float32) * 6.0).numpy()
    out = np.zeros((2, 4, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 7: float16 small 2D
    input = (torch.tensor([[0.5, 1.0], [2.0, 3.0]], dtype=torch.float16)).numpy()
    other = (torch.tensor([[0.0, 0.2], [5.0, 10.0]], dtype=torch.float16)).numpy()
    out = np.zeros((2, 2), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 8: Large/small values, float64
    input = torch.tensor([0.5, 1.0, 5.0, 10.0], dtype=torch.float64).numpy()
    other = torch.tensor([1e-8, 1.0, 50.0, 100.0], dtype=torch.float64).numpy()
    out = np.zeros((4,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 9: x zeros, float32
    input = (torch.tensor([[0.2, 0.5, 1.5],
                           [2.0, 3.5, 5.0]], dtype=torch.float32)).numpy()
    other = (torch.zeros((2, 3), dtype=torch.float32)).numpy()
    out = np.zeros((2, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 10: Non-contiguous views via slicing, float64
    base_a = (torch.rand((3, 4), dtype=torch.float64) + 0.2).numpy()
    base_x = (torch.rand((3, 4), dtype=torch.float64) * 8.0).numpy()
    input = base_a[:, ::2]
    other = base_x[:, ::2]
    out = np.zeros((3, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 11: Include infinity in x, float64
    input = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64).numpy()
    other = torch.tensor([0.0, 1.0, float('inf')], dtype=torch.float64).numpy()
    out = np.zeros((3,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # Input 12: 4D broadcasting, float32
    input = (torch.rand((2, 1, 3, 1), dtype=torch.float32) + 0.3).numpy()
    other = (torch.rand((1, 5, 1, 4), dtype=torch.float32) * 4.0).numpy()
    out = np.zeros((2, 5, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.igammac"] = igammac_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.igammac' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.igammac'.")


check_valid('torch.igammac', generated_inputs['torch.igammac'], lib="torch", suffix=0)
