
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def pdist_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = 2.0
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    p = 1.0
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[2.0, 3.0, 1.0], [5.0, 1.0, 4.0]])
    p = float('inf')
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 5.0, 6.0]])
    p = 0.0
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    p = 2.0
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-1.0, -2.0, 3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]])
    p = 2.0
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = 0.5
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    p = 2.0
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = 3.0
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[0.0, 0.0], [1.0, 1.0]])
    p = 2.0
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[2.5, 3.5], [1.5, 4.5], [0.5, 2.5]])
    p = 1.5
    input_dict = {"input": input_tensor, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pdist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pdist'.")


check_valid('torch.nn.functional.pdist', generated_inputs['torch.nn.functional.pdist'], lib="torch", suffix=0)
