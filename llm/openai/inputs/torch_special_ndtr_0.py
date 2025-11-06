
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def ndtr_inputs():
    list_of_inputs = []
    
    # Input 1: 1D float32 with negative/positive values
    input = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 2: 2D float64 with a range of values
    input = torch.tensor([[-10.0, -1e-9, 0.5],
                          [5.0, 8.0, 20.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 3: 0-D scalar float32
    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 4: 3D float16 random values
    input = torch.randn((2, 3, 4), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 5: Empty float32 array
    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 6: Values including infinities and very large magnitudes (float64)
    input = torch.tensor([float("-inf"), -1e10, -6.0, 0.0, 6.0, 1e10, float("inf")], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 7: Non-contiguous transposed 2D float32 array
    input = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 8: float16 with large/small values
    input = torch.tensor([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 9: Includes NaN and signed zeros (float32)
    input = torch.tensor([float("nan"), -0.0, 0.0, 2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 10: 4D float64 random array
    input = torch.randn((1, 2, 1, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 11: Linearly spaced values across a wide range (float32)
    input = torch.linspace(-6.0, 6.0, steps=13, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # Input 12: Very small subnormal values (float32)
    input = torch.tensor([1e-45, -1e-45], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.special.ndtr"] = ndtr_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.ndtr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.ndtr'.")


check_valid('torch.special.ndtr', generated_inputs['torch.special.ndtr'], lib="torch", suffix=0)
