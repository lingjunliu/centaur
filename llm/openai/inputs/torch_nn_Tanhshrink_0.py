
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def tanhshrink_inputs():
    list_of_inputs = []
    
    # 1: 0-D scalar float32
    input = torch.tensor(3.5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 2: 1-D array with negatives, zero, positives float32
    input = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 3: 2-D matrix float64
    input = torch.randn(2, 3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 4: 3-D tensor float16 with larger magnitudes
    input = (torch.randn(2, 3, 4, dtype=torch.float16) * 10).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 5: 4-D tensor with an empty dimension
    input = torch.zeros(1, 0, 3, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 6: 1-D array with extreme values
    input = torch.tensor([-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 7: Non-contiguous transpose (2-D)
    t = torch.arange(12, dtype=torch.float32).view(3, 4)
    input = t.t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 8: Strided slice (1-D)
    input = torch.arange(-5.0, 5.0, 0.5, dtype=torch.float32)[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 9: Empty 1-D array
    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 10: Contains NaN and Infs
    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 11: 5-D tensor
    input = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    # 12: Reshaped linspace to 3-D float64
    input = torch.linspace(-2, 2, steps=24, dtype=torch.float64).view(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.nn.Tanhshrink"] = tanhshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Tanhshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Tanhshrink'.")


check_valid('torch.nn.Tanhshrink', generated_inputs['torch.nn.Tanhshrink'], lib="torch", suffix=0)
