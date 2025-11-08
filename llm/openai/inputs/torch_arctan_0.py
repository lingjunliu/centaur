
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def arctan_inputs():
    list_of_inputs = []
    
    t = torch.tensor([-1.0, 0.0, 1.0, 10.0, -10.0], dtype=torch.float32)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.linspace(-1000, 1000, steps=6, dtype=torch.float64).reshape(2, 3)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.randn(2, 3, 4, dtype=torch.float16)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.tensor([[-3, -2, -1, 0, 1, 2, 3]], dtype=torch.int32)
    input_tensor = t.numpy()
    out_tensor = torch.empty(t.shape, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.arange(-6, 6, dtype=torch.int64).reshape(2, 3, 2)
    input_tensor = t.numpy()
    out_tensor = torch.empty(t.shape, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.tensor([1+1j, -1+2j, -3-4j, 0+0j], dtype=torch.complex64)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    a = torch.randn(2, 2, dtype=torch.float64)
    b = torch.randn(2, 2, dtype=torch.float64)
    t = torch.complex(a, b)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.tensor(0.5, dtype=torch.float32)
    input_tensor = t.numpy()
    out_tensor = torch.empty((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0], dtype=torch.float64)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.empty(0, dtype=torch.float32)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    t = base.t()
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.linspace(-3, 3, steps=24, dtype=torch.float32).reshape(2, 1, 3, 4)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))
    
    t = torch.arange(4, -1, -1, dtype=torch.float32)
    input_tensor = t.numpy()
    out_tensor = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out_tensor}))

    return list_of_inputs

generated_inputs["torch.arctan"] = arctan_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.arctan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arctan'.")


check_valid('torch.arctan', generated_inputs['torch.arctan'], lib="torch", suffix=0)
