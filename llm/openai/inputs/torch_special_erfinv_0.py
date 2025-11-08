
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def erfinv_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with boundaries
    t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 2: 2D float64 values within (-1, 1)
    t = torch.tensor([[-0.9, -0.8, -0.1],
                      [ 0.1,  0.8,  0.9]], dtype=torch.float64)
    input_arr = t.numpy()
    out_arr = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 3: 0-D scalar float32
    t = torch.tensor(0.2, dtype=torch.float32)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 4: 3D float16 random in (-0.9, 0.9)
    t = (torch.rand(2, 3, 4, dtype=torch.float16) * 1.8 - 0.9)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 5: 1D float64 with NaN and infinities and out-of-range values
    t = torch.tensor([-1.1, 1.2, float('nan'), -float('inf'), float('inf'), -0.0, 0.9999999], dtype=torch.float64)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 6: 2D non-contiguous (transpose) float32
    base = torch.linspace(-0.99, 0.99, steps=6, dtype=torch.float32).reshape(2, 3)
    t = base.t()
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 7: 4D float64 small range
    vals = torch.arange(6, dtype=torch.float64).view(1, 2, 1, 3)
    t = (vals - 2.5) / 5.0
    input_arr = t.numpy()
    out_arr = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 8: 1D float32 large vector near boundaries
    t = torch.linspace(-0.999, 0.999, steps=100, dtype=torch.float32)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 9: 1D single-element float16 at +1
    t = torch.tensor([1.0], dtype=torch.float16)
    input_arr = t.numpy()
    out_arr = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 10: 1D single-element float64 at -1
    t = torch.tensor([-1.0], dtype=torch.float64)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 11: 1D float32 values extremely close to +/-1
    t = torch.tensor([0.99999994, -0.99999994], dtype=torch.float32)
    input_arr = t.numpy()
    out_arr = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # Input 12: 3D float64 mixed values within [-1, 1]
    t = torch.tensor([[[ 0.0,  0.1],
                       [-0.1,  0.3]],
                      [[-0.3,  0.7],
                       [-0.7,  0.0]]], dtype=torch.float64)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.special.erfinv"] = erfinv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erfinv'.")


check_valid('torch.special.erfinv', generated_inputs['torch.special.erfinv'], lib="torch", suffix=0)
