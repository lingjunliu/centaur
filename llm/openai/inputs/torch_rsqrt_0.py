
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def rsqrt_inputs():
    list_of_inputs = []

    # 1: 1D float32 positive values
    input_arr = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float32).numpy()
    out_arr = torch.empty_like(torch.tensor([1.0, 4.0, 9.0], dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 2: 2D float64 with zero and negative
    input_arr = torch.tensor([[0.0, 0.25], [4.0, -1.0]], dtype=torch.float64).numpy()
    out_arr = torch.empty_like(torch.tensor([[0.0, 0.25], [4.0, -1.0]], dtype=torch.float64)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 3: 3D float16 random values (may include negatives)
    t = torch.randn(2, 3, 4, dtype=torch.float16)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 4: 0-D scalar float32
    t = torch.tensor(16.0, dtype=torch.float32)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 5: 4D float64 with varied shape
    t = torch.arange(12, dtype=torch.float64).view(2, 1, 3, 2) + 1.0
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 6: Non-contiguous slice float32
    base = torch.linspace(1.0, 20.0, steps=20, dtype=torch.float32).view(4, 5)
    t = base[:, ::2]
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 7: Contains inf, nan, zeros, negative, positive (float64)
    t = torch.tensor([float('inf'), float('nan'), 0.0, -0.0, -4.0, 16.0], dtype=torch.float64)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 8: Very small and very large positives (float32)
    t = torch.tensor([1e-12, 1e-20, 1e12, 1e20], dtype=torch.float32)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 9: Row vector zeros (float32)
    t = torch.zeros((1, 5), dtype=torch.float32)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 10: Column vector arange (float64)
    t = torch.arange(1, 6, dtype=torch.float64).view(5, 1)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 11: High-dimensional random (float32)
    t = torch.randn(3, 2, 2, 2, 2, dtype=torch.float32)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 12: 1D float16 with mix including zeros and negatives
    t = torch.tensor([0.0, 2.0, -3.0, 4.0, -0.0], dtype=torch.float16)
    input_arr = t.numpy()
    out_arr = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.rsqrt"] = rsqrt_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.rsqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsqrt'.")


check_valid('torch.rsqrt', generated_inputs['torch.rsqrt'], lib="torch", suffix=0)
