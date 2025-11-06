
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def torch_special_erf_inputs():
    list_of_inputs = []

    # 1) 1D float32 with negatives and positives
    input_arr = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 2) 2D float64 matrix
    input_arr = torch.linspace(-3, 3, steps=6, dtype=torch.float64).reshape(2, 3).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 3) 0D scalar float32
    input_arr = torch.tensor(0.123, dtype=torch.float32).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 4) 3D float32 tensor
    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 5) Non-contiguous (transpose) float32
    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 6) Empty 1D float32
    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 7) Special values float64 (inf, -inf, nan, 0)
    input_arr = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0], dtype=torch.float64).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 8) 2D float32 with large magnitude values
    input_arr = torch.tensor([[10.0, -10.0, 0.0], [5.5, -7.25, 0.001]], dtype=torch.float32).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 9) 4D float64 tensor
    input_arr = torch.linspace(-1, 1, steps=24, dtype=torch.float64).reshape(1, 2, 3, 4).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 10) 2D empty (0,3) float64
    input_arr = torch.empty(0, 3, dtype=torch.float64).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 11) 5D float32 tensor
    input_arr = torch.ones(1, 2, 1, 3, 4, dtype=torch.float32).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 12) 1D float64 with extreme small/large values
    input_arr = torch.tensor([1e-12, -1e-12, 1e8, -1e8], dtype=torch.float64).numpy()
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.special.erf"] = torch_special_erf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erf'.")


check_valid('torch.special.erf', generated_inputs['torch.special.erf'], lib="torch", suffix=0)
