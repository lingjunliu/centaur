
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def nansum_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, 2.0, float('nan'), 4.0]).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1.0, float('nan')], [3.0, 4.0]]).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1, 2, -3], [4, -5, 6]], dtype=torch.int64).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([
        [[1.0, float('nan')], [-2.0, -3.0]],
        [[float('nan'), 5.0], [6.0, float('nan')]]
    ]).numpy()
    input_dict = {"input": input_arr, "dim": -1, "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor(
        [[[[1.0, float('nan')], [float('inf'), -1.0], [2.0, 3.0]],
          [[-float('inf'), 0.0], [4.0, float('nan')], [-2.0, 5.0]]],
         [[[float('nan'), -3.0], [7.0, 8.0], [float('nan'), float('inf')]],
          [[9.0, -10.0], [float('-inf'), float('nan')], [0.0, 1.0]]]]
    ).numpy()
    input_dict = {"input": input_arr, "dim": 2, "keepdim": True, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([
        [[1, -2, 3, -4],
         [5, -6, 7, -8],
         [9, -10, 11, -12]],
        [[-1, 2, -3, 4],
         [-5, 6, -7, 8],
         [-9, 10, -11, 12]]
    ], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": True, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[True, False, True],
                              [False, False, True]], dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "dim": -2, "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32)
    t[0, 0, 1, 0, 2] = float('nan')
    t[1, 0, 2, 0, 3] = float('nan')
    input_arr = t.numpy()
    input_dict = {"input": input_arr, "dim": 3, "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    u = torch.tensor(np.arange(2*3*1*2, dtype=np.float32).reshape(2, 3, 1, 2))
    u[0, 2, 0, 1] = float('nan')
    input_arr = u.numpy()
    input_dict = {"input": input_arr, "dim": -3, "keepdim": True, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([1e4, -2e4, float('nan'), 3e4, -4e4], dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": True, "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-30000, 20000, -10000, 5000],
                              [4000, -3000, 2000, -1000]], dtype=torch.int16).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nansum_2"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_2'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_2'], lib="torch", suffix=2)
