
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def amax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, reduce to scalar
    input_arr = torch.tensor([1.0, -2.0, 3.5, 0.0, -1.2], dtype=torch.float32).numpy()
    out_arr = torch.empty((), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64, dim=1
    input_arr = (torch.arange(12, dtype=torch.float64).reshape(3, 4) - 6.0).numpy()
    out_arr = torch.empty((3,), dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": False, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int64, dim=0, keepdim=True
    input_arr = torch.tensor([[1, -2, 3, -4], [5, -6, 7, 8]], dtype=torch.long).numpy()
    out_arr = torch.empty((1, 4), dtype=torch.long).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": True, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32, dim=-1
    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    out_arr = torch.empty((2, 3), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": -1, "keepdim": False, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int8, dim=1, keepdim=True
    input_arr = torch.randint(-128, 128, (2, 3, 4), dtype=torch.int8).numpy()
    out_arr = torch.empty((2, 1, 4), dtype=torch.int8).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": True, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D uint8, dim=2
    input_arr = torch.randint(0, 256, (2, 3, 4, 5), dtype=torch.uint8).numpy()
    out_arr = torch.empty((2, 3, 5), dtype=torch.uint8).numpy()
    input_dict = {"input": input_arr, "dim": 2, "keepdim": False, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D float64, dim=-4, keepdim=True
    input_arr = torch.randn(2, 3, 4, 5, 6, dtype=torch.float64).numpy()
    out_arr = torch.empty((2, 1, 4, 5, 6), dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dim": -4, "keepdim": True, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D bool, dim=0
    input_arr = (torch.rand(4, 3) > 0.5).numpy()
    out_arr = torch.zeros((3,), dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 with NaNs, dim=2, keepdim=True
    input_arr = torch.tensor(
        [
            [[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]],
            [[-1.0, -2.0, -3.0], [float('nan'), 0.0, 2.0]],
        ],
        dtype=torch.float32
    ).numpy()
    out_arr = torch.empty((2, 2, 1), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": 2, "keepdim": True, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int32, dim=-1
    input_arr = torch.randint(-100, 100, (4, 1, 2), dtype=torch.int32).numpy()
    out_arr = torch.empty((4, 1), dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dim": -1, "keepdim": False, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D int64, dim=-1, keepdim=True
    input_arr = torch.randint(-1000, 1000, (1, 2, 1, 3), dtype=torch.long).numpy()
    out_arr = torch.empty((1, 2, 1, 1), dtype=torch.long).numpy()
    input_dict = {"input": input_arr, "dim": -1, "keepdim": True, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 3D float16, dim=0
    input_arr = torch.randn(3, 2, 2, dtype=torch.float16).numpy()
    out_arr = torch.empty((2, 2), dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "out": out_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.amax_1"] = amax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.amax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amax_1'.")


check_valid('torch.amax', generated_inputs['torch.amax_1'], lib="torch", suffix=1)
