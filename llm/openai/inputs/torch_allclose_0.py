
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def allclose_inputs():
    list_of_inputs = []
    # Input 1: 1D float32, small differences within tolerance
    input_arr = torch.tensor([1.0, 2.0, 3.00001], dtype=torch.float32).numpy()
    other_arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32, differences outside tolerance
    input_arr = torch.tensor([1.0, -5.0, 10.0], dtype=torch.float32).numpy()
    other_arr = torch.tensor([1.1, -4.7, 9.7], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-06,
        "atol": 1e-09,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64, large values to test relative tolerance
    input_arr = torch.tensor([[10000.0, 1e-08], [3.1415926535, -2.7182818284]], dtype=torch.float64).numpy()
    other_arr = torch.tensor([[10000.1, 1e-09], [3.1415926, -2.7182819]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 with NaNs, equal_nan=False
    input_arr = torch.tensor([[1.0, float('nan')], [2.0, 3.0]], dtype=torch.float64).numpy()
    other_arr = torch.tensor([[1.0, float('nan')], [2.0, 3.0000000001]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-07,
        "atol": 1e-09,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 with NaNs, equal_nan=True
    input_arr = torch.tensor([[1.0, float('nan')], [2.0, 3.0]], dtype=torch.float64).numpy()
    other_arr = torch.tensor([[1.0, float('nan')], [2.0, 3.0000000001]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-07,
        "atol": 1e-09,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int64 tensors, exactly equal
    input_arr = torch.tensor([[[-1, 0], [5, -7]], [[9, 10], [11, -12]]], dtype=torch.int64).numpy()
    other_arr = torch.tensor([[[-1, 0], [5, -7]], [[9, 10], [11, -12]]], dtype=torch.int64).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 0.0,
        "atol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D bool tensors
    input_arr = torch.tensor([[[True, False], [True, True]]], dtype=torch.bool).numpy()
    other_arr = torch.tensor([[[True, False], [True, False]]], dtype=torch.bool).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 0.0,
        "atol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 tensors with small differences
    input_arr = torch.tensor([1+1j, 2-3j, -0.5+0.25j], dtype=torch.complex64).numpy()
    other_arr = torch.tensor([1+1.0001j, 2.0002-3j, -0.5001+0.2502j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-03,
        "atol": 1e-04,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero-length tensors
    input_arr = torch.tensor([], dtype=torch.float32).numpy()
    other_arr = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Arrays with infinities
    input_arr = torch.tensor([float('inf'), -float('inf'), 1.0, -2.0], dtype=torch.float64).numpy()
    other_arr = torch.tensor([float('inf'), -float('inf'), 1.0 + 1e-10, -2.0 - 1e-10], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-06,
        "atol": 1e-12,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float16 tensors with moderate tolerance
    input_arr = torch.tensor([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.6]], dtype=torch.float16).numpy()
    other_arr = torch.tensor([[0.1005, 0.1995, 0.3005],
                              [0.4005, 0.5005, 0.5995]], dtype=torch.float16).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 1e-02,
        "atol": 1e-02,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D uint8 tensors, identical
    input_arr = torch.tensor([[[[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 10, 11, 12]]],
                              [[[13, 14, 15, 16],
                                [17, 18, 19, 20],
                                [21, 22, 23, 24]]]], dtype=torch.uint8).numpy()
    other_arr = torch.tensor([[[[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 10, 11, 12]]],
                              [[[13, 14, 15, 16],
                                [17, 18, 19, 20],
                                [21, 22, 23, 24]]]], dtype=torch.uint8).numpy()
    input_dict = {
        "input": input_arr,
        "other": other_arr,
        "rtol": 0.0,
        "atol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.allclose"] = allclose_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.allclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.allclose'.")


check_valid('torch.allclose', generated_inputs['torch.allclose'], lib="torch", suffix=0)
