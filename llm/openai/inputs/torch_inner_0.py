
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def inner_inputs():
    def result_shape(a, b):
        if a.ndim == 0 and b.ndim == 0:
            return ()
        if a.ndim == 0:
            return b.shape
        if b.ndim == 0:
            return a.shape
        return a.shape[:-1] + b.shape[:-1]

    list_of_inputs = []

    # 1
    input_arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()
    other_arr = torch.tensor([0.0, 2.0, 1.0], dtype=torch.float64).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 2
    input_arr = torch.tensor([3, -1, 2, 5], dtype=torch.int64).numpy()
    other_arr = torch.tensor([-2, 0, 7, 1], dtype=torch.int64).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 3
    input_arr = torch.randn(2, 3, dtype=torch.float32).numpy()
    other_arr = torch.randn(4, 3, dtype=torch.float32).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 4
    input_arr = torch.randn(5, 2, 7, dtype=torch.float32).numpy()
    other_arr = torch.randn(6, 7, dtype=torch.float32).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 5
    input_arr = torch.randint(-10, 10, (2, 7), dtype=torch.int32).numpy()
    other_arr = torch.randint(0, 5, (3, 4, 7), dtype=torch.int32).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 6
    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    other_arr = torch.randn(6, 7, 5, dtype=torch.float32).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 7
    input_arr = torch.tensor(2.5, dtype=torch.float32).numpy()
    other_arr = torch.randn(3, 2, dtype=torch.float32).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(other_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 8
    input_arr = torch.randint(-5, 5, (2, 3, 4), dtype=torch.int16).numpy()
    other_arr = torch.tensor(-3, dtype=torch.int16).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 9
    input_arr = torch.randn(3, dtype=torch.complex64).numpy()
    other_arr = torch.randn(3, dtype=torch.complex64).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 10
    input_arr = torch.empty(2, 0, dtype=torch.float32).numpy()
    other_arr = torch.empty(3, 0, dtype=torch.float32).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 11
    input_arr = torch.randn(2, 5, dtype=torch.complex128).numpy()
    other_arr = torch.randn(3, 5, dtype=torch.complex128).numpy()
    out_arr = torch.empty(result_shape(input_arr, other_arr), dtype=torch.from_numpy(input_arr).dtype).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.inner"] = inner_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.inner' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.inner'.")


check_valid('torch.inner', generated_inputs['torch.inner'], lib="torch", suffix=0)
