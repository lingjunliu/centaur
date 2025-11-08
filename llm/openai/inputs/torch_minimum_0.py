
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def minimum_inputs():
    list_of_inputs = []

    # 1: int64, 1D same shape
    input = torch.tensor([1, 2, -1], dtype=torch.int64).numpy()
    other = torch.tensor([3, 0, 4], dtype=torch.int64).numpy()
    out = torch.empty((3,), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 2: float32 with NaN and Inf, 2D same shape
    input = torch.tensor([[float("nan"), 2.0], [float("inf"), -3.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[1.5, float("nan")], [float("-inf"), -2.5]], dtype=torch.float32).numpy()
    out = torch.empty((2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 3: int32 broadcasting with scalar tensor
    input = torch.tensor([[10, -20, 30], [5, -5, 0]], dtype=torch.int32).numpy()
    other = torch.tensor(5, dtype=torch.int32).numpy()
    out = torch.empty((2, 3), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 4: float64 broadcasting (4,1,3) with (1,5,3) -> (4,5,3)
    input = torch.tensor(
        [[[1.0, -2.0, 3.5]],
         [[-4.2, 5.1, -6.3]],
         [[7.7, -8.8, 9.9]],
         [[0.0, -0.1, 0.2]]],
        dtype=torch.float64
    ).numpy()
    other = torch.tensor(
        [[[2.0, -3.0, 4.0],
          [5.0, -6.0, 7.0],
          [8.0, -9.0, 10.0],
          [11.0, -12.0, 13.0],
          [14.0, -15.0, 16.0]]],
        dtype=torch.float64
    ).numpy()
    out = torch.empty((4, 5, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 5: bool, 1D same shape
    input = torch.tensor([True, False, True, False], dtype=torch.bool).numpy()
    other = torch.tensor([False, False, True, True], dtype=torch.bool).numpy()
    out = torch.empty((4,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 6: uint8 broadcasting (3,1) with (1,4) -> (3,4)
    input = torch.tensor([[255], [100], [0]], dtype=torch.uint8).numpy()
    other = torch.tensor([[1, 50, 200, 255]], dtype=torch.uint8).numpy()
    out = torch.empty((3, 4), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 7: float64 broadcasting with negative zero (1,4) and (2,1,4) -> (2,1,4)
    input = torch.tensor([[-0.0, 0.0, -1.0, 1.0]], dtype=torch.float64).numpy()
    other = torch.tensor([[[0.5, -0.0, -2.0, 2.0]], [[-0.5, 0.0, 3.0, -3.0]]], dtype=torch.float64).numpy()
    out = torch.empty((2, 1, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 8: int16, 3D same shape
    input = torch.tensor([[[1, -2], [300, -400]], [[-32768 + 10, 100], [0, 32767 - 10]]], dtype=torch.int16).numpy()
    other = torch.tensor([[[2, -3], [200, -500]], [[-32768 + 5, 200], [-1, 32767 - 20]]], dtype=torch.int16).numpy()
    out = torch.empty((2, 2, 2), dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 9: float16 scalar with 2D tensor -> (3,3)
    input = torch.tensor(1.25, dtype=torch.float16).numpy()
    other = torch.tensor([[2.0, -1.5, 1.25], [0.0, 3.5, -4.0], [1.0, 1.3, 1.2]], dtype=torch.float16).numpy()
    out = torch.empty((3, 3), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 10: int32 non-contiguous via transpose, same shape
    base = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    input = base.t().numpy()
    other = (base.t() * -1).numpy()
    out = torch.empty((4, 3), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 11: float32 empty dimension (0,3), same shape
    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    other = torch.empty((0, 3), dtype=torch.float32).numpy()
    out = torch.empty((0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 12: float32 4D with empty dimension (2,0,4,5), same shape
    input = torch.empty((2, 0, 4, 5), dtype=torch.float32).numpy()
    other = torch.empty((2, 0, 4, 5), dtype=torch.float32).numpy()
    out = torch.empty((2, 0, 4, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.minimum"] = minimum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.minimum'.")


check_valid('torch.minimum', generated_inputs['torch.minimum'], lib="torch", suffix=0)
