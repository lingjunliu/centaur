
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D int64, same shape
    input = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    other = torch.tensor([3, 2, 1], dtype=torch.int64).numpy()
    out = torch.empty((3,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 2: 2D float32 with negatives
    input = torch.tensor([[1.0, -2.5, 0.0], [3.14, 4.2, -0.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[1.0, 2.5, 0.0], [3.14, -4.2, 0.0]], dtype=torch.float32).numpy()
    out = torch.empty((2, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 3: Broadcasting (2,1) vs (1,3) int32
    input = torch.tensor([[1], [2]], dtype=torch.int32).numpy()
    other = torch.tensor([[1, 2, 3]], dtype=torch.int32).numpy()
    out = torch.empty((2, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 4: Scalars (0-d) int16
    input = torch.tensor(5, dtype=torch.int16).numpy()
    other = torch.tensor(5, dtype=torch.int16).numpy()
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 5: Different dtypes float64 vs int32
    input = torch.tensor([0.0, -1.5, 2.5, 3.0], dtype=torch.float64).numpy()
    other = torch.tensor([0, -2, 3, 3], dtype=torch.int32).numpy()
    out = torch.empty((4,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 6: Boolean tensors
    input = torch.tensor([[True, False], [False, False]], dtype=torch.bool).numpy()
    other = torch.tensor([[False, False], [True, False]], dtype=torch.bool).numpy()
    out = torch.empty((2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 7: Complex64 tensors
    input = torch.tensor([1+2j, 3+0j, -1-0.5j], dtype=torch.complex64).numpy()
    other = torch.tensor([1+2j, 3+1j, -1-0.5j], dtype=torch.complex64).numpy()
    out = torch.empty((3,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 8: 3D broadcasting (2,1,3) vs (1,4,1) int64
    input = torch.tensor([[[1, 2, 3]], [[-1, 0, -3]]], dtype=torch.int64).numpy()
    other = torch.tensor([[[0], [1], [2], [3]]], dtype=torch.int64).numpy()
    out = torch.empty((2, 4, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 9: Empty tensors broadcasting (0,3) vs (0,1) float32
    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    other = torch.empty((0, 1), dtype=torch.float32).numpy()
    out = torch.empty((0, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 10: 4D broadcasting (1,2,1,3) vs (2,1,1,3) int8
    input = torch.tensor([[[[1, -2, 3]], [[4, 5, -6]]]], dtype=torch.int8).numpy()  # shape (1,2,1,3)
    other = torch.tensor([[[[1, 0, -3]]], [[[4, -5, 6]]]], dtype=torch.int8).numpy()  # shape (2,1,1,3)
    out = torch.empty((2, 2, 1, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 11: Mixed dtypes uint8 vs int16
    input = torch.tensor([[0, 255, 128], [1, 2, 3]], dtype=torch.uint8).numpy()
    other = torch.tensor([[0, -1, 128], [4, 2, -3]], dtype=torch.int16).numpy()
    out = torch.empty((2, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 12: Non-contiguous via transpose float64
    base = torch.arange(12, dtype=torch.float64).view(3, 4)
    input = base.t().numpy()
    other = (base + 1).t().numpy()
    out = torch.empty((4, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.not_equal_1"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.not_equal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.not_equal_1'.")


check_valid('torch.not_equal', generated_inputs['torch.not_equal_1'], lib="torch", suffix=1)
