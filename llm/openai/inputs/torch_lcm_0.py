
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def lcm_inputs():
    list_of_inputs = []

    # Input 1: 1D int64
    input = torch.tensor([5, 10, 15], dtype=torch.int64).numpy()
    other = torch.tensor([3, 4, 5], dtype=torch.int64).numpy()
    out = torch.empty(3, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 2: Broadcasting with scalar (0-d) other
    input = torch.tensor([5, 10, 15], dtype=torch.int64).numpy()
    other = torch.tensor(3, dtype=torch.int64).numpy()
    out = torch.empty(3, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 3: 2D int32 with negatives and zeros
    input = torch.tensor([[0, -4, 6], [8, -12, 0]], dtype=torch.int32).numpy()
    other = torch.tensor([[0, 6, -9], [14, 18, 21]], dtype=torch.int32).numpy()
    out = torch.empty(2, 3, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 4: uint8 2x2
    input = torch.tensor([[12, 15], [7, 0]], dtype=torch.uint8).numpy()
    other = torch.tensor([[8, 20], [14, 5]], dtype=torch.uint8).numpy()
    out = torch.empty(2, 2, dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 5: mixed dtypes int16 and int8
    input = torch.tensor([12, -9, 7, -128], dtype=torch.int16).numpy()
    other = torch.tensor([18, 6, -21, 64], dtype=torch.int8).numpy()
    out = torch.empty(4, dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 6: 3D broadcasting
    input = torch.tensor([[[2, 4, 6]], [[3, 5, 7]]], dtype=torch.int64).numpy()          # (2,1,3)
    other = torch.tensor([[[3], [4], [5], [6]]], dtype=torch.int64).numpy()               # (1,4,1)
    out = torch.empty(2, 4, 3, dtype=torch.int64).numpy()                                 # (2,4,3)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 7: Large int64 numbers
    input = torch.tensor([1234567890, 9876543210], dtype=torch.int64).numpy()
    other = torch.tensor([987654321, 123456789], dtype=torch.int64).numpy()
    out = torch.empty(2, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 8: zeros and negatives int32
    input = torch.tensor([0, -1, 0, -10, 21], dtype=torch.int32).numpy()
    other = torch.tensor([0, 2, -3, 5, -7], dtype=torch.int32).numpy()
    out = torch.empty(5, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 9: empty tensors
    input = torch.empty(0, dtype=torch.int64).numpy()
    other = torch.empty(0, dtype=torch.int64).numpy()
    out = torch.empty(0, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 10: transposed non-contiguous int16
    base = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.int16)
    input = base.t().numpy()  # (3,2)
    other = torch.tensor([[6, 10], [9, 12], [15, 18]], dtype=torch.int16).numpy()  # (3,2)
    out = torch.empty(3, 2, dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 11: 4D broadcasting
    input = torch.tensor([[[[2, 4, 6]], [[3, 6, 9]]]], dtype=torch.int16).numpy()  # (1,2,1,3)
    other = torch.arange(1, 5 * 1 * 4 * 1 + 1, dtype=torch.int16).view(5, 1, 4, 1).numpy()  # (5,1,4,1)
    out = torch.empty(5, 2, 4, 3, dtype=torch.int16).numpy()  # (5,2,4,3)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 12: mixed uint8 and int16 broadcasting
    input = torch.tensor([[12], [0], [25]], dtype=torch.uint8).numpy()           # (3,1)
    other = torch.tensor([[5, 10, 15, 20]], dtype=torch.int16).numpy()           # (1,4)
    out = torch.empty(3, 4, dtype=torch.int16).numpy()                           # (3,4)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 13: int8 with negatives
    input = torch.tensor([-12, -18, -27, -128], dtype=torch.int8).numpy()
    other = torch.tensor([6, -9, 0, 1], dtype=torch.int8).numpy()
    out = torch.empty(4, dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 14: both scalars (0-d)
    input = torch.tensor(-12, dtype=torch.int64).numpy()
    other = torch.tensor(18, dtype=torch.int64).numpy()
    out = torch.empty((), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.lcm"] = lcm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lcm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lcm'.")


check_valid('torch.lcm', generated_inputs['torch.lcm'], lib="torch", suffix=0)
