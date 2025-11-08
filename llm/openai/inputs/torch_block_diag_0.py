
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def block_diag_inputs():
    list_of_inputs = []

    # 1: 2x2 float32
    t = torch.tensor([[1.5, -2.0],
                      [3.0,  4.5]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": t}))

    # 2: 1D float64
    v = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": v}))

    # 3: 2x3 int64
    i1 = torch.tensor([[1, -2, 3],
                       [4,  5, -6]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": i1}))

    # 4: bool 2x3
    b = torch.tensor([[True, False, True],
                      [False, True, False]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": b}))

    # 5: complex64 2x2
    c = torch.tensor([[1+2j, 3-4j],
                      [-5+0.5j, 0+6j]], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": c}))

    # 6: uint8 3x1
    u = torch.tensor([[0],
                      [128],
                      [255]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": u}))

    # 7: empty 0x0 float32
    e00 = torch.empty((0, 0), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": e00}))

    # 8: zero-sized 0x3 float32
    e03 = torch.empty((0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": e03}))

    # 9: zero-sized 2x0 float32
    e20 = torch.empty((2, 0), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": e20}))

    # 10: float16 3x3 with negatives
    h = torch.tensor([[1.0, -2.0, 3.0],
                      [4.0,  5.0, -6.0],
                      [-7.5,  8.5,  9.5]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": h}))

    # 11: 1x1 float32
    s = torch.tensor([[42.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensors": s}))

    # 12: non-contiguous slice int32 -> numpy
    base = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    nc = base[::2, 1:6:2].numpy()  # shape (2,3)
    list_of_inputs.append(copy.deepcopy({"tensors": nc}))

    return list_of_inputs

generated_inputs["torch.block_diag"] = block_diag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.block_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.block_diag'.")


check_valid('torch.block_diag', generated_inputs['torch.block_diag'], lib="torch", suffix=0)
