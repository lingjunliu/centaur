
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, scalar values
    input_t = torch.tensor([-1.5, 0.0, 2.0], dtype=torch.float32)
    values_t = torch.tensor(0.5, dtype=torch.float32)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64, vector values
    input_t = torch.tensor([-1.5, 0.0, 2.0], dtype=torch.float64)
    values_t = torch.tensor([1.2, -2.0, 3.5], dtype=torch.float64)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int64, scalar values
    input_t = torch.tensor([[-1, 0, 2], [3, 0, -4]], dtype=torch.int64)
    values_t = torch.tensor(7, dtype=torch.int64)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int32, values broadcast along columns
    input_t = torch.tensor([[0, -1, 2], [0, 5, -6]], dtype=torch.int32)
    values_t = torch.tensor([[3, 4, 5]], dtype=torch.int32)  # shape (1,3)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float16, values broadcast from last dim
    input_t = torch.tensor([[-1.0, 0.0, 1.0],
                            [2.5, -3.3, 0.0]], dtype=torch.float16).reshape(2, 1, 3)
    values_t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0-D scalar float64
    input_t = torch.tensor(0.0, dtype=torch.float64)
    values_t = torch.tensor(-1.25, dtype=torch.float64)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: broadcast across multiple dims float32
    input_t = torch.tensor([
        [[-2.0, 0.0]],
        [[3.0, -1.0]],
        [[0.0, 0.5]],
        [[-0.7, 2.2]]
    ], dtype=torch.float32)  # shape (4,1,2)
    values_t = torch.tensor([[-1.0], [0.0], [0.5], [1.5], [-2.0]], dtype=torch.float32).reshape(1, 5, 1)  # shape (1,5,1)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape  # (4,5,2)
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: -0.0 and +0.0 cases float64
    input_t = torch.tensor([-0.0, 0.0, -1.0, 5.0], dtype=torch.float64)
    values_t = torch.tensor(0.3, dtype=torch.float64)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8 vector with negative/zero/positive
    input_t = torch.tensor([-5, -1, 0, 1, 5], dtype=torch.int8)
    values_t = torch.tensor(2, dtype=torch.int8)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16 3D with values varying per batch dim
    input_t = torch.tensor([
        [[-1, 0], [2, 0]],
        [[0, -3], [4, 5]]
    ], dtype=torch.int16)  # shape (2,2,2)
    values_t = torch.tensor([1, -3], dtype=torch.int16).reshape(2, 1, 1)  # shape (2,1,1)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: empty tensor float32
    input_t = torch.empty(0, 3, dtype=torch.float32)
    values_t = torch.tensor(0.9, dtype=torch.float32)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape  # (0,3)
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D float64 broadcasting with last dim
    input_t = torch.tensor([
        -5.0, 0.0, 2.0, -1.0,
        3.0, -4.0, 0.0, 7.0,
        -2.2, 0.0, 0.1, 9.9
    ], dtype=torch.float64).reshape(3, 1, 1, 4)
    values_t = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)  # shape (4,)
    bshape = torch.broadcast_tensors(input_t, values_t)[0].shape  # (3,1,1,4)
    out_t = torch.empty(bshape, dtype=input_t.dtype)
    input_dict = {"input": input_t.numpy(), "values": values_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.heaviside"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.heaviside' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.heaviside'.")


check_valid('torch.heaviside', generated_inputs['torch.heaviside'], lib="torch", suffix=0)
