
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def gammainc_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    t_input = torch.tensor([0.5, 1.0, 2.5], dtype=torch.float32)
    t_other = torch.tensor([0.1, 2.0, 5.0], dtype=torch.float32)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64
    t_input = torch.tensor([[1.2, 2.3, 3.4],
                            [4.5, 5.6, 6.7]], dtype=torch.float64)
    t_other = torch.tensor([[0.0, 0.5, 1.0],
                            [10.0, 20.0, 30.0]], dtype=torch.float64)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: broadcasting (3,1) x (1,4), float32
    t_input = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float32)
    t_other = torch.tensor([[0.1, 1.0, 2.0, 3.0]], dtype=torch.float32)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: scalar float64
    t_input = torch.tensor(3.5, dtype=torch.float64)
    t_other = torch.tensor(1.2, dtype=torch.float64)
    out = torch.empty((), dtype=torch.float64)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float16
    t_input = torch.tensor([[[0.5, 1.0],
                             [1.5, 2.0]],
                            [[2.5, 3.0],
                             [3.5, 4.0]]], dtype=torch.float16)
    t_other = torch.tensor([[[0.2, 0.4],
                             [0.6, 0.8]],
                            [[1.0, 2.0],
                             [3.0, 4.0]]], dtype=torch.float16)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: zeros in other, float32
    t_input = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float32)
    t_other = torch.zeros(4, dtype=torch.float32)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large x values, float32
    t_input = torch.tensor([[0.5, 1.0, 2.0],
                            [3.0, 4.0, 5.0]], dtype=torch.float32)
    t_other = torch.tensor([[50.0, 100.0, 150.0],
                            [200.0, 300.0, 400.0]], dtype=torch.float32)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: very small a and x, float64
    t_input = torch.tensor([1e-6, 1e-3, 1e-1], dtype=torch.float64)
    t_other = torch.tensor([1e-6, 1e-3, 1e-1], dtype=torch.float64)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: higher-dim broadcasting (1,3,1) x (4,1,5), float32
    t_input = torch.tensor([[[0.5],
                             [1.5],
                             [2.5]]], dtype=torch.float32)
    t_other = torch.tensor([[[0.1, 0.5, 1.0, 1.5, 2.0]],
                            [[2.5, 3.0, 3.5, 4.0, 4.5]],
                            [[5.0, 5.5, 6.0, 6.5, 7.0]],
                            [[7.5, 8.0, 8.5, 9.0, 9.5]]], dtype=torch.float32)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: vector a and scalar x, float64
    t_input = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    t_other = torch.tensor(1.5, dtype=torch.float64)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: matrix and length-1 vector broadcasting, float32
    t_input = torch.tensor([[0.5, 1.0, 1.5],
                            [2.0, 2.5, 3.0],
                            [3.5, 4.0, 4.5]], dtype=torch.float32)
    t_other = torch.tensor([2.0], dtype=torch.float32)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: larger 3D, float16
    t_input = torch.tensor([[[0.5, 0.75, 1.0, 1.25],
                             [1.5, 1.75, 2.0, 2.25]],
                            [[2.5, 2.75, 3.0, 3.25],
                             [3.5, 3.75, 4.0, 4.25]]], dtype=torch.float16)
    t_other = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                             [0.5, 0.6, 0.7, 0.8]],
                            [[0.9, 1.0, 1.1, 1.2],
                             [1.3, 1.4, 1.5, 1.6]]], dtype=torch.float16)
    b_inp, _ = torch.broadcast_tensors(t_input, t_other)
    out = torch.empty_like(b_inp)
    input_dict = {"input": t_input.numpy(), "other": t_other.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.gammainc"] = gammainc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.gammainc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.gammainc'.")


check_valid('torch.special.gammainc', generated_inputs['torch.special.gammainc'], lib="torch", suffix=0)
