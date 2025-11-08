
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def pow_inputs_3():
    list_of_inputs = []

    # Input 1
    self_val = 2.0
    exp_t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    self_val = -3.5
    exp_t = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    self_val = 0.5
    exp_t = torch.tensor([[[ -2.0, -1.0, 0.0],
                           [ 0.5,  1.5, 2.5]]], dtype=torch.float32)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (0-D exponent)
    self_val = 10.0
    exp_t = torch.tensor(3, dtype=torch.int32)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty((), dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (empty 1-D)
    self_val = 2.0
    exp_t = torch.empty((0,), dtype=torch.float32)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (float16 exponents)
    self_val = 1.5
    exp_t = torch.tensor([[0.0, 1.0, 2.0],
                          [-3.0, 0.5, -0.5]], dtype=torch.float16)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (float64 exponents)
    self_val = 4.0
    exp_t = torch.tensor([-1.0, 0.0, 0.5, 2.0], dtype=torch.float64)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (int8 exponents, 3-D)
    self_val = 2.0
    exp_t = torch.tensor([[[ -1,  0],
                           [  1,  2]],
                          [[  3,  4],
                           [ -2, -3]]], dtype=torch.int8)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (negative base with fractional exponents)
    self_val = -2.0
    exp_t = torch.tensor([0.5, -0.5, 1.5, 2.0, 3.0], dtype=torch.float32)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (complex exponents)
    self_val = 3.0
    exp_t = torch.tensor([1+0j, 2+0.5j, -1-1j], dtype=torch.complex64)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (empty in middle dimension)
    self_val = 2.0
    exp_t = torch.empty((2, 0, 3), dtype=torch.int64)
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (non-contiguous exponent)
    self_val = 7.0
    exp_base = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    exp_t = exp_base  # non-contiguous
    out_dtype = torch.result_type(torch.tensor(self_val), exp_t)
    out = torch.empty_like(exp_t, dtype=out_dtype).numpy()
    input_dict = {"self": self_val, "exponent": exp_t.numpy(), "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.pow_3"] = pow_inputs_3()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.pow_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pow_3'.")


check_valid('torch.pow', generated_inputs['torch.pow_3'], lib="torch", suffix=3)
