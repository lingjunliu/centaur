
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def arcsinh_inputs():
    list_of_inputs = []

    t1 = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    input = t1.numpy()
    out = torch.empty_like(t1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t2 = torch.tensor([[-10.0, -0.1, 0.0], [0.1, 10.0, 1000.0]], dtype=torch.float64)
    input = t2.numpy()
    out = torch.empty_like(t2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t3 = torch.tensor(-0.5, dtype=torch.float32)
    input = t3.numpy()
    out = torch.empty_like(t3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t4 = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    input = t4.numpy()
    out = torch.empty_like(t4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t5 = torch.tensor([[[[-1e-4, 0.0, 1e-4]], [[-2.5, 2.5, -3.5]]]], dtype=torch.float64)
    input = t5.numpy()
    out = torch.empty_like(t5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t6 = torch.tensor([1+1j, -2+0.5j, -1j, 3+0j], dtype=torch.complex64)
    input = t6.numpy()
    out = torch.empty_like(t6).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t7 = torch.tensor([[-1-1j, 2-3j], [4+0j, 0+2j]], dtype=torch.complex128)
    input = t7.numpy()
    out = torch.empty_like(t7).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t8 = torch.tensor([1e-6, -1e-6, 1e6, -1e6, 1e20], dtype=torch.float32)
    input = t8.numpy()
    out = torch.empty_like(t8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t9 = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0, -0.0], dtype=torch.float64)
    input = t9.numpy()
    out = torch.empty_like(t9).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    t10 = base.t()
    input = t10.numpy()
    out = torch.empty_like(t10).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t12 = torch.tensor([[[[[-0.0, 0.0, 7.5, -7.5]]]]], dtype=torch.float32)
    input = t12.numpy()
    out = torch.empty_like(t12).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.arcsinh"] = arcsinh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.arcsinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arcsinh'.")


check_valid('torch.arcsinh', generated_inputs['torch.arcsinh'], lib="torch", suffix=0)
