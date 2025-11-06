
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def gelu_inputs():
    list_of_inputs = []

    # 1: 1D float32 with mixed signs
    input_arr = torch.tensor([[-1.0, 0.0, 1.0, 3.14]]).flatten().numpy()
    input_dict = {
        "approximate": "none",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2: 2D float64 with diverse values
    input_arr = torch.tensor([[-3.0, -0.5, 0.0], [0.5, 2.0, 5.0]], dtype=torch.float64).numpy()
    input_dict = {
        "approximate": "tanh",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3: scalar float32
    input_arr = torch.tensor(0.75, dtype=torch.float32).numpy()
    input_dict = {
        "approximate": "none",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4: 3D float16 random
    input_arr = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    input_dict = {
        "approximate": "tanh",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5: 4D float32 NCHW
    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    input_dict = {
        "approximate": "none",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6: empty 1D float32
    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    input_dict = {
        "approximate": "tanh",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7: zero-size dimension 3D float32
    input_arr = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    input_dict = {
        "approximate": "none",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8: non-contiguous via transpose then numpy
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    input_arr = base.t().numpy()
    input_dict = {
        "approximate": "tanh",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9: large magnitude values float32
    input_arr = torch.tensor([-1000.0, -10.0, 0.0, 10.0, 1000.0], dtype=torch.float32).numpy()
    input_dict = {
        "approximate": "none",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10: very small magnitude values float64
    input_arr = torch.tensor([-1e-6, -1e-8, 0.0, 1e-8, 1e-6], dtype=torch.float64).numpy()
    input_dict = {
        "approximate": "tanh",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11: 5D float32
    input_arr = torch.randn(1, 2, 3, 1, 4, dtype=torch.float32).numpy()
    input_dict = {
        "approximate": "none",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12: ones vector float16
    input_arr = torch.ones(10, dtype=torch.float16).numpy()
    input_dict = {
        "approximate": "tanh",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 13: normal distribution std=5, float32
    input_arr = (torch.randn(7, dtype=torch.float32) * 5.0).numpy()
    input_dict = {
        "approximate": "tanh",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 14: linspace -3 to 3, float32
    input_arr = torch.linspace(-3.0, 3.0, steps=50, dtype=torch.float32).numpy()
    input_dict = {
        "approximate": "none",
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.GELU"] = gelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GELU'.")


check_valid('torch.nn.GELU', generated_inputs['torch.nn.GELU'], lib="torch", suffix=0)
