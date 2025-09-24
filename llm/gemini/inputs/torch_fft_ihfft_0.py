
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ihfft_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": -1, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specifying n
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input_tensor, "n": 5, "dim": -1, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Specifying dim
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": 0, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Specifying norm = "forward"
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": -1, "norm": "forward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Specifying norm = "backward"
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": -1, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Specifying norm = "ortho"
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": -1, "norm": "ortho", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: With out tensor
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": -1, "norm": None, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Different dimension size
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": 1, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: n smaller than input size
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {"input": input_tensor, "n": 3, "dim": -1, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: different data type
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": -1, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: negative values
    input_tensor = torch.tensor([-1.0, 2.0, -3.0]).numpy()
    input_dict = {"input": input_tensor, "n": None, "dim": -1, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.ihfft"] = ihfft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.ihfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ihfft'.")

check_valid('torch.fft.ihfft', generated_inputs['torch.fft.ihfft'], lib="torch", suffix=0)
