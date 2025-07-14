
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def randint_like_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    low = 0
    high = 5
    dtype = np.int32
    layout = "strided"
    requires_grad = False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int32
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.ones((2, 2), dtype=torch.float32).numpy()
    low = -2
    high = 3
    dtype = np.int64
    layout = "strided"
    requires_grad = False # Changed to False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int64
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.zeros((1, 3, 1), dtype=torch.int8).numpy()
    low = 0
    high = 10
    dtype = np.int8
    layout = "strided"
    requires_grad = False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int8
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.tensor([[-1, -2], [3, 4]], dtype=torch.int32).numpy()
    low = -5
    high = 0
    dtype = np.int32
    layout = "strided"
    requires_grad = False # Changed to False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int32
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.rand(5, dtype=torch.float64).numpy()
    low = 1
    high = 100
    dtype = np.int16
    layout = "strided"
    requires_grad = False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int16
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randint(-10, 10, size=(2, 3, 4), dtype=torch.int64).numpy()
    low = -1
    high = 2
    dtype = np.int64
    layout = "strided"
    requires_grad = False # Changed to False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int64
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.tensor(5, dtype=torch.int32).numpy()
    low = 1
    high = 6
    dtype = np.int32
    layout = "strided"
    requires_grad = False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int32
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    low = 0
    high = 2
    dtype = np.int8
    layout = "strided"
    requires_grad = False # Changed to False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int8
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = torch.zeros((2, 5), dtype=torch.float64).numpy()
    low = -5
    high = 5
    dtype = np.int16
    layout = "strided"
    requires_grad = False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int16
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = torch.ones((1, 1, 1, 1), dtype=torch.int16).numpy()
    low = 0
    high = 1
    dtype = np.int8
    layout = "strided"
    requires_grad = False # Changed to False
    input_dict = {"input": input_tensor, "low": low, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    input_dict["dtype"] = torch.int8
    input_dict["layout"] = torch.strided
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.randint_like_1"] = randint_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.randint_like_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.randint_like_1'.")

check_valid('torch.randint_like', generated_inputs['torch.randint_like_1'], lib="torch", suffix=1)
