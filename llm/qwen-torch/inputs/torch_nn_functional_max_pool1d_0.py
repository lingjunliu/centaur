
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def max_pool1d_inputs():
    list_of_inputs = []
    
    # Input 1 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).numpy()
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]).numpy()
    kernel_size = 4
    stride = 1
    padding = 0
    dilation = 2
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).numpy()
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]).numpy()
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]).numpy()
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]]).numpy()
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]).numpy()
    kernel_size = 4
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]]).numpy()
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]]]).numpy()
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool1d"] = max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool1d'.")


check_valid('torch.nn.functional.max_pool1d', generated_inputs['torch.nn.functional.max_pool1d'], lib="torch", suffix=0)
