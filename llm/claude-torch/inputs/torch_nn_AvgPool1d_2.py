
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def avgpool1d_inputs():
    list_of_inputs = []
    
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": np.array([[[1., 2., 3., 4., 5., 6., 7.]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (3,),
        "stride": (1,),
        "padding": (1,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": np.array([[[1., 2., 3., 4., 5.]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (2,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": True,
        "count_include_pad": True,
        "input": np.array([[[1., 2., 3., 4., 5.]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (3,),
        "stride": (1,),
        "padding": (1,),
        "ceil_mode": False,
        "count_include_pad": False,
        "input": np.array([[[2., 4., 6., 8.]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (2,),
        "stride": (1,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": np.array([[[1., 2., 3., 4., 5.]], [[6., 7., 8., 9., 10.]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (5,),
        "stride": (3,),
        "padding": (2,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": np.array([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (2,),
        "stride": (1,),
        "padding": (1,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": np.array([[[3., 5.]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (4,),
        "stride": (2,),
        "padding": (1,),
        "ceil_mode": True,
        "count_include_

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool1d_2'.")


check_valid('torch.nn.AvgPool1d', generated_inputs['torch.nn.AvgPool1d_2'], lib="torch", suffix=2)
