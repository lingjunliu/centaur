
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def where_inputs():
    list_of_inputs = []

    # Helper function to create inputs with consistent dtypes
    def create_input(shape, dtype, broadcast_input=None, broadcast_other=None):
        condition = np.random.choice([True, False], size=shape)
        
        input_shape = broadcast_input if broadcast_input is not None else shape
        other_shape = broadcast_other if broadcast_other is not None else shape
        
        # Create numpy arrays from random values, handling the scalar case
        input_val = np.random.rand(*input_shape) * 100
        other_val = np.random.rand(*other_shape) * 100

        input_tensor = np.array(input_val, dtype=dtype)
        other_tensor = np.array(other_val, dtype=dtype)

        # Determine the output shape after broadcasting
        out_shape = np.broadcast_shapes(condition.shape, input_tensor.shape, other_tensor.shape)
        
        # For mixed dtypes, torch promotes, but for the out parameter, the type must match
        # the promoted type. For simplicity, we ensure all are the same.
        result_dtype = np.result_type(input_tensor, other_tensor)

        return {
            "condition": condition,
            "input": input_tensor.astype(result_dtype),
            "other": other_tensor.astype(result_dtype),
            "out": np.empty(out_shape, dtype=result_dtype)
        }

    # Input 1: float32, basic
    list_of_inputs.append(copy.deepcopy(create_input((3, 4), np.float32)))

    # Input 2: float64, basic
    list_of_inputs.append(copy.deepcopy(create_input((2, 5), np.float64)))

    # Input 3: int32, basic
    list_of_inputs.append(copy.deepcopy(create_input((4, 4), np.int32)))

    # Input 4: int16, basic
    list_of_inputs.append(copy.deepcopy(create_input((5, 2), np.int16)))

    # Input 5: Broadcasting with float32
    list_of_inputs.append(copy.deepcopy(create_input((3, 4), np.float32, broadcast_input=(4,), broadcast_other=(3, 1))))
    
    # Input 6: Broadcasting with float64, one input is a scalar
    list_of_inputs.append(copy.deepcopy(create_input((2, 3, 4), np.float64, broadcast_input=(3, 4), broadcast_other=())))
    
    # Input 7: Broadcasting with float32, other input is a scalar
    list_of_inputs.append(copy.deepcopy(create_input((4, 2), np.float32, broadcast_input=(), broadcast_other=(4, 2))))
    
    # Input 8: Higher dimensions (4D) with float32
    list_of_inputs.append(copy.deepcopy(create_input((2, 3, 2, 4), np.float32)))
    
    # Input 9: All True condition with int32
    input_dict_true = create_input((3, 3), np.int32)
    input_dict_true["condition"] = np.ones((3, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy(input_dict_true))

    # Input 10: All False condition with float64
    input_dict_false = create_input((2, 6), np.float64)
    input_dict_false["condition"] = np.zeros((2, 6), dtype=bool)
    list_of_inputs.append(copy.deepcopy(input_dict_false))

    return list_of_inputs

generated_inputs["torch.where_2"] = where_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.where_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.where_2'.")

check_valid('torch.where', generated_inputs['torch.where_2'], lib="torch", suffix=2)
