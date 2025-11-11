
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_convert_to_tensor_inputs():
    list_of_inputs = []
    
    input_dict = {
        "value": [1, 2, 3, 4, 5],
        "dtype": np.int32,
        "dtype_hint": None,
        "name": "tensor_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [[1.0, 2.0], [3.0, 4.0]],
        "dtype": np.float32,
        "dtype_hint": None,
        "name": "tensor_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        "dtype": np.float64,
        "dtype_hint": None,
        "name": "tensor_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [-1, -2, -3, 0, 1, 2],
        "dtype": np.int64,
        "dtype_hint": None,
        "name": "tensor_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [5.5],
        "dtype": np.float32,
        "dtype_hint": None,
        "name": "tensor_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [True, False, True, False],
        "dtype": np.bool_,
        "dtype_hint": None,
        "name": "tensor_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [[1, 2, 3], [4, 5, 6]],
        "dtype": None,
        "dtype_hint": np.int32,
        "name": "tensor_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [[-1.5, 2.3], [-4.7, 5.1]],
        "dtype": np.float64,
        "dtype_hint": None,
        "name": "tensor_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]],
        "dtype": np.int16,
        "dtype_hint": None,
        "name": "tensor_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": [0, 1, 2],
        "dtype": np.uint8,
        "dtype_hint": None,
        "name": "tensor_11"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.convert_to_tensor_2"] = tf_convert_to_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.convert_to_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.convert_to_tensor_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.convert_to_tensor', generated_inputs['tf.convert_to_tensor_2'], lib="tf", suffix=2)
