
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_nth_element_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    input_tensor = np.array([[1, 4, 2, 3], [5, 8, 6, 7]], dtype=np.int32)
    n_tensor = np.array(1, dtype=np.int32)
    reverse_bool = False
    name_str = "basic_int"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with floats
    input_tensor = np.array([[1.0, 4.0, 2.0, 3.0], [5.0, 8.0, 6.0, 7.0]], dtype=np.float32)
    n_tensor = np.array(2, dtype=np.int32)
    reverse_bool = False
    name_str = "basic_float"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Reverse is True
    input_tensor = np.array([[1, 4, 2, 3], [5, 8, 6, 7]], dtype=np.int32)
    n_tensor = np.array(1, dtype=np.int32)
    reverse_bool = True
    name_str = "reverse_true"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D input
    input_tensor = np.array([1, 4, 2, 3], dtype=np.int32)
    n_tensor = np.array(1, dtype=np.int32)
    reverse_bool = False
    name_str = "1d_input"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type (int64)
    input_tensor = np.array([[1, 4, 2, 3], [5, 8, 6, 7]], dtype=np.int64)
    n_tensor = np.array(1, dtype=np.int32)
    reverse_bool = False
    name_str = "int64_type"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Different data type (uint64)
    input_tensor = np.array([[1, 4, 2, 3], [5, 8, 6, 7]], dtype=np.uint64)
    n_tensor = np.array(1, dtype=np.int32)
    reverse_bool = False
    name_str = "uint64_type"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D input
    input_tensor = np.array([[[1, 4, 2, 3], [5, 8, 6, 7]], [[9, 12, 10, 11], [13, 16, 14, 15]]], dtype=np.int32)
    n_tensor = np.array(1, dtype=np.int32)
    reverse_bool = False
    name_str = "3d_input"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Different data type (float64)
    input_tensor = np.array([[1.0, 4.0, 2.0, 3.0], [5.0, 8.0, 6.0, 7.0]], dtype=np.float64)
    n_tensor = np.array(2, dtype=np.int32)
    reverse_bool = False
    name_str = "float64_type"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    input_tensor = np.array([[1, 4, 2, 3], [5, 8, 6, 7]], dtype=np.uint8)
    n_tensor = np.array(1, dtype=np.int32)
    reverse_bool = False
    name_str = "uint8_type"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half
    input_tensor = np.array([[1.0, 4.0, 2.0, 3.0], [5.0, 8.0, 6.0, 7.0]], dtype=np.float16)
    n_tensor = np.array(2, dtype=np.int32)
    reverse_bool = False
    name_str = "half_type"

    input_dict = {
        "input": input_tensor,
        "n": n_tensor,
        "reverse": reverse_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NthElement"] = tf_raw_ops_nth_element_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NthElement' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NthElement'.")

check_valid('tf.raw_ops.NthElement', generated_inputs['tf.raw_ops.NthElement'], lib="tf", suffix=0)
