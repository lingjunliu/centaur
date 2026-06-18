
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_boolean_mask_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "tensor": np.array([10, 20, 30], dtype=np.int64),
        "mask": np.array([True, False, True], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "tensor": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "mask": np.array([False, True], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "tensor": np.array([[[1], [2]], [[3], [4]]], dtype=np.int32),
        "mask": np.array([True, True], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "tensor": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "mask": np.array([[True, False], [True, True]], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "tensor": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "mask": np.array([[True, False], [False, True]], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "tensor": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32),
        "mask": np.array([True, False, True], dtype=bool),
        "axis": np.array(1, dtype=np.int32),
        "name": "case6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "tensor": np.array([1.5, -2.5, 3.5], dtype=np.float64),
        "mask": np.array([True, True, False], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "tensor": np.array([[[1.1, 1.2], [1.3, 1.4]], [[2.1, 2.2], [2.3, 2.4]]], dtype=np.float32),
        "mask": np.array([True, False], dtype=bool),
        "axis": np.array(1, dtype=np.int32),
        "name": "case8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "tensor": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "mask": np.array([True, True, False, False], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "tensor": np.arange(24).reshape(2, 3, 4).astype(np.int32),
        "mask": np.array([True, False, True], dtype=bool),
        "axis": np.array(1, dtype=np.int32),
        "name": "case10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.boolean_mask"] = tf_boolean_mask_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.boolean_mask' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.boolean_mask'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.boolean_mask', generated_inputs['tf.boolean_mask'], lib="tf", suffix=0)
