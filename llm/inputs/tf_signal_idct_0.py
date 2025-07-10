
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_idct_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    type_val = 2
    n_val = None
    axis_val = -1
    norm_val = None
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    type_val = 3
    n_val = None
    axis_val = -1
    norm_val = 'ortho'
    name_val = "idct_op"

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    type_val = 1
    n_val = None
    axis_val = -1
    norm_val = None
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    type_val = 2
    n_val = None
    axis_val = -1
    norm_val = 'ortho'
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    type_val = 3
    n_val = None
    axis_val = -1
    norm_val = None
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_tensor = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    type_val = 4
    n_val = None
    axis_val = -1
    norm_val = 'ortho'
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
    type_val = 2
    n_val = None
    axis_val = -1
    norm_val = None
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    type_val = 3
    n_val = None
    axis_val = -1
    norm_val = 'ortho'
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1,2,3], dtype=np.float32)
    type_val = 1
    n_val = None
    axis_val = -1
    norm_val = None
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1,2,3], dtype=np.float64)
    type_val = 4
    n_val = None
    axis_val = -1
    norm_val = "ortho"
    name_val = None

    input_dict = {
        "input": input_tensor,
        "type": type_val,
        "n": n_val,
        "axis": axis_val,
        "norm": norm_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.idct"] = tf_signal_idct_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.idct' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.idct'.")

check_valid('tf.signal.idct', generated_inputs['tf.signal.idct'], lib="tf", suffix=0)
