
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Erfinv_inputs():
    list_of_inputs = []

    # Input 1: float32, simple
    x = np.array([0.5], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, negative values
    x = np.array([-0.5, -0.25, 0, 0.25, 0.5], dtype=np.float64)
    input_dict = {"x": x, "name": "erfinv_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, multiple dimensions
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, larger values (within range)
    x = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    input_dict = {"x": x, "name": "erfinv_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, more negative values
    x = np.array([-0.9, -0.8, -0.7], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, zero value
    x = np.array([0.0], dtype=np.float64)
    input_dict = {"x": x, "name": "erfinv_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D tensor
    x = np.random.uniform(-0.9, 0.9, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, a broader range of values
    x = np.random.uniform(-0.9, 0.9, size=(5,)).astype(np.float64)
    input_dict = {"x": x, "name": "erfinv_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, large array
    x = np.random.uniform(-0.9, 0.9, size=(100,)).astype(np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, a more complex tensor structure
    x = np.random.uniform(-0.9, 0.9, size=(2, 2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "name": "erfinv_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Erfinv"] = tf_raw_ops_Erfinv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erfinv'.")

check_valid('tf.raw_ops.Erfinv', generated_inputs['tf.raw_ops.Erfinv'], lib="tf", suffix=0)
