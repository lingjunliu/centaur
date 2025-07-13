
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Erf_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array, positive values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 2D array, mixed values
    x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 1D array, negative values
    x = np.array([-1.5, -2.5, -3.5], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar value
    x = np.array(0.75, dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bfloat16, 1D array
    x = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half (float16), 2D array
    x = np.array([[-0.5, 0.5], [1.5, -1.5]], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float64, 3D array, negative and positive values
    x = np.random.uniform(-5, 5, size=(2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, array with zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, large values
    x = np.array([100.0, -100.0], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Erf"] = tf_raw_ops_Erf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erf'.")

check_valid('tf.raw_ops.Erf', generated_inputs['tf.raw_ops.Erf'], lib="tf", suffix=0)
