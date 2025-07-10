
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_erfinv_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32 within range
    x = np.float32(0.5)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar float64 within range
    x = np.float64(-0.8)
    name = "erfinv_input_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32 array
    x = np.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=np.float32)
    name = "erfinv_input_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array
    x = np.array([[-0.7, -0.3], [0.3, 0.7]], dtype=np.float64)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array
    x = np.random.uniform(-0.99, 0.99, size=(2, 2, 2)).astype(np.float32)
    name = "erfinv_input_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Tensor with a different shape and name
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).reshape((2, 2))
    name = "input_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger array with float64 type
    x = np.random.uniform(-0.95, 0.95, size=(5, 5)).astype(np.float64)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array containing both positive and negative values close to the limits
    x = np.array([-0.99, 0.99, -0.01, 0.01], dtype=np.float32)
    name = "input_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Scalar close to the limit with float32
    x = np.float32(0.999)
    name = "input_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Scalar close to the limit with float64 and a long name
    x = np.float64(-0.99999)
    name = "a_very_long_and_descriptive_name"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.erfinv"] = tf_math_erfinv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.erfinv'.")

check_valid('tf.math.erfinv', generated_inputs['tf.math.erfinv'], lib="tf", suffix=0)
