
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_asin_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "name": "asin_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 1D array
    x = np.array([-0.2, 0.0, 0.2, 0.4], dtype=np.float64)
    input_dict = {"x": x, "name": "asin_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, 2D array
    x = np.array([[-0.8, 0.1], [0.3, 0.9]], dtype=np.float16)
    input_dict = {"x": x, "name": "asin_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 3D array
    x = np.array([[[0.5, -0.5], [0.2, -0.2]], [[0.8, -0.8], [0.9, -0.9]]], dtype=np.float16)
    input_dict = {"x": x, "name": "asin_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, scalar
    x = np.complex64(0.5 + 0.5j)
    input_dict = {"x": x, "name": "asin_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 1D array
    x = np.array([0.1 + 0.1j, 0.2 - 0.2j, -0.3 + 0.3j], dtype=np.complex128)
    input_dict = {"x": x, "name": "asin_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, values close to 1 and -1
    x = np.array([0.999, -0.999], dtype=np.float32)
    input_dict = {"x": x, "name": "asin_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, larger array
    x = np.random.uniform(-1, 1, size=(5, 5)).astype(np.float64)
    input_dict = {"x": x, "name": "asin_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: complex64, array with zero imaginary part
    x = np.array([0.2+0j, 0.5+0j, -0.1+0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "asin_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: bfloat16, single negative value
    x = np.array([-0.75], dtype=np.float16)
    input_dict = {"x": x, "name": "asin_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Asin"] = tf_raw_ops_asin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Asin'.")

check_valid('tf.raw_ops.Asin', generated_inputs['tf.raw_ops.Asin'], lib="tf", suffix=0)
