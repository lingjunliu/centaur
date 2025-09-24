
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Acos_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0]).astype(np.float32)
    input_dict = {"x": x, "name": "acos_basic_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64
    x = np.array([0.2, -0.3, 0.8, -0.9]).astype(np.float64)
    input_dict = {"x": x, "name": "acos_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Half
    x = np.array([0.1, -0.6, 0.4, -0.2]).astype(np.float16)
    input_dict = {"x": x, "name": "acos_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64
    x = np.array([0.1 + 0.1j, -0.2 - 0.2j, 0.3 - 0.3j, -0.4 + 0.4j]).astype(np.complex64)
    input_dict = {"x": x, "name": "acos_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128
    x = np.array([0.5 + 0.5j, -0.6 - 0.6j, 0.7 - 0.7j, -0.8 + 0.8j]).astype(np.complex128)
    input_dict = {"x": x, "name": "acos_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional float32
    x = np.array([[0.1, 0.2], [-0.3, -0.4]]).astype(np.float32)
    input_dict = {"x": x, "name": "acos_multi_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.5, -0.6], [-0.7, -0.8]]]).astype(np.float32)
    input_dict = {"x": x, "name": "acos_3d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float32 with values close to 1 and -1
    x = np.array([0.99, -0.99, 0.999, -0.999]).astype(np.float32)
    input_dict = {"x": x, "name": "acos_extreme_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Empty array float32
    x = np.array([]).astype(np.float32)
    input_dict = {"x": x, "name": "acos_empty_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Acos"] = tf_raw_ops_Acos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Acos'.")

check_valid('tf.raw_ops.Acos', generated_inputs['tf.raw_ops.Acos'], lib="tf", suffix=0)
