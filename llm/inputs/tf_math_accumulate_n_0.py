
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_accumulate_n_inputs():
    list_of_inputs = []

    def create_input_dict(inputs, shape, tensor_dtype, name):
        return {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}

    # Input 1: Basic example with integers
    inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    shape = [2, 2]
    tensor_dtype = np.int32
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, shape, tensor_dtype, "accumulate_basic")))

    # Input 2: Different shape, float dtype
    inputs = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0])]
    shape = [3]
    tensor_dtype = np.float32
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, shape, tensor_dtype, "accumulate_float")))

    # Input 3: 3D array, int64
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    shape = [2, 2, 2]
    tensor_dtype = np.int64
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, shape, tensor_dtype, "accumulate_3d")))

    # Input 4: Negative values, int16
    inputs = [np.array([[-1, -2], [-3, -4]]), np.array([[5, 6], [7, 8]])]
    shape = [2, 2]
    tensor_dtype = np.int16
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, shape, tensor_dtype, "accumulate_negative")))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.accumulate_n"] = tf_math_accumulate_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.accumulate_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.accumulate_n'.")

check_valid('tf.math.accumulate_n', generated_inputs['tf.math.accumulate_n'], lib="tf", suffix=0)
