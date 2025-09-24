
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_relu6_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array, positive values
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array, negative and positive values
    features = np.array([-1.0, 2.0, -3.0, 4.0, -5.0]).astype(np.float32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, 2D array, with negative values
    features = np.array([[-1.0, 2.0], [-3.0, 4.0], [-5.0, -6.0]]).astype(np.float32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D array
    features = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32, 1D array
    features = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32, 1D array, negative values
    features = np.array([-1, -2, -3, -4, -5]).astype(np.int32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64, 2D array
    features = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int64)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, scalar
    features = np.array(3.14159).astype(np.float64)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16, 1D array
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float16)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Relu6"] = tf_raw_ops_relu6_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Relu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Relu6'.")

check_valid('tf.raw_ops.Relu6', generated_inputs['tf.raw_ops.Relu6'], lib="tf", suffix=0)
