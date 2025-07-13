
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_selu_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    features = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    features = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    input_dict = {"features": features, "name": "selu_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, 3D array
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float16)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, scalar
    features = np.array(-2.5, dtype=np.float32)
    input_dict = {"features": features, "name": "scalar_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, large values
    features = np.array([100.0, -100.0, 0.0], dtype=np.float32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, all zeros
    features = np.zeros((2, 2), dtype=np.float64)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, negative values
    features = np.array([-0.1, -0.5, -1.0], dtype=np.float16)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, positive values
    features = np.array([0.1, 0.5, 1.0], dtype=np.float32)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, mixed values
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"features": features, "name": "mixed_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, 1D with large positive and negative values
    features = np.array([-1000.0, 1000.0], dtype=np.float64)
    input_dict = {"features": features, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Selu"] = tf_raw_ops_selu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Selu'.")

check_valid('tf.raw_ops.Selu', generated_inputs['tf.raw_ops.Selu'], lib="tf", suffix=0)
