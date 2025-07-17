
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_softsign_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    features = np.array(1.0, dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    features = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    features = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    features = np.array(-2.0, dtype=np.float64)
    input_dict = {"features": features, "name": "softsign_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, 1D array
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {"features": features, "name": "softsign_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 2D array
    features = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    input_dict = {"features": features, "name": "softsign_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half, scalar
    features = np.array(0.5, dtype=np.float16)
    input_dict = {"features": features, "name": "softsign_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, 1D array
    features = np.array([-0.5, 0.0, 0.5], dtype=np.float16)
    input_dict = {"features": features, "name": "softsign_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, 3D array
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, 3D array
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"features": features, "name": "softsign_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Softsign"] = tf_raw_ops_softsign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Softsign'.")

check_valid('tf.raw_ops.Softsign', generated_inputs['tf.raw_ops.Softsign'], lib="tf", suffix=0)
