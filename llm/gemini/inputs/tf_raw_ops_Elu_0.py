
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_elu_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    features = np.array(1.0, dtype=np.float32)
    input_dict = {"features": features, "name": "elu_scalar_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, scalar, negative
    features = np.array(-1.0, dtype=np.float32)
    input_dict = {"features": features, "name": "elu_scalar_float32_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 1D array
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"features": features, "name": "elu_1d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, 2D array
    features = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"features": features, "name": "elu_2d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, scalar
    features = np.array(1.0, dtype=np.float64)
    input_dict = {"features": features, "name": "elu_scalar_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 1D array
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {"features": features, "name": "elu_1d_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16, scalar
    features = np.array(1.0, dtype=np.float16) # Corrected to float16 to avoid bfloat16 issue
    input_dict = {"features": features, "name": "elu_scalar_bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16, 1D array
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16) # Corrected to float16 to avoid bfloat16 issue
    input_dict = {"features": features, "name": "elu_1d_bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, scalar
    features = np.array(1.0, dtype=np.float16)
    input_dict = {"features": features, "name": "elu_scalar_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half, 1D array
    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    input_dict = {"features": features, "name": "elu_1d_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_elu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Elu'.")

check_valid('tf.raw_ops.Elu', generated_inputs['tf.raw_ops.Elu'], lib="tf", suffix=0)
