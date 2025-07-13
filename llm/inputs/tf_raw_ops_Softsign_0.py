
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_softsign_inputs():
    list_of_inputs = []

    # Input 1: float32, single element
    features = np.array(1.0, dtype=np.float32)
    name = "softsign_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, multiple elements, positive
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "softsign_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, multiple elements, negative
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "softsign_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, mixed positive and negative
    features = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    name = "softsign_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, positive
    features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "softsign_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, negative
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    name = "softsign_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, positive
    features = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    name = "softsign_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: float16, negative
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float16)
    name = "softsign_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16, positive
    features = np.array([1.0, 2.0, 3.0], dtype=tf.bfloat16.as_numpy_dtype)
    name = "softsign_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16, negative
    features = np.array([-1.0, -2.0, -3.0], dtype=tf.bfloat16.as_numpy_dtype)
    name = "softsign_10"
    input_dict = {"features": features, "name": name}
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
