
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_softsign_inputs():
    list_of_inputs = []

    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "softsign_test_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "softsign_test_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "softsign_test_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.0], dtype=np.float64)
    name = "softsign_test_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "softsign_test_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    name = "softsign_test_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.5, -2.5, 3.5], dtype=np.float16)
    name = "softsign_test_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    name = "softsign_test_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([100.0, -100.0, 0.0], dtype=np.float32)
    name = "softsign_test_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    name = "softsign_test_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.softsign"] = tf_nn_softsign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.softsign'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.softsign', generated_inputs['tf.nn.softsign'], lib="tf", suffix=0)
