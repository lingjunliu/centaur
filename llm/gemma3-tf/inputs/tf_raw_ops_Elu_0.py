
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_elu_inputs():
    list_of_inputs = []

    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "elu_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "elu_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "elu_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.0], dtype=np.float64)
    name = "elu_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    name = "elu_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[1.0], [-1.0]], [[2.0], [-2.0]]], dtype=np.float32)
    name = "elu_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1000.0, -1000.0], dtype=np.float32)
    name = "elu_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    name = "elu_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([2.71828, -2.71828], dtype=np.float32)
    name = "elu_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    name = "elu_11"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_elu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Elu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Elu', generated_inputs['tf.raw_ops.Elu'], lib="tf", suffix=0)
